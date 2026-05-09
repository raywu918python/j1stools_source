import pandas as pd
import numpy as np


def detect_n_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基於標準 N 字型結構產生特徵：
        A → B 上漲
        B → C 回調（C > A）
        C → D 上漲，D 突破 B → 確認 N 字（D1）
        D1 之後為做多訊號區

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close, high, low, volume

    Returns
    -------
    pd.DataFrame
        原始長表格 + f_ 開頭特徵欄位
    """
    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")
    volume_wide = df.pivot(index="date", columns="stock_id", values="volume")

    # 用來近似 A B C D 四個點的窗口設定
    w_ab = 10  # A→B 上漲段長度
    w_bc = 10  # B→C 回調段長度
    w_cd = 10  # C→D 再漲段長度
    w_total = w_ab + w_bc + w_cd  # 整個 N 字視窗 = 30 根

    # ── 近似四個關鍵價位 ──────────────────────────────────────────────── #
    # B：w_ab 段內的最高點（第一波高點）
    B = high_wide.shift(w_bc + w_cd).rolling(w_ab).max()

    # A：B 之前的最低點（起漲點）
    A = low_wide.shift(w_bc + w_cd + w_ab).rolling(w_ab).min()

    # C：B 之後回調段的最低點
    C = low_wide.shift(w_cd).rolling(w_bc).min()

    # D：當前 C→D 段的最高點（今天附近）
    D = high_wide.rolling(w_cd).max()

    # 當前收盤
    cur_close = close_wide

    # ── 核心條件：D1 確認 ─────────────────────────────────────────────── #
    # D 突破 B，且 C > A（N 字結構成立）
    d1_confirmed = (D > B) & (C > A)

    # ── Feature 1: f_n_ab_gain ────────────────────────────────────────── #
    # A→B 的漲幅，衡量第一波力道
    f_n_ab_gain = (B - A) / (A + 1e-9)

    # ── Feature 2: f_n_bc_retracement ────────────────────────────────── #
    # B→C 回調比例 = (B - C) / (B - A)
    # 黃金回調約 0.382~0.618，太深代表結構可能破壞
    ab_range = (B - A).clip(lower=1e-9)
    f_n_bc_retracement = (B - C) / ab_range

    # ── Feature 3: f_n_c_above_a ─────────────────────────────────────── #
    # C 比 A 高多少（N 字必要條件：C > A）
    f_n_c_above_a = (C - A) / (A + 1e-9)

    # ── Feature 4: f_n_d_breakout_strength ───────────────────────────── #
    # D 突破 B 的幅度，越大代表突破越有力
    f_n_d_breakout_strength = (D - B) / (B + 1e-9)

    # ── Feature 5: f_n_cd_vs_ab_momentum ─────────────────────────────── #
    # C→D 漲幅 vs A→B 漲幅的比值
    # > 1 代表第二波力道強於第一波（加速上漲）
    cd_gain = (D - C) / (C + 1e-9)
    f_n_cd_vs_ab_momentum = cd_gain / (f_n_ab_gain + 1e-9)

    # ── Feature 6: f_n_close_vs_b ────────────────────────────────────── #
    # 當前收盤相對 B 點的位置
    # > 0 代表已站上 B（D1 之後持續強勢）
    f_n_close_vs_b = (cur_close - B) / (B + 1e-9)

    # ── Feature 7: f_n_structure_score ───────────────────────────────── #
    # 綜合 N 字品質分數（只在 d1_confirmed 時有意義）
    # 回調淺 + C>A 幅度大 + 突破力道強
    f_n_structure_score = (
        (1 - f_n_bc_retracement.clip(0, 1)) * 0.3  # 回調越淺越好
        + f_n_c_above_a.clip(0) * 0.3  # C 高於 A 越多越好
        + f_n_d_breakout_strength.clip(0) * 0.4  # 突破力道
    ) * d1_confirmed.astype(
        float
    )  # 未確認 N 字則為 0

    # ── Feature 8: f_n_d1_confirmed ──────────────────────────────────── #
    # D1 確認旗標（布林 → 0/1）
    f_n_d1_confirmed = d1_confirmed.astype(float)

    # ── Feature 9: f_n_volume_on_breakout ────────────────────────────── #
    # 突破 B 當段的成交量相對 BC 回調段平均量的放大倍數
    vol_cd = volume_wide.rolling(w_cd).mean()
    vol_bc = volume_wide.shift(w_cd).rolling(w_bc).mean()
    f_n_volume_on_breakout = vol_cd / (vol_bc + 1e-9)

    # ── Feature 10: f_n_days_since_d1 ────────────────────────────────── #
    # D1 確認後已過幾根（越近越新鮮）
    # 用 cumsum trick：確認日 reset，否則累加
    def days_since_signal(bool_wide: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bool_wide.index, columns=bool_wide.columns, dtype=float)
        arr = bool_wide.values.astype(float)
        out = np.full_like(arr, np.nan)
        counter = np.full(arr.shape[1], np.nan)
        for i in range(len(arr)):
            signal = arr[i]
            counter = np.where(signal == 1, 0, np.where(np.isnan(counter), np.nan, counter + 1))
            out[i] = counter
        return pd.DataFrame(out, index=bool_wide.index, columns=bool_wide.columns)

    f_n_days_since_d1 = days_since_signal(d1_confirmed)

    # ── 整合回長表格 ──────────────────────────────────────────────────── #
    feature_wides = {
        "f_n_ab_gain": f_n_ab_gain,
        "f_n_bc_retracement": f_n_bc_retracement,
        "f_n_c_above_a": f_n_c_above_a,
        "f_n_d_breakout_strength": f_n_d_breakout_strength,
        "f_n_cd_vs_ab_momentum": f_n_cd_vs_ab_momentum,
        "f_n_close_vs_b": f_n_close_vs_b,
        "f_n_structure_score": f_n_structure_score,
        "f_n_d1_confirmed": f_n_d1_confirmed,
        "f_n_volume_on_breakout": f_n_volume_on_breakout,
        "f_n_days_since_d1": f_n_days_since_d1,
    }

    feature_longs = []
    for feat_name, feat_wide in feature_wides.items():
        feat_long = feat_wide.stack().reset_index().rename(columns={0: feat_name})
        feature_longs.append(feat_long.set_index(["date", "stock_id"]))

    features_df = pd.concat(feature_longs, axis=1).reset_index()
    result = df.merge(features_df, on=["date", "stock_id"], how="left")
    return result
