import pandas as pd
import numpy as np
from numba import njit


@njit
def _find_abcd_segments(
    high: np.ndarray,
    low: np.ndarray,
    seg: int = 7,  # 每段 K 根數，建議 5~9
):
    """
    用固定分段高低點找 ABCD：
        seg1：A = 最低點，B = 最高點
        seg2：C = 最低點（C > A，否則結構無效）
        seg3：D = 最高點，D > B 則 N 字成立

    每根 bar 往回看三段（3 * seg 根），計算當下的 ABCD。

    Returns
    -------
    A_arr, B_arr, C_arr, D_arr, valid_arr : np.ndarray shape=(T,)
        valid = 1 表示 C > A 且 D > B（N 字成立）
    """
    T = len(high)
    A_arr = np.full(T, np.nan)
    B_arr = np.full(T, np.nan)
    C_arr = np.full(T, np.nan)
    D_arr = np.full(T, np.nan)
    valid_arr = np.zeros(T)

    total = seg * 3  # 需要回看的總根數

    for i in range(total - 1, T):
        # seg1：最早的一段 → A（低）、B（高）
        seg1_high = high[i - total + 1 : i - 2 * seg + 1]
        seg1_low = low[i - total + 1 : i - 2 * seg + 1]

        # seg2：中間一段 → C（低）
        seg2_high = high[i - 2 * seg + 1 : i - seg + 1]
        seg2_low = low[i - 2 * seg + 1 : i - seg + 1]

        # seg3：最近一段 → D（高）
        seg3_high = high[i - seg + 1 : i + 1]
        seg3_low = low[i - seg + 1 : i + 1]

        A = np.min(seg1_low)
        B = np.max(seg1_high)
        C = np.min(seg2_low)
        D = np.max(seg3_high)

        A_arr[i] = A
        B_arr[i] = B
        C_arr[i] = C
        D_arr[i] = D

        # N 字成立條件：C > A（不跌破起漲點）且 D > B（突破前高）
        if C > A and D > B:
            valid_arr[i] = 1.0

    return A_arr, B_arr, C_arr, D_arr, valid_arr


def detect_n_shape_features(
    df: pd.DataFrame,
    seg: int = 7,  # 每段 K 根數，可調整 5~9
) -> pd.DataFrame:
    """
    用固定分段高低點計算 N 字型特徵。

    每根 bar 往回看三段（3 * seg 根）：
        segment 1（最早）→ A 低點、B 高點
        segment 2（中間）→ C 低點
        segment 3（最近）→ D 高點

    N 字成立條件：C > A 且 D > B

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close, high, low, volume
    seg : int
        每段的 K 根數，建議 5~9（預設 7）

    Returns
    -------
    pd.DataFrame
        原始長表格 + f_ 開頭特徵欄位
    """
    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")
    volume_wide = df.pivot(index="date", columns="stock_id", values="volume")

    stocks = close_wide.columns.tolist()
    dates = close_wide.index

    # 每檔跑一次分段計算
    res = {}
    for sid in stocks:
        h = high_wide[sid].values.astype(np.float64)
        l = low_wide[sid].values.astype(np.float64)
        A_arr, B_arr, C_arr, D_arr, valid_arr = _find_abcd_segments(h, l, seg)
        res[sid] = {
            "A": A_arr,
            "B": B_arr,
            "C": C_arr,
            "D": D_arr,
            "valid": valid_arr,
        }

    def to_wide(key):
        return pd.DataFrame({sid: res[sid][key] for sid in stocks}, index=dates)

    A_wide = to_wide("A")
    B_wide = to_wide("B")
    C_wide = to_wide("C")
    D_wide = to_wide("D")
    valid_wide = to_wide("valid").astype(bool)

    # ── 特徵計算 ──────────────────────────────────────────────────────── #

    # AB 漲幅（第一波力道）
    f_n_ab_gain = (B_wide - A_wide) / (A_wide + 1e-9)

    # BC 回調比例（相對 AB 幅度，黃金回調 0.38~0.62）
    ab_range = (B_wide - A_wide).clip(lower=1e-9)
    f_n_bc_retracement = (B_wide - C_wide) / ab_range

    # C 高於 A 的幅度（越高結構越健康）
    f_n_c_above_a = (C_wide - A_wide) / (A_wide + 1e-9)

    # D 突破 B 的力道
    f_n_d_breakout_strength = (D_wide - B_wide) / (B_wide + 1e-9)

    # CD 漲幅 vs AB 漲幅（> 1 代表第二波更強）
    cd_gain = (D_wide - C_wide) / (C_wide + 1e-9)
    f_n_cd_vs_ab_momentum = cd_gain / (f_n_ab_gain + 1e-9)

    # 收盤相對 B 的位置（> 0 代表站上前高）
    f_n_close_vs_b = (close_wide - B_wide) / (B_wide + 1e-9)

    # 綜合品質分（只在 N 字成立時有值）
    f_n_structure_score = (
        (1 - f_n_bc_retracement.clip(0, 1)) * 0.3 + f_n_c_above_a.clip(0) * 0.3 + f_n_d_breakout_strength.clip(0) * 0.4
    ).where(valid_wide)

    # N 字成立旗標
    f_n_confirmed = valid_wide.astype(float)

    # 突破段量 vs 回調段量
    vol_seg3 = volume_wide.rolling(seg).mean()
    vol_seg2 = volume_wide.rolling(seg).mean().shift(seg)
    f_n_volume_confirm = (vol_seg3 / (vol_seg2 + 1e-9)).where(valid_wide)

    # ── 有效結構 mask：所有特徵在 N 字不成立時設為 NaN ──────────────── #
    feature_wides_raw = {
        "f_n_ab_gain": f_n_ab_gain,
        "f_n_bc_retracement": f_n_bc_retracement,
        "f_n_c_above_a": f_n_c_above_a,
        "f_n_d_breakout_strength": f_n_d_breakout_strength,
        "f_n_cd_vs_ab_momentum": f_n_cd_vs_ab_momentum,
        "f_n_close_vs_b": f_n_close_vs_b,
        "f_n_structure_score": f_n_structure_score,
        "f_n_confirmed": f_n_confirmed,
        "f_n_volume_confirm": f_n_volume_confirm,
    }

    feature_wides = {name: feat.where(valid_wide) for name, feat in feature_wides_raw.items()}
    # f_n_confirmed 本身就是 0/1，不需要遮
    feature_wides["f_n_confirmed"] = f_n_confirmed

    # ── 整合回長表格 ──────────────────────────────────────────────────── #
    feature_longs = []
    for feat_name, feat_wide in feature_wides.items():
        feat_long = feat_wide.stack().reset_index()
        feat_long.columns = ["date", "stock_id", feat_name]
        feature_longs.append(feat_long.set_index(["date", "stock_id"]))

    features_df = pd.concat(feature_longs, axis=1).reset_index()
    result = df.merge(features_df, on=["date", "stock_id"], how="left")
    return result


import pandas as pd
import numpy as np


def detect_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    趨勢相關特徵，用來輔助 N 字型訊號過濾。
    N 字在上升趨勢裡成功率更高，這些特徵幫助模型判斷趨勢強弱。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close, high, low, volume

    Returns
    -------
    pd.DataFrame
        原始長表格 + f_trend_ 開頭特徵欄位
    """
    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")
    volume_wide = df.pivot(index="date", columns="stock_id", values="volume")

    # ── Feature 1: f_trend_ma20_slope ────────────────────────────────── #
    # MA20 斜率：(今日MA20 - N日前MA20) / N日前MA20
    # 正值代表均線向上，趨勢健康
    ma20 = close_wide.rolling(20).mean()
    f_trend_ma20_slope = (ma20 - ma20.shift(5)) / (ma20.shift(5) + 1e-9)

    # ── Feature 2: f_trend_close_vs_ma20 ─────────────────────────────── #
    # 收盤相對 MA20 的位置
    # > 0 代表站上均線，N 字在均線上方成功率更高
    f_trend_close_vs_ma20 = (close_wide - ma20) / (ma20 + 1e-9)

    # ── Feature 3: f_trend_close_vs_ma60 ─────────────────────────────── #
    # 收盤相對 MA60 的位置（中期趨勢）
    ma60 = close_wide.rolling(60).mean()
    f_trend_close_vs_ma60 = (close_wide - ma60) / (ma60 + 1e-9)

    # ── Feature 4: f_trend_ma20_vs_ma60 ──────────────────────────────── #
    # 短均線 vs 長均線（黃金交叉/死亡交叉）
    # > 0 代表 MA20 在 MA60 上方，多頭排列
    f_trend_ma20_vs_ma60 = (ma20 - ma60) / (ma60 + 1e-9)

    # ── Feature 5: f_trend_high_breakout ─────────────────────────────── #
    # 近期是否創 N 日新高（突破前高是上升趨勢的特徵）
    rolling_high_60 = high_wide.rolling(60).max().shift(1)
    f_trend_high_breakout = (close_wide - rolling_high_60) / (rolling_high_60 + 1e-9)

    # ── Feature 6: f_trend_volume_trend ──────────────────────────────── #
    # 成交量趨勢：近期均量 vs 前期均量
    # > 1 代表量能放大，趨勢有支撐
    vol_ma10 = volume_wide.rolling(10).mean()
    vol_ma30 = volume_wide.rolling(30).mean()
    f_trend_volume_trend = vol_ma10 / (vol_ma30 + 1e-9)

    # ── Feature 7: f_trend_atr_ratio ─────────────────────────────────── #
    # ATR 比率：近期波動 vs 長期波動
    # N 字啟動時通常波動放大
    daily_range = high_wide - low_wide
    atr10 = daily_range.rolling(10).mean()
    atr30 = daily_range.rolling(30).mean()
    f_trend_atr_ratio = atr10 / (atr30 + 1e-9)

    # ── Feature 8: f_trend_momentum_20 ───────────────────────────────── #
    # 20日動能（價格動量）
    f_trend_momentum_20 = close_wide.pct_change(20)

    # ── Feature 9: f_trend_rsi_14 ────────────────────────────────────── #
    # RSI(14)：衡量超買超賣，N 字啟動時 RSI 通常從 40~60 向上
    delta = close_wide.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    f_trend_rsi_14 = 100 - (100 / (1 + rs))

    # ── Feature 10: f_trend_above_ma_days ────────────────────────────── #
    # 近 20 根中有幾根收盤在 MA20 以上（趨勢穩定度）
    above_ma20 = (close_wide > ma20).astype(float)
    f_trend_above_ma_days = above_ma20.rolling(20).sum() / 20

    # ── 整合回長表格 ──────────────────────────────────────────────────── #
    feature_wides = {
        "f_trend_ma20_slope": f_trend_ma20_slope,
        "f_trend_close_vs_ma20": f_trend_close_vs_ma20,
        "f_trend_close_vs_ma60": f_trend_close_vs_ma60,
        "f_trend_ma20_vs_ma60": f_trend_ma20_vs_ma60,
        "f_trend_high_breakout": f_trend_high_breakout,
        "f_trend_volume_trend": f_trend_volume_trend,
        "f_trend_atr_ratio": f_trend_atr_ratio,
        "f_trend_momentum_20": f_trend_momentum_20,
        "f_trend_rsi_14": f_trend_rsi_14,
        "f_trend_above_ma_days": f_trend_above_ma_days,
    }

    feature_longs = []
    for feat_name, feat_wide in feature_wides.items():
        feat_long = feat_wide.stack().reset_index()
        feat_long.columns = ["date", "stock_id", feat_name]
        feature_longs.append(feat_long.set_index(["date", "stock_id"]))

    features_df = pd.concat(feature_longs, axis=1).reset_index()
    result = df.merge(features_df, on=["date", "stock_id"], how="left")
    return result
