import pandas as pd
import numpy as np
from numba import njit

# ── 狀態定義 ──────────────────────────────────────────────────────────────── #
IDLE = 0  # 等待起漲
AB = 1  # A→B 上漲段
BC = 2  # B→C 回調段
CD = 3  # C→D 再漲段
D1 = 4  # N 字確認後


@njit
def _run_state_machine(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
    """
    逐 bar 執行 N 字型狀態機，回傳每根 bar 的狀態與 ABCD 關鍵價位。

    破壞條件：
      - 任何階段跌破 A → reset 回 IDLE
      - CD 段（尚未突破 B）若跌破 C → reset 回 IDLE

    Parameters
    ----------
    high, low, close, volume : np.ndarray  shape=(T,)

    Returns
    -------
    tuple of np.ndarray, each shape=(T,):
        state_arr, A_arr, B_arr, C_arr, D_arr
    """
    T = len(close)
    state_arr = np.full(T, np.nan)
    A_arr = np.full(T, np.nan)
    B_arr = np.full(T, np.nan)
    C_arr = np.full(T, np.nan)
    D_arr = np.full(T, np.nan)

    state = IDLE
    A = np.nan
    B = np.nan
    C = np.nan
    D = np.nan

    for i in range(T):
        h = high[i]
        l = low[i]
        c = close[i]

        if state == IDLE:
            # 以當根低點作為潛在 A，下一根若創新高則進入 AB 段
            A = l
            B = h
            state = AB

        elif state == AB:
            # ── 破壞條件：跌破 A ──
            if l < A:
                # reset：以當根重新作為起點
                A = l
                B = h
                state = AB
                state_arr[i] = state
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 持續更新 B 高點
            if h > B:
                B = h

            # 判斷是否開始回調（收盤低於前一根低點，簡單以收盤拉回判斷）
            # 用高點已確立、當根收盤低於 B 的一定比例作為進入 BC 的信號
            # 這裡用「收盤跌破 B 的 1%」作為轉折判斷
            if c < B * 0.99:
                state = BC
                C = l  # 初始化 C

        elif state == BC:
            # ── 破壞條件：跌破 A ──
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                state = AB
                state_arr[i] = state
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 持續更新 C 低點
            if l < C:
                C = l

            # 判斷回調結束、開始上漲（收盤突破回調高點）
            # 用「收盤站回 C 上方一定幅度」作為進入 CD 的信號
            if c > C * 1.01:
                # 確認 C > A，否則 N 字無效，reset
                if C <= A:
                    A = l
                    B = h
                    C = np.nan
                    D = np.nan
                    state = AB
                    state_arr[i] = state
                    A_arr[i] = A
                    B_arr[i] = B
                    continue
                state = CD
                D = h  # 初始化 D

        elif state == CD:
            # ── 破壞條件 1：跌破 A ──
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                state = AB
                state_arr[i] = state
                A_arr[i] = A
                B_arr[i] = B
                continue

            # ── 破壞條件 2：未突破 B 卻跌破 C ──
            if D < B and l < C:
                A = l
                B = h
                C = np.nan
                D = np.nan
                state = AB
                state_arr[i] = state
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 更新 D 高點
            if h > D:
                D = h

            # D 突破 B → N 字確認（D1）
            if D > B:
                state = D1

        elif state == D1:
            # ── 破壞條件：跌破 A（最嚴格，結構全毀）──
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                state = AB
                state_arr[i] = state
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 也可選擇跌破 C 就重來（更嚴格版本，可依需求開啟）
            # if l < C:
            #     A = l; B = h; C = np.nan; D = np.nan; state = AB; continue

        state_arr[i] = state
        A_arr[i] = A
        B_arr[i] = B
        C_arr[i] = C if not np.isnan(C) else np.nan
        D_arr[i] = D if not np.isnan(D) else np.nan

    return state_arr, A_arr, B_arr, C_arr, D_arr


def detect_n_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基於狀態機的 N 字型特徵，支援結構破壞自動 reset：
        破壞條件 1：任何階段跌破 A → 重來
        破壞條件 2：CD 段尚未突破 B 就跌破 C → 重來

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

    stocks = close_wide.columns.tolist()
    dates = close_wide.index

    # 每個股票跑一次狀態機
    results = {}
    for sid in stocks:
        h = high_wide[sid].values.astype(np.float64)
        l = low_wide[sid].values.astype(np.float64)
        c = close_wide[sid].values.astype(np.float64)
        v = volume_wide[sid].values.astype(np.float64)

        state_arr, A_arr, B_arr, C_arr, D_arr = _run_state_machine(h, l, c, v)
        results[sid] = {
            "state": state_arr,
            "A": A_arr,
            "B": B_arr,
            "C": C_arr,
            "D": D_arr,
        }

    def to_wide(key):
        return pd.DataFrame({sid: results[sid][key] for sid in stocks}, index=dates)

    state_wide = to_wide("state")
    A_wide = to_wide("A")
    B_wide = to_wide("B")
    C_wide = to_wide("C")
    D_wide = to_wide("D")

    # D1 確認旗標
    d1_mask = state_wide == D1

    # ── 特徵計算（只在有效狀態下有值）────────────────────────────────── #

    # f_n_ab_gain：A→B 漲幅
    f_n_ab_gain = (B_wide - A_wide) / (A_wide + 1e-9)

    # f_n_bc_retracement：B→C 回調比例
    ab_range = (B_wide - A_wide).clip(lower=1e-9)
    f_n_bc_retracement = (B_wide - C_wide) / ab_range

    # f_n_c_above_a：C 高於 A 的幅度
    f_n_c_above_a = (C_wide - A_wide) / (A_wide + 1e-9)

    # f_n_d_breakout_strength：D 突破 B 的力道
    f_n_d_breakout_strength = (D_wide - B_wide) / (B_wide + 1e-9)

    # f_n_cd_gain：C→D 漲幅
    cd_gain = (D_wide - C_wide) / (C_wide + 1e-9)

    # f_n_cd_vs_ab_momentum：第二波 vs 第一波力道
    f_n_cd_vs_ab_momentum = cd_gain / (f_n_ab_gain + 1e-9)

    # f_n_close_vs_b：收盤相對 B 的位置
    f_n_close_vs_b = (close_wide - B_wide) / (B_wide + 1e-9)

    # f_n_structure_score：綜合品質分（只在 D1 後有值）
    f_n_structure_score = (
        (1 - f_n_bc_retracement.clip(0, 1)) * 0.3 + f_n_c_above_a.clip(0) * 0.3 + f_n_d_breakout_strength.clip(0) * 0.4
    ).where(d1_mask)

    # f_n_d1_confirmed：D1 旗標
    f_n_d1_confirmed = d1_mask.astype(float)

    # f_n_volume_on_breakout：CD 段量 vs BC 段量
    # 用狀態紀錄無法直接取窗口，用當前 state 做近似
    vol_roll10 = volume_wide.rolling(10).mean()
    vol_roll10_prev = vol_roll10.shift(10)
    f_n_volume_on_breakout = (vol_roll10 / (vol_roll10_prev + 1e-9)).where(state_wide.isin([CD, D1]))

    # f_n_days_since_d1：D1 確認後幾根
    def days_since_signal(bool_wide: pd.DataFrame) -> pd.DataFrame:
        arr = bool_wide.values.astype(float)
        out = np.full_like(arr, np.nan)
        counter = np.full(arr.shape[1], np.nan)
        for i in range(len(arr)):
            signal = arr[i]
            counter = np.where(signal == 1, 0, np.where(np.isnan(counter), np.nan, counter + 1))
            out[i] = counter
        return pd.DataFrame(out, index=bool_wide.index, columns=bool_wide.columns)

    f_n_days_since_d1 = days_since_signal(d1_mask)

    # f_n_state：目前所處階段（1=AB, 2=BC, 3=CD, 4=D1）
    f_n_state = state_wide.where(state_wide > IDLE)

    # ── 有效結構 mask：非 IDLE 狀態才輸出特徵 ───────────────────────── #
    valid = state_wide > IDLE

    feature_wides_raw = {
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
        "f_n_state": f_n_state,
    }

    feature_wides = {name: feat.where(valid) for name, feat in feature_wides_raw.items()}

    # ── 整合回長表格 ──────────────────────────────────────────────────── #
    feature_longs = []
    for feat_name, feat_wide in feature_wides.items():
        feat_long = feat_wide.stack().reset_index()
        feat_long.columns = ["date", "stock_id", feat_name]  # 強制命名
        feature_longs.append(feat_long.set_index(["date", "stock_id"]))

    features_df = pd.concat(feature_longs, axis=1).reset_index()
    result = df.merge(features_df, on=["date", "stock_id"], how="left")
    return result
