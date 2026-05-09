import pandas as pd
import numpy as np
from numba import njit

# ── 狀態定義 ──────────────────────────────────────────────────────────────── #
IDLE = 0
AB = 1  # A→B 上漲段
BC = 2  # B→C 回調段
CD = 3  # C→D 再漲段
D1 = 4  # N 字確認後


@njit
def _run_state_machine(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    max_ab_bars: int = 10,  # AB 段必須在 N 根內完成上漲
    max_bc_bars: int = 10,  # BC 回調必須在 N 根內止跌
    max_cd_bars: int = 10,  # CD 再漲必須在 N 根內突破 B
):
    """
    N 字型狀態機：
        AB：max_ab_bars 根內持續上漲，找到 B 高點
        BC：max_bc_bars 根內回調止跌，不跌破 A，找到 C 低點（C > A）
        CD：max_cd_bars 根內再漲，D 突破 B → D1 確認

    破壞條件：
        1. 任何階段跌破 A → reset
        2. CD 段未突破 B 卻跌破 C → reset
        3. 各段超過 bar 上限 → reset
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
    bar_count = 0

    for i in range(T):
        h = high[i]
        l = low[i]
        c = close[i]

        # ── IDLE → AB ────────────────────────────────────────────────── #
        if state == IDLE:
            A = l
            B = h
            bar_count = 0
            state = AB

        # ── AB 段：在 max_ab_bars 根內找到高點 B ─────────────────────── #
        elif state == AB:
            bar_count += 1

            # 超時：N 根內沒完成上漲，以當根重設起點（含第 N 根）
            if bar_count >= max_ab_bars:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 破壞：跌破 A
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 持續更新 B 高點
            if h > B:
                B = h

            # 收盤回調（跌破 B 的 1%）→ 進入 BC
            # 不要求漲幅，只要在 N 根內有上漲且開始回調即可
            if c < B * 0.99:
                state = BC
                bar_count = 0
                C = l

        # ── BC 段：max_bc_bars 根內止跌，不跌破 A ────────────────────── #
        elif state == BC:
            bar_count += 1

            # 超時：回調超過 N 根，reset（含第 N 根）
            if bar_count >= max_bc_bars:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 破壞：跌破 A
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 更新 C 低點
            if l < C:
                C = l

            # 收盤站回 C 上方 → 止跌，準備進 CD
            if c > C * 1.01:
                # C 必須高於 A
                if C <= A:
                    A = l
                    B = h
                    C = np.nan
                    D = np.nan
                    bar_count = 0
                    state_arr[i] = AB
                    A_arr[i] = A
                    B_arr[i] = B
                    continue
                state = CD
                bar_count = 0
                D = h

        # ── CD 段：max_cd_bars 根內 D 突破 B ─────────────────────────── #
        elif state == CD:
            bar_count += 1

            # 超時：N 根內沒突破 B，reset（含第 N 根，所以用 >=）
            if bar_count >= max_cd_bars:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 破壞 1：跌破 A
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            # 破壞 2：未突破 B 卻跌破 C
            if D < B and l < C:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

            if h > D:
                D = h

            # D 突破 B → N 字確認
            if D > B:
                state = D1
                bar_count = 0

        # ── D1：確認後持續追蹤 ───────────────────────────────────────── #
        elif state == D1:
            bar_count += 1

            # 破壞：跌破 A
            if l < A:
                A = l
                B = h
                C = np.nan
                D = np.nan
                bar_count = 0
                state_arr[i] = AB
                A_arr[i] = A
                B_arr[i] = B
                continue

        state_arr[i] = state
        A_arr[i] = A
        B_arr[i] = B
        C_arr[i] = C
        D_arr[i] = D

    return state_arr, A_arr, B_arr, C_arr, D_arr


def detect_n_shape_features(
    df: pd.DataFrame,
    max_ab_bars: int = 10,  # AB 段最多幾根
    max_bc_bars: int = 10,  # BC 段最多幾根
    max_cd_bars: int = 10,  # CD 段最多幾根
) -> pd.DataFrame:
    """
    基於狀態機的 N 字型特徵。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close, high, low, volume
    max_ab_bars : AB 段最大 bar 數（預設 10）
    max_bc_bars : BC 段最大 bar 數（預設 10）
    max_cd_bars : CD 段最大 bar 數（預設 10）
    """
    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")
    volume_wide = df.pivot(index="date", columns="stock_id", values="volume")

    stocks = close_wide.columns.tolist()
    dates = close_wide.index

    results = {}
    for sid in stocks:
        h = high_wide[sid].values.astype(np.float64)
        l = low_wide[sid].values.astype(np.float64)
        c = close_wide[sid].values.astype(np.float64)
        v = volume_wide[sid].values.astype(np.float64)

        state_arr, A_arr, B_arr, C_arr, D_arr = _run_state_machine(h, l, c, v, max_ab_bars, max_bc_bars, max_cd_bars)
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

    d1_mask = state_wide == D1

    # ── 特徵計算 ──────────────────────────────────────────────────────── #

    # AB 漲幅
    f_n_ab_gain = (B_wide - A_wide) / (A_wide + 1e-9)

    # BC 回調比例（相對 AB 幅度）
    ab_range = (B_wide - A_wide).clip(lower=1e-9)
    f_n_bc_retracement = (B_wide - C_wide) / ab_range

    # C 高於 A 的幅度
    f_n_c_above_a = (C_wide - A_wide) / (A_wide + 1e-9)

    # D 突破 B 的力道
    f_n_d_breakout_strength = (D_wide - B_wide) / (B_wide + 1e-9)

    # CD 漲幅 vs AB 漲幅
    cd_gain = (D_wide - C_wide) / (C_wide + 1e-9)
    f_n_cd_vs_ab_momentum = cd_gain / (f_n_ab_gain + 1e-9)

    # 收盤相對 B 的位置
    f_n_close_vs_b = (close_wide - B_wide) / (B_wide + 1e-9)

    # 綜合品質分（只在 D1 有值）
    f_n_structure_score = (
        (1 - f_n_bc_retracement.clip(0, 1)) * 0.3 + f_n_c_above_a.clip(0) * 0.3 + f_n_d_breakout_strength.clip(0) * 0.4
    ).where(d1_mask)

    # D1 旗標
    f_n_d1_confirmed = d1_mask.astype(float)

    # 突破段量 vs 回調段量
    vol_roll = volume_wide.rolling(max_cd_bars).mean()
    vol_roll_prev = vol_roll.shift(max_bc_bars)
    f_n_volume_on_breakout = (vol_roll / (vol_roll_prev + 1e-9)).where(state_wide.isin([CD, D1]))

    # D1 確認後幾根
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

    # 當前所在階段
    f_n_state = state_wide.where(state_wide > IDLE)

    # ── 有效結構 mask ─────────────────────────────────────────────────── #
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
        feat_long.columns = ["date", "stock_id", feat_name]
        feature_longs.append(feat_long.set_index(["date", "stock_id"]))

    features_df = pd.concat(feature_longs, axis=1).reset_index()
    result = df.merge(features_df, on=["date", "stock_id"], how="left")
    return result
