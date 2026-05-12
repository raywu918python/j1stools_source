import random

from matplotlib import pyplot as plt
from j1stools import parquet_db

import pandas as pd
import numpy as np
from numba import njit


def detect_triangle_samples(
    df: pd.DataFrame,
    lookback: int = 30,  # 往回取幾根 K 線作為 CNN 輸入
    min_pivots: int = 4,  # 至少需要幾個轉折點（2高2低）
    hold_days: int = 10,  # 突破後持有幾天判斷真假突破
    profit_target: float = 0.05,  # 真突破：突破後漲幅 > 5%
    stop_loss: float = -0.03,  # 假突破：跌回超過 3%
) -> tuple:
    """
    從多股票長表格中找出三角收斂突破的樣本。

    流程：
        1. 用 ZigZag 高低點找出高點序列和低點序列
        2. 判斷是否符合三角收斂條件（高點遞減、低點遞增）
        3. 偵測突破（收盤突破上方趨勢線）
        4. 切出突破當天往回 lookback 根的 OHLCV 作為樣本
        5. 標記真假突破（label）

    Parameters
    ----------
    df : pd.DataFrame
        需已執行 detect_zigzag，包含：
        date, stock_id, open, high, low, close, volume,
        f_zz_type, f_zz_price
    lookback  : CNN 輸入的 K 線根數（預設 30）
    min_pivots: 最少轉折點數量（預設 4，即 2高2低）
    hold_days : 判斷真假突破的持有天數
    profit_target : 真突破的漲幅門檻
    stop_loss : 假突破的跌幅門檻

    Returns
    -------
    X : np.ndarray shape=(N, lookback, 5)
        CNN 輸入，已歸一化的 OHLCV
    y : np.ndarray shape=(N,)
        label，1=真突破，0=假突破
    meta : pd.DataFrame
        每個樣本的 date, stock_id，用於追蹤
    """
    df = df.copy().sort_values(["stock_id", "date"]).reset_index(drop=True)

    X_list = []
    y_list = []
    meta_list = []

    for sid, grp in df.groupby("stock_id"):
        grp = grp.reset_index(drop=True)
        n = len(grp)

        # 取出所有轉折點
        pivot_mask = grp["f_zz_type"] != 0
        pivot_idx = grp.index[pivot_mask].tolist()
        pivot_types = grp.loc[pivot_mask, "f_zz_type"].values
        pivot_prices = grp.loc[pivot_mask, "f_zz_price"].values

        if len(pivot_idx) < min_pivots:
            continue

        # 逐個轉折點，往前看是否形成三角收斂
        for pi in range(min_pivots - 1, len(pivot_idx)):
            recent_idx = pivot_idx[pi - min_pivots + 1 : pi + 1]
            recent_types = pivot_types[pi - min_pivots + 1 : pi + 1]
            recent_prices = pivot_prices[pi - min_pivots + 1 : pi + 1]

            # 分離高點和低點
            highs = [(recent_idx[j], recent_prices[j]) for j in range(len(recent_types)) if recent_types[j] == 1]
            lows = [(recent_idx[j], recent_prices[j]) for j in range(len(recent_types)) if recent_types[j] == -1]

            if len(highs) < 2 or len(lows) < 2:
                continue

            # ── 三角收斂條件 ───────────────────────────────────────── #
            # 高點遞減
            high_prices = [h[1] for h in highs]
            if not all(high_prices[i] > high_prices[i + 1] for i in range(len(high_prices) - 1)):
                continue

            # 低點遞增
            low_prices = [l[1] for l in lows]
            if not all(low_prices[i] < low_prices[i + 1] for i in range(len(low_prices) - 1)):
                continue

            # 只用最近兩個高點和最近兩個低點連線
            # 這樣趨勢線一定過轉折點，不會偏移
            base = recent_idx[0]

            h1_x, h1_y = highs[-1]  # 最近高點
            h2_x, h2_y = highs[-2]  # 次近高點
            l1_x, l1_y = lows[-1]  # 最近低點
            l2_x, l2_y = lows[-2]  # 次近低點

            # 用兩點決定直線：y = slope * (x - x1) + y1
            upper_slope = (h1_y - h2_y) / (h1_x - h2_x + 1e-9)
            lower_slope = (l1_y - l2_y) / (l1_x - l2_x + 1e-9)

            # 轉成 np.polyfit 格式 (slope, intercept)，x 用相對 base
            h1_x_rel = h1_x - base
            l1_x_rel = l1_x - base
            upper_coef = np.array([upper_slope, h1_y - upper_slope * h1_x_rel])
            lower_coef = np.array([lower_slope, l1_y - lower_slope * l1_x_rel])

            price_scale = grp.loc[recent_idx[-1], "close"]
            upper_slope_norm = upper_coef[0] / (price_scale + 1e-9)
            lower_slope_norm = lower_coef[0] / (price_scale + 1e-9)

            # 只要求兩線收斂（上斜率 < 下斜率），支援對稱/上升/下降三角
            if upper_slope >= lower_slope:
                continue  # 兩線發散，不是三角

            # 上方線必須在下方線之上（起點時）
            upper_at_0 = np.polyval(upper_coef, 0)
            lower_at_0 = np.polyval(lower_coef, 0)
            if upper_at_0 <= lower_at_0:
                continue

            # 初始寬度不能太大（超過價格的 15% 代表通道太鬆）
            price_ref = grp.loc[recent_idx[-1], "close"]
            width_init_check = upper_at_0 - lower_at_0
            if width_init_check <= 0 or width_init_check / price_ref > 0.15:
                continue

            # ── 偵測突破 ───────────────────────────────────────────── #
            last_pivot_bar = recent_idx[-1]
            search_start = last_pivot_bar + 1
            search_end = min(last_pivot_bar + 20, n - hold_days - 1)

            # 確認三角的第一個轉折點在 lookback 窗口內
            first_pivot_bar = recent_idx[0]

            # 計算兩條趨勢線的交叉點
            # upper: y = a1*x + b1, lower: y = a2*x + b2
            # 交叉：x = (b2 - b1) / (a1 - a2)
            denom = upper_coef[0] - lower_coef[0]
            if abs(denom) < 1e-9:
                continue  # 平行線，不收斂
            cross_x_rel = (lower_coef[1] - upper_coef[1]) / denom
            cross_x_abs = cross_x_rel + base

            # 交叉點必須在最後一個轉折點之後、合理範圍內（不能太遠）
            if cross_x_abs < last_pivot_bar or cross_x_abs > last_pivot_bar + 40:
                continue  # 收斂點太遠或已過，不是有效三角

            for bar in range(search_start, search_end):
                upper_price = np.polyval(upper_coef, bar - base)
                lower_price = np.polyval(lower_coef, bar - base)

                # 收斂條件：突破點的上下線距離要小於初始距離的 60%
                upper_at_base = np.polyval(upper_coef, 0)
                lower_at_base = np.polyval(lower_coef, 0)
                width_now = upper_price - lower_price
                width_init = upper_at_base - lower_at_base

                if width_init <= 0 or width_now <= 0:
                    continue
                if width_now / width_init > 0.6:
                    continue  # 收斂不夠

                # 突破點的收盤要在兩條線附近（綠線不能比收盤低超過 15%）
                close_at_bar = grp.loc[bar, "close"]
                if lower_price < close_at_bar * 0.90:
                    continue  # 支撐線太低，不是有效三角

                # 突破條件：收盤站上趨勢線
                if close_at_bar > upper_price:

                    # ── 取樣本 ─────────────────────────────────────── #
                    sample_start = bar - lookback + 1
                    if sample_start < 0:
                        break

                    # 確認所有轉折點都在 lookback 窗口內
                    if first_pivot_bar < sample_start:
                        break  # 三角太老，轉折點不在窗口內

                    # 第一個轉折點要在窗口前半段
                    if (first_pivot_bar - sample_start) > lookback // 2:
                        break  # 轉折點集中在後半段，不是有效三角

                    # 確認突破點在交叉點之前（還沒完全收斂就突破）
                    if bar >= cross_x_abs:
                        break  # 已過收斂點，不算有效突破

                    # 確認價格大部分時間在兩條趨勢線之間
                    # 用每根 bar 的絕對 index 轉成相對 base 的 x 座標
                    close_arr = grp.loc[sample_start:bar, "close"].values
                    x_coords = np.arange(sample_start - base, bar - base + 1, dtype=float)

                    upper_line = np.polyval(upper_coef, x_coords)
                    lower_line = np.polyval(lower_coef, x_coords)

                    # 印出診斷資訊
                    above_lower = (close_arr >= lower_line).mean()
                    below_upper = (close_arr <= upper_line).mean()
                    in_channel = above_lower * below_upper  # 簡化：兩個條件各自的比例

                    in_channel_mask = (close_arr >= lower_line) & (close_arr <= upper_line)
                    if in_channel_mask.mean() < 0.7:
                        break

                    ohlcv = grp.loc[sample_start:bar, ["open", "high", "low", "close", "volume"]].values.astype(
                        np.float64
                    )

                    if len(ohlcv) < lookback:
                        break

                    # 歸一化
                    close_ref = ohlcv[-1, 3]
                    vol_ref = ohlcv[:, 4].max()
                    ohlcv_norm = ohlcv.copy()
                    ohlcv_norm[:, :4] = ohlcv[:, :4] / (close_ref + 1e-9)
                    ohlcv_norm[:, 4] = ohlcv[:, 4] / (vol_ref + 1e-9)

                    # ── Label ──────────────────────────────────────── #
                    future = grp.loc[bar + 1 : bar + hold_days, "close"].values
                    if len(future) < hold_days:
                        break

                    entry = grp.loc[bar, "close"]
                    max_ret = (future.max() - entry) / (entry + 1e-9)
                    min_ret = (future.min() - entry) / (entry + 1e-9)

                    label = 1 if (max_ret >= profit_target and min_ret > stop_loss) else 0

                    X_list.append(ohlcv_norm)
                    y_list.append(label)
                    # 計算趨勢線在 sample 窗口內（0~lookback-1）的值，並歸一化
                    x_plot = np.arange(0, lookback, dtype=float)
                    x_abs_plot = x_plot + (sample_start - base)
                    close_ref = grp.loc[bar, "close"]

                    upper_plot = np.polyval(upper_coef, x_abs_plot) / close_ref
                    lower_plot = np.polyval(lower_coef, x_abs_plot) / close_ref

                    meta_list.append(
                        {
                            "date": grp.loc[bar, "date"],
                            "stock_id": sid,
                            "upper_slope": upper_slope_norm,
                            "lower_slope": lower_slope_norm,
                            "upper_plot": upper_plot.tolist(),  # 已歸一化，直接畫
                            "lower_plot": lower_plot.tolist(),
                        }
                    )
                    break  # 每個三角只取一個突破點

    if len(X_list) == 0:
        print("找不到三角收斂樣本，請調整參數")
        return np.array([]), np.array([]), pd.DataFrame()

    X = np.array(X_list)  # shape=(N, lookback, 5)
    y = np.array(y_list)  # shape=(N,)
    meta = pd.DataFrame(meta_list)

    print(f"找到樣本數：{len(X)}")
    print(f"真突破（1）：{y.sum()}  假突破（0）：{(y==0).sum()}")
    print(f"真突破比例：{y.mean():.2%}")

    return X, y, meta


@njit
def _zigzag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_bars: int = 5,  # 轉折點左右各至少幾根確認
    min_change: float = 0.03,  # 最小變動幅度（3%），過濾雜訊
):
    """
    ZigZag 高低點偵測。

    規則：
    - 高點：左右各 min_bars 根的最高點，且與前一個低點變動幅度 > min_change
    - 低點：左右各 min_bars 根的最低點，且與前一個高點變動幅度 > min_change
    - 高低點交替出現

    Parameters
    ----------
    high, low, close : np.ndarray shape=(T,)
    min_bars : 轉折點確認所需的左右根數
    min_change : 最小變動幅度

    Returns
    -------
    pivot_type : np.ndarray shape=(T,)
        1  = 高點（swing high）
        -1 = 低點（swing low）
        0  = 非轉折點
    pivot_price : np.ndarray shape=(T,)
        轉折點的價格（高點用 high，低點用 low）
    """
    T = len(close)
    pivot_type = np.zeros(T)
    pivot_price = np.full(T, np.nan)

    last_pivot_type = 0  # 上一個轉折點的類型（1=高, -1=低）
    last_pivot_price = 0.0  # 上一個轉折點的價格
    last_pivot_idx = -1  # 上一個轉折點的位置

    for i in range(min_bars, T - min_bars):
        h = high[i]
        l = low[i]

        # ── 判斷是否為高點 ──────────────────────────────────────────── #
        is_high = True
        for j in range(i - min_bars, i + min_bars + 1):
            if j != i and high[j] >= h:
                is_high = False
                break

        # ── 判斷是否為低點 ──────────────────────────────────────────── #
        is_low = True
        for j in range(i - min_bars, i + min_bars + 1):
            if j != i and low[j] <= l:
                is_low = False
                break

        # ── 高點處理 ────────────────────────────────────────────────── #
        if is_high:
            # 第一個轉折點
            if last_pivot_type == 0:
                pivot_type[i] = 1
                pivot_price[i] = h
                last_pivot_type = 1
                last_pivot_price = h
                last_pivot_idx = i

            # 上一個也是高點 → 保留較高的那個
            elif last_pivot_type == 1:
                if h > last_pivot_price:
                    pivot_type[last_pivot_idx] = 0  # 取消舊高點
                    pivot_price[last_pivot_idx] = np.nan
                    pivot_type[i] = 1
                    pivot_price[i] = h
                    last_pivot_price = h
                    last_pivot_idx = i

            # 上一個是低點 → 確認變動幅度夠大才算
            elif last_pivot_type == -1:
                if (h - last_pivot_price) / (last_pivot_price + 1e-9) >= min_change:
                    pivot_type[i] = 1
                    pivot_price[i] = h
                    last_pivot_type = 1
                    last_pivot_price = h
                    last_pivot_idx = i

        # ── 低點處理 ────────────────────────────────────────────────── #
        if is_low:
            # 第一個轉折點
            if last_pivot_type == 0:
                pivot_type[i] = -1
                pivot_price[i] = l
                last_pivot_type = -1
                last_pivot_price = l
                last_pivot_idx = i

            # 上一個也是低點 → 保留較低的那個
            elif last_pivot_type == -1:
                if l < last_pivot_price:
                    pivot_type[last_pivot_idx] = 0
                    pivot_price[last_pivot_idx] = np.nan
                    pivot_type[i] = -1
                    pivot_price[i] = l
                    last_pivot_price = l
                    last_pivot_idx = i

            # 上一個是高點 → 確認變動幅度夠大才算
            elif last_pivot_type == 1:
                if (last_pivot_price - l) / (last_pivot_price + 1e-9) >= min_change:
                    pivot_type[i] = -1
                    pivot_price[i] = l
                    last_pivot_type = -1
                    last_pivot_price = l
                    last_pivot_idx = i

    return pivot_type, pivot_price


def detect_zigzag(
    df: pd.DataFrame,
    min_bars: int = 5,
    min_change: float = 0.03,
) -> pd.DataFrame:
    """
    對多股票長表格計算 ZigZag 高低點。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, high, low, close
    min_bars : 轉折點確認所需的左右根數（預設 5）
    min_change : 最小變動幅度，過濾雜訊（預設 3%）

    Returns
    -------
    pd.DataFrame
        新增欄位：
        f_zz_type  : 1=高點, -1=低點, 0=非轉折點
        f_zz_price : 轉折點價格
    """
    df = df.copy().sort_values(["stock_id", "date"]).reset_index(drop=True)

    all_types = np.zeros(len(df))
    all_prices = np.full(len(df), np.nan)

    for sid, grp in df.groupby("stock_id"):
        idx = grp.index.values
        h = grp["high"].values.astype(np.float64)
        l = grp["low"].values.astype(np.float64)
        c = grp["close"].values.astype(np.float64)

        pivot_type, pivot_price = _zigzag(h, l, c, min_bars, min_change)

        all_types[idx] = pivot_type
        all_prices[idx] = pivot_price

    df["f_zz_type"] = all_types
    df["f_zz_price"] = all_prices

    return df


def get_pivot_history(df: pd.DataFrame, n_pivots: int = 6) -> pd.DataFrame:
    """
    對每根 bar，往回取最近 n_pivots 個轉折點的價格和類型。
    用於後續三角收斂特徵計算。

    Parameters
    ----------
    df : pd.DataFrame
        需已執行 detect_zigzag，包含 f_zz_type, f_zz_price
    n_pivots : 往回取幾個轉折點（預設 6，3高3低）

    Returns
    -------
    pd.DataFrame
        新增欄位: f_zz_p1_price, f_zz_p1_type, ..., f_zz_p6_price, f_zz_p6_type
        p1 = 最近的轉折點，p6 = 最舊的轉折點
    """
    df = df.copy()

    for sid, grp in df.groupby("stock_id"):
        pivot_mask = grp["f_zz_type"] != 0
        pivot_idx = grp.index[pivot_mask].values

        for i, row_idx in enumerate(grp.index):
            # 找這根 bar 之前的所有轉折點
            past_pivots = pivot_idx[pivot_idx <= row_idx]

            for p in range(1, n_pivots + 1):
                col_price = f"f_zz_p{p}_price"
                col_type = f"f_zz_p{p}_type"

                if len(past_pivots) >= p:
                    pidx = past_pivots[-p]
                    df.at[row_idx, col_price] = grp.at[pidx, "f_zz_price"]
                    df.at[row_idx, col_type] = grp.at[pidx, "f_zz_type"]
                else:
                    df.at[row_idx, col_price] = np.nan
                    df.at[row_idx, col_type] = np.nan

    return df


def test():

    # df = parquet_db.query_price(["0050"], "2015-01-01", "2099-0   1-01")
    stocks = parquet_db.query_stocks_ids_list()
    stocks = random.sample(parquet_db.query_stocks_no_etf(), 100)
    df = parquet_db.query_price(stocks, "2025-01-01", "2099-01-01")

    df_triangle = detect_zigzag(df, min_bars=5, min_change=0.02)
    # 看有沒有任何轉折點
    print(f"轉折點數量：{(df_triangle['f_zz_type'] != 0).sum()}")
    print(f"資料筆數：{len(df)}")

    X, y, meta = detect_triangle_samples(
        df_triangle,
        lookback=30,
        min_pivots=4,
        hold_days=10,
        profit_target=0.05,
        stop_loss=-0.03,
    )

    print(meta[["upper_slope", "lower_slope"]].describe())

    print(f"樣本數：{len(X)}")
    print(f"真突破（1）：{y.sum()}")
    print(f"假突破（0）：{(y==0).sum()}")
    print(f"真突破比例：{y.mean():.2%}")
    print(f"X shape：{X.shape}")  # 應該是 (N, 30, 5)
    #############################################################
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    true_idx = np.where(y == 1)[0][:3]
    false_idx = np.where(y == 0)[0][:3]

    for ax, idx in zip(axes.flatten(), list(true_idx) + list(false_idx)):
        m = meta.iloc[idx]

        close_data = X[idx, :, 3]
        upper = m["upper_plot"]
        lower = m["lower_plot"]

        ax.plot(close_data, color="blue")
        ax.plot(upper, "r--", alpha=0.7)
        ax.plot(lower, "g--", alpha=0.7)
        ax.axvline(x=29, color="r", linestyle=":", alpha=0.5)

        # 固定 y 軸範圍，只看價格附近
        y_min = min(close_data.min(), min(lower)) * 0.99
        y_max = max(close_data.max(), max(upper)) * 1.01
        ax.set_ylim(y_min, y_max)

        label_str = "真突破✅" if y[idx] == 1 else "假突破❌"
        ax.set_title(f"{label_str} {m['stock_id']} {str(m['date'])[:10]}")

    plt.tight_layout()
    plt.savefig("triangle/triangle_v4.png")
    plt.show()
    #############################################################
    # 取第一個樣本，直接印出關鍵數字
    idx = 0
    m = meta.iloc[idx]
    sid = m["stock_id"]
    date = m["date"]

    # 原始資料
    mask = (df_triangle["stock_id"] == sid) & (df_triangle["date"] <= date)
    grp = df_triangle[mask].tail(30).reset_index(drop=True)

    print("收盤價範圍：", grp["close"].min(), "~", grp["close"].max())
    print("突破日收盤：", grp["close"].iloc[-1])
    print()
    print("upper_plot 範圍：", min(m["upper_plot"]), "~", max(m["upper_plot"]))
    print("lower_plot 範圍：", min(m["lower_plot"]), "~", max(m["lower_plot"]))
    print()
    print("X[0,:,3] 範圍：", X[idx, :, 3].min(), "~", X[idx, :, 3].max())


# test()
