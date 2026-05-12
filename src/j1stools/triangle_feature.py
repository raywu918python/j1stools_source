import pandas as pd
import numpy as np

from j1stools import parquet_db


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

            # 計算上方趨勢線斜率（高點連線）
            high_x = np.array([h[0] for h in highs], dtype=float)
            high_y = np.array([h[1] for h in highs], dtype=float)
            upper_coef = np.polyfit(high_x, high_y, 1)

            # 計算下方趨勢線斜率（低點連線）
            low_x = np.array([l[0] for l in lows], dtype=float)
            low_y = np.array([l[1] for l in lows], dtype=float)
            lower_coef = np.polyfit(low_x, low_y, 1)

            price_scale = grp.loc[recent_idx[-1], "close"]
            upper_slope_norm = upper_coef[0] / (price_scale + 1e-9)
            lower_slope_norm = lower_coef[0] / (price_scale + 1e-9)

            # 上斜率為負，下斜率為正 → 對稱三角
            if upper_slope_norm >= 0 or lower_slope_norm <= 0:
                continue

            # ── 偵測突破 ───────────────────────────────────────────── #
            last_pivot_bar = recent_idx[-1]
            search_start = last_pivot_bar + 1
            search_end = min(last_pivot_bar + 20, n - hold_days - 1)

            for bar in range(search_start, search_end):
                upper_price = np.polyval(upper_coef, bar)

                # 突破條件：收盤站上趨勢線
                if grp.loc[bar, "close"] > upper_price:

                    # ── 取樣本 ─────────────────────────────────────── #
                    sample_start = bar - lookback + 1
                    if sample_start < 0:
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
                    meta_list.append(
                        {
                            "date": grp.loc[bar, "date"],
                            "stock_id": sid,
                            "upper_slope": upper_slope_norm,
                            "lower_slope": lower_slope_norm,
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


import pandas as pd
import numpy as np
from numba import njit


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

    # df = parquet_db.query_price(["0050"], "2015-01-01", "2099-01-01")
    stocks = parquet_db.query_stocks_no_etf()
    df = parquet_db.query_price(stocks, "2015-01-01", "2099-01-01")

    df_triangle = detect_zigzag(df, min_bars=5, min_change=0.03)
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

    print(f"樣本數：{len(X)}")
    print(f"真突破（1）：{y.sum()}")
    print(f"假突破（0）：{(y==0).sum()}")
    print(f"真突破比例：{y.mean():.2%}")
    print(f"X shape：{X.shape}")  # 應該是 (N, 30, 5)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    true_idx = np.where(y == 1)[0][:3]
    false_idx = np.where(y == 0)[0][:3]

    for i, idx in enumerate(true_idx):
        axes[0, i].plot(X[idx, :, 3])  # 收盤價
        stock = meta.iloc[idx]["stock_id"]
        date = meta.iloc[idx]["date"]
        axes[0, i].set_title(f"真突破 {stock} {date}")
        axes[0, i].axvline(x=29, color="r", linestyle="--")  # 突破點

    for i, idx in enumerate(false_idx):
        axes[1, i].plot(X[idx, :, 3])
        stock = meta.iloc[idx]["stock_id"]
        date = meta.iloc[idx]["date"]
        axes[1, i].set_title(f"假突破 {stock} {date}")
        axes[1, i].axvline(x=29, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig("triangle/triangle_samples.png")
    plt.show()

    import matplotlib.pyplot as plt
    import numpy as np

    # 取一個真突破樣本，畫出高低點和趨勢線
    idx = np.where(y == 1)[0][0]
    sample = X[idx, :, 3]  # 收盤價

    # 從 meta 取 stock_id 和 date
    print(meta.iloc[idx])

    plt.figure(figsize=(12, 5))
    plt.plot(sample, label="收盤價")
    plt.axvline(x=29, color="r", linestyle="--", label="突破點")
    plt.title(f"{meta.iloc[idx]['stock_id']} {meta.iloc[idx]['date']}")
    plt.legend()
    plt.show()


test()
