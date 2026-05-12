import pandas as pd


def profit_label(df: pd.DataFrame, hold_days=10, profit_target=0.10, stop_loss=-0.10) -> pd.DataFrame:
    """
    三分類標籤方案：
    2: 成功（漲）- 達到 profit_target 且過程中未觸及 stop_loss
    1: 失敗（跌）- 觸及 stop_loss
    0: 盤整（不漲不跌）- 持有期滿，既未達標也未停損
    """
    # 1. 進場基準價 (以今日收盤預期明日進場)
    df["entry_price"] = df["close"]

    # 2. 獲取未來持有期間的最高與最低價
    df["f_max"] = df.groupby("stock_id")["high"].transform(
        lambda x: x.shift(-hold_days).rolling(window=hold_days, min_periods=1).max()
    )
    df["f_min"] = df.groupby("stock_id")["low"].transform(
        lambda x: x.shift(-hold_days).rolling(window=hold_days, min_periods=1).min()
    )

    # 3. 計算最大盈虧比
    df["max_ret"] = (df["f_max"] - df["entry_price"]) / df["entry_price"]
    df["min_ret"] = (df["f_min"] - df["entry_price"]) / df["entry_price"]

    # 4. 邏輯判定
    # 先預設全部為 0 (不漲不跌)
    df["target"] = 0

    # 判定為 1 (跌)：只要最低跌幅低於停損點，就算失敗
    df.loc[df["min_ret"] <= stop_loss, "target"] = 1

    # 1. 基本獲利判定
    success_mask = (df["max_ret"] >= profit_target) & (df["min_ret"] > stop_loss)
    # 2. 進場過濾條件 (當天要紅K)
    # 3. 只有同時符合「未來會漲」且「當天紅K」才標為 2
    df.loc[success_mask, "target"] = 2

    # 5. 過濾掉無法買入的情況 (例如漲停)
    # 如果當天接近漲停 (9.5%)，這筆資料標籤改為 0 或直接丟棄，避免模型學到買不到的股票

    limit_up_mask = df.groupby("stock_id")["close"].pct_change(1, fill_method=None) > 0.095
    df.loc[limit_up_mask, "target"] = 0

    # 移除暫存欄位
    df.drop(
        ["entry_price", "f_max", "f_min", "max_ret", "min_ret"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    return df


def power_label(
    df: pd.DataFrame,
    hold_days: int = 5,
) -> pd.DataFrame:
    """
    LGBM 排名用 label：未來 N 日報酬率在當天全市場的排名（0~1）。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close
    hold_days : int
        持有天數（預設 5）

    Returns
    -------
    pd.DataFrame
        新增 target 欄位（0~1 的排名）
    """
    df = df.copy()

    future_close = df.groupby("stock_id")["close"].shift(-hold_days)
    df["future_return"] = (future_close - df["close"]) / (df["close"] + 1e-9)
    df["target"] = df.groupby("date")["future_return"].rank(pct=True)

    # future_return 保留在 df，不要 drop
    return df


def create_label(df: pd.DataFrame, threshold=0.1, period=1):
    # 確保排序
    df = df.sort_values(["stock_id", "date"])

    # 計算未來 10 日的最高價 (注意要 shift(-period) 往回拉)
    # 我們在 T 日，要看 T+1 到 T+10 的最高價
    def get_future_max(group):
        # 滾動取得未來 N 天最大值，並往回推移
        return group["high"].shift(-period).rolling(window=period, min_periods=1).max()

    df["future_max"] = df.groupby("stock_id").apply(lambda x: get_future_max(x)).reset_index(0, drop=True)

    # 建立標籤：未來最高價比今日收盤價高出 threshold
    df["target"] = (df["future_max"] / df["close"] > (1 + threshold)).astype(int)

    # 重要：最後 N 天的資料沒有未來資訊，必須刪除，不能參與訓練
    df = df.dropna(subset=["future_max"])

    return df


def test():
    pass
    # df = stock("3363")
    # df: pd.DataFrame = Label.add_label(df)
    # df = df.loc["2017", :]

    # # print(df.head().T)
    # # print(df.describe().T)
    # print(df["target"].value_counts())


# Label.test()
