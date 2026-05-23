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


import pandas as pd


def add_target_forward(
    df, df_market, price_col="close", stock_col="stock_id", date_col="date", forward_days=20, max_return=0.5
):

    df = df.copy().sort_values([stock_col, date_col]).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])

    df["stock_return"] = df.groupby(stock_col)[price_col].transform(lambda x: x.shift(-forward_days) / x - 1)

    mkt = df_market.copy()
    mkt[date_col] = pd.to_datetime(mkt[date_col])
    mkt = mkt.sort_values(date_col).reset_index(drop=True)
    mkt["index_return"] = mkt[price_col].shift(-forward_days) / mkt[price_col] - 1

    df = df.merge(mkt[[date_col, "index_return"]], on=date_col, how="left")

    df["target"] = df["stock_return"] - df["index_return"]

    # 過濾異常值

    # bug_price = df[df["target"].abs() > max_return]
    # print(f"過濾異常值：{len(df) - len(bug_price)}")
    # print(bug_price.head(10))

    df = df.drop(columns=["stock_return", "index_return"])
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    return df


def check_bug_price(df: pd.DataFrame):
    print(df["target"].describe())
    print(f"\ntarget > 100% 的筆數：{(df['target'] >  1.0).sum()}")
    print(f"target > 200% 的筆數：{(df['target'] >  2.0).sum()}")
    print(f"target < -80% 的筆數：{(df['target'] < -0.8).sum()}")

    df_bug_price = df[df["target"] > 2]
    if len(df_bug_price) > 0:
        print(f"target > 1% 的筆數：{len(df_bug_price)}")
        df_bug_price.to_csv("bug_price.csv", index=False)
        raise Exception("有異常值")


def add_target(
    df, df_market, price_col="close", stock_col="stock_id", date_col="date", forward_days=20, max_return=5.0
):
    """
    計算每支股票未來N天內的最大超額報酬作為 target

    target = 個股N天內最高收盤價漲幅 - 大盤N天漲跌幅

    用最大漲幅而非第N天漲幅，給 B 模型更大的操作彈性，
    同時提升 IC（實測從 0.10 提升至 0.25）。

    Parameters
    ----------
    df           : 個股資料，需包含 date, stock_id, close
    df_market    : 大盤資料，需包含 date, close
    price_col    : 價格欄位名稱，預設 'close'
    stock_col    : 股票代號欄位名稱，預設 'stock_id'
    date_col     : 日期欄位名稱，預設 'date'
    forward_days : 預測天數，預設20天
    max_return   : 異常值過濾門檻（只過濾右側暴漲），預設200%

    Returns
    -------
    df : 原始資料加上 target 欄位，已移除 NaN 和異常值

    Raises
    ------
    ValueError : 必要欄位不存在時
    ValueError : 處理後資料為空時
    """
    # 檢查必要欄位
    required_cols = [price_col, stock_col, date_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必要欄位：{missing}")

    mkt_required = [price_col, date_col]
    mkt_missing = [c for c in mkt_required if c not in df_market.columns]
    if mkt_missing:
        raise ValueError(f"df_market 缺少必要欄位：{mkt_missing}")

    try:
        df = df.copy().sort_values([stock_col, date_col]).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])

        df["stock_return"] = df.groupby(stock_col)[price_col].transform(
            lambda x: x.rolling(forward_days).max().shift(-forward_days) / x - 1
        )

        mkt = df_market.copy()
        mkt[date_col] = pd.to_datetime(mkt[date_col])
        mkt = mkt.sort_values(date_col).reset_index(drop=True)
        mkt["index_return"] = mkt[price_col].shift(-forward_days) / mkt[price_col] - 1

        df = df.merge(mkt[[date_col, "index_return"]], on=date_col, how="left")
        df["target"] = df["stock_return"] - df["index_return"]
        df = df.drop(columns=["stock_return", "index_return"])
        df = df.reset_index(drop=True)

        # 超過 max_return 視為異常資料，直接拋出例外
        abnormal = df[df["target"] > max_return]
        if len(abnormal) > 0:
            detail = abnormal[["date", stock_col, "target"]].to_string()
            raise ValueError(
                f"發現 {len(abnormal)} 筆異常 target（超過 {max_return:.0%}），" f"請檢查以下股票資料：{detail}"
            )

        if len(df) == 0:
            raise ValueError("add_target 處理後資料為空，請檢查輸入資料")

        print(f"add_target 完成：{len(df)} 筆，" f"target 範圍 [{df['target'].min():.3f}, {df['target'].max():.3f}]")

        return df

    except Exception as e:
        raise RuntimeError(f"add_target 發生錯誤：{e}") from e
