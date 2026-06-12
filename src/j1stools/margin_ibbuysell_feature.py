import pandas as pd
import numpy as np


def _rolling_zscore(s, window=120, min_periods=20):
    m = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / std.replace(0, np.nan)


# new
def add_feature(df, df_ibbuysell, df_market, mode="train"):
    """
    計算所有特徵，回傳 f_ 開頭的特徵欄位

    特徵分為六類：
      1. 大盤環境特徵  : 市場波動率、成交量熱度
      2. 法人籌碼特徵  : 外資、投信、自營商淨買超及滾動累計
      3. 融資融券特徵  : 餘額變化、使用率、券資比
      4. 價格動能特徵  : 各週期漲跌幅、振幅、成交量比
      5. 均線特徵      : 乖離率、均線斜率、動能交叉
      6. 相對大盤特徵  : 個股相對強弱、成交量佔比、波動率比

    Parameters
    ----------
    df           : 個股資料，欄位包含：
                   date, stock_id, close, open, high, low, volume
                   margin_purchase_buy, margin_purchase_cash_repayment,
                   margin_purchase_limit, margin_purchase_sell,
                   margin_purchase_today_balance, margin_purchase_yesterday_balance,
                   offset_loan_and_short, short_sale_buy, short_sale_cash_repayment,
                   short_sale_limit, short_sale_sell,
                   short_sale_today_balance, short_sale_yesterday_balance
    df_ibbuysell : 法人買賣超，欄位包含：
                   date, stock_id, name, buy, sell
                   name 類別：Dealer_Hedging, Dealer_self,
                              Foreign_Investor, Investment_Trust
    df_market    : 大盤資料，欄位包含：
                   date, close, open, high, low, volume

    Returns
    -------
    result : date, stock_id, target 加上所有 f_ 特徵欄位
    """
    if mode not in ("train", "predict"):
        raise ValueError(f"mode 必須是 'train' 或 'predict'，收到：{mode}")

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.drop_duplicates(subset=["date", "stock_id"])
    base = base.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 大盤特徵（只保留環境特徵，不放報酬率避免干擾超額報酬預測）
    mkt = df_market.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)
    mkt["f_market_volatility_20d"] = mkt["close"].pct_change(1, fill_method=None).rolling(20).std()
    mkt["f_market_volume_20d"] = mkt["volume"] / mkt["volume"].rolling(20).mean()
    mkt["_mkt_return_5d"] = mkt["close"].pct_change(5, fill_method=None)
    mkt["_mkt_return_20d"] = mkt["close"].pct_change(20, fill_method=None)
    mkt_merge = mkt[
        [
            "date",
            "f_market_volatility_20d",
            "f_market_volume_20d",
            "close",
            "volume",
            "_mkt_return_5d",
            "_mkt_return_20d",
        ]
    ].rename(columns={"close": "_mkt_close", "volume": "_mkt_volume"})
    base = base.merge(mkt_merge, on="date", how="left")

    # 法人籌碼特徵
    ib = df_ibbuysell.copy()
    ib["date"] = pd.to_datetime(ib["date"])
    ib["net"] = ib["buy"] - ib["sell"]
    ib_pivot = (
        ib.pivot_table(index=["date", "stock_id"], columns="name", values="net", aggfunc="sum").fillna(0).reset_index()
    )
    # 原始絕對值用 _ 前綴（中間變數，不進 f_cols）
    name_map = {
        "Dealer_Hedging": "_net_dealer_hedging",
        "Dealer_self": "_net_dealer_self",
        "Foreign_Investor": "_net_foreign",
        "Investment_Trust": "_net_trust",
    }
    ib_pivot = ib_pivot.rename(columns=name_map)
    for col in name_map.values():
        if col not in ib_pivot.columns:
            ib_pivot[col] = 0
    ib_pivot["_net_institutional_total"] = (
        ib_pivot["_net_dealer_hedging"]
        + ib_pivot["_net_dealer_self"]
        + ib_pivot["_net_foreign"]
        + ib_pivot["_net_trust"]
    )
    base = base.merge(ib_pivot, on=["date", "stock_id"], how="left")

    # 法人資料 shift 1天（法人資料收盤後才公布，實際交易用昨天的資料）
    ib_cols = ["_net_dealer_hedging", "_net_dealer_self", "_net_foreign", "_net_trust", "_net_institutional_total"]
    for col in ib_cols:
        base[col] = base.groupby("stock_id")[col].transform(lambda x: x.shift(1))

    # 融資融券原始欄位 shift 1天（融資融券隔天早上才公布）
    margin_cols = [
        "margin_purchase_buy",
        "margin_purchase_cash_repayment",
        "margin_purchase_limit",
        "margin_purchase_sell",
        "margin_purchase_today_balance",
        "margin_purchase_yesterday_balance",
        "offset_loan_and_short",
        "short_sale_buy",
        "short_sale_cash_repayment",
        "short_sale_limit",
        "short_sale_sell",
        "short_sale_today_balance",
        "short_sale_yesterday_balance",
    ]
    for col in margin_cols:
        if col in base.columns:
            base[col] = base.groupby("stock_id")[col].transform(lambda x: x.shift(1))

    g = base.groupby("stock_id")

    # 法人買超比例化（相對個股成交量，讓大小股可以比較）
    # 當日成交量極小（停牌/首掛）時比值會爆炸，用前20日均量做分母保護
    vol_denom = g["volume"].transform(lambda x: x.rolling(20, min_periods=1).mean()).replace(0, np.nan)
    raw_to_pct = {
        "_net_foreign": "f_net_foreign_pct",
        "_net_trust": "f_net_trust_pct",
        "_net_dealer_self": "f_net_dealer_self_pct",
        "_net_dealer_hedging": "f_net_dealer_hedging_pct",
        "_net_institutional_total": "f_net_institutional_total_pct",
    }
    for raw_col, pct_col in raw_to_pct.items():
        base[pct_col] = base[raw_col] / vol_denom

    # 法人滾動累計：原始值只當中間變數，最終特徵用 f_ 前綴（大小股可比）
    for raw_col, fname in [
        ("_net_foreign", "f_net_foreign"),
        ("_net_trust", "f_net_trust"),
        ("_net_institutional_total", "f_net_institutional_total"),
    ]:
        _5d = g[raw_col].transform(lambda x: x.rolling(5).sum())
        _10d = g[raw_col].transform(lambda x: x.rolling(10).sum())
        base[f"{fname}_streak"] = g[raw_col].transform(
            lambda x: (x.groupby((x <= 0).cumsum()).cumcount() + 1).where(x > 0, 0)
        )
        base[f"{fname}_5d_z"] = _5d.groupby(base["stock_id"]).transform(_rolling_zscore)
        base[f"{fname}_10d_z"] = _10d.groupby(base["stock_id"]).transform(_rolling_zscore)

    # 融資融券特徵（絕對張數只當中間變數，不進 f_cols）
    base["_margin_balance_change"] = base["margin_purchase_today_balance"] - base["margin_purchase_yesterday_balance"]
    base["_short_balance_change"] = base["short_sale_today_balance"] - base["short_sale_yesterday_balance"]
    base["f_margin_balance_change_pct"] = base["_margin_balance_change"] / base[
        "margin_purchase_yesterday_balance"
    ].replace(0, np.nan)
    base["f_short_balance_change_pct"] = base["_short_balance_change"] / base["short_sale_yesterday_balance"].replace(
        0, np.nan
    )
    base["f_short_margin_ratio"] = base["short_sale_today_balance"] / base["margin_purchase_today_balance"].replace(
        0, np.nan
    )
    base["f_margin_balance_change_5d_pct"] = g["margin_purchase_today_balance"].transform(
        lambda x: (x - x.shift(5)) / x.shift(5).replace(0, np.nan)
    )
    base["f_margin_balance_change_10d_pct"] = g["margin_purchase_today_balance"].transform(
        lambda x: (x - x.shift(10)) / x.shift(10).replace(0, np.nan)
    )

    # 價格動能特徵
    base["f_return_1d"] = g["close"].transform(lambda x: x.pct_change(1, fill_method=None))
    base["f_return_5d"] = g["close"].transform(lambda x: x.pct_change(5, fill_method=None))
    base["f_return_10d"] = g["close"].transform(lambda x: x.pct_change(10, fill_method=None))
    base["f_return_20d"] = g["close"].transform(lambda x: x.pct_change(20, fill_method=None))
    base["f_return_60d"] = g["close"].transform(lambda x: x.pct_change(60, fill_method=None))
    base["f_amplitude"] = (base["high"] - base["low"]) / base["close"]
    base["f_volume_change_pct"] = g["volume"].transform(lambda x: x.replace(0, np.nan).pct_change(1, fill_method=None))
    base["f_volume_ratio_5d"] = base["volume"] / g["volume"].transform(lambda x: x.rolling(5).mean())
    base["f_stock_volatility_20d"] = g["close"].transform(lambda x: x.pct_change(1, fill_method=None).rolling(20).std())

    # 均線特徵
    ma5 = g["close"].transform(lambda x: x.rolling(5).mean())
    ma20 = g["close"].transform(lambda x: x.rolling(20).mean())
    ma60 = g["close"].transform(lambda x: x.rolling(60).mean())
    base["f_bias_ma5"] = (base["close"] - ma5) / ma5
    base["f_bias_ma20"] = (base["close"] - ma20) / ma20
    base["f_bias_ma60"] = (base["close"] - ma60) / ma60
    base["f_ma5_slope"] = ma5.groupby(base["stock_id"]).transform(lambda x: x.pct_change(5, fill_method=None))
    base["f_ma20_slope"] = ma20.groupby(base["stock_id"]).transform(lambda x: x.pct_change(10, fill_method=None))
    base["f_ma60_slope"] = ma60.groupby(base["stock_id"]).transform(lambda x: x.pct_change(20, fill_method=None))
    base["f_momentum_cross"] = base["f_return_5d"] - base["f_return_20d"]

    # 相對大盤特徵
    base["f_relative_strength_5d"] = base["f_return_5d"] - base["_mkt_return_5d"]
    base["f_relative_strength_20d"] = base["f_return_20d"] - base["_mkt_return_20d"]
    base["f_volume_market_ratio"] = base["volume"] / base["_mkt_volume"].replace(0, np.nan)
    base["f_volatility_vs_market"] = base["f_stock_volatility_20d"] / base["f_market_volatility_20d"].replace(0, np.nan)

    drop_cols = [
        "_mkt_close",
        "_mkt_volume",
        "_mkt_return_5d",
        "_mkt_return_20d",
        "_net_foreign",
        "_net_trust",
        "_net_dealer_self",
        "_net_dealer_hedging",
        "_net_institutional_total",
        "_margin_balance_change",
        "_short_balance_change",
    ]
    base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    # Winsorize：截斷極端值，避免停牌/低流動性日造成 inf 或爆炸數值進入模型
    # _pct 類：法人流量/均量，正常範圍 -5~5（超過代表當日幾乎無成交）
    pct_clip_cols = [
        "f_net_foreign_pct",
        "f_net_trust_pct",
        "f_net_dealer_self_pct",
        "f_net_dealer_hedging_pct",
        "f_net_institutional_total_pct",
    ]
    for col in pct_clip_cols:
        if col in base.columns:
            base[col] = base[col].clip(-5, 5)
    # 融資融券 _pct：正常日變化 < 100%，極端情況截到 -1~10
    for col in [
        "f_margin_balance_change_pct",
        "f_short_balance_change_pct",
        "f_margin_balance_change_5d_pct",
        "f_margin_balance_change_10d_pct",
    ]:
        if col in base.columns:
            base[col] = base[col].clip(-1, 10)
    # 成交量相關：去掉極端值
    if "f_volume_market_ratio" in base.columns:
        base["f_volume_market_ratio"] = base["f_volume_market_ratio"].clip(0, 10)
    if "f_volume_change_pct" in base.columns:
        base["f_volume_change_pct"] = base["f_volume_change_pct"].clip(-1, 10)
    # 其餘 inf → NaN
    f_cols_now = [c for c in base.columns if c.startswith("f_")]
    base[f_cols_now] = base[f_cols_now].replace([np.inf, -np.inf], np.nan)

    # 截面百分位排名：在每個交易日對所有股票做排名(0~1)，消除大小股規模差異
    # 只對「需要跨股比較」且「未已標準化」的特徵做排名
    # _pct / _z / _streak / utilization / ratio 類已相對化，額外加 xrank 讓模型有兩個視角
    xrank_cols = [
        # 法人籌碼：_pct 已標準化，但仍做 xrank 消除股本差異
        "f_net_foreign_pct",
        "f_net_trust_pct",
        "f_net_dealer_self_pct",
        "f_net_dealer_hedging_pct",
        "f_net_institutional_total_pct",
        # 法人 z-score：已個股自身標準化，再做截面排名讓跨股可比
        "f_net_foreign_5d_z",
        "f_net_foreign_10d_z",
        "f_net_trust_5d_z",
        "f_net_trust_10d_z",
        "f_net_institutional_total_5d_z",
        "f_net_institutional_total_10d_z",
        # 融資融券：百分比已相對化，截面排名消除各股絕對額度差異
        "f_margin_balance_change_pct",
        "f_short_balance_change_pct",
        "f_margin_balance_change_5d_pct",
        "f_margin_balance_change_10d_pct",
        "f_short_margin_ratio",
        # 價格動能：報酬率本身已無量綱，截面排名去除個股波動率差異
        "f_return_1d",
        "f_return_5d",
        "f_return_10d",
        "f_return_20d",
        "f_relative_strength_5d",
        "f_relative_strength_20d",
        # 成交量相對特徵
        "f_volume_ratio_5d",
        "f_volume_change_pct",
        # 均線特徵
        "f_bias_ma5",
        "f_bias_ma20",
        "f_bias_ma60",
        "f_ma5_slope",
        "f_ma20_slope",
        "f_ma60_slope",
        "f_momentum_cross",
        # 波動率
        "f_stock_volatility_20d",
        "f_volatility_vs_market",
    ]
    for col in xrank_cols:
        if col in base.columns:
            base[f"{col}_xrank"] = base.groupby("date")[col].rank(pct=True, na_option="keep")

    f_cols = [c for c in base.columns if c.startswith("f_")]

    if mode == "train":
        # 訓練模式：過濾掉 target = NaN 的資料（最後20天沒有未來價格）
        has_target = base["target"].notna()
        result = base[has_target][["date", "stock_id", "target"] + f_cols].copy()
    else:
        # 預測模式：保留全部資料包含今天，不管 target 有沒有或是否為 NaN
        cols = ["date", "stock_id"] + f_cols
        if "target" in base.columns:
            cols = ["date", "stock_id", "target"] + f_cols
        result = base[cols].copy()

    return result.reset_index(drop=True)


# old
def add_feature_old(df, df_ibbuysell, df_market, mode="train"):
    """
    計算所有特徵，回傳 f_ 開頭的特徵欄位

    特徵分為六類：
      1. 大盤環境特徵  : 市場波動率、成交量熱度
      2. 法人籌碼特徵  : 外資、投信、自營商淨買超及滾動累計
      3. 融資融券特徵  : 餘額變化、使用率、券資比
      4. 價格動能特徵  : 各週期漲跌幅、振幅、成交量比
      5. 均線特徵      : 乖離率、均線斜率、動能交叉
      6. 相對大盤特徵  : 個股相對強弱、成交量佔比、波動率比

    Parameters
    ----------
    df           : 個股資料，欄位包含：
                   date, stock_id, close, open, high, low, volume
                   margin_purchase_buy, margin_purchase_cash_repayment,
                   margin_purchase_limit, margin_purchase_sell,
                   margin_purchase_today_balance, margin_purchase_yesterday_balance,
                   offset_loan_and_short, short_sale_buy, short_sale_cash_repayment,
                   short_sale_limit, short_sale_sell,
                   short_sale_today_balance, short_sale_yesterday_balance
    df_ibbuysell : 法人買賣超，欄位包含：
                   date, stock_id, name, buy, sell
                   name 類別：Dealer_Hedging, Dealer_self,
                              Foreign_Investor, Investment_Trust
    df_market    : 大盤資料，欄位包含：
                   date, close, open, high, low, volume

    Returns
    -------
    result : date, stock_id, target 加上所有 f_ 特徵欄位
    """
    if mode not in ("train", "predict"):
        raise ValueError(f"mode 必須是 'train' 或 'predict'，收到：{mode}")

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.drop_duplicates(subset=["date", "stock_id"])
    base = base.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 大盤特徵（只保留環境特徵，不放報酬率避免干擾超額報酬預測）
    mkt = df_market.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)
    mkt["f_market_volatility_20d"] = mkt["close"].pct_change(1).rolling(20).std()
    mkt["f_market_volume_20d"] = mkt["volume"] / mkt["volume"].rolling(20).mean()
    mkt_merge = mkt[["date", "f_market_volatility_20d", "f_market_volume_20d", "close", "volume"]].rename(
        columns={"close": "_mkt_close", "volume": "_mkt_volume"}
    )
    base = base.merge(mkt_merge, on="date", how="left")

    # 法人籌碼特徵
    ib = df_ibbuysell.copy()
    ib["date"] = pd.to_datetime(ib["date"])
    ib["net"] = ib["buy"] - ib["sell"]
    ib_pivot = (
        ib.pivot_table(index=["date", "stock_id"], columns="name", values="net", aggfunc="sum").fillna(0).reset_index()
    )
    name_map = {
        "Dealer_Hedging": "f_net_dealer_hedging",
        "Dealer_self": "f_net_dealer_self",
        "Foreign_Investor": "f_net_foreign",
        "Investment_Trust": "f_net_trust",
    }
    ib_pivot = ib_pivot.rename(columns=name_map)
    for col in name_map.values():
        if col not in ib_pivot.columns:
            ib_pivot[col] = 0
    ib_pivot["f_net_institutional_total"] = (
        ib_pivot["f_net_dealer_hedging"]
        + ib_pivot["f_net_dealer_self"]
        + ib_pivot["f_net_foreign"]
        + ib_pivot["f_net_trust"]
    )
    base = base.merge(ib_pivot, on=["date", "stock_id"], how="left")

    # 法人資料 shift 1天（法人資料收盤後才公布，實際交易用昨天的資料）
    ib_cols = ["f_net_dealer_hedging", "f_net_dealer_self", "f_net_foreign", "f_net_trust", "f_net_institutional_total"]
    for col in ib_cols:
        base[col] = base.groupby("stock_id")[col].transform(lambda x: x.shift(1))

    # 融資融券原始欄位 shift 1天（融資融券隔天早上才公布）
    margin_cols = [
        "margin_purchase_buy",
        "margin_purchase_cash_repayment",
        "margin_purchase_limit",
        "margin_purchase_sell",
        "margin_purchase_today_balance",
        "margin_purchase_yesterday_balance",
        "offset_loan_and_short",
        "short_sale_buy",
        "short_sale_cash_repayment",
        "short_sale_limit",
        "short_sale_sell",
        "short_sale_today_balance",
        "short_sale_yesterday_balance",
    ]
    for col in margin_cols:
        if col in base.columns:
            base[col] = base.groupby("stock_id")[col].transform(lambda x: x.shift(1))

    # 法人買超比例化（相對個股成交量，讓大小股可以比較）
    for col in ["f_net_foreign", "f_net_trust", "f_net_dealer_self", "f_net_institutional_total"]:
        base[f"{col}_pct"] = base[col] / base["volume"].replace(0, np.nan)

    g = base.groupby("stock_id")

    # 法人滾動累計（連續買超比單日更有意義）
    for col in ["f_net_foreign", "f_net_trust", "f_net_institutional_total"]:
        base[f"{col}_5d"] = g[col].transform(lambda x: x.rolling(5).sum())
        base[f"{col}_10d"] = g[col].transform(lambda x: x.rolling(10).sum())
        base[f"{col}_streak"] = g[col].transform(lambda x: x.groupby((x <= 0).cumsum()).cumcount().where(x > 0, 0))

    # 融資融券特徵
    base["f_margin_balance_change"] = base["margin_purchase_today_balance"] - base["margin_purchase_yesterday_balance"]
    base["f_short_balance_change"] = base["short_sale_today_balance"] - base["short_sale_yesterday_balance"]
    base["f_margin_balance_change_pct"] = base["f_margin_balance_change"] / base[
        "margin_purchase_yesterday_balance"
    ].replace(0, np.nan)
    base["f_short_balance_change_pct"] = base["f_short_balance_change"] / base["short_sale_yesterday_balance"].replace(
        0, np.nan
    )
    base["f_margin_utilization"] = base["margin_purchase_today_balance"] / base["margin_purchase_limit"].replace(
        0, np.nan
    )
    base["f_short_utilization"] = base["short_sale_today_balance"] / base["short_sale_limit"].replace(0, np.nan)
    base["f_short_margin_ratio"] = base["short_sale_today_balance"] / base["margin_purchase_today_balance"].replace(
        0, np.nan
    )
    base["f_margin_balance_change_5d"] = g["margin_purchase_today_balance"].transform(lambda x: x - x.shift(5))
    base["f_margin_balance_change_10d"] = g["margin_purchase_today_balance"].transform(lambda x: x - x.shift(10))

    # 價格動能特徵
    base["f_return_1d"] = g["close"].transform(lambda x: x.pct_change(1))
    base["f_return_5d"] = g["close"].transform(lambda x: x.pct_change(5))
    base["f_return_10d"] = g["close"].transform(lambda x: x.pct_change(10))
    base["f_return_20d"] = g["close"].transform(lambda x: x.pct_change(20))
    base["f_return_60d"] = g["close"].transform(lambda x: x.pct_change(60))
    base["f_amplitude"] = (base["high"] - base["low"]) / base["close"]
    base["f_volume_change_pct"] = g["volume"].transform(lambda x: x.pct_change(1))
    base["f_volume_ratio_5d"] = base["volume"] / g["volume"].transform(lambda x: x.rolling(5).mean())
    base["f_stock_volatility_20d"] = g["close"].transform(lambda x: x.pct_change(1).rolling(20).std())

    # 均線特徵
    ma5 = g["close"].transform(lambda x: x.rolling(5).mean())
    ma20 = g["close"].transform(lambda x: x.rolling(20).mean())
    ma60 = g["close"].transform(lambda x: x.rolling(60).mean())
    base["f_bias_ma5"] = (base["close"] - ma5) / ma5
    base["f_bias_ma20"] = (base["close"] - ma20) / ma20
    base["f_bias_ma60"] = (base["close"] - ma60) / ma60
    base["f_ma5_slope"] = ma5.groupby(base["stock_id"]).transform(lambda x: x.pct_change(5))
    base["f_ma20_slope"] = ma20.groupby(base["stock_id"]).transform(lambda x: x.pct_change(10))
    base["f_momentum_cross"] = base["f_return_5d"] - base["f_return_20d"]

    # 相對大盤特徵
    base["f_relative_strength_5d"] = base["f_return_5d"] - base["_mkt_close"].pct_change(5)
    base["f_relative_strength_20d"] = base["f_return_20d"] - base["_mkt_close"].pct_change(20)
    base["f_volume_market_ratio"] = base["volume"] / base["_mkt_volume"].replace(0, np.nan)
    base["f_volatility_vs_market"] = base["f_stock_volatility_20d"] / base["f_market_volatility_20d"].replace(0, np.nan)

    base = base.drop(columns=["_mkt_close", "_mkt_volume"])
    f_cols = [c for c in base.columns if c.startswith("f_")]

    if mode == "train":
        # 訓練模式：過濾掉 target = NaN 的資料（最後20天沒有未來價格）
        has_target = base["target"].notna()
        result = base[has_target][["date", "stock_id", "target"] + f_cols].copy()
    else:
        # 預測模式：保留全部資料包含今天，不管 target 有沒有或是否為 NaN
        cols = ["date", "stock_id"] + f_cols
        if "target" in base.columns:
            cols = ["date", "stock_id", "target"] + f_cols
        result = base[cols].copy()

    return result.reset_index(drop=True)
