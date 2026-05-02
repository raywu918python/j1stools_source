import pandas as pd
import pandas
import pandas_ta as ta
from pytest import mark

import parquet_db as parquet_db


class MarketFeature:
    market_df = None

    def add_feature(df: pandas.DataFrame):
        """
        df: 個股資料
        market_df: 大盤資料 (需包含 close)
        'f_market_close_change' (大盤今日漲跌幅)
        'f_price_close_change' (個股今日漲跌幅)
        - f_market_bias_slope
        - f_market_close_change
        - f_market_ret3
        - f_market_rs_10d

        - f_market_rs_1d
        - f_market_rs_3d
        - f_market_rsi
        - f_market_rsi_high
        - f_market_rsi_low
        - f_market_stock_rsi
        """
        if MarketFeature.market_df is None:
            df0050 = parquet_db.query_price(["0050"])
            df0050.set_index("date", drop=False, inplace=True)
            df0050["m_close_change"] = df0050["close"].pct_change()
            df0050["m_ema20"] = df0050["close"].ewm(span=20).mean()
            df0050["m_close_change_3"] = df0050["close"].pct_change(3)
            df0050["f_market_close_change"] = df0050["close"].pct_change()

        dfclose = df.pivot(index="date", columns="stock_id", values="close")
        #
        # f_market_rs_10d,f_market_rs_3d,f_market_rs_1d
        #
        stockchange = dfclose.pct_change(10, fill_method=None)
        marketchange = df0050["close"].pct_change(10)
        f_market_rs_10d = stockchange.sub(marketchange, axis=0)
        stockchange = dfclose.pct_change(3, fill_method=None)
        marketchange = df0050["close"].pct_change(3)
        f_market_rs_3d = stockchange.sub(marketchange, axis=0)
        stockchange = dfclose.pct_change(1, fill_method=None)
        marketchange = df0050["close"].pct_change(1)
        f_market_rs_1d = stockchange.sub(marketchange, axis=0)
        #
        # f_market_stock_rsi
        #
        dfstockrsi = dfclose.apply(lambda x: ta.rsi(x, length=14))
        df0050rsi = ta.rsi(df0050["close"], length=14)
        f_market_stock_rsi = (dfstockrsi.sub(df0050rsi, axis=0)) * 0.01
        #
        # f_market_dist,f_market_bias_slope
        #
        stockchange = dfclose.pct_change(1, fill_method=None)
        marketchange = df0050["close"].pct_change(1)
        f_market_rs_1d = stockchange.sub(marketchange, axis=0)
        f_market_dist = f_market_rs_1d.div(marketchange, axis=0)
        f_market_bias_slope = f_market_dist.diff(3)
        #
        # set to value
        #
        df = df.set_index(["date", "stock_id"])
        df["f_market_rs_10d"] = f_market_rs_10d.stack(future_stack=True)
        df["f_market_rs_3d"] = f_market_rs_3d.stack(future_stack=True)
        df["f_market_rs_1d"] = f_market_rs_1d.stack(future_stack=True)
        df["f_market_stock_rsi"] = f_market_stock_rsi.stack(future_stack=True)
        df["f_market_dist"] = f_market_dist.stack(future_stack=True)
        df["f_market_bias_slope"] = f_market_bias_slope.stack(future_stack=True)
        df = df.reset_index()
        df["f_market_rsi_high"] = df["date"].map((df0050rsi > 0.70).astype(int))
        df["f_market_rsi_low"] = df["date"].map((df0050rsi < 0.30).astype(int))
        df["f_market_rsi"] = df["date"].map(df0050rsi) * 0.01
        df["f_market_rsi"] = df["date"].map(df0050rsi) * 0.01
        df["f_market_ret3"] = df["date"].map(df0050["close"].pct_change(3))
        df["f_market_close_change"] = df["date"].map(df0050["f_market_close_change"])

        df0050 = df0050.reset_index(drop=True)
        #############################################################

        # df = df.merge(df0050[["date", "m_ema20", "m_close_change", "m_close_change_3"]], on="date", how="left")
        # df["close_change"] = df.groupby("stock_id")["close"].pct_change()
        # df["f_market_dist"] = (df["close_change"] - df["m_close_change"]) / df["m_close_change"]
        # # df["f_market_dist_3"] = (df["close_change"] - df["m_close_change_3"]) / df["m_close_change_3"]
        # # df["f_market_dist_ema"] = (df["close"] - df["m_ema20"]) / df["m_ema20"]

        # ####
        # # 1. 乖離率斜率：正確！使用 groupby 避免跨個股計算 diff
        # df["f_market_bias_slope"] = df.groupby("stock_id")["f_market_dist"].diff(3)
        # # 2. 映射大盤資料：map 是正確的，但不需要 groupby("stock_id")
        # # 因為 date 對應的大盤數據是不分股票的，直接用 map 即可
        # temp = df0050.set_index("date")["f_market_close_change"]
        # df["f_market_close_change"] = df["date"].map(temp)

        # # 3. 大盤報酬率：不能賦值給 groupby 物件
        # # 先在 market_df 算好，再 map 回主表
        # market_temp = df0050.set_index("date")
        # df["f_market_ret3"] = df["date"].map(market_temp["close"].pct_change(3))
        # df["ret_10d_market"] = df["date"].map(market_temp["close"].pct_change(10))

        # # 4. 個股報酬率：正確！使用 groupby
        # df["ret_10d_stock"] = df.groupby("stock_id")["close"].pct_change(10)

        # # 5. 相對強度 (RS)：直接相減即可
        # df["f_market_rs_10d"] = df["ret_10d_stock"] - df["ret_10d_market"]
        # ##############
        # if "f_price_close_change" not in df.columns:
        #     df["f_price_close_change"] = df.groupby("stock_id")["close"].pct_change()
        # df["f_market_rs_1d"] = df["f_price_close_change"] - df["f_market_close_change"]
        # df["ret_3d_stock"] = df.groupby("stock_id")["close"].pct_change(3)
        # # 建議先準備一個以日期為索引的 Series
        # market_close_series = df0050.set_index("date")["close"]
        # df["ret_3d_market"] = df["date"].map(market_close_series.pct_change(3))
        # df["f_market_rs_3d"] = df["ret_3d_stock"] - df["ret_3d_market"]
        # rsi_series = ta.rsi(market_close_series, length=14) / 100
        # df["f_market_rsi"] = df["date"].map(rsi_series)
        # df["f_market_rsi_high"] = (df["f_market_rsi"] > 0.70).astype(int)  # 過熱區
        # ##########
        # df["f_market_rsi_low"] = (df["f_market_rsi"] < 0.30).astype(int)  # 超跌區
        # # # 大盤 10 日乖離變化 (Market Bias Change)：
        # df["rsi"] = df.groupby("stock_id")["close"].transform(MarketFeature.calc_rsi) / 100
        # df["f_market_stock_rsi"] = df["rsi"] - df["f_market_rsi"]  # 個股比大盤強多少

        # df.drop(
        #     ["m_ema20", "m_close_change", "m_close_change_3"],
        #     axis=1,
        #     inplace=True,
        # )
        return df

    def calc_rsi(x):
        return ta.rsi(x, length=14)

    # def add_relative_strength_features(df):
    #     """
    #     df 需包含:
    #     'f_price_close_change' (個股今日漲跌幅)
    #     'f_market_close_change' (大盤今日漲跌幅)
    #     """
    #     # 1. 超短期相對強度 (1日)：今天的抗跌/領漲力
    #     df["f_rs_1d"] = df["f_price_close_change"] - df["f_market_close_change"]

    #     # 2. 短期相對強度 (3日)：累積三天的強度
    #     # 先算出個股與大盤的 3 日累積漲幅
    #     df["ret_3d_stock"] = df["close"].pct_change(3)
    #     df["ret_3d_market"] = df["market_close"].pct_change(3)  # 假設你有大盤收盤價欄位
    #     df["f_rs_3d"] = df["ret_3d_stock"] - df["ret_3d_market"]

    #     # 3. 中期相對強度 (10日)：波段的慣性
    #     df["ret_10d_stock"] = df["close"].pct_change(10)
    #     df["ret_10d_market"] = df["market_close"].pct_change(10)
    #     df["f_rs_10d"] = df["ret_10d_stock"] - df["ret_10d_market"]

    #     # 移除中間計算用的欄位
    #     df.drop(
    #         ["ret_3d_stock", "ret_3d_market", "ret_10d_stock", "ret_10d_market"],
    #         axis=1,
    #         inplace=True,
    #     )

    #     return df

    def test():

        # df = stock("1215")
        # df = stock("2072")  # 資料量很少
        # df = stock("6269")  # 資料量很少
        # print(df.shape)
        # print(df.isnull().any())
        df = parquet_db.query_price(["2330", "2360"])
        # PriceFeature.add_feature(df)
        df = MarketFeature.add_feature(df)
        # df["rsi"] = pandas_ta.rsi(df["close"], length=14)  # / 100
        # print(df.head(20))
        # JS_market_feature.add_feature(df)
        # print(df.describe().T)

        # print(df["f_market_rs_10d"][0])
        # "['f_market_rs_10d', 'f_market_rs_3d', 'f_market_rs_1d', 'f_market_stock_rsi', 'f_market_rsi', 'f_market_bias_slope'] not in index"
        # print(df.describe().T)
        print(df[100:].head().T)

    def check_multi_window_corr(stock_df, market_df):
        """
        傳入包含日期索引 (Datetime Index) 的 DataFrame
        """
        # 1. 計算收益率並對齊日期
        # 使用 inner join 確保日期完全一致
        combined = pd.concat(
            [stock_df["close"].pct_change(), market_df["close"].pct_change()],
            axis=1,
            join="inner",
        ).dropna()

        combined.columns = ["stock", "market"]

        # 如果對齊後的資料長度不足最大窗口 (120)，直接剔除
        if len(combined) < 120:
            return False, None

        windows = [20, 60, 120]
        corrs = {}

        # 2. 針對對齊後的資料進行計算
        for w in windows:
            # 取得最後 w 筆資料
            subset = combined.tail(w)
            corrs[f"w{w}"] = subset["stock"].corr(subset["market"])

        return corrs

    def test1():
        pass
        # stock_df = stock("2330")
        # rs = MarketFeature.check_multi_window_corr(stock_df, market_df)
        # print(rs)


# MarketFeature.test()
# JS_market_feature.test()
# JS_market_feature.test1()
