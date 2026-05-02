import random
from traceback import print_tb

import numpy as np
from pandas import DataFrame

import pandas as pd
from scipy.signal import argrelextrema
import pandas_ta as ta

import parquet_db as parquet_db


class MacdFeature:

    def gen_macd(df, fast=12, slow=26, signal=9):
        """計算macd"""
        # --- 1. 計算 MACD (手寫 EMA) ---
        # df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["hist"] = df["macd"] - df["signal"]
        df.drop(
            ["ema_fast", "ema_slow"],
            axis=1,
            inplace=True,
        )

    def macd(df):
        # --- 1. MACD / close (價格標度化) ---
        # 解決地雷 A：讓 1000 元的台積電與 50 元的聯電 MACD 具有可比性
        df["f_macd_h_ratio"] = df["hist"] / df["close"]

        # --- 2. MACD_Hist / ATR (波動率標準化) ---
        # 目的：消除「股性」差異。有些股平時波動大，MACD 柱狀體自然大。
        # 用 ATR (真實波幅) 來除，能看出現在的動能相對於平時波動是否「異常放大」
        # 假設你已有 atr 欄位，若無可簡單用 close.diff().abs().rolling(14).mean() 代替
        df["atr"] = df["close"].diff().abs().rolling(window=14).mean()
        df["atr"] = df["atr"].replace(0, df["atr"].median())
        df["f_macd_h_norm"] = np.where(df["atr"] > 0, df["hist"] / df["atr"], 0)

        # --- 3. MACD_Slope (變化率/斜率) ---
        # 目的：捕捉動能轉向的速度。
        # 注意：直接相除 (MACD - lag) / lag 在 MACD 跨越 0 軸時會出錯 (分母為0或正負反轉)
        # 建議改用：過去 N 天的數值差異 (diff) 再除以價格，或是直接用 diff
        df["f_macd_slope"] = df["macd"].diff(periods=3) / df["close"]  # 3日斜率相對股價比例

        # df.rename(columns={"atr": "f_macd_atr"}, inplace=True)

        return df

    def hist(df):
        """
        df 需包含: 'macdh' (MACD 柱狀體)
        """
        # 1. 柱體正負方向 (1: 正向/紅柱, -1: 負向/綠柱)
        df["f_macd_h_sign"] = np.sign(df["hist"])

        # 2. 柱體是否比昨天「高」 (Direction)
        # 不管是綠柱變短(回升)還是紅柱變長(噴發)，這都代表動能轉向上
        df["f_macd_h_direction"] = (df["hist"].diff() > 0).astype(int)

        # 3. 柱體連續上升天數 (Consecutive Up)
        # 這能讓模型知道「現在是剛翻紅」還是「已經漲了5天快力竭了」
        # 邏輯：如果今天比昨天高，計數+1，否則歸零
        h_dir = df["hist"].diff() > 0
        df["f_macd_h_rising_days"] = h_dir.groupby((h_dir != h_dir.shift()).cumsum()).cumcount() + 1
        # 若當天沒上升，則天數設為 0
        df.loc[~h_dir, "f_macd_h_rising_days"] = 0

        # 4. 柱體上升斜率 (標準化後的變動率)
        # 使用之前提到的 ATR 標準化，避免股價高低影響
        df["f_macd_h_slope_norm"] = np.where(df["atr"] > 0, df["hist"].diff() / df["atr"], 0)

        return df

    def lag_features(df, lags=[1, 2, 3]):
        # 使用你之前標準化過的 macdh_norm (MACD_Hist / ATR)
        base_col = "f_macd_h_norm"
        for lag in lags:
            # 1. 數值滯後
            f_name1 = f"f_macd_h_lag{lag}"
            df[f_name1] = df[base_col].shift(lag)

            # 2. 變化量滯後 (一階差分)
            f_name2 = f"f_macd_h_diff_lag{lag}"
            df[f_name2] = df[base_col].diff().shift(lag)

            #

        # 3. 加速度 (二階差分) - 捕捉轉折力道
        ft_name = "f_macd_h_accel"
        df[ft_name] = df[base_col].diff().diff()

        # 4. 過去 3 天是否全部都在上升 (連續性強化)
        ft_name = "f_macd_h_rising_3d"
        df[ft_name] = (df[base_col].diff() > 0).rolling(window=3).min().fillna(0).astype(int)

        return df

    def lag_feature_delay_2day(df: pd.DataFrame):
        base_name = "f_macd_hist_bull_lag"
        delay_day = 2
        hist_column = f"{base_name}0"
        df[hist_column] = 0
        MacdFeature.find_all_hist_divergences(df, hist_column)

        df[hist_column] = df[hist_column].shift(delay_day).fillna(0).astype(int)
        for i in range(3):
            lag_column = f"{base_name}{i+1}"
            df[lag_column] = df[hist_column].shift(i + 1).fillna(0).astype(int)

        return df

    def fix_macd(df):
        """定義需要修正的 MACD 相關特徵"""

        macd_lag_cols = [col for col in df.columns if "macdh" in col or "macd_h" in col]

        # 將數值限制在 -5 到 5 之間（超過 5 倍 ATR 的動能都視為 5 倍）
        df[macd_lag_cols] = df[macd_lag_cols].clip(lower=-5, upper=5)

    def add_feature(df: pd.DataFrame):
        dfclose = df.pivot(index="date", columns="stock_id", values=["close"])
        fast = 12
        slow = 26
        signal = 9
        ema_fast = dfclose.ewm(span=fast, adjust=False).mean()
        ema_slow = dfclose.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal
        #############################################################
        f_hist_change = (hist.diff() > 0).astype(int)
        f_hist_is_0 = (hist > 0).astype(int)

        #############################################################
        df = df.set_index(["date", "stock_id"])
        df["f_macdh_change"] = f_hist_change.stack(future_stack=True)
        df["f_macdh_is_0_up"] = f_hist_is_0.stack(future_stack=True)
        df = df.reset_index()
        return df

    def add_feature_bak(df: pd.DataFrame):

        MacdFeature.gen_macd(df)

        df = MacdFeature.macd(df)
        df = MacdFeature.hist(df)
        df = MacdFeature.lag_features(df)
        df = MacdFeature.lag_feature_delay_2day(df)

        MacdFeature.fix_macd(df)

        return df

    import pandas as pd
    import numpy as np
    from scipy.signal import argrelextrema

    def find_all_hist_divergences(df: pd.DataFrame, hist_column_name="f_macd_bull_lag0", delay_day=2, order_val=5):
        MacdFeature.gen_macd(df)

        # --- 2. 取得波峰與波谷的「索引」 (解決你遇到的 ValueError) ---
        # argrelextrema 回傳的是 tuple，我們取第一個元素 [0]
        peaks_idx = argrelextrema(df["hist"].values, np.greater, order=order_val)[0]
        troughs_idx = argrelextrema(df["hist"].values, np.less, order=order_val)[0]

        divergence_results = []

        if 0:
            # --- 3. 偵測頂背離 (價格新高 vs 柱體波峰萎縮) ---
            for i in range(1, len(peaks_idx)):
                curr, prev = peaks_idx[i], peaks_idx[i - 1]

                # 條件：柱體都在 0 軸以上 (真正的頂背離通常發生在正值區)

                if df["hist"].iloc[curr] > 0 and df["hist"].iloc[prev] > 0:
                    # 價格創新高，但柱體波峰下降
                    if df["high"].iloc[curr] > df["high"].iloc[prev] and df["hist"].iloc[curr] < df["hist"].iloc[prev]:
                        divergence_results.append(
                            {
                                "日期": df.index[curr],
                                "類型": "頂背離 (Bearish)",
                                "價格": df["high"].iloc[curr],
                                "Hist值": df["hist"].iloc[curr],
                            }
                        )

        # --- 4. 偵測底背離 (價格新低 vs 柱體波谷萎縮) ---
        for i in range(1, len(troughs_idx)):
            curr, prev = troughs_idx[i], troughs_idx[i - 1]

            # 條件：柱體都在 0 軸以下
            if df["hist"].iloc[curr] < 0 and df["hist"].iloc[prev] < 0:
                # 價格創新低，但柱體負值縮小 (例如從 -10 變成 -5)
                if (
                    (df["low"].iloc[curr] < df["low"].iloc[prev])
                    # (df["close"].iloc[curr] < df["close"].iloc[prev])
                ) and df["hist"].iloc[curr] > df["hist"].iloc[prev]:
                    # df.at["2026-02-26", "aa"] = 1

                    # print(f"curr = {df.index[curr]}背離了,但要{delay_day}天後才知今天是低點,所以是 {df.index[curr+delay_day]}")
                    # print(f"建特徵不用delay,交易時需要")
                    #
                    # 2026-03-05
                    # for i in range(4):
                    # df.loc[df.index[curr + i], f"f_macd_bull_lag{i}"] = 1
                    # df.to_csv("macd.csv")

                    # df.loc[df.index[curr], hist_column_name] = 1 #忘記要幹嘛

                    divergence_results.append(
                        {
                            "st": df.index[prev],
                            "end": df.index[curr],
                            "stock_id": df["stock_id"].iloc[0],
                            "類型": "底背離 (Bullish)",
                            "價格": df["low"].iloc[curr],
                            "Hist值": df["hist"].iloc[curr],
                        }
                    )

        return pd.DataFrame(divergence_results)

        # --- 使用方法 ---
        # df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
        # all_divergences = find_all_hist_divergences(df, order_val=5)
        # print(all_divergences)

        # 使用範例
        # df = pd.read_csv('your_data.csv', index_col='Date', parse_dates=True)
        # divergence_list = get_macd_hist_divergence(df)
        # print(divergence_list)

    def test():
        df = parquet_db.query_price(["2330", "0050"])
        df = MacdFeature.add_feature(df)
        print(df.tail())

        # print(df[100:].head().T)
        # df = stock("1210")
        # df["f_macd_bull_lag0"] = 0
        # MacdFeature.find_all_hist_divergences(df, "f_macd_bull_lag0")

        # for i in range(3):
        #     df[f"f_macd_bull_lag{i+1}"] = df["f_macd_bull_lag0"].shift(i + 1).fillna(0).astype(int)
        # print(
        #     df[
        #         [
        #             "f_macd_bull_lag0",
        #             "f_macd_bull_lag1",
        #             "f_macd_bull_lag2",
        #             "f_macd_bull_lag3",
        #         ]
        #     ].tail(25)
        # )

    def test_addfeature():
        pass
        # df = stock("1210")
        # df: pd.DataFrame = MacdFeature.add_feature(df)
        # # print(df.columns)
        # # print(df.tail().T)
        # # print(df.head().T)
        # print(df.describe().T)

    def test_macd():
        # count = 0
        # # 9946有錯
        # for i in random.sample(parquet_db.query_stocks_ids_list(), 100):
        #     # for i in ["3357"]:
        #     # for i in get_file_list():
        #     try:
        #         df = MacdFeature.find_all_hist_divergences(stock(i))
        #     except:
        #         continue
        #     if df.shape[0] == 0:
        #         continue
        #     df: pd.DataFrame = df[
        #         (df["end"] > "2026-01-01")
        #         & (df["end"] <= "2027")
        #         # (df["日期end"] > "2026-04") & (df["日期end"] <= "2027")
        #     ]

        #     if df.shape[0] > 0:
        #         print("*" * 30, i)
        #         # df = df[["日期end"]]
        #         print(df["end"].tail().T)
        #         # print(df.shape[0])
        #     else:
        #         continue
        #     count += df.shape[0]
        #     # if count >= 5:
        #     # break

        # print(count)
        pass

    def test_add_macd_to_df():
        # i = "3357"
        # df = stock(i)
        # df_macd = MacdFeature.find_all_hist_divergences(df)
        # macd_criteria = df_macd.rename(columns={"end": "date"})
        # macd_criteria["is_macd_bull"] = 1
        # print("macd_criteria\n", macd_criteria.head())
        # print("macd_criteria\n", macd_criteria.head())
        # df = df.merge(macd_criteria, on=["date", "stock_id"], how="left")
        # df["is_macd_bull"] = df["is_macd_bull"].fillna(0).astype(int)
        # df["is_macd_bull"] = (df["date"].isin(df_macd["end"]) & df["stock_id"].isin(df_macd["stock_id"])).astype(int)
        # # df["is_macd_bull"] = df["is_macd_bull"].rolling(window=5, min_periods=1).max()
        # df = df[df["is_macd_bull"] == 1]
        # print(f"8ffffff", df.shape[0])
        # print(df.tail().T)
        pass


# MacdFeature.test()
# JS_macd_feature.test()
# MacdFeature.test_addfeature()
# MacdFeature.test_macd()
# MacdFeature.test_add_macd_to_df()
