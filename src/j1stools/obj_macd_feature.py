import random
from traceback import print_tb

import numpy as np
from pandas import DataFrame

import pandas as pd
from scipy.signal import argrelextrema
import pandas_ta as ta

import j1stools.parquet_db as parquet_db


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
        # f_macdh_update = (hist.diff() > 0).astype(int)
        f_macdh_change = hist.pct_change()
        # f_macdh_up_0down = (f_macdh_change > 0) & (hist < 0)
        # f_macdh_down_up = (
        #     (hist > hist.shift(1))
        #     & (hist.shift(1) > hist.shift(2))
        #     & (hist.shift(2) > hist.shift(3))
        #     & (hist.shift(3) < hist.shift(4))
        f_macdh_slope01 = hist - hist.shift(1)
        f_macdh_slope12 = hist.shift(1) - hist.shift(2)
        f_macdh_slope23 = hist.shift(2) - hist.shift(3)
        f_macdh_slope_up = ((f_macdh_slope01 > f_macdh_slope12) & (f_macdh_slope12 > f_macdh_slope23)).astype(int)
        f_hist_is_0 = (hist > 0).astype(int)
        #############################################################
        # 結合 MACD 柱狀體當前的位置
        # 如果 hist < 0 且在上升，通常是跌深反彈
        # 如果 hist > 0 且在上升，通常是強勢噴發
        f_hist_pos_direction = (hist > hist.shift(1)).astype(int) * np.sign(hist)
        #############################################################
        # 動能化：計算「連續上升天數」
        # 1. 判斷是否上升 (1 為上升, 0 為持平或下降)
        is_rise = (hist > hist.shift(1)).astype(int)

        # 2. 計算連續天數 (寬表格向量化技巧)
        # 原理：利用累積加總減去「最近一次歸零點」的累積值，達成歸零重計的效果
        def get_consecutive_rise_wide(df_is_rise):
            cumsum = df_is_rise.cumsum()
            # 找到 0 的位置，並記錄當時的 cumsum 值，其餘補缺失值後用 ffill 往下填
            reset_points = cumsum.where(df_is_rise == 0).ffill().fillna(0)
            return cumsum - reset_points

        # 產生特徵
        f_hist_rise_days = get_consecutive_rise_wide(is_rise)
        #############################################################
        # 如果你想更進一步抓「起漲點」
        # 只有在 MACD 柱狀體位於零軸以下、且連續上升天數為 1 或 2 時，才標記為潛在起漲點
        f_potential_start = ((hist < 0) & (f_hist_rise_days >= 1) & (f_hist_rise_days <= 2)).astype(int)
        #############################################################
        # 今天斜率比昨天斜率增加多少（加速度）
        f_macdh_accel = f_macdh_slope01 - f_macdh_slope01.shift(1)
        #############################################################
        # 只有在 hist 為負時的斜率，這能過濾掉高檔震盪的干擾
        f_bottom_slope = f_macdh_slope01.where(hist < 0, 0)
        #############################################################
        # 背離
        # 1. 取得過去 20 天的價格與 hist 最低點（不含當天，避免當天就是最低點導致無法比較）
        past_price_min = dfclose.shift(1).rolling(window=20).min()
        past_hist_min = hist.shift(1).rolling(window=20).min()

        # 2. 定義底背離結構
        # 條件 A：股價創 20 日新低（或接近新低）
        # 條件 B：MACD 柱狀體比過去 20 天的最低點還高（能量墊高）
        # 條件 C：hist 必須在零軸下（水底背離才有意義）
        f_div_signal = (((dfclose <= past_price_min) & (hist > past_hist_min) & (hist < 0))).astype(int)

        # 3. 數值化特徵：背離強度 (Divergence Strength)
        # 原理：(當前 hist - 過去最低 hist) -> 數值越大，代表能量墊高的力道越強
        f_div_strength = (hist - past_hist_min) * f_div_signal
        ##############################################################
        # 強力起漲訊號：有背離結構 + hist 剛好反轉向上
        f_div_with_slope = (f_div_signal & (hist > hist.shift(1))).astype(int)
        #############################################################
        # 2. 進化版特徵：背離深度與能量比（數值型）不要只給 0 或 1，給模型「程度」的資訊，這能顯著提升 LightGBM/XGBoost 的特徵重要性：
        # 1. 股價與 hist 的波段低點
        past_price_min = dfclose.shift(1).rolling(window=20).min()
        past_hist_min = hist.shift(1).rolling(window=20).min()

        # 2. 計算「背離程度」：數值越高，代表價格創新低但動能墊得越高
        # 即使 close 還沒完全跌破 past_price_min，接近時也可以計算強度
        f_div_score = (hist - past_hist_min) / (abs(past_hist_min) + 1e-9)

        # 3. 結合條件：只有在股價接近低點 (例如低點 3% 內) 且 hist 在水底時才保留數值
        # 這樣可以過濾掉無意義的區域，讓特徵更集中在「起漲區」
        f_div_refined = f_div_score.where((dfclose <= past_price_min * 1.03) & (hist < 0), 0)
        #############################################################
        # 3. 加入「乖離率」增強效果要達成 30 天 15% 的漲幅，通常起漲點會伴隨著負乖離過大。將背離與乖離結合，是模型最愛的特徵組合：
        # 股價與 20 日均線的距離
        f_bias_20 = (dfclose - dfclose.rolling(20).mean()) / dfclose.rolling(20).mean()

        # 組合特徵：低乖離 + 強背離
        # 這能讓模型學到：當股價跌得很慘 (Bias 負很多) 且 MACD 已經背離時，就是 +15% 的候選者

        #############################################################
        df = df.set_index(["date", "stock_id"])
        df["f_bias_20"] = f_bias_20.stack(future_stack=True)
        df["f_div_refined"] = f_div_refined.stack(future_stack=True)
        df["f_div_score"] = f_div_score.stack(future_stack=True)
        df["f_div_with_slope"] = f_div_with_slope.stack(future_stack=True)
        df["f_div_strength"] = f_div_strength.stack(future_stack=True)
        df["f_div_signal"] = f_div_signal.stack(future_stack=True)
        df["f_bottom_slope"] = f_bottom_slope.stack(future_stack=True)
        df["f_macdh_accel"] = f_macdh_accel.stack(future_stack=True)
        df["f_hist_is_0"] = f_hist_is_0.stack(future_stack=True)
        df["f_macdh_slope_up"] = f_macdh_slope_up.stack(future_stack=True)
        df["f_potential_start"] = f_potential_start.stack(future_stack=True)
        df["f_hist_pos_direction"] = f_hist_pos_direction.stack(future_stack=True)
        df["f_hist_rise_days"] = f_hist_rise_days.stack(future_stack=True)
        df["f_macdh_slope01"] = f_macdh_slope01.stack(future_stack=True)
        df["f_macdh_slope12"] = f_macdh_slope12.stack(future_stack=True)
        df["f_macdh_slope23"] = f_macdh_slope23.stack(future_stack=True)
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
