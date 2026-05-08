import numpy as np
import pandas as pd
from pandas import DataFrame

from j1stools.obj_macd_feature import MacdFeature


class FilterData:
    def get_data(df: pd.DataFrame, log=False, is_del_ich=False, is_del_atr=True) -> DataFrame:
        """
        定義你的「進場門檻」
        只有符合這個門檻的資料，我們才關心它的 Label
        """

        before = df["target"].copy().value_counts().sort_index().tolist()

        # df = FilterData.volume_price_break(df)  # 看起來沒用
        # df = FilterData.hw3_vol520(df)
        # df = FilterData.top_10k(df)

        # df = FilterData.ma1020(df)  # 做多比重提升
        # df = FilterData.price_break_3d(df)  # 做多比重提升
        # df = FilterData.atr_breakout(df)  # 做多比重提升
        # df = FilterData.abcd(df)

        # 刪除 df
        df = FilterData.ichimoku(df, is_del_ich)  # 做多比重提升
        df = FilterData.always_del_limit_up(df)
        df = FilterData.always_del_atr(df, is_del_atr)

        if log:
            after = df["target"].copy().value_counts().sort_index().tolist()
            FilterData.print_log(before, after)
        return df

    def filter_ma(df):
        if "ema10" not in df.columns:
            df["ema10"] = df["close"].ewm(span=10).mean()

        if "ema20" not in df.columns:
            df["ema20"] = df["close"].ewm(span=20).mean()

        if "ema60" not in df.columns:
            df["ema60"] = df["close"].ewm(span=60).mean()

        df["is_ma_bull"] = ((df["ema10"] > df["ema20"]) & (df["close"] > df["ema10"])).astype(int)
        df["is_ma_bull"] = df["is_ma_bull"].rolling(window=5, min_periods=1).max()

        df = df[(df["is_macd_bull"] == 1)]

        return df

    def set_is_above_mid(df, name="f_price_is_above_mid", window=5):
        df["prev_5d_high"] = df["high"].shift(1).rolling(window=window).max()
        df["prev_5d_low"] = df["low"].shift(1).rolling(window=window).min()

        # 2. 計算前5天的中間價（最高與最低的平均）
        df["prev_5d_mid"] = (df["prev_5d_high"] + df["prev_5d_low"]) / 2
        # df[name] = (df["close"] - df["prev_5d_mid"]) / df["prev_5d_mid"]
        df[name] = (df["close"] > df["prev_5d_mid"]).astype(int)

        return df

    def add_macd_label(df):
        """macd 背離後 5 天，交易"""

        df_macd: pd.DataFrame = MacdFeature.find_all_hist_divergences(df)
        macd_criteria = df_macd[["end", "stock_id"]].rename(columns={"end": "date"}).drop_duplicates()

        macd_criteria["is_macd_bull"] = 1
        df = df.merge(macd_criteria, on=["date", "stock_id"], how="left")
        cols = ["stock_id", "date", "is_macd_bull"]  # 定義你想看的欄位
        # print(df.loc[df["is_macd_bull"] == 1, cols].head(11))

        df["is_macd_bull"] = macd_criteria["is_macd_bull"].fillna(0).astype(int)
        df["is_macd_bull"] = (df["date"].isin(macd_criteria["date"]) & df["stock_id"].isin(df_macd["stock_id"])).astype(
            int
        )

        # df["is_macd_bull_next_break"] = df["is_macd_bull"]
        df["is_macd_bull_next_break"] = (
            (df["is_macd_bull"].shift(1) == 1)
            # & (df["close"] > df["low"].shift(1))
            & (df["close"] > df["high"].shift(1))
            & (df["volume"] > df["volume"].rolling(5).mean())
        ).astype(int)

        df = df.set_index("date", drop=False).rename_axis("Date")
        return df

    def find_pivots_n(df, n=5):
        df = df.copy()

        # center=True
        df["pivot_high"] = df["high"] == df["high"].rolling(n, center=True).max()

        df["pivot_low"] = df["low"] == df["low"].rolling(n, center=True).min()

        df["pivot"] = np.where(df["pivot_high"], "H", np.where(df["pivot_low"], "L", None))

        return df

    def filter_volume(df):
        df["vol20"] = df["volume"].rolling(20).mean()
        df = df[df["volume"] < (df["vol20"] * 0.5)]
        return df

    def detect_n_pattern(df, min_rise=0.05, min_pullback=0.02):
        # troughs_idx = argrelextrema(df["hist"].values, np.less, order=order_val)[0]

        pivots = df.dropna(subset=["pivot"])
        # pd.DataFrame(pivots[["date", "pivot"]]).to_csv("2330tmp.csv")
        # print(pivots[["date", "pivot"]].head())
        # print(pivots[pivots["date", "pivot"]])
        signals = []

        for i in range(2, len(pivots)):
            A = pivots.iloc[i - 2]
            B = pivots.iloc[i - 1]
            C = pivots.iloc[i]

            # 只找 L H L
            if A.pivot == "L" and B.pivot == "H" and C.pivot == "L":

                # N字最小漲幅 5%
                cond1 = (B.high - A.low) / A.low >= min_rise

                # 回檔不破 2%
                cond2 = (C.low - A.low) / A.low >= min_pullback

                # 條件
                # cond1 = B.high > A.low
                # cond2 = C.low > A.low

                if cond1 and cond2:

                    # 往後找突破B
                    future = df.loc[C.name :]

                    for idx, row in future.iterrows():

                        # 如果先出現新的 pivot → 結構失敗
                        if idx != C.name and row["pivot"] in ["H", "L"]:
                            break

                        # 突破B
                        if row["close"] > B.high:
                            signals.append(idx)

                            # signals.append(
                            #     str(idx)
                            #     + " "
                            #     + str(C["date"])
                            #     + " "
                            #     + str(B["date"])
                            #     + " "
                            #     + str(A["date"])
                            # )

                            break

                    # breakout = future[future["high"] > B.high]

                    # if (
                    #     len(breakout) > 0
                    #     and str(breakout.index[0]) == "2026-01-06 00:00:00"
                    # ):
                    #     signals.append(
                    #         str(breakout.index[0])
                    #         + " "
                    #         + str(C["date"])
                    #         + " "
                    #         + str(B["date"])
                    #         + " "
                    #         + str(A["date"])
                    #     )

        return signals

    def abcd(df: pd.DataFrame):

        df = FilterData.find_pivots_n(df, n=7)  # ← 這裡調整天數
        # signals = FilterData.detect_n_pattern(df)
        date_list = FilterData.detect_n_pattern_fast(df, max_days=20)
        # print(len(signals))
        # for s in signals:
        #     print(s)

        if FilterData.IS_DELETE_DATA:
            df = df[df["date"].isin(date_list)]
        else:
            df["f_is_abcd"] = np.where(df["date"].isin(date_list), 1, 0)
        return df

    def test_abcd():
        pass
        # df = stock("2330")
        # # df = df[df["date"] > "2024-12"]
        # df: pd.DataFrame = FilterData.abcd(df)
        # print("df.shape[0]", df.shape[0], sep="\n")
        # print("df.shape[0]", df.tail(), sep="\n")

    def detect_n_pattern_fast(df, min_rise=0.10, min_pullback=0.03, max_days=5):
        # 1. 向量化取得 A, B 點資訊
        pivots = df.dropna(subset=["pivot"]).copy()
        pivots["A_low"] = pivots["low"].shift(2)
        pivots["B_high"] = pivots["high"].shift(1)
        pivots["A_pivot"] = pivots["pivot"].shift(2)
        pivots["B_pivot"] = pivots["pivot"].shift(1)

        # 2. 篩選符合條件的 C 點 (L-H-L)
        mask = (
            (pivots["A_pivot"] == "L")
            & (pivots["B_pivot"] == "H")
            & (pivots["pivot"] == "L")
            & ((pivots["B_high"] - pivots["A_low"]) / pivots["A_low"] >= min_rise)
            & ((pivots["low"] - pivots["A_low"]) / pivots["A_low"] >= min_pullback)
        )
        valid_c_points = pivots[mask]

        signals = []

        # 3. 遍歷符合條件的 C 點尋找突破
        for c_idx_label, c_row in valid_c_points.iterrows():
            c_pos = df.index.get_loc(c_idx_label)
            b_high = c_row["B_high"]

            # 搜尋範圍：從 C 的下一筆開始，長度為 max_days
            # 但不能超過 df 總長度
            search_end = min(c_pos + 1 + max_days, len(df))
            future = df.iloc[c_pos + 1 : search_end]

            if future.empty:
                continue

            # 檢查區間內是否出現新的 pivot (結構失效)
            # 找出第一個 pivot 的「整數位置」
            pivot_in_future = future["pivot"].dropna()
            if not pivot_in_future.empty:
                # 如果有新 pivot，只看新 pivot 之前的資料
                first_pivot_label = pivot_in_future.index[0]
                limit_pos = df.index.get_loc(first_pivot_label)
                # 更新搜尋區間 (只到新 pivot 出現那一列為止)
                future = df.iloc[c_pos + 1 : limit_pos + 1]

            # 尋找第一次收盤價突破 B 高點
            breakout = future[future["close"] > b_high]

            if not breakout.empty:
                # log = str(breakout.index[0]) + " " + str(c_row["date"])
                # print(log)

                signals.append(breakout.index[0])  # 取得突破當日的 Index
        df.drop(columns=["pivot_high", "pivot_low", "pivot"], inplace=True)
        return signals

    import pandas as pd

    def filter_low_volume_half(df):
        """
        刪除成交量 (volume) 小於 20日均量 (vol20) 一半的資料，並直接回傳結果
        """
        # 計算 20 日移動平均量
        vol20 = df["volume"].rolling(window=20).mean()

        # 篩選條件：保留 volume >= (vol20 * 0.5) 的資料
        # 注意：前 19 筆資料因無法計算 vol20 (NaN) 會被自動剔除
        df_filtered = df[df["volume"] >= (vol20 * 0.5)].copy()

        return df_filtered

    # 使用方式：
    # df = filter_low_volume_half(df)

    import pandas as pd

    def filter_above_any_ma(df):
        """
        只要 close 在 MA5, MA10 或 MA20 其中一個之上（或持平），就保留。
        只有當 close 同時低於這三條線時，才會被刪除。
        """
        # 1. 計算均線
        ma5 = df["close"].rolling(window=5).mean()
        ma10 = df["close"].rolling(window=10).mean()
        ma20 = df["close"].rolling(window=20).mean()

        # 2. 設定篩選條件 (使用 | 代表 OR)
        condition = (df["close"] >= ma5) | (df["close"] >= ma10) | (df["close"] >= ma20)

        # 3. 篩選並回傳
        return df[condition].copy()

    # 使用範例：
    # df_result = filter_above_any_ma(df)

    import pandas as pd

    def price_break_3d(df):
        """
        篩選條件：
        1. 量能：volume > vol5 且 volume > vol20
        2. 突破：今日 close > 前 5 日中至少 3 個交易日的高點 (high)
        3. 趨勢：今日 close > ma10 且今日 close > ma20
        """
        # 避免影響原始資料
        df = df.copy()

        # --- 1. 計算均線與均量 ---
        df["vol5"] = df["volume"].rolling(window=5).mean()
        df["vol20"] = df["volume"].rolling(window=20).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()

        # --- 2. 判斷價格突破條件 ---
        # 我們要檢查今日 close 是否大於前 5 日的 high
        # shift(1) 是為了不包含今天的 high，只看過去 5 天
        prev_5h = pd.concat([df["high"].shift(i) for i in range(1, 6)], axis=1)

        # 計算今日收盤價大於過去 5 天中幾個高點
        # (prev_5h.lt(df['close'], axis=0)) 會回傳布林值矩陣，sum(axis=1) 加總 True 的數量
        breakout_count = prev_5h.lt(df["close"], axis=0).sum(axis=1)

        # --- 3. 綜合條件篩選 ---
        cond_vol = (df["volume"] > df["vol5"]) & (df["volume"] > df["vol20"])
        cond_trend = (df["close"] > df["ma10"]) & (df["close"] > df["ma20"])
        cond_breakout = breakout_count >= 3

        # 合併所有條件
        final_condition = cond_vol & cond_trend & cond_breakout
        # 執行過濾並回傳

        if FilterData.IS_DELETE_DATA:
            df = df[final_condition].copy()
        else:
            df["f_price_break_3d"] = final_condition.astype(int)

        return df
        # 使用範例：
        # df_signal = filter_breakout_strategy(df)

    def volume_price_break(df):
        """
        昨天價格上漲，且量破 5日均量，且今日價格不破昨天中間價
        """

        # 1. 計算 5 日均量
        df["VMA5"] = df["volume"].rolling(window=5).mean()

        # 2. 定義「昨日」的相關數據 (將資料下移一列)
        yesterday_up = df["close"].shift(1) > df["open"].shift(1)  # 昨天上漲
        yesterday_vol_ok = df["volume"].shift(1) > df["VMA5"].shift(1)  # 昨天量破 5 日均量
        yesterday_mid = (df["high"].shift(1) + df["low"].shift(1)) / 2  # 昨天中間價

        # 3. 判斷今日條件
        # 今天最低價不破昨天中間價，且昨天符合上漲與放量條件
        condition = (df["low"] >= yesterday_mid) & yesterday_up & yesterday_vol_ok

        # 4. 寫入欄位：成立為 1，不成立為 2
        if FilterData.IS_DELETE_DATA:
            df = df[condition].copy()
        else:
            df["f_volume_price_break"] = np.where(condition, 1, 0)
        return df

    def ma1020(df):
        """
        整合篩選條件：
        1. 趨勢：價格在 MA10 或 MA20 之上 (OR)
        2. 量能：成交量大於 Vol5 或 Vol20 (OR)
        3. 方向：今日價格大於昨日價格 (上漲)
        """
        # 複製一份 df 避免影響原始資料
        df = df.copy()

        # --- 1. 計算技術指標 ---
        # 使用 min_periods=1 確保資料前期不因 NaN 被剔除
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["vol5"] = df["volume"].rolling(window=5).mean()
        df["vol20"] = df["volume"].rolling(window=20).mean()

        # --- 2. 定義三大濾網 ---
        # 條件一：價格在 MA10 或 MA20 以上
        cond_price_ma = (df["close"] >= df["ma10"]) | (df["close"] >= df["ma20"])

        # 條件二：量大於 Vol5 或 Vol20
        cond_vol_ma = (df["volume"] > df["vol5"]) | (df["volume"] > df["vol20"])

        # 條件三：今日價格上漲 (Close > Prev Close)
        cond_is_up = df["close"] > df["close"].shift(1)

        # --- 3. 執行合併篩選 (AND 連結三大主條件) ---
        final_condition = cond_price_ma & cond_vol_ma & cond_is_up

        # 篩選並刪除輔助用的計算欄位（如果想保留欄位，可註解掉 drop）
        if FilterData.IS_DELETE_DATA:
            df = df[final_condition]
        else:
            df["f_ma1020"] = final_condition.astype(int)

        df = df.drop(columns=["ma10", "ma20", "vol5", "vol20"])
        return df

    # 使用範例：
    # df_filtered = filter_combined_strategy(df)

    # 使用方式：
    # df = keep_above_ma10_or_20(df)

    import pandas as pd

    def ichimoku(df: pd.DataFrame, is_del=False):
        # 確保資料按 stock_id 和時間排序，這是 groupby 操作的基礎
        df = df.sort_values(["stock_id", "date"])
        grouped = df.groupby("stock_id")

        # --- 1. 計算一目均衡表指標 ---
        # 使用 transform 配合 rolling 可以保持跟原 df 一樣的索引長度
        def get_rolling_mid(series, window):
            return (series.rolling(window).max() + series.rolling(window).min()) / 2

        tenkan = grouped["high"].transform(lambda x: x.rolling(9).max()) + grouped["low"].transform(
            lambda x: x.rolling(9).min()
        )
        tenkan /= 2

        kijun = grouped["high"].transform(lambda x: x.rolling(26).max()) + grouped["low"].transform(
            lambda x: x.rolling(26).min()
        )
        kijun /= 2

        # 先行帶 A 與 B：必須在 group 內 shift，否則會跨股票
        # Senkou A 是 (轉+基)/2 往後推 26 期
        senkou_a = ((tenkan + kijun) / 2).groupby(df["stock_id"]).shift(26)

        # Senkou B 是 52日中位數 往後推 26 期
        span_b_mid = (
            grouped["high"].transform(lambda x: x.rolling(52).max())
            + grouped["low"].transform(lambda x: x.rolling(52).min())
        ) / 2
        senkou_b = span_b_mid.groupby(df["stock_id"]).shift(26)

        # --- 2. 判定條件 ---
        # 雲層頂部
        kumo_top = np.maximum(senkou_a, senkou_b)

        # 條件 1: 今日收盤價 > 今日雲層頂部
        cond_price_above_kumo = df["close"] > kumo_top

        # 條件 2: 滯後指標 (Chikou Span) 在雲之上
        # 邏輯：26 天前的收盤價 > 26 天前的雲層頂部
        # 注意：這裡直接用已經算好的布林值進行 group shift 最安全
        chikou_above_kumo = cond_price_above_kumo.groupby(df["stock_id"]).shift(26)

        # --- 3. 執行篩選 ---
        # 確保沒有 NaN 的干擾（shift產生的前26筆會是 NaN）
        final_cond = (cond_price_above_kumo == True) & (chikou_above_kumo == True)

        if is_del:
            return df[final_cond].copy()
        else:
            # df["f_ich"] = np.where(final_cond, 1, 0)
            return df

    import pandas as pd

    def hw3_vol520(df):
        """
        篩選活躍度條件（滿足其一即保留）：
        1. 量能：成交量 (volume) > Vol5 或 > Vol20
        2. 波動：當日高低價振幅 (high-low) / yesterday_close > 3%
        """
        df = df.copy()

        # --- 1. 計算均量 ---
        vol5 = df["volume"].rolling(window=5, min_periods=1).mean()
        vol20 = df["volume"].rolling(window=20, min_periods=1).mean()

        # --- 2. 計算波動率 (振幅) ---
        # 振幅通常以昨日收盤價為基點，若無昨日收盤則以今日開盤計
        prev_close = df["close"].shift(1)
        amplitude = (df["high"] - df["low"]) / prev_close

        # --- 3. 定義條件 ---
        # 條件 A：量大於 5日或 20日均量
        cond_vol = (df["volume"] > vol5) | (df["volume"] > vol20)

        # 條件 B：高低點波動大於 3%
        cond_amplitude = amplitude > 0.03

        # --- 4. 執行篩選 (OR 邏輯) ---
        # 只要符合量大 OR 波動大，就保留

        if FilterData.IS_DELETE_DATA:
            df = df[cond_vol & cond_amplitude]
        else:
            # 確保 cond_vol 和 cond_amplitude 的長度與 df 一致
            df["f_hw3_vol520"] = (cond_vol.values & cond_amplitude.values).astype(int)

        return df

    def top_10k(df):
        """
        篩選條件：
        1. 拿今日 Close 與 (前1日High, 前2日High, ..., 前10日High) 這 10 個數值比對。
        2. 今日 Close > 這 10 個 High 的次數 >= 5。
        3. 今日 Close < 這 10 個 High 的次數 >= 5。
        """
        df = df.copy()

        # 1. 取得前 1 到前 10 日的 High (產生 10 個 Column)
        # 我們用一個 DataFrame 來存放這 10 天的高點
        prev_highs = pd.concat([df["high"].shift(i) for i in range(1, 11)], axis=1)

        # 2. 逐一比對今日 Close
        # 比對今日 Close 是否大於這 10 天的高點
        # lt 代表 less than，這裡是用來判斷「前 10 日 High 是否小於今日 Close」
        above_count = prev_highs.lt(df["close"], axis=0).sum(axis=1)

        # 比對今日 Close 是否小於這 10 天的高點
        # gt 代表 greater than，判斷「前 10 日 High 是否大於今日 Close」
        below_count = prev_highs.gt(df["close"], axis=0).sum(axis=1)

        # 3. 執行篩選
        # 條件：高於的次數至少 5 次，且低於的次數至少 5 次
        # (這代表收盤價正好落在這 10 天高點的中間分佈區)
        condition = (above_count >= 5) & (below_count >= 5)

        if FilterData.IS_DELETE_DATA:
            df = df[condition]
        else:
            # df["f_top_10k"] = np.where((top_10k >= 6) & (below_count >= 3), 1, 0)
            df["f_top_10k"] = condition.astype(int)

        return df

    def atr_breakout(df):
        """
        篩選條件：
        1. ATR 變化量：今日 ATR > 過去 10 日 ATR 平均值
        2. 量能：volume > vol5 且 volume > vol20
        3. 趨勢：close > ma10 且 close > ma20
        """
        df = df.copy()

        # --- 1. 計算 ATR (14日為標準週期) ---
        high_low = df["high"] - df["low"]
        high_pc = (df["high"] - df["close"].shift(1)).abs()
        low_pc = (df["low"] - df["close"].shift(1)).abs()

        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # --- 2. 計算其餘指標 ---
        df["atr_ma10"] = df["atr"].rolling(window=10).mean()
        df["vol5"] = df["volume"].rolling(window=5).mean()
        df["vol20"] = df["volume"].rolling(window=20).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()

        # --- 3. 執行條件篩選 ---
        # 條件 1: 今日 ATR 大於前 10 日 ATR 平均
        cond_atr = df["atr"] > df["atr_ma10"]

        # 條件 2: 量能大於 5日與 20日均量 (AND)
        cond_vol = (df["volume"] > df["vol5"]) & (df["volume"] > df["vol20"])

        # 條件 3: 價格站上 10日與 20日均線 (AND)
        cond_trend = (df["close"] > df["ma10"]) & (df["close"] > df["ma20"])

        # 綜合篩選
        final_condition = cond_atr & cond_vol & cond_trend

        if FilterData.IS_DELETE_DATA:
            df = df[final_condition]
        else:
            # df["date"].isin(macd_criteria["date"])
            df["f_atr_breakout"] = np.where(final_condition, 1, 0)

        # 移除計算用欄位回傳
        df.drop(columns=["atr_ma10", "vol5", "vol20", "ma10", "ma20"])

        return df

    def always_del_atr(df, is_del_atr):
        """
        篩選條件：
        1. ATR 變化量：今日 ATR > 過去 10 日 ATR 平均值
        """

        # --- 1. 計算 ATR (14日為標準週期) ---
        # 1. 先計算 TR (這部分不需要 group，因為是橫向 row-based 運算)
        high_low = df["high"] - df["low"]
        last_close = df.groupby("stock_id")["close"].shift(1)
        high_pc = (df["high"] - last_close).abs()
        low_pc = (df["low"] - last_close).abs()

        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

        # 2. 【關鍵修正】計算 ATR 時必須按 stock_id 分組
        df["del_atr"] = tr.groupby(df["stock_id"]).rolling(window=14).mean().reset_index(level=0, drop=True)
        df["del_atr"] = df["del_atr"] / df["close"]  # ATR 相對值（ATR / 收盤價）
        condition = df["del_atr"] > 0.05  # ATR 相對值大於 2% 的條件
        if is_del_atr:
            df = df[condition]
        else:
            # df["f_atr_just"] = condition.astype(int)
            df["f_atr_just"] = df["del_atr"]

        # 移除計算用欄位回傳
        # df.drop(columns=["atr"])

        return df

    import pandas as pd

    def always_del_limit_up(df: pd.DataFrame):
        """
        刪除漲停的資料（以漲幅 9.9% 作為判定門檻，避免浮點數誤差）
        """

        condition = df["close"].pct_change() < 0.09
        return df[condition]

    def print_log(before, after):
        try:
            print("*" * 30, "filter")
            while len(before) != 3:
                before.append(0)
            while len(after) != 3:
                after.append(0)
            print(before, int(before[2] / sum(before) * 100), "%", int(before[1] / sum(before) * 100), "%")
            print(after, int(after[2] / sum(after) * 100), "%", int(after[1] / sum(after) * 100), "%")
        except Exception as e:
            print(e)


# FilterData.IS_DELETE_DATA = True
# for stock_id in random.sample(get_file_list(), 5):
#     df: pd.DataFrame = stock(stock_id)
#     df: pd.DataFrame = FilterData.get_data(df, True)
#
#
#
# FilterData.IS_DELETE_DATA = True
# df = stock("2485")
# df: pd.DataFrame = FilterData.atr_just(df)
# print(df.tail(200).to_csv("tmp.csv"))
#
#
#
# FilterData.IS_DELETE_DATA = True
# df: pd.DataFrame = stock("2330")
# df = Label.add_label(df)
# print(df[].value_counts())
# df = FilterData.atr_breakout(df)
# print(df.shape[0])
# print(df["f_atr_breakout"].value_counts())
