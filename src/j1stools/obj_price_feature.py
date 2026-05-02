import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas

from j1stools.feature_utils import add_lag
from j1stools.obj_label import Label


class PriceFeature:
    def add_feature(df: pd.DataFrame):

        # df = PriceFeature.remove_breakdown_intervals(df)

        # df = PriceFeature.price_down(df)

        return df

    def remove_price_down(df):
        """
        昨日量大於5日平均，且大於前日2倍，且今日價格跌破昨日收盤價
        """

        import numpy as np

        # 1. 計算 5 日均量
        df["vma5"] = df["volume"].rolling(window=5).mean()

        # 2. 定義「昨日」相關數據 (使用 shift(1))
        # 昨天成交量 > 昨天當下的 5日均量
        vol_gt_ma5 = df["volume"].shift(1) > df["vma5"].shift(1)

        # 昨天成交量 > 前天成交量 (shift(2)) 的 2 倍
        vol_gt_2x_prev = df["volume"].shift(1) > (df["volume"].shift(2) * 2)

        # 昨天價格上漲 (收盤 > 開盤)
        yesterday_up = df["close"].shift(1) > df["open"].shift(1)

        # 3. 定義「今日」條件
        # 今日收盤價破昨日收盤價
        price_break_yesterday_close = df["close"] < df["close"].shift(1)

        # 4. 綜合判斷並寫入欄位
        # 條件：昨日上漲 + 昨日放量(>ma5 & >2倍) + 今日跌破昨收 + 今日守住昨中價
        condition = yesterday_up & vol_gt_ma5 & vol_gt_2x_prev & price_break_yesterday_close

        df["f_price_down"] = np.where(condition, 1, 0)

        df = df[df["f_price_down"] == 0]
        return df

    def add_feature_bak(df: pd.DataFrame):
        df["f_price_close_change"] = df["close"].pct_change()
        df = add_lag("f_price_close_change", df)

        df["f_price_hl_range"] = abs(df["high"] - df["low"]) / df["low"]
        df = add_lag("f_price_hl_range", df)

        # 昨天跌,今天漲
        df["f_price_trun_around"] = (
            (df["close"] > df["open"])
            & (df["close"].shift(1) < df["open"].shift(1))
            & ((df["high"] - df["low"]) > (df["high"].shift(1) - df["low"].shift(1)))
        ).astype(int)
        # df.dropna(inplace=True)  # 注意這裡不用加 ["欄位名"]
        # 剔除掉數據異常值，只保留合理的漲跌幅範圍
        # df = df[
        #     (df["f_price_close_change"] >= -0.11) & (df["f_price_close_change"] <= 0.11)
        # ]

        # df["f_price_position"] = (df["high"] - df["low"]) / (
        #     df["high"] - df["low"]
        # ).replace(0, 0.001)
        # df = add_lag("f_price_position", df)

        # # 2. 實體棒比例與方向 (-1~1)
        df["f_body_direction"] = (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, 0.001)
        df = add_lag("f_body_direction", df)

        # 3. 乖離率 (相對於月線)
        df["f_bias_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).mean()
        df = add_lag("f_bias_20", df)
        df["f_bias_5"] = (df["close"] - df["close"].rolling(5).mean()) / df["close"].rolling(5).mean()
        df = add_lag("f_bias_5", df)
        df["f_bias_10"] = (df["close"] - df["close"].rolling(10).mean()) / df["close"].rolling(10).mean()
        df = add_lag("f_bias_10", df)

        # df = PriceFeature.set_is_above_mid(df, "f_price_is_above_mid")
        # df = add_lag("f_price_is_above_mid", df)

        return df

    def test2():
        pass
        # list_df = []
        # list_df.append(PriceFeature.add_feature(stock("2330")))
        # # list_df.append(PriceFeature.add_feature(stock("0050")))
        # df = pd.concat(list_df).sort_values("date").reset_index(drop=True)
        # print(df.tail().T)
        # # final = pd.concat(df, axis=0).sort_values("date").reset_index(drop=True)

    def test():
        # # ['2374', '1229', '8039', '3406', '2449']
        # df = pd.read_csv("log_training_stock0.csv", dtype={1: str})

        # # for stock_id in df.iloc[:, 1]:
        # for stock_id in ["8039"]:
        #     # df = stock("8039")
        #     df = stock(stock_id)
        #     df = df.loc["2022-01":"2023-12"]

        #     df: pd.DataFrame = PriceFeature.add_feature(df)
        #     f_cols = [col for col in df.columns if col.startswith("f")]

        #     rs = df[df[f_cols] > 10]

        #     rs = rs.select_dtypes(exclude=["datetime64", "datetime"]).any()
        #     rs = rs.any()
        #     # if rs:
        #     print(f"***************", stock_id)
        #     print(df[f_cols].describe().T)
        #     break
        pass

    def debug():
        l = [
            [1, 2, 3, pd.Timestamp.now()],
            [44, 55, 66, pd.Timestamp.now()],
            [7, 8, 9, pd.Timestamp.now()],
        ]
        df = pd.DataFrame(columns=["a", "b", "c", "t"], data=l)
        # print(df.head())
        # print(df.info())
        # rs = df[["a", "b"]] > 10
        rs = df[df[["a", "b"]] > 10]  # OK
        # rs = df[df["a", "b"] > 1] # error
        # print(rs)
        # rs = rs.any(axis=0)
        # rs = rs.any()
        rs = rs.select_dtypes(exclude=["datetime64", "datetime"]).any()
        rs = rs.any()
        print(rs)
        # if rs.iloc[0]:
        # print("rs.....", rs.iloc[0])
        # else:
        print("rs.....", rs)

    import numpy as np
    import pandas as pd

    def remove_breakdown_intervals(df: pd.DataFrame):
        """
        沒有用了，因為用到未來數劇

        尋找這類型的區間，價格在一天內同時跌破ma10,ma20
        且在ma10,ma20之下，直到有爆大量同時大於vol5,vol10,前一日vol的2倍
        且大跌超過5%
        找到這區間，從df中刪掉
        """

        # 1. 準備必要指標
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["vma5"] = df["volume"].rolling(window=5).mean()
        df["vma10"] = df["volume"].rolling(window=10).mean()
        df["change_pct"] = df["close"].pct_change()

        # 2. 定義條件
        # 開始點：收盤同時跌破 MA10 & MA20
        cond_start = (
            (df["close"] < df["ma10"])
            & (df["close"] < df["ma20"])
            & ((df["close"].shift(1) >= df["ma10"].shift(1)) | (df["close"].shift(1) >= df["ma20"].shift(1)))
        )

        # 結束點 (爆量大跌)
        cond_end = (
            (df["close"] < df["ma10"])
            & (df["close"] < df["ma20"])
            & (
                (
                    (df["volume"] > df["vma5"])
                    | (df["volume"] > df["vma10"])
                    | (df["volume"] > df["volume"].shift(1) * 2)
                )
                & (df["change_pct"] <= -0.03)
            )
        )

        # 漲回條件：收盤價站回 MA10 或 MA20
        cond_recover = (df["close"] > df["ma10"]) | (df["close"] > df["ma20"])

        # 3. 區間標記邏輯
        drop_mask = np.zeros(len(df), dtype=bool)
        temp_indices = []  # 暫存目前區間的索引

        for i in range(len(df)):
            # 如果還沒進入區間，檢查是否開始跌破
            if not temp_indices:
                if cond_start.iloc[i]:
                    temp_indices.append(i)
            else:
                # 已在區間內，先檢查是否「漲回」 (失效機制)
                if cond_recover.iloc[i]:
                    temp_indices = []  # 清空暫存，不標記刪除，重新尋找
                # 檢查是否「符合結束條件」 (成功捕捉到目標區間)
                elif cond_end.iloc[i]:
                    temp_indices.append(i)
                    for idx in temp_indices:  # 確保留下這段，標記到 mask
                        drop_mask[idx] = True
                    temp_indices = []  # 處理完畢，重置
                else:
                    # 既沒漲回也沒爆量，持續在區間內
                    temp_indices.append(i)

        # 4. 執行刪除
        print(df[drop_mask].to_csv("3357.csv"))
        new_df = df[~drop_mask].copy()

        return new_df


# 使用範例
# df_cleaned = remove_breakdown_intervals(df)

# PriceFeature.test2()
# PriceFeature.test()
# JS_price_feature.debug()


# df = stock("3357")
# df: pd.DataFrame = Label.add_label(df)
# print(df["target"].value_counts().sort_index())
# # df = PriceFeature.price_up(df)
# # df = PriceFeature.remove_price_down(df)
# df = PriceFeature.remove_breakdown_intervals(df)
# print(df["target"].value_counts().sort_index())
