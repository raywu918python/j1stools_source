from traceback import print_tb

import numpy as np
import pandas as pd

from j1stools.feature_utils import add_lag


class AtrFeature:
    def add_feature(df, window=14):
        # 1. 計算真實波幅 (True Range)
        high_low = df["high"] - df["low"]
        high_close_prev = np.abs(df["high"] - df["close"].shift(1))
        low_close_prev = np.abs(df["low"] - df["close"].shift(1))

        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # 2. 計算 ATR (通常使用 RMA 或 SMA)
        df["atr"] = tr.rolling(window=window).mean()

        # 3. 計算 ATR 比例 (%)
        df["f_atr_percent"] = (df["atr"] / df["close"]) * 100
        df = add_lag("f_atr_percent", df)

        # df["f_atr_break"] = ((df["close"] - df["open"]) > (df["atr"] * 1.5)).astype(int)
        # df = add_lag("f_atr_break", df)

        return df

    def test():
        pass
        # pd.read_csv()
        # df = parqu("2330")

        # df = AtrFeature.add_feature(df)
        # print(df.describe().T)
        # print(df.tail().T)


# AtrFeature.test()
