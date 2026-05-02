from logging import addLevelName

import numpy as np
import pandas as pd
from feature_utils import add_lag
import vectorbt as vbt
import quantstats as qs
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta


class VolumeFeature:

    def init(df: pd.DataFrame):
        df["vol20"] = df["volume"].rolling(window=20).mean()
        df["vol5"] = df["volume"].rolling(window=5).mean()

    def add_feature(df: pd.DataFrame):
        # 量增
        avg_volume20 = df["volume"].rolling(window=20).mean()
        volume_break_avg = df["volume"] > (avg_volume20 * 1.5)
        # 過去 3 天都是漲的
        # 計算與前一筆的差值，並判斷是否大於 0
        is_rising_volume = df["volume"].diff() > 0
        is_up_close = df["close"].diff() > 0
        is_down_close = df["close"].diff() < 0
        # 檢查最後 3 筆資料（也就是過去 3 天）是否全部為 True
        volume_break_3day = is_rising_volume.tail(3).all()
        close_up_3day = is_up_close.tail(3).all()
        close_down_3day = is_down_close.tail(3).all()
        df["f_volume_p_up_3day"] = (close_up_3day & volume_break_3day).astype(int)
        df["f_volume_p_down_3day"] = (close_down_3day & volume_break_3day).astype(int)

        # 價漲
        price_up = (df["close"].pct_change() > 0.1).astype(int)
        # add feature
        df["f_volume_change"] = df["volume"].pct_change()  # .clip(lower=-10, upper=3)
        df["f_volume_price_up"] = (volume_break_avg & price_up).astype(int)

        # 1. 取得 PVT
        # df["f_volume_pvt"] = ta.pvt(df["close"], df["volume"])
        VolumeFeature.set_pvt(df)

        # 2. 取得 PVO
        df = VolumeFeature.set_pvo(df, fast=12, slow=26, signal=9)

        return df

    import pandas as pd

    def set_pvt(df):
        # 1. 計算每日漲跌幅
        pct_change = df["close"].pct_change()

        # 2. 計算當日的 PVT 增量 (漲跌幅 * 當日成交量)
        pvt_daily_change = pct_change * df["volume"]

        # 3. 透過累加 (cumsum) 得到 PVT 指標
        df["pvt"] = pvt_daily_change.cumsum()

        # 4. 加上一條 9 日的 PVT 訊號線 (幫助看交叉)
        df["pvt_signal"] = df["pvt"].rolling(window=9).mean()

        df["pvt_scaled"] = (df["pvt"] - df["pvt"].min()) / (df["pvt"].max() - df["pvt"].min())

        # 3. 加上 20 日移動平均線
        df["pvt_ema"] = df["pvt_scaled"].ewm(span=20, adjust=False).mean()

        # 4. 同時畫出這兩條線，走勢與黃金交叉就會一目了然
        # df[["pvt_scaled", "pvt_ema"]].plot(figsize=(12, 6))
        df["f_volume_pvt_change"] = df["pvt_scaled"] - df["pvt_ema"]
        df = add_lag("f_volume_pvt_change", df)

    def set_pvo(df, fast=12, slow=26, signal=9):
        """
        計算 PVO (Percentage Volume Oscillator)
        """
        volume_series = df["volume"]
        # pvo = ta.pvo(df["volume"], fast=12, slow=26, signal=9)
        # 1. 計算成交量的短、長週期 EMA
        ema_fast = volume_series.ewm(span=fast, adjust=False).mean()
        ema_slow = volume_series.ewm(span=slow, adjust=False).mean()
        # 2. 計算 PVO 主線 (%)
        # 注意：分母若為 0 會產生 inf，這裡使用 ema_slow 作為分母
        pvo = ((ema_fast - ema_slow) / ema_slow) * 100
        # 3. 計算 訊號線 (Signal Line)
        pvo_signal = pvo.ewm(span=signal, adjust=False).mean()

        # 4. 計算 柱狀圖 (Histogram)
        pvo_hist = pvo - pvo_signal

        # 整合回 DataFrame
        pvo_df = pd.DataFrame(
            {"pvo": pvo, "pvos": pvo_signal, "pvoh": pvo_hist},
            index=volume_series.index,
        )

        df = pd.concat([df, pvo_df], axis=1)

        # 4. 畫圖觀察 PVO 與 Signal 的黃金交叉與死亡交叉
        df["f_volume_pvo_change"] = df["pvo"] - df["pvoh"]
        df = add_lag("f_volume_pvo_change", df)
        return df

    def test():
        # df = query("0050")
        # df = df.loc["2020":]

        # df = VolumeFeature.init(df)
        # print(df[-30:].T)
        # # print(df[["date", "PVOh_12_26_9"]][:30])
        # # print(
        # #     df[
        # #         [
        # #             "date",
        # #             # "f_volume_pvoh_change",
        # #             "PVOs_12_26_9",
        # #             "PVO_12_26_9",
        # #             "PVOh_12_26_9",
        # #             # "PVO_Signal",
        # #             # "PVO",
        # #             # "PVO_Hist",
        # #             "volume",
        # #         ]
        # #     ][:40]
        # # )
        # print(df.describe().T)
        # # df2 = stock("2646")
        # # JS_volume_feature.add_feature(df2)
        # # print(df2.describe().T)

        # # df1.add(df2)
        # # print(df1.describe().T)
        # # df.to_csv("tmp_volume.csv")
        pass


# VolumeFeature.test()
