from traceback import print_tb

import pandas as pd
import pandas_ta as ta


def help(df: pd.DataFrame):
    """寬表格用法"""
    ma20 = df["close"].ewm(span=5, adjust=False).mean()

    ma20a = ma20.stack(future_stack=True).rename("close_ma20")  # 單一欄位
    ma20a = ma20a.reset_index()  # 單一欄位
    print(ma20a.head())
    ma20a.columns = [f"{col}_ma20" for col in ma20a.columns]  # 多欄位
    print(ma20a.head())
    # 4. 併回原表
    df = df.merge(ma20a.reset_index(), on=["date", "stock_id"], how="left")


class VolumePriceFeature:
    def add_feature(df: pd.DataFrame):
        # df.set_index(pd.to_datetime(df["date"]), inplace=True)
        # print(df.head())
        dfw = df.pivot(index="date", columns="stock_id", values=["high", "low", "close", "volume"])

        #
        #
        #
        f_price_close_change = dfw["close"].pct_change()
        #
        #
        #
        vwap, f_vwap_gap, f_vwap_roc = VolumePriceFeature.get_vwap_features(
            dfw["high"], dfw["low"], dfw["close"], dfw["volume"]
        )
        pvt, f_pvt_gap, f_pvt_roc = VolumePriceFeature.get_pvt_features(dfw["close"], dfw["volume"])

        # print(df[["stock_id", "close", "volume", "f_vwap"]][100:].head().T)
        # f_vwap = ta.vwap(dfw.high, dfw.low, dfw.close, dfw.volume)

        # 計算 PVT (Price Volume Trend)
        # df["PVT"] = ta.pvt(df.close, df.volume)

        # 甚至可以直接計算均線並與其結合

        # ma = adjust=False, ema = adjust=True

        df = df.set_index(["date", "stock_id"])
        df["f_vwap_gap"] = f_vwap_gap.stack(future_stack=True)
        df["f_vwap_roc"] = f_vwap_roc.stack(future_stack=True)
        df["f_pvt_gap"] = f_pvt_gap.stack(future_stack=True)
        df["f_pvt_roc"] = f_pvt_roc.stack(future_stack=True)
        df["f_price_close_change"] = f_price_close_change.stack(future_stack=True)
        df = df.reset_index()
        # print(df[100:].head().T)
        return df

    def test():
        pass
        # # pd.read_csv()
        # df = stock("2330")

        # df = MaFeature.add_feature(df)
        # print(df.describe().T)
        # # print(df.head().T)

    import pandas as pd

    def get_vwap_features(df_high, df_low, df_close: pd.DataFrame, df_volume: pd.DataFrame, window=20):
        """
        計算 VWAP 相關特徵（包含原始值、乖離率、滾動變化率）
        """
        # 1. 原始 VWAP 計算 (分子與分母滾動總和相除)
        typical_price = (df_high + df_low + df_close) / 3
        rolling_vol_price = (typical_price * df_volume).rolling(window=window).sum()

        # rolling_vol_price = (df_close * df_volume).rolling(window=window).sum()#簡單版
        rolling_vol = df_volume.rolling(window=window).sum()
        vwap_df = rolling_vol_price / rolling_vol

        # 2. 方案 A：乖離率 (Price Deviation)
        vwap_gap = (df_close - vwap_df) / vwap_df

        # 3. 方案 B：滾動變化率 (ROC, 百分比)
        vwap_roc = vwap_df.pct_change(axis=0, fill_method=None) * 100

        return vwap_df, vwap_gap, vwap_roc

    def get_pvt_features(df_close: pd.DataFrame, df_volume: pd.DataFrame, window=20):
        """
        計算 PVT 相關特徵（包含原始值、乖離率、變化量 Z-Score）
        """
        # 1. 原始 PVT 計算
        returns = df_close.pct_change(axis=0, fill_method=None)
        pvt_delta = returns * df_volume
        pvt_df = pvt_delta.cumsum(axis=0)

        # 2. 方案 A：PVT 乖離率 (Gap)
        pvt_ma = pvt_df.rolling(window=window).mean()
        pvt_gap = (pvt_df - pvt_ma) / pvt_ma

        # 3. 方案 B：PVT 變化量 Z-Score
        pvt_diff = pvt_df.diff(axis=0)
        mean = pvt_diff.rolling(window=window).mean()
        std = pvt_diff.rolling(window=window).std()
        pvt_roc = (pvt_diff - mean) / std

        return pvt_df, pvt_gap, pvt_roc


# MaFeature.test()
