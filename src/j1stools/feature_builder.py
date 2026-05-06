from time import time

import pandas as pd

from j1stools.obj_hv_feature import HvFeature
from j1stools.obj_ma_feature import MaFeature
from j1stools.obj_macd_feature import MacdFeature
from j1stools.obj_market_feature import MarketFeature
from j1stools.obj_random_feature import RandomFeature
from j1stools.obj_vwap_pvt_feature import VolumePriceFeature

IS_DAY_TRADE = False


def gen_feature(df) -> pd.DataFrame:
    st = time()

    # df = VolumeFeature.init(df)
    # df = PriceFeature.add_feature(df)
    #
    # df = AtrFeature.add_feature(df)
    #############################################################
    if IS_DAY_TRADE:
        return generate_features_t1(df)
    else:
        df = HvFeature.add_feature(df)
        df = MaFeature.add_feature(df)
        df = MacdFeature.add_feature(df)
        df = MarketFeature.add_feature(df)
        df = VolumePriceFeature.add_feature(df)
        df = RandomFeature.add_feature(df)
    print(f"gen_feature: {time() - st:.2f} 秒")

    return df


def pick_feature(x, is_using_rfc=False):
    ft = [
        "f_HV_squeeze",
        "f_div_refined",
        "f_div_score",
        "f_bottom_slope",
        "f_hist_pos_direction",
        "f_hist_rise_days",
        "f_macdh_slope12",
        "f_macdh_slope23",
        "f_market_rs_10d",
        "f_market_rs_3d",
        "f_market_rs_1d",
        "f_market_stock_rsi",
        "f_market_ret3",
        "f_market_close_change",
        "f_vwap_gap",
        "f_vwap_roc",
        "f_pvt_gap",
        "f_pvt_roc",
    ]
    if is_using_rfc:
        ft.append("f_rfc")
    return x[ft]


import numpy as np
import pandas as pd


def generate_features_t1(df: pd.DataFrame):
    """計算短線特徵

    參數:
    df_dict (dict): 包含 'Open', 'High', 'Low', 'Close', 'Volume' 鍵值的字典，
                    每個值都是寬表格 (DataFrame: index=日期, columns=股票代碼)
    """

    df.sort_values(by=["date", "stock_id"], inplace=True)
    df_dict = df.pivot(index="date", columns="stock_id", values=["high", "low", "close", "volume"])

    Close = df_dict["close"]
    High = df_dict["high"]
    Low = df_dict["low"]
    Volume = df_dict["volume"]

    # 1. ret_1d / ret_3d : 過去 1 天與 3 天的報酬率 (動能)
    ret_1d = Close.pct_change(1)
    ret_3d = Close.pct_change(3)

    # 2. v_ratio_5d : 今日成交量 / 過去 5 日均量 (資金異常流入)
    # 使用 5 日移動平均處理，若包含當日則需注意有無未來函數，這裡使用 rolling(5).mean()
    ma5_volume = Volume.rolling(5).mean()
    v_ratio_5d = Volume / ma5_volume

    # 3. upper_shadow_pct : (最高價 - 收盤價) / 股價 (賣壓過濾)
    upper_shadow_pct = (High - Close) / Close

    # 4. dist_to_ma5 : (收盤價 - MA5) / MA5 (判斷是否過度拉抬)
    ma5_price = Close.rolling(5).mean()
    dist_to_ma5 = (Close - ma5_price) / ma5_price

    # 5. atr_ratio : 當前波動幅度 / ATR (判斷波動是否異常放大)
    # 計算真實波動幅度 (TR)
    tr1 = High - Low
    tr2 = (High - Close.shift(1)).abs()
    tr3 = (Low - Close.shift(1)).abs()

    # 逐一比較三者的最大值
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # 接著計算 ATR
    atr = tr.rolling(14).mean()
    # 當前波動幅度取 High - Low 或 Close 的每日絕對波動
    current_vol = (High - Low).abs()
    atr_ratio = current_vol / atr

    MA60 = Close.rolling(60).mean()
    dist_to_ma60 = (Close - MA60) / MA60

    df = df.set_index(["date", "stock_id"])
    df["f_dist_to_ma60"] = dist_to_ma60.stack(future_stack=True)
    df["f_ret_1d"] = (ret_1d).stack(future_stack=True)
    df["f_ret_3d"] = (ret_3d).stack(future_stack=True)
    df["f_v_ratio_5d"] = (v_ratio_5d).stack(future_stack=True)
    df["f_upper_shadow_pct"] = (upper_shadow_pct).stack(future_stack=True)
    df["f_dist_to_ma5"] = (dist_to_ma5).stack(future_stack=True)
    df["f_atr_ratio"] = (atr_ratio).stack(future_stack=True)
    df = df.reset_index()

    f = df.select_dtypes(include="number").describe().T.round(2)
    f.to_csv("tmp.csv")

    return df


def time_series_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """對寬表格進行時間序列的滾動 Z-score 標準化

    參數:
    - df: 寬表格 DataFrame，Index 為日期，Columns 為股票代碼
    - window: 計算平均與標準差的天數（例如 60 天）
    """
    # 計算滾動平均值與滾動標準差
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()

    # # 避免標準差為 0 造成除以零的錯誤（填補微小值）
    rolling_std = rolling_std.replace(0, np.nan).fillna(1e-5)

    # # 計算 Z-score
    zscore_df = (df - rolling_mean) / rolling_std

    zscore_df = zscore_df.clip(lower=-3, upper=3)

    return df
