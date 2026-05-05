from time import time

import pandas as pd

from j1stools.obj_hv_feature import HvFeature
from j1stools.obj_ma_feature import MaFeature
from j1stools.obj_macd_feature import MacdFeature
from j1stools.obj_market_feature import MarketFeature
from j1stools.obj_random_feature import RandomFeature
from j1stools.obj_vwap_pvt_feature import VolumePriceFeature


def gen_feature(df) -> pd.DataFrame:
    st = time()

    # df = VolumeFeature.init(df)
    # df = PriceFeature.add_feature(df)
    #
    # df = AtrFeature.add_feature(df)
    #############################################################
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
