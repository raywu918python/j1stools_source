from time import time

import pandas as pd

from j1stools.obj_hv_feature import HvFeature
from j1stools.obj_ma_feature import MaFeature
from j1stools.obj_market_feature import MarketFeature
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
    # df = MacdFeature.add_feature(df)
    df = MarketFeature.add_feature(df)
    df = VolumePriceFeature.add_feature(df)
    # df = RandomFeature.RandomFeature.add_feature(df)
    print(f"gen_feature: {time() - st:.2f} 秒")

    return df
