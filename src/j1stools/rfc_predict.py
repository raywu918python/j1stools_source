import random
from time import time

import joblib
import numpy as np
import pandas as pd

# from obj_label import Label  # 我可，套件不可
from j1stools.obj_label import Label
from j1stools.obj_ma_feature import MaFeature  # 套作可，我不可

from j1stools.obj_hv_feature import HvFeature
from j1stools.obj_market_feature import MarketFeature
from j1stools.obj_random_feature import RandomFeature
from j1stools.obj_vwap_pvt_feature import VolumePriceFeature
from j1stools.obj_filter_data import FilterData
from j1stools import parquet_db


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
    df = RandomFeature.add_feature(df)
    print(f"gen_feature: {time() - st:.2f} 秒")

    return df


def drop_na_inf(x, y):
    # 1. 把 inf 換成 NaN
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 2. 找出哪些列是乾淨的（沒有 NaN）
    # 注意：xtrain 和 ytrain 的列必須同步刪除，否則 index 會對不起來
    clean_mask = x.isnull().any(axis=1) == False
    x = x[clean_mask]
    y = y[clean_mask]
    return x, y


def predict():

    model = joblib.load("models/rfc_2015_2020.joblib")
    stocks = random.sample(parquet_db.query_stocks_no_etf(), 10)
    THRESHOLD = 0.6
    df = parquet_db.query_price(stocks, "2024-01-01", "2025-01-01")
    df = gen_feature(df)
    df = Label.add_label(df)
    df = FilterData.get_data(df, True, False)

    x = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["target"]
    x, y = drop_na_inf(x, y)

    expected_features = model.feature_names_in_
    x = x[expected_features]

    yproba = model.predict_proba(x)
    print(yproba)
    return yproba


# predict()
