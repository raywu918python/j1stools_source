from calendar import c
from datetime import datetime, timedelta
from pdb import run
import random
from time import time
from math import e


from attr import field
from numpy import mod, sign
import pandas as pd
from requests import head
from sklearn.model_selection import train_test_split
from sympy import rf
from websockets import Data

from j1stools import feature_builder, label_builder
from j1stools import data_filter
from j1stools.CONFIG import (
    MACDDataBuilterConfig,
    NormalDataBuilderConfig,
    RfcTrainConfig,
    TodayDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, TRAIN_TYPE
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_builder import gen_rfc_model
from j1stools.model_utils import drop_na_inf
import j1stools.parquet_db as parquet_db
import joblib

from j1stools.train_flow import (
    batter_predict,
    get_full_name,
    keep_latest_ten_files,
    print_ft_important,
    start_train,
)
from j1stools.train_function import rfc_train_function

import numpy as np

USE_IB_FEATURES = True  # True → 訓練 rfc_macd_ib_6xx.joblib，False → 原本 rfc_macd_6xx.joblib


def _add_ib_features(df: pd.DataFrame, stocks: list, st: str, end: str) -> pd.DataFrame:
    ib = parquet_db.query_ib(stocks, st, end)
    ib["net"] = ib["buy"] - ib["sell"]
    foreign = (
        ib[ib["name"] == "Foreign_Investor"].groupby(["date", "stock_id"])["net"].sum().reset_index(name="_net_foreign")
    )
    df = df.merge(foreign, on=["date", "stock_id"], how="left")
    df["_net_foreign"] = df["_net_foreign"].fillna(0)

    g = df.groupby("stock_id")
    df["_net_foreign"] = g["_net_foreign"].transform(lambda x: x.shift(1))

    vol_denom = g["volume"].transform(lambda x: x.rolling(20, min_periods=1).mean()).replace(0, np.nan)
    df["f_ib_net_foreign_pct"] = (df["_net_foreign"] / vol_denom).clip(-5, 5)

    def _zscore(s, window=120, min_periods=20):
        m = s.rolling(window, min_periods=min_periods).mean()
        std = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
        return (s - m) / std

    _5d = g["_net_foreign"].transform(lambda x: x.rolling(5).sum())
    _10d = g["_net_foreign"].transform(lambda x: x.rolling(10).sum())
    df["f_ib_net_foreign_5d_z"] = _5d.groupby(df["stock_id"]).transform(_zscore)
    df["f_ib_net_foreign_10d_z"] = _10d.groupby(df["stock_id"]).transform(_zscore)
    df["f_ib_net_foreign_streak"] = g["_net_foreign"].transform(
        lambda x: (x.groupby((x <= 0).cumsum()).cumcount() + 1).where(x > 0, 0)
    )
    return df.drop(columns=["_net_foreign"])


def predict_today():
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 100)
    # stocks = parquet_db.query_stocks_ids_list()
    # today = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    # model = joblib.load("models/rfct0.joblib")
    signal = exec(
        stocks,
        today,
        "2099-01-01",
        model=model,
        is_del_atr=True,
        pick_import_feature=False,
        run_type=TRAIN_TYPE.predict,
    )
    signal = signal[signal[2] > 0.6]
    signal.sort_values(by=["date", "y_proba"], inplace=True)
    signal.to_csv("signal_today.csv", index=False)


# def predict(
#     stocks,
#     st,
#     end,
#     pick_import_feature=False,
# ):

#     train = RfcTrainConfig()
#     train.train_type = TRAIN_TYPE.predict
#     train.model_type = MODEL_TYPE.rfc
#     train.model = joblib.load("models/rfc_macd.joblib")
#     #############################################################
#     data = MACDDataBuilterConfig()
#     data.train_type = TRAIN_TYPE.predict
#     data.stocks = stocks
#     data.train_config = train
#     data.trainging_idx = 0
#     data.st = st
#     data.end = end
#     data.pick_import_feature = pick_import_feature
#     return start_train(cfg=data)


#
# predict
#
# rfc = RFCModel()
# rfc.model = joblib.load("models/20260417/model20260417_140212_4.joblib")
# rfc.THRESHOLD = 0.6
# rfc.is_print_import_ft = False
# rfc.is_add_noise = True
# rfc.set_split_date(trainging_idx=0)
# _, _, xtest, ytest = rfc.prepare_df()
# gold_singal = rfc.batter_predict(xtest, ytest)
# print(gold_singal.head())
# print(gold_singal[gold_singal["gold_signal"] > 0.6].shape)
# print(gold_singal[gold_singal["gold_signal"] > 0.6].head())


def main():
    stocks = parquet_db.query_stocks_ids_list()
    st = "2015-01-01"
    end = "2024-01-01"
    # signal = train(stocks=stocks, st=st, end=end)

    predict(stocks=stocks, st=st, end=end)


# predict(
# model=joblib.load("models/rfc.joblib"),
# df=parquet_db.query_price(stocks, st, end),
# )

#############################################################
# signal.sort_values(by=["date", "yproba"], inplace=True)
# signal.to_csv("signal_today.csv", index=False)


def optimize():
    model = joblib.load("model/rfc20260504_233715_0.joblib")
    exec(
        stocks=parquet_db.query_stocks_no_etf(),
        st="2018-01-01",
        end="2021-01-01",
        model=model,
        run_type=TRAIN_TYPE.predict,
        pick_import_feature=True,
    )


def prepare_data(stocks, st, end, atr_filter_type=FILTER_TYPE.add_, use_ib=USE_IB_FEATURES):
    df = parquet_db.query_price(stocks, st, end)
    print("filter.before:", df.shape)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    if use_ib:
        df = _add_ib_features(df, stocks, st, end)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, atr_filter_type)
    print("filter.after:", df.shape)
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)
    return df


def train(
    stocks=None,
    st="2015-01-01",
    end="2024-01-01",
    trainging_idx=0.8,
    pick_import_feature=False,
):
    if stocks is None:
        stocks = parquet_db.query_stocks_no_etf()
    print(f"=" * 60, "rfc start")
    keep_latest_ten_files("./model")
    model = gen_rfc_model()

    df = prepare_data(stocks, st, end)

    xtrain, xtest, ytrain, ytest = rfc_split_date(df, True, True, trainging_idx)
    xtrain, ytrain = drop_na_inf(xtrain, ytrain)
    xtest, ytest = drop_na_inf(xtest, ytest)
    print("drop_na_inf:", len(xtrain), len(ytrain), len(xtest), len(ytest))

    if pick_import_feature:
        xtrain = feature_builder.pick_feature(xtrain)
        xtest = feature_builder.pick_feature(xtest)

    xtrain = xtrain[[col for col in xtrain.columns if col.startswith("f_")]] if xtrain is not None else None
    xtest = xtest[[col for col in xtest.columns if col.startswith("f_")]] if xtest is not None else None
    print("=" * 60, "train")
    rfc_train_function(xtrain, ytrain, model, df)
    joblib.dump(model, get_full_name("rfc"))
    print("=" * 60, "test")

    expected_features = model.feature_names_in_
    xtest = xtest[expected_features]
    dfyproba = batter_predict(model, xtest, ytest)
    print("=" * 60, "signal")
    print(dfyproba.head())
    print_ft_important(model)


def predict(stocks, st, end):

    model = joblib.load("models/rfc_macd_6xx.joblib")
    df = prepare_data(stocks, st, end)
    _, x, _, y = rfc_split_date(df, is_gen_test=True, is_gen_train=False, trainging_idx=0)
    x = x[[col for col in x.columns if col.startswith("f_")]] if x is not None else None
    x, y = drop_na_inf(x, y)
    expected_features = model.feature_names_in_
    x = x[expected_features]
    return batter_predict(model, x, y)


def model_release():
    model_name = "rfc_macd_ib_6xx" if USE_IB_FEATURES else "rfc_macd_6xx"
    stocks = parquet_db.activate_stocks()
    st = "2015-01-01"
    end = "2024-01-01"
    df = prepare_data(stocks, st, end, use_ib=USE_IB_FEATURES)
    x, _, y, _ = rfc_split_date(df, is_gen_train=True, trainging_idx=1)
    x = x[[col for col in x.columns if col.startswith("f_")]] if x is not None else None
    x, y = drop_na_inf(x, y)
    model = gen_rfc_model()
    model.fit(x, y)
    joblib.dump(model, f"models/{model_name}.joblib")
    print(f"saved → models/{model_name}.joblib")


if __name__ == "__main__":
    # train()
    model_release()

# predict_today()
# main()
# predict()
# optimize()
# model_release()
# main()
