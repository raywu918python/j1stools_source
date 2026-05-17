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
from torch import mode
from websockets import Data

from j1stools import feature_builder, label_builder
from j1stools import data_filter
from j1stools.CONFIG import (
    MACDDataBuilterConfig,
    NormalDataBuilderConfig,
    RfcTrainConfig,
    TodayDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, MODEL_TYPE, TRAIN_TYPE
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


def prepare_data(stocks, st, end):
    df = parquet_db.query_price(stocks, st, end)
    print("filter.before:", df.shape)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    print("filter.after:", df.shape)
    # 資料在這裡刪
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)
    return df


def train(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    pick_import_feature=False,
):
    print(f"=" * 60, "rfc start")
    keep_latest_ten_files("./model")
    model = gen_rfc_model() if model is None else model

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
    stocks = parquet_db.query_stocks_ids_list()
    st = "2015-01-01"
    end = "2024-01-01"
    df = prepare_data(stocks, st, end)
    x, _, y, _ = rfc_split_date(df, is_gen_train=True, trainging_idx=1)
    x = x[[col for col in x.columns if col.startswith("f_")]] if x is not None else None
    x, y = drop_na_inf(x, y)
    model = gen_rfc_model()
    model.fit(x, y)
    joblib.dump(model, "models/rfc_macd_6xx.joblib")


# predict_today()
# main()
# predict()
# optimize()
# model_release()
# main()
