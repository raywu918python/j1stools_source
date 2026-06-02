import random
from time import time

import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
import pandas as pd
import joblib

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
from j1stools.train_flow import (
    batter_predict,
    get_full_name,
    keep_latest_ten_files,
    print_ft_important,
    start_train,
)
from j1stools.train_function import rfc_train_function


def prepare_data(stocks, st, end):
    df = parquet_db.query_price(stocks, st, end)
    print("filter.before:", df.shape)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    print("filter.after:", df.shape)
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)
    return df


def predict(stocks, st, end):
    model = joblib.load("models/rfc_macd_6xx.joblib")
    # MACD divergence 需要足夠歷史，往前多抓 120 天暖機
    warmup_st = (pd.Timestamp(st) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    df = prepare_data(stocks, warmup_st, end)
    _, x, _, y = rfc_split_date(df, is_gen_test=True, is_gen_train=False, trainging_idx=0)
    x = x[[col for col in x.columns if col.startswith("f_")]] if x is not None else None
    x, y = drop_na_inf(x, y)
    expected_features = model.feature_names_in_
    x = x[expected_features]
    result = batter_predict(model, x, y)
    result["date"] = pd.to_datetime(result["date"])
    return result[result["date"] >= pd.Timestamp(st)].reset_index(drop=True)


def train(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    pick_import_feature=False,
):
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


if __name__ == "__main__":
    stocks = parquet_db.activate_stocks()
    predict(stocks, "2026-01-01", "2099-01-01")
