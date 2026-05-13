from calendar import c
from datetime import datetime, timedelta
from pdb import run
import random
from time import time
from math import e
from venv import create


from attr import field
from requests import head
from sklearn.model_selection import train_test_split
from sympy import rf
from websockets import Data

from j1stools.CONFIG import (
    MACDDataBuilterConfig,
    NormalDataBuilderConfig,
    RfcTrainConfig,
    TodayDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, MODEL_TYPE, TRAIN_TYPE
import j1stools.parquet_db as parquet_db
import joblib

from j1stools.train_flow import start_train_flow


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
    signal = signal[signal["y_proba"] > 0.6]
    signal.sort_values(by=["date", "y_proba"], inplace=True)
    signal.to_csv("signal_today.csv", index=False)


def predict(
    stocks,
    st,
    end,
    pick_import_feature=False,
):

    train = RfcTrainConfig()
    train.train_type = TRAIN_TYPE.predict
    train.model_type = MODEL_TYPE.rfc
    train.model = joblib.load("models/rfc_macd.joblib")
    #############################################################
    data = MACDDataBuilterConfig()
    data.train_type = TRAIN_TYPE.predict
    data.stocks = stocks
    data.train_config = train
    data.trainging_idx = 0
    data.st = st
    data.end = end
    data.pick_import_feature = pick_import_feature
    return start_train_flow(cfg=data)


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
    train = RfcTrainConfig()
    train.model_type = MODEL_TYPE.rfc
    # train.model = (joblib.load("models/rfc_macd.joblib"),)
    #############################################################
    data = MACDDataBuilterConfig()
    data.feature_type = FEATURE_TYPE.macd
    # data.atrcfg = FILTER_CONFIG.none_
    # data.feature_type = FEATURE_TYPE.abcd
    data.train_config = train
    data.st = "2024-01-01"
    data.end = "2099-01-01"
    signal = start_train_flow(cfg=data)
    #############################################################
    signal = signal[signal["y_proba"] > 0.5]
    signal.sort_values(by=["date", "y_proba"], inplace=True)
    signal.to_csv("signal_today.csv", index=False)


def optimize():
    model = joblib.load("model/rfc20260504_233715_0.joblib")
    exec(
        stocks=parquet_db.query_stocks_no_etf(),
        st="2024-01-01",
        end="2099-01-01",
        model=model,
        run_type=TRAIN_TYPE.predict,
        pick_import_feature=True,
    )


# predict_today()
# main()
# predict()
# optimize()
