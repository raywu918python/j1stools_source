from re import L
from time import time

import random

from j1stools.CONFIG import LgbmTrainConfig, MACDDataBuilterConfig
from j1stools.TYPE import FEATURE_TYPE, TRAIN_TYPE, MODEL_TYPE


from j1stools.model_training_process import run_train_process
import joblib


def query_no_rfc(stocks, st, end):

    model = joblib.load("models/lgbm_20231231_no_rfc.joblib")
    return exec(
        stocks,
        st,
        end,
        0,
        model,
    )


def predict(stocks, st, end):
    model = joblib.load("models/lgbm.joblib")
    return exec(
        stocks=stocks,
        st=st,
        end=end,
        model=model,
        is_using_rfc=True,
        pick_import_feature=False,
        run_type=TRAIN_TYPE.predict,
    )


def main():
    train = LgbmTrainConfig()
    # train.model = (joblib.load("models/rfc_macd.joblib"),)
    #############################################################
    data = MACDDataBuilterConfig()
    # data.is_continuous = True
    data.feature_type = FEATURE_TYPE.today
    # data.atrcfg = FILTER_CONFIG.del_
    data.train_config = train
    data.st = "2021-01-01"
    data.end = "2024-01-01"
    signal = run_train_process(cfg=data)
    #############################################################
    signal = signal[signal["y_proba"] > 0.5]
    signal.sort_values(by=["date", "y_proba"], inplace=True)
    signal.to_csv("signal_today.csv", index=False)


main()
# predict(
#     stocks=random.sample(parquet_db.query_stocks_no_etf(), 500),
#     st="2024-01-01",
#     end="2099-01-01",
# )
