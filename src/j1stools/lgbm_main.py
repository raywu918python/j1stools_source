from re import L
from time import time

import random

from numpy import add
import pandas as pd

from j1stools import rfc_main
from j1stools.CONFIG import (
    FILTER_CONFIG,
    BaseDataBuilderConfig,
    BaseLabelConfig,
    LgbmTrainConfig,
    MACDDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, TRAIN_TYPE, MODEL_TYPE


from j1stools.model_training_process import run_train_process
import joblib


def add_rfc_feature(df, data: BaseDataBuilderConfig):
    signal = rfc_main.predict(
        stocks=data.stocks,
        st=data.st,
        end=data.end,
        pick_import_feature=data.pick_import_feature,
    )
    signal.rename(columns={"y_proba": "f_rfc"}, inplace=True)
    signal = signal[["date", "stock_id", "f_rfc"]]
    #
    return pd.merge(
        df,
        signal[["date", "stock_id", "f_rfc"]],
        on=["date", "stock_id"],
        how="left",  # 只取索引部分  # 以全時段為準
    ).fillna(
        0
    )  # 沒預測到的（ATR太小的）補 0


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
    train.is_use_rfc = False
    # train.rfc_proba = add_rfc_feature(df, data)
    # train.model = (joblib.load("models/rfc_macd.joblib"),)
    ##############################################################
    # l = BaseLabelConfig()
    # l.hold_days = 20
    # l.profit_target = 0.1
    # l.stop_loss = -0.1
    #############################################################
    data = MACDDataBuilterConfig()
    # data.is_continuous = True
    # data.label_cfg = l
    data.feature_type = FEATURE_TYPE.abcd | FEATURE_TYPE.margin
    # data.feature_type = FEATURE_TYPE.today
    data.atrcfg = FILTER_CONFIG.none_
    data.train_config = train
    data.st = "2024-01-01"
    data.end = "2099-01-01"
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
