from re import L
from time import time

import random

from numpy import add
import pandas as pd

from j1stools import rfc_main
from j1stools.CONFIG import (
    BaseDataBuilderConfig,
    BaseLabelConfig,
    LgbmTrainConfig,
    MACDDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, TRAIN_TYPE, MODEL_TYPE


from j1stools.model_builder import gen_lgbm_r_model
from j1stools.train_flow import start_train
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
    train.model_type = MODEL_TYPE.lgbm_c
    train.model = gen_lgbm_r_model()
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
    data.feature_type = FEATURE_TYPE.power
    # data.feature_type = FEATURE_TYPE.today
    data.atrcfg = FILTER_TYPE.none_
    data.train_config = train
    data.st = "2024-01-01"
    data.end = "2099-01-01"
    signal = start_train(cfg=data)
    #############################################################
    top = signal[signal["predicted_rank"] >= 0.8]
    print(f"選出股票數：{len(top)}")
    print(f"平均未來報酬：{top['future_return'].mean():.2%}")
    print(f"勝率（>0）：{(top['future_return'] > 0).mean():.2%}")
    # 看結果
    print(top.reset_index()[["date", "stock_id", "predicted_rank"]])


# main()
# predict(
#     stocks=random.sample(parquet_db.query_stocks_no_etf(), 500),
#     st="2024-01-01",
#     end="2099-01-01",
# )
