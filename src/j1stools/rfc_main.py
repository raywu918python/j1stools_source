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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from websockets import Data

from j1stools.CONFIG import (
    BaseDataBuilderConfig,
    MACDDataBuilterConfig,
    NormalDataBuilderConfig,
    TodayDataBuilterConfig,
)
from j1stools.data_builder import DataBuilder
from j1stools.obj_filter_data import FILTER_CONFIG
import j1stools.parquet_db as parquet_db
from j1stools.obj_base_model import MODEL_TYPE, MODEL_RUN_TYPE, BaseModel
from j1stools.feature_builder import gen_feature, pick_feature
import joblib


class RFCModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = MODEL_TYPE.rfc


#
# training
#
def gen_model():
    return RandomForestClassifier(
        n_estimators=1000,
        min_samples_leaf=15,  # 稍微下修，對 1000 筆數據較友善
        max_depth=10,  # 數據量小，建議深度再淺一點 (12 -> 10)，防過擬合
        max_features=0.3,  # 40個特徵，抽 8 個比較能抓到關鍵特徵 (0.1太少)
        # 手動指定權重，避免自動權重在小樣本下的極端波動
        # class_weight={0: 2.5, 1: 2.5, 2: 1},
        # class_weight={0: 1, 1: 2.5, 2: 6},
        # class_weight={0: 1, 1: 1, 2: 10},
        # 增加 OOB 評估，讓你在訓練完可以直接看 OOB Score 準不準
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        criterion="entropy",
        n_jobs=-1,
    )


def function_train(xtrain, ytrain, model, df):
    sample_weights = ytrain.map({0: 1, 1: 1, 2: 3})
    # m = xtrain["f_atr_just"]
    # xtrain = xtrain.drop(["f_atr_just"], axis=1)
    # print(m.head(10))

    model.fit(
        xtrain,
        ytrain,
        # sample_weight=m.values**2,
        # sample_weight=sample_weights,
    )


def exec(
    model=None,
    cfg: BaseDataBuilderConfig = None,
):
    print(f"*" * 60, "rfc start")
    m = RFCModel()
    m.trainging_idx = cfg.trainging_idx
    m.model_run_type = cfg.model_run_type
    m.is_print_import_ft = True
    if model:
        m.model = model
    else:
        m.model = gen_model()

    d = DataBuilder(cfg).build()

    acc_list = []
    for i in range(1):
        y_proba = m.train_model(
            d.xtrain,
            d.xtest,
            d.ytrain,
            d.ytest,
            i,
            function_train=function_train,
        )
        if m.model_run_type == MODEL_RUN_TYPE.create_model:
            return
        acc_list.append(y_proba[:, 2])
    # print(f"平均:", np.average(acc_list))
    signal = m.gen_gold_signal(d.xtest, y_proba).iloc[:, [0, 1, 4]]
    signal.rename(columns={2: "y_proba"}, inplace=True)
    signal["date"] = pd.to_datetime(signal["date"])
    return signal


def predict_today():
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 100)
    stocks = parquet_db.query_stocks_ids_list()
    today = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    model = joblib.load("models/rfct0.joblib")
    signal = exec(
        stocks,
        today,
        "2099-01-01",
        model=model,
        is_del_atr=True,
        pick_import_feature=False,
        run_type=MODEL_RUN_TYPE.predict,
    )
    signal = signal[signal["y_proba"] > 0.6]
    signal.sort_values(by=["date", "y_proba"], inplace=True)
    signal.to_csv("signal_today.csv", index=False)


def predict(
    stocks,
    st,
    end,
    pick_import_feature=True,
):
    model = joblib.load("models/rfc.joblib")
    return exec(
        stocks,
        st,
        end,
        trainging_idx=0,
        model=model,
        pick_import_feature=pick_import_feature,
        run_type=MODEL_RUN_TYPE.predict,
    )


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
    cfg = MACDDataBuilterConfig()
    cfg.is_continuous = True
    cfg.st = "2018-01-01"
    cfg.end = "2021-01-01"
    # cfg = NormalDataBuilderConfig()
    # cfg = TodayDataBuilterConfig()
    signal = exec(
        model=None,
        cfg=cfg,
    )
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
        run_type=MODEL_RUN_TYPE.predict,
        pick_import_feature=True,
    )


# predict_today()
# main()
# predict()
# optimize()
