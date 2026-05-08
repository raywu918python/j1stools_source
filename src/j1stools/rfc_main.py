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

from j1stools import feature_builder
from j1stools.obj_filter_data import FilterData
import j1stools.parquet_db as parquet_db
from j1stools.obj_base_model import MODEL_TYPE, RUN_TYPE, BaseModel
from j1stools.obj_label import Label
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


def split_date(df: pd.DataFrame, trainging_idx=0.8, is_gen_train=True, is_gen_test=True):
    # group
    df.sort_values(by=["date", "stock_id"], inplace=True)
    x = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["target"]

    if trainging_idx and is_gen_train:
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=trainging_idx, random_state=42)
        return xtrain, xtest, ytrain, ytest
    elif is_gen_train:
        return x, None, y, None
    elif is_gen_test:
        return None, x, None, y
    else:
        raise Exception("參數錯誤")


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
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    model=None,
    pick_import_feature=True,
    is_del_atr=True,
    run_type=RUN_TYPE.train,
):
    print(f"*" * 60, "rfc start")
    m = RFCModel()
    m.trainging_idx = trainging_idx
    m.run_type = run_type
    m.is_print_import_ft = True
    if model:
        m.model = model
    else:
        m.model = gen_model()

    if run_type == RUN_TYPE.train:
        is_gen_train_data = True
        is_gen_test_data = True
    elif run_type == RUN_TYPE.create_model:
        is_gen_train_data = True
        is_gen_test_data = False
    elif run_type == RUN_TYPE.predict:
        is_gen_train_data = False
        is_gen_test_data = True

    df = parquet_db.query_price(stocks, st, end)
    print("delete.before:", df.shape)
    df = feature_builder.gen_feature(df)
    df = Label.add_label(df)
    df = FilterData.get_data(df, True, False, is_del_atr)
    # parquet_db.create_features(df)
    # 資料在這裡刪
    df.set_index(["date", "stock_id"], inplace=True)
    xtrain, xtest, ytrain, ytest = split_date(df, trainging_idx, is_gen_train_data, is_gen_test_data)
    if is_gen_train_data:
        xtrain, ytrain = m.drop_na_inf(xtrain, ytrain)
        if pick_import_feature:
            xtrain = feature_builder.pick_feature(xtrain)
    if is_gen_test_data:
        xtest, ytest = m.drop_na_inf(xtest, ytest)
        if pick_import_feature:
            xtest = feature_builder.pick_feature(xtest)

    acc_list = []
    for i in range(1):
        y_proba = m.train_model(
            xtrain,
            xtest,
            ytrain,
            ytest,
            i,
            function_train=function_train,
            df=df,
        )
        if m.run_type == RUN_TYPE.create_model:
            return
        acc_list.append(y_proba[:, 2])
    # print(f"平均:", np.average(acc_list))
    signal = m.gen_gold_signal(xtest, y_proba).iloc[:, [0, 1, 4]]
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
        run_type=RUN_TYPE.predict,
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
        run_type=RUN_TYPE.predict,
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
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 100)
    stocks = parquet_db.query_stocks_ids_list()
    signal = exec(
        stocks,
        st="2024-01-01",  # 2024-01-01
        end="2099-01-01",
        trainging_idx=0.8,
        model=None,
        pick_import_feature=False,
        is_del_atr=False,
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
        run_type=RUN_TYPE.predict,
        pick_import_feature=True,
    )


# predict_today()
# main()
# predict()
# optimize()
