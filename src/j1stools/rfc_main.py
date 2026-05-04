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
from j1stools.obj_base_model import BaseModel
from j1stools.obj_label import Label
from j1stools.feature_builder import gen_feature
import joblib


class RFCModel(BaseModel):
    def __init__(self):
        super().__init__()


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
        oob_score=True,
        random_state=42,
        criterion="entropy",
        n_jobs=-1,
    )


def split_date(df: pd.DataFrame, trainging_idx=0.8):
    # group
    df.sort_values(by=["date", "stock_id"], inplace=True)
    x = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["target"]
    if trainging_idx == 0:
        return None, x, None, y
    elif trainging_idx == 1:
        return x, None, y, None
    else:
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=trainging_idx, random_state=42)
        return xtrain, xtest, ytrain, ytest


def exec(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    model=None,
    threshold=0.6,
):
    print(f"*" * 60, "rfc start")
    rfc = RFCModel()
    rfc.trainging_idx = trainging_idx
    rfc.is_training = True and rfc.trainging_idx
    rfc.is_print_import_ft = True
    rfc.THRESHOLD = threshold
    if model:
        rfc.model = model
    else:
        rfc.model = gen_model()

    df = parquet_db.query_price(stocks, st, end)
    print("delete.before:", df.shape)
    df = feature_builder.gen_feature(df)
    df = Label.add_label(df)
    df = FilterData.get_data(df, True, False)
    # parquet_db.create_features(df)
    # 資料在這裡刪
    df.set_index(["date", "stock_id"], inplace=True)
    xtrain, xtest, ytrain, ytest = split_date(df, trainging_idx)
    if rfc.is_training:
        xtrain, ytrain = rfc.drop_na_inf(xtrain, ytrain)
        xtrain = feature_builder.pick_feature(xtrain)
    xtest, ytest = rfc.drop_na_inf(xtest, ytest)
    xtest = feature_builder.pick_feature(xtest)

    print("delete.after:", df.shape)
    acc_list = []
    for i in range(1):
        y_proba = rfc.train_model(
            xtrain,
            xtest,
            ytrain,
            ytest,
            i,
        )
        acc_list.append(y_proba[:, 2])

    # print(f"平均:", np.average(acc_list))
    signal = rfc.gen_gold_signal(xtest, y_proba).iloc[:, [0, 1, 4]]
    signal.rename(columns={2: "y_proba"}, inplace=True)
    signal["date"] = pd.to_datetime(signal["date"])
    return signal


def predict(stocks, st, end):
    model = joblib.load("models/rfc.joblib")
    return exec(stocks, st, end, trainging_idx=0, model=model)


# create_model()

# delete.before: (482543, 7)
# ****************************** filter
# [318923, 64674, 98946] 20 % 13 %
# [22611, 12431, 14682] 29 % 25 %
# delete.after: (49724, 32)
# main(
#     stocks=parquet_db.stocks(),
#     st="2015-01-01",
#     end="2018-01-01",
#     trainging_idx=0.8,
# )

#
#
#
#
# main(stocks=["2330", "2360"])
# main(parquet_db.query_stocks_no_etf(), st="2015-01-01", end="2018-01-01")
# main(random.sample(parquet_db.query_stocks_no_etf(), 10), st="2015-01-01", end="2018-01-01")


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
