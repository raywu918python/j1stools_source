from time import time
from math import e
from venv import create


from requests import head
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from j1stools.obj_filter_data import FilterData
from obj_market_feature import MarketFeature
from obj_random_feature import RandomFeature
from obj_vwap_pvt_feature import VolumePriceFeature
import parquet_db as parquet_db
from obj_base_model import BaseModel
from obj_hv_feature import HvFeature
from obj_label import Label
import obj_ma_feature
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


def gen_feature(df) -> pd.DataFrame:
    st = time()

    # df = VolumeFeature.init(df)
    # df = PriceFeature.add_feature(df)
    #
    # df = AtrFeature.add_feature(df)
    #############################################################
    df = HvFeature.add_feature(df)
    df = obj_ma_feature.MaFeature.add_feature(df)
    # df = MacdFeature.add_feature(df)
    df = MarketFeature.add_feature(df)
    df = VolumePriceFeature.add_feature(df)
    df = RandomFeature.add_feature(df)
    print(f"gen_feature: {time() - st:.2f} 秒")

    return df


def split_date(df: pd.DataFrame, trainging_idx=0.8):
    # group
    df.sort_values(by=["date", "stock_id"], inplace=True)
    X = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["target"]
    if trainging_idx == 0:
        return None, X, None, y
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainging_idx, random_state=42)
    return X_train, X_test, y_train, y_test


def init(trainging_idx=0.8, model=None):
    rfc = RFCModel()
    rfc.trainging_idx = trainging_idx
    rfc.is_training = True and rfc.trainging_idx
    rfc.is_print_import_ft = True
    rfc.THRESHOLD = 0.5
    if model:
        rfc.model = model
    else:
        rfc.model = gen_model()
    # rfc.model = joblib.load("models/20260417/model20260421_232454_0.joblib")
    # rfc.model = joblib.load("models/20260417/model20260417_140212_4.joblib")
    return rfc


def main(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    model=None,
):
    print(f"*" * 60, "rfc start")
    model = init(trainging_idx=trainging_idx, model=model)
    df = parquet_db.query_price(stocks, st, end)
    print("delete.before:", df.shape)
    df = gen_feature(df)
    df = Label.add_label(df)
    df = FilterData.get_data(df, True, False)
    # parquet_db.create_features(df)
    # 資料在這裡刪
    df.set_index(["date", "stock_id"], inplace=True)
    xtrain, xtest, ytrain, ytest = split_date(df, trainging_idx)
    if model.is_training:
        xtrain, ytrain = model.drop_na_inf(xtrain, ytrain)
    xtest, ytest = model.drop_na_inf(xtest, ytest)

    print("delete.after:", df.shape)
    acc_list = []
    for i in range(1):
        y_proba = model.train_model(
            xtrain,
            xtest,
            ytrain,
            ytest,
            i,
        )
        acc_list.append(y_proba[:, 2])

    # print(f"平均:", np.average(acc_list))
    signal = model.get_gold_signal(xtest, y_proba).iloc[:, [0, 1, 4]]
    signal.to_csv("gold_signal_rfc.csv")
    # print(signal.shape)
    return signal


def query(stocks, st, end):
    model = joblib.load("models/rfc_20201231.joblib")
    return main(stocks, st, end, trainging_idx=0, model=model)


def create_model():
    """
    訓練模型並儲存
    "2015-01-01", "2020-12-31"
    """
    main(parquet_db.query_stocks_ids_list(), "2015-01-01", "2020-12-31", trainging_idx=0.99)


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
