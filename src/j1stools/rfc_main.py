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


def prepare_data(stocks, st, end, atr_filter_type=FILTER_TYPE.add_):
    df = parquet_db.query_price(stocks, st, end)
    print("filter.before:", df.shape)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, atr_filter_type)
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


def train_best_model(
    stocks=None,
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    cv=3,
):
    """
    Pipeline 架構 — 每個模型包在 Pipeline([("clf", model)]) 裡，未來可以直接在前面插入 scaler 或 feature selector
    四個模型 — RFC、LightGBM、GradientBoosting、XGBoost（如果沒裝 xgboost 會自動跳過）
    GridSearchCV — cv=3（可調）、scoring="f1_weighted"（適合多分類不平衡資料），每個模型各有獨立的 param_grid
    結果比較 — 全部跑完後印出比較表，找出最高 cv_score 的 winner
    自動存檔 — winner pipeline 存到 model/best_{name}_時間戳.joblib
    holdout 評估 — 用既有的 batter_predict 跑測試集
    """
    import numpy as np
    import joblib as _joblib
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from lightgbm import LGBMClassifier

    try:
        from xgboost import XGBClassifier

        _has_xgb = True
    except ImportError:
        _has_xgb = False
        print("xgboost 未安裝，跳過 XGB")

    if stocks is None:
        stocks = parquet_db.query_stocks_no_etf()

    print("=" * 60, "prepare data")
    # df = prepare_data(stocks, st, end, FILTER_TYPE.add_and_del)
    df = prepare_data(stocks, st, end, atr_filter_type=FILTER_TYPE.add_and_del)
    xtrain, xtest, ytrain, ytest = rfc_split_date(df, True, True, trainging_idx)
    xtrain, ytrain = drop_na_inf(xtrain, ytrain)
    xtest, ytest = drop_na_inf(xtest, ytest)

    f_cols = [col for col in xtrain.columns if col.startswith("f_")]
    xtrain, xtest = xtrain[f_cols], xtest[f_cols]
    print(f"features: {len(f_cols)}, train: {len(xtrain)}, test: {len(xtest)}")

    pipelines = {
        "rfc": Pipeline([("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))]),
        "lgbm": Pipeline([("clf", LGBMClassifier(random_state=42, n_jobs=1, class_weight="balanced", verbose=-1))]),
        "gbm": Pipeline([("clf", GradientBoostingClassifier(random_state=42))]),
    }
    if _has_xgb:
        pipelines["xgb"] = Pipeline(
            [
                ("clf", XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss", use_label_encoder=False)),
            ]
        )

    param_grids = {
        "rfc": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [8, 12],
            "clf__min_samples_leaf": [10, 20],
            "clf__max_features": [0.2, 0.4],
        },
        "lgbm": {
            "clf__n_estimators": [300, 600],
            "clf__num_leaves": [31, 63],
            "clf__max_depth": [6, 10],
            "clf__learning_rate": [0.05, 0.1],
        },
        "gbm": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
            "clf__min_samples_leaf": [10, 20],
        },
        "xgb": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
            "clf__min_child_weight": [5, 10],
        },
    }

    best_score = -np.inf
    best_name = None
    best_pipeline = None
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*60} GridSearch: {name}")
        gs = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=1,
            verbose=1,
            refit=True,
        )
        try:
            with _joblib.parallel_backend("threading"):
                gs.fit(xtrain, ytrain)
        except Exception as e:
            print(f"  [SKIP] {name} 訓練失敗: {e}")
            continue
        results[name] = {"best_params": gs.best_params_, "cv_score": gs.best_score_}
        print(f"  best_params: {gs.best_params_}")
        print(f"  cv_score:    {gs.best_score_:.4f}")
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_name = name
            best_pipeline = gs.best_estimator_

    print(f"\n{'='*60} 結果比較")
    for name, r in results.items():
        mark = " <-- winner" if name == best_name else ""
        print(f"  {name}: {r['cv_score']:.4f}  {r['best_params']}{mark}")

    save_path = get_full_name(f"best_{best_name}")
    joblib.dump(best_pipeline, save_path)
    print(f"\nsaved: {save_path}")

    print("=" * 60, "test on holdout")
    dfyproba = batter_predict(best_pipeline, xtest, ytest)
    print(dfyproba.head())

    return best_pipeline, best_name, results


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


if __name__ == "__main__":
    from j1stools.rfc_main import train_best_model

    pipeline, best_name, results = train_best_model(
        st="2023-01-01",
        end="2024-01-01",
        cv=3,
    )


# predict_today()
# main()
# predict()
# optimize()
# model_release()
# main()
