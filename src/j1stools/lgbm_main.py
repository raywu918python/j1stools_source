from re import L
import signal
from time import time

import random

from numpy import add
import pandas as pd
import numpy as np
from regex import P

from j1stools import data_filter, feature_builder, label_builder, lite_db, parquet_db, rfc_main
from j1stools.CONFIG import (
    BaseDataBuilderConfig,
    BaseLabelConfig,
    LgbmTrainConfig,
    MACDDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, TRAIN_TYPE, MODEL_TYPE


from j1stools.j1s_split_date import lgbm_split_date
from j1stools.market_filter import add_market_filter, apply_filter
from j1stools.model_builder import gen_lgbm_orgin_model, gen_lgbm_r_model
from j1stools.model_utils import drop_na_inf
from j1stools.train_flow import get_full_name, keep_latest_ten_files, start_train
import joblib

from j1stools.train_function import lgbm_r_function_train


def add_rfc_feature(df, data: BaseDataBuilderConfig):
    signal = rfc_main.test(
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


def predict(
    stocks,
    st,
    end,
):

    models = joblib.load("models/lgbm_timeseries_ensemble.joblib")

    df = prepare_data(stocks, st, end)

    features = [col for col in df.columns if col.startswith("f_")]
    # df["pred_return"] = np.max([m.predict(df[features]) for m in models], axis=0)
    df["pred_return"] = np.mean([m.predict(df[features]) for m in models], axis=0)

    df = df[["date", "stock_id", "pred_return"]]
    # print(signal.head(10))
    signal = df[df["pred_return"] > 0.05]
    signal.sort_values(by=["date", "pred_return"], inplace=True)
    signal.to_csv("lgbm_signal.csv", index=False)
    # print(signal.sort_values(by=["pred_return"], ascending=False).head(10))


def main():
    stocks = list(set(parquet_db.query_stocks_ids_list()) - set(["0050", "0052", "0056"]))
    st = "2015-01-01"
    end = "2024-01-01"
    models, scores = train(stocks=stocks, st=st, end=end)
    # joblib.dump(models, "models/lgbm_timeseries_ensemble.joblib")
    # predict(stocks=stocks, st=st, end=end)


def prepare_data(stocks, st, end):
    df_margin = lite_db.margin(stocks, st, end)
    df_market = parquet_db.query_price(["0050"], st, end)
    df_ibbuysell = lite_db.ibbuysell(stocks, st, end)
    df_margin = label_builder.add_target(df_margin, df_market=df_market)
    df_feature = feature_builder.gen_feature(
        None,
        FEATURE_TYPE.margin_ibbuysell,
        dfs=[df_margin, df_ibbuysell, df_market],
    )

    fold_ranges = [
        ("Fold1", "2016-07-06", "2017-12-22"),
        ("Fold2", "2017-12-25", "2019-06-24"),
        ("Fold3", "2019-06-25", "2020-12-11"),
        ("Fold4", "2020-12-14", "2022-06-13"),
        ("Fold5", "2022-06-14", "2023-12-01"),
    ]

    for name, start, end in fold_ranges:
        mask = (df_feature["date"] >= start) & (df_feature["date"] <= end)
        vol = df_feature.loc[mask, "f_market_volatility_20d"].mean()
        print(f"{name}: 平均波動率 {vol:.4f}")

    print("filter.before:", df_feature.shape)

    # df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    print("filter.after:", df_feature.shape)
    # 資料在這裡刪
    # df.set_index(["date", "stock_id"], inplace=True)
    # df.sort_index(level=["date", "stock_id"], inplace=True)
    return df_feature


def train(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2024-01-01",
    pick_import_feature=False,
):
    print(f"=" * 60, "lgbm start")
    keep_latest_ten_files("./model")
    # model = gen_lgbm_orgin_model()

    # df = prepare_data(stocks, st, end)

    # xtrain, xval, xtest, ytrain, yval, ytest = lgbm_split_date(df, True, True, trainging_idx)
    # xtrain, ytrain = drop_na_inf(xtrain, ytrain)
    # xval, yval = drop_na_inf(xval, yval)
    # xtest, ytest = drop_na_inf(xtest, ytest)

    # print("drop_na_inf:", len(xtrain), len(ytrain), len(xval), len(yval), len(xtest), len(ytest))

    if pick_import_feature:
        xtrain = feature_builder.pick_feature(xtrain)
        xtest = feature_builder.pick_feature(xtest)

    # xtrain = xtrain[[col for col in xtrain.columns if col.startswith("f_")]] if xtrain is not None else None
    # xtest = xtest[[col for col in xtest.columns if col.startswith("f_")]] if xtest is not None else None
    # xval = xval[[col for col in xtest.columns if col.startswith("f_")]] if xtest is not None else None
    print("=" * 60, "train")
    # lgbm_r_function_train(xtrain, ytrain, xval, yval, model, df)
    # lgbm_r_function_train(xtrain, ytrain, xval, yval, model)

    df_margin = lite_db.margin(stocks, st, end)
    df_market = parquet_db.query_price(["0050"], st, end)
    df_ibbuysell = lite_db.ibbuysell(stocks, st, end)
    df_margin = label_builder.add_target(df_margin, df_market=df_market)
    df_feature = feature_builder.gen_feature(
        None,
        FEATURE_TYPE.margin_ibbuysell,
        dfs=[df_margin, df_ibbuysell, df_market],
    )

    # 加入過濾欄位
    df = add_market_filter(df_feature, df_market)

    # 查看各 fold 的可交易比例

    fold_ranges = [
        ("Fold1", "2016-07-06", "2017-12-22"),
        ("Fold2", "2017-12-25", "2019-06-24"),
        ("Fold3", "2019-06-25", "2020-12-11"),
        ("Fold4", "2020-12-14", "2022-06-13"),
        ("Fold5", "2022-06-14", "2023-12-01"),
    ]

    for name, start, end in fold_ranges:
        mask = (df["date"] >= start) & (df["date"] <= end)
        ratio = df.loc[mask, "can_trade"].mean()
        print(f"{name}: 可交易比例 {ratio:.1%}")

    # 訓練時只用可交易的資料
    df_filtered = apply_filter(df)

    return walk_forward_train(df_filtered, params)

    # joblib.dump(model, get_full_name("lgbm_r"))
    # print("=" * 60, "test")

    # expected_features = model.feature_names_in_
    # xtest = xtest[expected_features]
    # model.train(xtest, ytest)
    # dfyproba = batter_predict(model, xtest, ytest)
    # print("=" * 60, "signal")
    # print(dfyproba.head())
    # print_ft_important(model)


def main_bak():
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
    data.feature_type = FEATURE_TYPE.test_lgbm_feature
    # data.feature_type = FEATURE_TYPE.today
    data.atrcfg = FILTER_TYPE.none_
    data.train_config = train
    data.st = "2024-01-01"
    data.end = "2099-01-01"
    signal = start_train(cfg=data)
    #############################################################
    signal = signal[signal["y_proba"] > 0.5]
    signal.sort_values(by=["date", "y_proba"], inplace=True)
    signal.to_csv("signal_today.csv", index=False)


from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,
    "learning_rate": 0.03,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1,
}


def walk_forward_train(df, params, n_splits=5):
    """
    Fold 2: 2017-12 ~ 2019-06
    2018年發生了：
    - 中美貿易戰開打
    - 台股從11000點跌到9000點
    - 全年跌幅約 -8%
    - 很多技術指標和籌碼訊號完全失效
    Fold 3: 2019-06 ~ 2020-12
    包含了：
    - 2019年反彈大多頭
    - 2020年疫情急跌後的V型反彈
    - 動能和籌碼訊號在這段時間特別有效
    """
    df = df.sort_values("date").reset_index(drop=True)

    dates = df["date"].unique()
    tss = TimeSeriesSplit(n_splits=n_splits)

    feature_cols = [c for c in df.columns if c.startswith("f_")]

    scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(dates)):
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]

        train = df[df["date"].isin(train_dates)]
        val = df[df["date"].isin(val_dates)]

        X_train, y_train = train[feature_cols], train["target"]
        X_val, y_val = val[feature_cols], val["target"]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(50),
            ],
        )

        pred = model.predict(X_val)  # ← 修正這裡
        score = np.corrcoef(pred, y_val)[0, 1]

        print(f"Fold {fold+1} IC: {score:.4f}")
        scores.append(score)
        models.append(model)

    # 用最後一個 fold 的模型看
    last_model = models[-1]

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": last_model.feature_importances_},
    ).sort_values("importance", ascending=False)

    print(importance.head(20))

    dates = df["date"].unique()
    tss = TimeSeriesSplit(n_splits=5)

    # for fold, (train_idx, val_idx) in enumerate(tss.split(dates)):
    #     val_dates = dates[val_idx]
    #     print(f"Fold {fold+1}: {val_dates.min()} ~ {val_dates.max()}")

    print(f"\n平均 IC: {np.mean(scores):.4f}")
    return models, scores


def select_stocks(df_today, models, top_n=20):
    """
    傳入今天的特徵，回傳買進清單
    """
    feature_cols = [c for c in df_today.columns if c.startswith("f_")]

    # 用所有 fold 的模型平均預測（ensemble）
    preds = np.mean([m.predict(df_today[feature_cols]) for m in models], axis=0)

    df_today["pred_score"] = preds

    # 取前 N 名
    top_stocks = df_today[["stock_id", "pred_score"]].sort_values("pred_score", ascending=False).head(top_n)

    return top_stocks


main()
# predict(
#     stocks=random.sample(parquet_db.query_stocks_no_etf(), 500),
#     st="2024-01-01",
#     end="2099-01-01",
# )
