from time import time

import random
import lightgbm as lgb
from lightgbm import LGBMClassifier
from pygments.unistring import No

from j1stools.obj_base_model import RUN_TYPE, BaseModel

from j1stools import feature_builder, obj_filter_data, rfc_main
from j1stools import parquet_db
from click import File
from numpy.testing import print_assert_equal
from pandas import DataFrame
from regex import D
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_score,
)

from j1stools.obj_label import Label
import joblib


class LgbmModel(BaseModel):
    def __init__(self):
        super().__init__()


def split_date(
    df: pd.DataFrame,
    is_gen_train_data=True,
    is_gen_test_data=True,
):
    # group
    df.sort_values(by=["date", "stock_id"], inplace=True)

    X = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["target"]
    if is_gen_train_data and is_gen_test_data:
        xremain, xtest, yremain, ytest = train_test_split(X, y, test_size=0.15, random_state=42)
        xtrain, xval, ytrain, yval = train_test_split(xremain, yremain, test_size=0.176, random_state=42)

        print(f"訓練集大小: {len(xtrain)}")
        print(f"驗證集大小: {len(xval)}")
        print(f"測試集大小: {len(xtest)}")
        return xtrain, xval, xtest, ytrain, yval, ytest
    elif is_gen_train_data:
        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.15, random_state=42)
        return xtrain, xval, None, ytrain, yval, None
    elif is_gen_test_data:
        return None, None, X, None, None, y


def get_model():
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return model


def lgbm_train(xtrain, xval, ytrain, yval, model):
    model.fit(
        xtrain,
        ytrain,
        eval_set=[(xval, yval)],
        eval_metric="multi_logloss",
        callbacks=[
            # 如果 50 輪內沒進步就停止
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )


#         joblib.dump(model, model_save_name)

#     #
#     # 5. 預測與評估
#     #
#     # 獲取模型預期的特徵名稱（適用於 scikit-learn 1.0+）
#     expected_features = model.feature_names_in_
#     # 強制調整 X_test 的欄位順序與名稱
#     X_test = X_test[expected_features]

#     def base_pred():
#         y_pred = model.predict(X_test)
#         precisions = precision_score(y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
#         accuracy = precisions[2]
#         return y_pred, accuracy

#     if DEBUG:
#         y_pred, accuracy = base_pred()
#         print_model_info(y_test=y_test, y_proba=y_pred, accuracy_label2=accuracy)

#     # 在 batter_pred 和 check_better_pred 呼叫處修改
#     def batter_pred():

#         y_proba = model.predict_proba(X_test)
#         y_pred_threshold = (y_proba >= THRESHOLD).astype(int)
#         y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
#         precision_per_class = precision_score(y_test_binarized, y_pred_threshold, average=None)
#         return y_proba, precision_per_class[2]

#     y_pred, accuracy = batter_pred()
#     print_model_info(y_test=y_test, y_proba=y_pred, accuracy_label2=accuracy, is_better=True)

#     def find_batter_model():

#         import matplotlib.pyplot as plt
#         from sklearn.metrics import precision_score

#         thresholds = np.arange(0.5, 0.9, 0.05)

#         precisions = []
#         counts = []

#         for t in thresholds:
#             preds = (prob_label2 > t).astype(int)
#             precisions.append(precision_score(y_test, preds, zero_division=0))
#             counts.append(sum(preds))

#         # 將資料打包成 DataFrame
#         results_df = pd.DataFrame({"Precision": precisions, "Count": counts})

#         # 按 Precision 排序
#         results_df = results_df.sort_values(by="Precision")

#         print(f"{'*' * 30} find_batter_model {'*' * 30}")
#         print(results_df.to_string(index=False))

#         # 畫圖：左軸勝率，右軸次數
#         if 0:
#             fig, ax1 = plt.subplots()
#             ax1.plot(thresholds, precisions, "b-", label="Precision")
#             ax2 = ax1.twinx()
#             ax2.bar(thresholds, counts, alpha=0.3, label="Signal Count")
#             plt.show()

#     # find_batter_model()

#     def check_dup_features(df):
#         corr_matrix = df.corr()
#         print(corr_matrix)
#         pd.DataFrame(corr_matrix).to_csv("features.csv")

#     def print_ft_important():
#         feat_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)

#         print("*" * 30, f"feat_importances {feat_importances.size}")
#         print(feat_importances.sort_values(ascending=False))
#         with open("log_import_ft.txt", "w", encoding="utf-8") as f:
#             f.write(str(feat_importances.sort_values(ascending=False).index.tolist()))

#         # print(model.feature_names_in_)

#     def print_describe():
#         print(f"x" * 88, "describe\n", df[features].describe().T)

#     # print_describe()

#     def check_better_pred(test_df_full: pd.DataFrame, probabilities, threshold=0.65):
#         df = test_df_full.copy()
#         df["pred_prob"] = probabilities  # 這是 Label 2 的機率

#         # 篩選高信心訊號
#         gold_signals = df[df["pred_prob"] > threshold]

#         # print(f"實戰精準度 (預測為 2 且實際為 2 的比例):")
#         if len(gold_signals) > 0:
#             # 這裡計算的是在訊號中，target 真的等於 2 的比例
#             actual_win_rate = (gold_signals["target"] == 2).mean()
#             # print(f"Label 2 Precision: {actual_win_rate:.2%}")
#             # 1. 檢查訊號分布
#             print(f"門檻設定: {threshold}")
#             print("訊號分布的天數:", gold_signals["date"].nunique())
#             print("訊號包含的股票數:", gold_signals["stock_id"].nunique())  # 你的欄位是 stock_id
#             gold_signals["date"].to_csv("log_trading_date.csv")

#         return gold_signals

#     probs = model.predict_proba(X_test)[:, 2]
#     check_better_pred(test_df, probs, THRESHOLD)

#     # print_describe()

#     return y_pred


def backtest_with_daily_cap(gold_signals, cap=3):
    """
    gold_signals: 已經篩選過門檻 (如 > 0.75) 的 DataFrame
    cap: 每天最多交易幾支
    """
    # 1. 確保資料按日期和預測機率排序（機率由高到低）
    signals_sorted = gold_signals.sort_values(by=["date", "pred_prob"], ascending=[True, False])

    # 2. 執行每日限額：每一天只取前 cap 名
    capped_signals = signals_sorted.groupby("date").head(cap).copy()

    # 3. 重新計算實戰指標
    real_precision = capped_signals["target"].mean()
    real_growth = capped_signals["real_growth"].mean()
    total_trades = len(capped_signals)
    unique_days = capped_signals["date"].nunique()

    print(f"--- 每日限額 {cap} 支 回測結果 ---")
    print(f"總交易次數: {total_trades}")
    print(f"訊號分布天數: {unique_days}")
    print(f"實戰精準度 (Precision): {real_precision:.2%}")
    print(f"實戰平均表現: {real_growth:.4f}")

    return capped_signals


def get_next_day_prediction(model, df: pd.DataFrame, threshold=0.6):
    # expected_features = get_features_name(df)

    # # 1. 只取最後一筆資料 (最新的一天)
    # last_row = df.tail(1).copy()

    # # 2. 取得這筆資料的日期與推算明天
    # current_date = last_row.index.max()
    # next_trading_day = (current_date + pd.offsets.BDay(1)).date()

    # # 3. 準備特徵 (假設 prepare_features 已經處理好最後一天的特徵)
    # # 注意：這裡丟進 model 的只有這一列
    # expected_features = model.feature_names_in_
    # X_latest = last_row[expected_features]

    # # 4. 取得模型對「明天」的信心機率 [:, 1]
    # prob = model.predict_proba(X_latest)[0, 1]  # 因為只有一筆，直接拿第 0 個

    # # 5. 判斷進場條件 (漲幅過濾等)
    # curr_return = (last_row["current_price"] - last_row["last_close"]) / last_row["last_close"]
    # curr_return = curr_return.iloc[0]  # 轉成數值

    # stock_id = last_row["stock_id"].iloc[0]

    # if prob > threshold and curr_return < 0.07:
    #     print(f"🔥 [買入訊號] 標的: {stock_id}", end="")
    #     print(
    #         f"📅 依據日期: {current_date.date()} | 🎯 交易日期: {next_trading_day}",
    #         end="",
    #     )
    #     print(f"📈 模型信心: {prob:.2%}")
    #     with open("log_signals.csv", "a", encoding="utf-8") as f:
    #         f.write(
    #             f"依據日期: {current_date.date()} | 🎯 交易日期: {next_trading_day} | 標的: {stock_id} | 信心度: {prob:.2%}\n"
    #         )

    #     return {"stock_id": stock_id, "trade_date": next_trading_day}
    # else:
    #     print(f"Fail 標的: {stock_id}")
    return None


def pred_tomorrow():
    """查看最後一天的預測結果 (針對明天)"""

    # model = joblib.load("model_rf_0406v1.joblib")
    # stocks = random.sample(get_file_list(), 100)
    # for stock_id in stocks:
    #     if str(stock_id).startswith("00"):
    #         continue
    #     df = stock(stock_id)
    #     if df.shape[0] < 100:
    #         continue

    #     df["current_price"] = df["close"]
    #     df["last_close"] = df["close"]

    #     get_next_day_prediction(model, df, 0.7)


#
# 執行時間: 定位在 13:10 盤中，用當前價模擬收盤。
#
def test(n=0):
    # model = None
    # stocks = random.sample(get_file_list(), 500)
    # # stocks = get_csv_list()  # [65:70]
    # # model = joblib.load(model_save_name)
    # # stocks = ["3701"]
    # # print(stocks)
    # pd.DataFrame(stocks).to_csv(f"log_training_stock{n}.csv")

    # year = 2015 + n
    # from_year = year
    # to_year = year + 2
    # print(f"*" * 60, f"第 {n+1} 次訓練 {from_year} - {to_year}")
    # start_time = time.time()
    # all = []
    # for stock_id in stocks:
    #     if str(stock_id).startswith("00"):
    #         continue
    #     # df = gen_data(stock_id)[200:400]
    #     df = gen_data(stock_id)
    #     if df is None:
    #         continue
    #     df = df.loc[f"{from_year}-01":f"{to_year}-12"]

    #     all.append(df)
    #     # trade(df, p)
    # end_time = time.time()  # 結束計時
    # duration = end_time - start_time
    # print(f"建立特徵總共耗時: {duration:.2f} 秒")

    # if len(all) != 0:
    #     final = pd.concat(all, axis=0).sort_values("date").reset_index(drop=True)
    #     start_time = time.time()
    #     p = train_model(final, model)
    #     end_time = time.time()  # 結束計時
    #     duration = end_time - start_time
    #     print(f"訓練總共耗時: {duration:.2f} 秒")
    pass


def query_no_rfc(stocks, st, end):

    model = joblib.load("models/lgbm_20231231_no_rfc.joblib")
    return exec(
        stocks,
        st,
        end,
        0,
        model,
    )


def exec(
    stocks=random.sample(parquet_db.query_stocks_ids_list(), 100),
    st="2024-01-01",
    end="2099-01-01",
    model=None,
    is_using_rfc=False,
    trainging_idx=0.7,
    run_type=RUN_TYPE.train,
    pick_feature=True,
):
    print(f"*" * 60, "lgbm start")

    m = LgbmModel()
    if model:
        m.model = model
    else:
        m.model = get_model()
    m.THRESHOLD = 0.6
    m.is_print_import_ft = False
    m.is_add_noise = False
    m.trainging_idx = trainging_idx
    m.run_type = run_type

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

    df = feature_builder.gen_feature(df)
    if is_using_rfc:
        df = add_rfc_feature(df, stocks, st, end)

    df = Label.add_label(df)
    df = obj_filter_data.FilterData.get_data(df, True, False)
    df.set_index(["date", "stock_id"], inplace=True)
    xtrain, xval, xtest, ytrain, yval, ytest = split_date(
        df,
        is_gen_train_data=is_gen_train_data,
        is_gen_test_data=is_gen_test_data,
    )
    if is_gen_train_data:
        xtrain, ytrain = m.drop_na_inf(xtrain, ytrain)
        xval, yval = m.drop_na_inf(xval, yval)
        if pick_feature:
            xtrain = feature_builder.pick_feature(xtrain)
            xval = feature_builder.pick_feature(xval)
    if is_gen_test_data:
        xtest, ytest = m.drop_na_inf(xtest, ytest)
        if pick_feature:
            xtest = feature_builder.pick_feature(xtest)

    acc_list = []
    for i in range(1):
        y_proba = m.train_model(
            xtrain,
            xtest,
            ytrain,
            ytest,
            xval=xval,
            yval=yval,
            n=i,
            function_train=lgbm_train,
        )
        if m.run_type == RUN_TYPE.create_model:
            return
        acc_list.append(y_proba[:, 2])

    # print(f"平均:", np.average(acc_list))
    signal = m.gen_gold_signal(xtest, y_proba).iloc[:, [0, 1, 4]]
    signal.rename(columns={2: "y_proba"}, inplace=True)
    signal["date"] = pd.to_datetime(signal["date"])
    return signal


def predict(stocks, st, end):
    model = joblib.load("models/lgbm.joblib")
    return exec(
        stocks=stocks,
        st=st,
        end=end,
        model=model,
        is_using_rfc=True,
        run_type=RUN_TYPE.predict,
    )


def add_rfc_feature(df, stocks, st, end):
    signal = rfc_main.predict(stocks, st, end)
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


# create_model()
# main()
