from abc import ABC
from datetime import datetime
from pyexpat import model
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.preprocessing import label_binarize
from pathlib import Path

import joblib

from j1stools.CONFIG import BaseDataBuilderConfig, BaseTrainConfig
from j1stools.RESULT import DataBuilderResult, TrainResult
from j1stools.TYPE import MODEL_TYPE, TRAIN_TYPE
from j1stools.data_builder import DataBuilder


def print_matrix(ytest, yproba, accuracy_label2, is_better=False):
    if is_better:
        # 設定門檻值範圍：從 0.5 到 0.9，包含 0.9 所以終點設為 0.91
        thresholds = np.arange(0.5, 0.91, 0.05)
        for thresh in thresholds:
            print(f"\n" + "=" * 40)
            print(f"信心度門檻: {thresh:.2f}")
            # 根據當前門檻判定預測結果
            y_pred_threshold = (yproba >= thresh).astype(int)
            conf_matrix = np.zeros((3, 3), dtype=int)
            for i in range(len(ytest)):
                actual_label = ytest.iloc[i]
                # 找出所有超過門檻的類別索引
                predicted_indices = np.where(y_pred_threshold[i] == 1)[0]
                for pred_label in predicted_indices:
                    conf_matrix[actual_label, pred_label] += 1
            df_cm = pd.DataFrame(
                conf_matrix,
                index=["Actual 0", "Actual 1", "Actual 2"],
                columns=["Pred 0", "Pred 1", "Pred 2"],
            )
            # 計算該門檻下的進場勝率 (以 Label 2 為例)
            # 勝率 = 預測為 2 且實際為 2 / 總共預測為 2 的次數
            total_pred_2 = conf_matrix[:, 2].sum()
            win_rate_2 = (conf_matrix[2, 2] / total_pred_2) if total_pred_2 > 0 else 0
            print(df_cm)
            print(f"在此信心度下，Label 2 的進場勝率: {win_rate_2:.2%}")
    else:
        cm = confusion_matrix(ytest, yproba)
        # print("cm", cm, sep="\n")
        result = [[0] * 3 for _ in range(3)]
        # 2. 用雙層迴圈把原始資料填入左上角
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                result[i][j] = int(cm[i][j])
        cm = np.array(result)
        # print("cm", result, sep="\n")
        tp_2 = cm[2, 2]  # 真實為 2，預測也為 2 (賺錢)
        fp_2 = cm[0, 2] + cm[1, 2]  # 真實為 0 或 1，但預測為 2 (虧錢或沒賺)
        print(f"*" * 30, f"三分類混淆矩陣\n", cm)
        print(f"預測會漲但實際沒漲/跌 (FP): {fp_2}")
        print(f"預測會漲且實際達標 (TP): {tp_2}")
        # precision_2 = tp_2 / (tp_2 + fp_2) if (tp_2 + fp_2) > 0 else 0
        print(f"進場勝率: {accuracy_label2:.2%}")


def get_full_name(model_name: str):
    now = datetime.now()
    t = now.strftime("%Y%m%d_%H%M%S")
    return f"model/{model_name}{t}.joblib"


def base_predict(model, xtest, ytest):
    # base
    y_proba = model.predict(xtest)
    accuracy = accuracy_score(ytest, y_proba)
    precisions = precision_score(ytest, y_proba, average=None, zero_division=0)
    if len(precisions) > 2:
        accuracy = precisions[2]
    else:
        accuracy = 0
    print_matrix(ytest=ytest, yproba=y_proba, accuracy_label2=accuracy)
    return accuracy


def batter_predict(model, xtest, ytest, threshold=0.5):
    yproba = model.predict_proba(xtest)
    dfyproba = pd.DataFrame(yproba, index=xtest.index)
    y_pred_threshold = (yproba >= threshold).astype(int)
    y_test_binarized = label_binarize(ytest, classes=[0, 1, 2])
    try:
        accuracy = precision_score(y_test_binarized, y_pred_threshold, average=None, zero_division=0)[2]
    except Exception as e:
        print("y_test_binarized", y_test_binarized[:5], sep="\n")
        print("y_pred_threshold", y_pred_threshold[:5], sep="\n")
        raise e
    print_matrix(ytest=ytest, yproba=yproba, accuracy_label2=accuracy, is_better=True)
    # print_trading_date(threshold, dfyproba)
    # print_target_counts(ytest)
    return dfyproba


# def start_train_lgbm_r(xtrain, ytrain, xtest, ytest, xval, yval, model, model_name="lgbm_r", function_train=None, df=None):
#     keep_latest_ten_files("./model")

#     xtest = xtest[[col for col in xtest.columns if col.startswith("f_")]] if xtest is not None else None
#     xtrain = xtrain[[col for col in xtrain.columns if col.startswith("f_")]] if xtrain is not None else None
#     xval = xval[[col for col in xval.columns if col.startswith("f_")]] if xval is not None else None

#     t1 = time()
#     #############################################################train
#     function_train(xtrain, ytrain, model, df)
#     joblib.dump(model, get_full_name(model_name))
#     print(f"訓練時間: {time()-t1:.2f} 秒")
#     #############################################################test
#     expected_features = model.feature_names_in_
#     xtest = xtest[expected_features]
#     y_pred = model.predict(xtest)  # 輸出 0~1 的排名預
#     print(xtrain.isna().mean().sort_values(ascending=False
#     corr = xtrain.corrwith(data.ytrain)
#     print(corr.sort_values(
#     print(data.ytrain.describe(
#     print(pd.Series(y_pred).describe())
#     print(f">=0.8 的數量：{(y_pred >= 0.8).sum()}
#     result = xtest.copy()
#     result["predicted_rank"] = y_pred
#     result["future_return"] = xtest_future_retu
#     return TrainResult(result, None)


def start_train(
    xtrain,
    ytrain,
    xtest,
    ytest,
    model,
    function_train,
    model_name="rfc",
    df=None,
):
    keep_latest_ten_files("./model")

    xtest = xtest[[col for col in xtest.columns if col.startswith("f_")]] if xtest is not None else None
    xtrain = xtrain[[col for col in xtrain.columns if col.startswith("f_")]] if xtrain is not None else None

    t1 = time()
    #############################################################train
    function_train(xtrain, ytrain, model, df)
    joblib.dump(model, get_full_name(model_name))
    #############################################################test
    expected_features = model.feature_names_in_
    xtest = xtest[expected_features]
    base_predict(model, xtest, ytest)
    accuracy = batter_predict(model, xtest, ytest)
    print(f"訓練時間: {time()-t1:.2f} 秒")
    return accuracy


def flow(data: DataBuilderResult, train_cfg: BaseTrainConfig):
    keep_latest_ten_files("./model")

    xtest = xtest[[col for col in xtest.columns if col.startswith("f_")]] if xtest is not None else None
    xtrain = xtrain[[col for col in xtrain.columns if col.startswith("f_")]] if xtrain is not None else None
    xval = xval[[col for col in xval.columns if col.startswith("f_")]] if xval is not None else None

    print(f"*" * 30, f"第 {train_cfg.n} 次訓練")
    st = time()
    # train
    model = train_cfg.model
    if train_cfg.train_type == TRAIN_TYPE.train or train_cfg.train_type == TRAIN_TYPE.create_model:
        train_cfg.function_train(data, model)
        joblib.dump(model, get_full_name(train_cfg))
        print(f"訓練時間: {time()-st:.2f} 秒")
    #############################################################
    if train_cfg.train_type == TRAIN_TYPE.train or train_cfg.train_type == TRAIN_TYPE.predict:
        # test
        expected_features = train_cfg.model.feature_names_in_
        xtest = xtest[expected_features]
        if train_cfg.train_type == TRAIN_TYPE.train:
            if train_cfg.model_type == MODEL_TYPE.lgbm_r:
                y_pred = model.predict(xtest)  # 輸出 0~1 的排名預測值

                print(xtrain.isna().mean().sort_values(ascending=False))

                corr = xtrain.corrwith(data.ytrain)
                print(corr.sort_values())

                print(data.ytrain.describe())

                print(pd.Series(y_pred).describe())
                print(f">=0.8 的數量：{(y_pred >= 0.8).sum()}")

                result = xtest.copy()
                result["predicted_rank"] = y_pred
                result["future_return"] = xtest_future_return

                return TrainResult(result, None)
        else:
            print("222 base_predict")
            base_predict(train_cfg.model, xtest, data.ytest)
            accuracy = batter_predict(train_cfg, xtest, data.ytest)
        print(f"回測時間: {time()-st:.2f} 秒")
        return TrainResult(None, accuracy)


def print_target_counts(ytest):
    print("*" * 30, "value_counts")
    value_df = pd.concat(
        keys=["test"],
        objs=[ytest.value_counts()],
        # objs=[ytest["target"].value_counts()],
        # objs=[train_df["target"].value_counts(), test_df["target"].value_counts()],
        axis=1,
    ).sort_index()
    print(value_df)


def check_null_inf(xtrain):
    # 檢查是否有無限大 (inf)
    print(f"*" * 30, "isinf\n", np.isinf(xtrain).any())
    # 檢查是否有空值 (NaN)
    print(f"*" * 30, "isnull\n", xtrain.isnull().any())


def print_ft_important(model):
    feat_importances = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_,
    )
    print("=" * 60, f"feat_importances {feat_importances.size}")
    print(feat_importances.sort_values(ascending=False))
    with open(f"log/log_import_ft.py", "w", encoding="utf-8") as f:
        f.write(str(feat_importances.sort_values(ascending=False).index.tolist()))


# def gen_gold_signal(xtest, yproba: pd.Series) -> pd.DataFrame:
#     yproba.reset_index(drop=False, inplace=True)
#     yproba.rename(columns={2: "yproba"}, inplace=True)
#     yproba.to_csv("signal.csv")

#     return yproba


def print_trading_date(threshold, dfyproba: pd.DataFrame):
    print(f"*" * 30, "print_trading_date")
    dfyproba.reset_index(drop=False, inplace=True)
    gold_signals = dfyproba[dfyproba.loc[:, 2] > threshold]

    print(f"門檻設定: {threshold}")
    print(f"訊號量: {len(gold_signals)}")
    if len(gold_signals) > 0:
        print("訊號分布的天數:", gold_signals["date"].nunique())
        print("訊號包含的股票數:", gold_signals["stock_id"].nunique())  # 你的欄位是 stock_id
    return gold_signals


def keep_latest_ten_files(directory_path: str):
    """
    不論檔名，保留資料夾中最後修改時間最新的 10 個檔案，其餘刪除。
    """
    folder = Path(directory_path)
    if not folder.exists():
        return

    # 取得資料夾內的所有檔案（排除子資料夾）
    files = [f for f in folder.iterdir() if f.is_file()]

    # 依最後修改時間排序（最新到最舊）
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # 如果檔案數超過 10 個，將第 11 個（索引 10）之後的舊檔案刪除
    if len(files) > 10:
        for file in files[10:]:
            file.unlink()
            print(f"已刪除舊檔案: {file.name}")


def start_train_flow_bak(cfg: BaseDataBuilderConfig):
    print(f"*" * 60, f"{cfg.train_config.model_type.value} start")

    data = DataBuilder(cfg).build()
    train = cfg.train_config
    for i in range(1):
        train.n = i
        result: TrainResult = flow(data, train)
        top = result.top
        y_proba = result.yproba
        if train.train_type == TRAIN_TYPE.create_model:
            return

    if train.model_type == MODEL_TYPE.lgbm_r:
        return top
    else:
        signal = gen_gold_signal(xtest, y_proba).iloc[:, [0, 1, 4]]
        signal.rename(columns={2: "y_proba"}, inplace=True)
        signal["date"] = pd.to_datetime(signal["date"])

        return signal
