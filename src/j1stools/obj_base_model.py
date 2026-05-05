from abc import ABC
from datetime import datetime
from enum import Enum
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.preprocessing import label_binarize

import joblib


class RUN_TYPE(Enum):
    train = 1
    create_model = 2
    predict = 3


class MODEL_TYPE(Enum):
    rfc = 1
    lgbm = 2


class BaseModel(ABC):

    def __init__(self):
        self.DEBUG = True
        self.THRESHOLD = 0.5
        now = datetime.now()
        self.t = now.strftime("%Y%m%d_%H%M%S")
        self.trainging_idx = 0
        self.run_type = RUN_TYPE.train
        self.model_type = MODEL_TYPE.rfc

    def drop_na_inf(self, x, y):
        # 1. 把 inf 換成 NaN
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 2. 找出哪些列是乾淨的（沒有 NaN）
        # 注意：xtrain 和 ytrain 的列必須同步刪除，否則 index 會對不起來
        clean_mask = x.isnull().any(axis=1) == False
        x = x[clean_mask]
        y = y[clean_mask]
        return x, y

    # def load_stocks(self):
    #     all_close = []
    #     all = []
    #     self.list_stocks = list(set(self.list_stocks) - set(["0050", "0052"]))

    #     for df in stock_in(self.list_stocks):
    #         try:
    #             # df = stock(stock_id, is_add_noise=self.is_add_noise)
    #             df = self.gen_feature(df)
    #             # 建ema相關的feature會日期有誤差
    #             df = df[30:]
    #             all_close.append(df[["date", "close", "stock_id"]])
    #             df = FilterData.get_data(df)
    #         except Exception as e:
    #             print(f"__load_stock_data > {e}")
    #             continue

    #         if df is None:
    #             continue
    #         all.append(df)

    #     if len(all) == 0:
    #         raise Exception("no data")
    #     else:
    #         self.all_close_df = pd.concat(all_close, axis=0).sort_values("date")
    #         return pd.concat(all, axis=0).sort_values("date")  # .reset_index(drop=True)

    def print_matrix(self, ytest, yproba, accuracy_label2, is_better=False):
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

    def get_full_name(self):
        return f"model/{self.model_type.name}{self.t}_{self.n}.joblib"

    # def prepare_df(self):

    #     df: pd.DataFrame = self.load_stocks()

    #     if (df.shape[0]) < 500:
    #         raise Exception("no data")

    #     # check df
    #     df.dropna(inplace=True)
    #     df = df.sort_values("date")

    #     features = self.__get_features_name(df)

    #     split_date_df = self.__split_date(df)
    #     self.train_df, self.test_df = split_date_df[0], split_date_df[1]

    #     # X_train = self.train_df[features]
    #     # y_train = self.train_df["target"]
    #     X_test = self.test_df[features]
    #     y_test = self.test_df["target"]

    #     return None, None, X_test, y_test

    def base_predict(self, xtest, ytest):
        # base
        y_proba = self.model.predict(xtest)
        accuracy = accuracy_score(ytest, y_proba)
        precisions = precision_score(ytest, y_proba, average=None, zero_division=0)
        if len(precisions) > 2:
            accuracy = precisions[2]
        else:
            accuracy = 0
        self.print_matrix(ytest=ytest, yproba=y_proba, accuracy_label2=accuracy)

        return accuracy

    def batter_predict(self, xtest, ytest):

        yproba = self.model.predict_proba(xtest)
        y_pred_threshold = (yproba >= self.THRESHOLD).astype(int)
        y_test_binarized = label_binarize(ytest, classes=[0, 1, 2])
        try:
            accuracy = precision_score(y_test_binarized, y_pred_threshold, average=None, zero_division=0)[2]
        except Exception as e:
            print("y_test_binarized", y_test_binarized[:5], sep="\n")
            print("y_pred_threshold", y_pred_threshold[:5], sep="\n")
            raise e
        self.print_matrix(ytest=ytest, yproba=yproba, accuracy_label2=accuracy, is_better=True)

        self.print_trading_date(xtest, yproba)
        self.print_target_counts(ytest)
        # import ft
        if self.is_print_import_ft:
            self.print_ft_important()

        return yproba

    def train_model(
        self,
        xtrain,
        xtest,
        ytrain,
        ytest,
        xval=None,
        yval=None,
        n=0,
        function_train=None,
    ):
        self.n = n
        print(f"*" * 30, f"第 {self.n + 1} 次訓練")
        st = time()
        # train
        model = self.model
        if self.run_type == RUN_TYPE.train or self.run_type == RUN_TYPE.create_model:
            if function_train:
                if self.model_type == MODEL_TYPE.lgbm:
                    function_train(xtrain, xval, ytrain, yval, model)
                else:
                    function_train(xtrain, ytrain, model)
            else:
                model.fit(xtrain, ytrain)
            joblib.dump(model, self.get_full_name())
            print(f"訓練時間: {time()-st:.2f} 秒")

        if self.run_type == RUN_TYPE.train or self.run_type == RUN_TYPE.predict:
            # test
            expected_features = self.model.feature_names_in_
            xtest = xtest[expected_features]
            if self.run_type == RUN_TYPE.train:
                self.base_predict(xtest, ytest)

            accuracy = self.batter_predict(xtest, ytest)
            print(f"回測時間: {time()-st:.2f} 秒")
            return accuracy

    def print_target_counts(self, ytest):
        print("*" * 30, "value_counts")
        value_df = pd.concat(
            keys=["test"],
            objs=[ytest.value_counts()],
            # objs=[ytest["target"].value_counts()],
            # objs=[train_df["target"].value_counts(), test_df["target"].value_counts()],
            axis=1,
        ).sort_index()
        print(value_df)

    def check_null_inf(self, X_train):
        # 檢查是否有無限大 (inf)
        print(f"*" * 30, "isinf\n", np.isinf(X_train).any())
        # 檢查是否有空值 (NaN)
        print(f"*" * 30, "isnull\n", X_train.isnull().any())

    def print_ft_important(self):
        feat_importances = pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_,
        )

        print("*" * 30, f"feat_importances {feat_importances.size}")
        print(feat_importances.sort_values(ascending=False))
        with open(f"log/log_import_ft{self.n}.py", "w", encoding="utf-8") as f:
            f.write(str(feat_importances.sort_values(ascending=False).index.tolist()))

    def gen_gold_signal(self, xtest, y_proba: pd.Series) -> pd.DataFrame:
        results = pd.DataFrame(y_proba, index=xtest.index)
        results.reset_index(drop=False, inplace=True)
        # print(results.head())
        signal_label_2 = results.iloc[:, 4]
        return results

    def print_trading_date(self, xtest: pd.DataFrame, y_proba):
        print(f"*" * 30, "print_trading_date")
        xtest = self.gen_gold_signal(xtest, y_proba)
        gold_signals = xtest[xtest.iloc[:, 4] > self.THRESHOLD]

        print(f"門檻設定: {self.THRESHOLD}")
        print(f"訊號量: {len(gold_signals)}")
        if len(gold_signals) > 0:
            print("訊號分布的天數:", gold_signals["date"].nunique())
            print("訊號包含的股票數:", gold_signals["stock_id"].nunique())  # 你的欄位是 stock_id
        return gold_signals
