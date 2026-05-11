import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

from j1stools import feature_builder, parquet_db
from j1stools.label_builder import profit_label
from j1stools.train_flow import BaseTrainConfig


from j1stools.obj_filter_data import FilterData
from j1stools import rfc_main as rfc


class FeatureCheck:

    def __init__(self):
        pass

    def fin_dup_heatmap(self, X_train: pd.DataFrame):
        """
        check 重複（高度相關）的特徵
        方法 A:相關係數矩陣 (Heatmap)
        關係數矩陣 (Correlation Matrix)
        """
        # 假設 df 是你的資料框
        corr_matrix = X_train.corr().abs()

        # 繪製熱力圖
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.show()

        # 找出相關性大於 0.9 的特徵對
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        print(f"建議刪除的重複特徵: {to_drop}")

    def find_dup_vif(self, X_train):
        """
        check 重複（高度相關）的特徵
        方法 B:VIF (Variance Inflation Factor)
        VIF 衡量一個特徵能被其他特徵「解釋」的程度。通常 VIF > 5 或 10 就代表存在嚴重的共線性（重複）。
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # 計算每個特徵的 VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_train.columns
        vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
        print(vif_data)

    # 初步篩選：先用 Lasso 或 相關係數 剔除掉那些明顯是雜訊的特徵。
    # 精細調整：在剩下的特徵中，用 RFE 搭配你最終要用的模型（如 XGBoost 或隨機森林）來找出最黃金的組合。
    def find_import_by_l1_lasso(self, X_train, y_train):
        """
        篩選掉沒用的特徵
        L1 正規化 (Lasso Regression)
        原理： L1 正規化會在損失函數中加入特徵係數的絕對值之和作為懲罰項。它的神奇之處在於，它會把「不重要」特徵的係數直接壓縮到 0。這等於自動幫你把特徵踢出模型。
        """
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        # Lasso 對數值大小很敏感，一定要先做標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # 初始化 Lasso，alpha 就是懲罰強度 lambda
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_scaled, y_train)

        # 找出係數不為 0 的特徵
        important_features = [f for f, coef in zip(X_train.columns, lasso.coef_) if coef != 0]
        print(f"Lasso 留下的特徵: {important_features}")

    def find_import_by_RFECV(self, X_train, y_train):
        """
        篩選掉沒用的特徵
        2. 遞迴特徵消除 (Recursive Feature Elimination, RFE)
        原理：
        這是一種「貪婪演算法」。它先用所有特徵訓練模型，然後算出每個特徵的重要性，把最不重要的那個踢掉。接著重複這個過程，直到剩下的特徵數量達到你的要求。
        """

        print("find_import_by_RFECV...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 建立一個基礎模型
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        # 建立並訓練 RFECV
        # min_features_to_select 可以防止它把特徵刪到只剩 1 個
        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            min_features_to_select=5,
        )
        rfecv.fit(X_train_scaled, y_train)

        # 4. 直接從 RFECV 取得結果
        print(f"最理想的特徵數量是: {rfecv.n_features_}")

        # 哪些特徵被選中了？
        selected_features = X_train.columns[rfecv.support_]
        print(f"被選中的特徵: {list(selected_features)}")

        # 查看排名 (1 代表被選中的特徵)
        ranking = dict(zip(X_train.columns, rfecv.ranking_))
        # 只印出排名前 10 的特徵方便觀察
        sorted_ranking = sorted(ranking.items(), key=lambda x: x[1])
        top = 10
        print(f"特徵排名前 {top} 精簡版: {sorted_ranking[:top]}")

    def get_non_dup_ft_VIF(self, x_train, thresh=10):
        cols = x_train.columns.tolist()
        while True:
            vif = [variance_inflation_factor(x_train[cols].values, i) for i in range(len(cols))]
            max_vif = max(vif)
            if max_vif > thresh:
                max_idx = vif.index(max_vif)
                print(f"正在刪除 {cols[max_idx]}，其 VIF 為 {max_vif}")
                del cols[max_idx]
            else:
                break
        return x_train[cols]

        # 套用在訓練集上
        df_train_reduced = reduce_vif(X_train)

    def get_ft_count(self, X_train_scaled, y_train):
        from sklearn.feature_selection import RFECV

        # 自動幫你找最佳特徵數量
        rfecv = RFECV(estimator=RandomForestRegressor(), step=1, cv=5)
        rfecv.fit(X_train_scaled, y_train)

        print(f"最理想的特徵數量是: {rfecv.n_features_}")
        return rfecv.n_features_


def check():
    fc = FeatureCheck()
    # df = parquet_db.query_price(parquet_db.query_stocks_ids_list(), "2024-01-01", "2099-01-01")
    stocks = random.sample(parquet_db.query_stocks_ids_list(), 10)
    # stocks = ["2355"]
    print(stocks)
    df = parquet_db.query_price(stocks, "2024-01-01", "2099-01-01")
    df = feature_builder.gen_feature(df)
    import pandas as pd

    df = profit_label(df)
    df = FilterData.get_data(df, True, False)
    # parquet_db.create_features(df)
    # 資料在這裡刪
    df.set_index(["date", "stock_id"], inplace=True)
    xtrain, xtest, ytrain, ytest = rfc.split_date(df, 0.8)

    xtrain, ytrain = BaseTrainConfig().drop_na_inf(xtrain, ytrain)
    xtest, ytest = BaseTrainConfig().drop_na_inf(xtest, ytest)

    print(xtrain.select_dtypes(include="number").describe().T.round(2))
    xtrain: pd.DataFrame = fc.get_non_dup_ft_VIF(xtrain)
    print(xtrain.columns)
    fc.find_import_by_RFECV(xtrain, ytrain)


# check()
