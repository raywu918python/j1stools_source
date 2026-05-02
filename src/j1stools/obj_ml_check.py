from idna import check_label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from j1stools.obj_label import Label


class MlCheck:
    def check_label(df: pd.DataFrame, label_name="label", is_print=True):
        # 使用範例

        label_counts = df[label_name].value_counts()
        label_percentages = df[label_name].value_counts(normalize=True) * 100

        if is_print:
            print("標籤數量統計：")
            print(label_counts)
            print("\n標籤比例統計 (%)：")
            print(label_percentages)

        def chat(df):
            import matplotlib.pyplot as plt

            df["label"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
            plt.title("Label Distribution (0: Fail, 1: Success)")
            plt.xlabel("Target")
            plt.ylabel("Count")
            plt.show()

        # return label_counts[1] / (label_counts[1] + label_counts[0])
        return 0 if label_percentages.size == 1 else label_percentages[1]

    def check_feature(df: pd.DataFrame):
        sns.set(style="whitegrid")

        # 1. 单变量分布 - 了解每个特征自身的分布情况
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 创建2x2的画布
        features = df.columns

        for i, (ax, feature) in enumerate(zip(axes.flat, features)):
            # 绘制直方图（分布）与核密度估计曲线
            sns.histplot(df[feature], kde=True, ax=ax, bins=20)
            ax.set_title(f"{feature} 的分布", fontsize=14)
            ax.set_xlabel(feature)
            ax.set_ylabel("频数")

        plt.tight_layout()
        plt.show()

        # 2. 箱线图 - 查看数据分布与异常值（更直观）
        plt.figure(figsize=(10, 6))
        # 选择数值列绘制箱线图
        df_box = df.drop(columns=["species"])  # 假设'species'是文本标签列，先去掉
        sns.boxplot(data=df_box)
        plt.title("各数值特征的箱线图（查看分布与异常值）", fontsize=14)
        plt.xticks(rotation=45)
        plt.show()

        # 3. 变量间关系 - 散点图矩阵
        print("\n绘制特征间关系的散点图矩阵...（这能帮助我们发现特征之间的关联）")
        # 使用Seaborn的pairplot， hue参数可以根据类别着色（如鸢尾花的品种）
        sns.pairplot(df, hue="species", height=2.5)
        plt.suptitle("特征关系散点图矩阵（按种类着色）", y=1.02, fontsize=16)
        plt.show()

        # 4. 相关性热力图 - 量化特征间的线性关系
        plt.figure(figsize=(8, 6))
        # 计算数值特征之间的相关系数
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, square=True)
        plt.title("特征相关性热力图", fontsize=14)
        plt.show()

    # def check_dup_feature(df: pd.DataFrame):
    #     corr_matrix = df.corr()
    #     print(corr_matrix)

    def check_dup_feature_bak(df: pd.DataFrame):
        # 1. 相關係數矩陣 (Correlation Matrix)
        # 這是最簡單的方法。計算所有特徵兩兩之間的相關係數（Pearson Correlation）。
        # 判斷標準： 如果兩個特徵的相關係數 > 0.9（或很嚴格的話 > 0.85），就代表它們「太像了」，建議二選一。

        import seaborn as sns
        import matplotlib.pyplot as plt

        # 假設你的特徵都在 features_df 裡面
        corr_matrix = df.corr()

        # 畫成熱圖 (Heatmap)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="RdBu", fmt=".2f")
        plt.show()

        # 2. 變異數膨脹因子 (VIF, Variance Inflation Factor)
        # 如果你想更嚴謹地檢查「某個特徵是否能被其他特徵組合出來」，可以用 VIF。
        # 判斷標準： VIF > 5 代表有中度相關，> 10 代表嚴重共線性，必須刪除。
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # 準備特徵數據（不含標籤）
        X = df.dropna()

        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

        print(vif_data.sort_values(by="VIF", ascending=False))

    def test(chk=1):
        pass
        # if chk == 1:
        #     df = stock("2330")
        #     Label.add_label(df)
        #     ch = MlCheck.check_label(df, is_print=True)
        #     print(ch)
        # elif chk == 2:
        #     df = stock("2330")
        #     print(df.columns)
        #     MlCheck.check_dup_feature1(df)


# JS_ml_check.test(2)
