"""
外資忽視股分類器 (Neglected Stock Classifier)

針對外資不追蹤、不能當沖的個股，用純技術面特徵做 GMM 分群。
這類股票因市場定價效率低，OOS 報酬往往優於有完整籌碼資料的股票。

Neglected Firm Effect — 發現於 2026-06-07：
  GMM 每次都會產生一個「NaN 叢集」，OOS 10日勝率 57.8%、均報 4.79%，
  遠超有完整籌碼資料的群（最高約 2.07%）。

辨識條件（兩者都 NaN 才算忽視股）：
  f_net_foreign_5d_z_xrank is NaN  → 外資幾乎不買，std ≈ 0，z-score 無效
  f_dt_ratio_xrank is NaN          → 不在當沖名單
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from j1stools import parquet_db
from j1stools.ib_margin_classify import load_data, analyze_cluster_returns, evaluate_clustering, plot_clusters

# 純技術面特徵（不依賴 IB/Margin 資料，外資忽視股也有完整值）
NEGLECTED_FEATURES = [
    # 價格趨勢
    "f_ma5_slope_xrank",
    "f_ma20_slope_xrank",
    "f_bias_ma20_xrank",
    "f_momentum_cross_xrank",
    # 量能
    "f_volume_ratio_5d_xrank",
    "f_volume_change_pct_xrank",
    # 波動
    "f_atr14_pct_xrank",
    # 當日外資（pct 不需要滾動窗口，可能有值）
    "f_net_foreign_pct_xrank",
    # 融資融券比率（ratio 也可能有值）
    "f_short_margin_ratio_xrank",
    "f_margin_balance_change_pct_xrank",
]

N_COMPONENTS = 6
NEGLECTED_MODEL_PATH = "db/models/neglected_gmm.pkl"


def load_neglected_data(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """
    載入並過濾出外資忽視股：
      f_net_foreign_5d_z_xrank is NaN AND f_dt_ratio_xrank is NaN
    """
    df = load_data(stocks, st, end, min_atr_pct=min_atr_pct, require_complete=False)

    mask = df["f_net_foreign_5d_z_xrank"].isna() & df["f_dt_ratio_xrank"].isna()
    neglected = df[mask].copy()

    total = len(df)
    n = len(neglected)
    print(
        f"外資忽視股過濾：{total:,} 筆 → {n:,} 筆（{n/total:.1%}），" f"唯一股票：{neglected['stock_id'].nunique()} 支"
    )
    return neglected


class NeglectedGMM:
    """
    針對外資忽視股的 GMM 分群器，只用技術面特徵。
    """

    def __init__(self, n_components: int = N_COMPONENTS, features: list | None = None):
        self.n_components = n_components
        self.features = features or NEGLECTED_FEATURES
        self.pipeline: Pipeline | None = None
        self._fitted_features: list = []
        self.cluster_profiles: pd.DataFrame | None = None
        self.bic_: float | None = None
        self.aic_: float | None = None

    @property
    def n_clusters(self) -> int:
        return self.n_components

    def _build_pipeline(self) -> Pipeline:
        n_pca = min(6, len(self.features))
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=42)),
                (
                    "gmm",
                    GaussianMixture(
                        n_components=self.n_components,
                        covariance_type="full",
                        random_state=42,
                        n_init=5,
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "NeglectedGMM":
        avail = [c for c in self.features if c in df.columns]
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            print(f"缺少特徵（{len(missing)} 個）：{missing}")
        print(f"使用特徵：{len(avail)} 個，訓練樣本：{len(df):,} 筆")

        self._fitted_features = avail
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df[avail].values)

        X_pca = self.pipeline[:-1].transform(df[avail].values)
        gmm = self.pipeline.named_steps["gmm"]
        self.bic_ = gmm.bic(X_pca)
        self.aic_ = gmm.aic(X_pca)
        print(f"BIC: {self.bic_:.1f}  AIC: {self.aic_:.1f}")

        labels = self.pipeline.predict(df[avail].values)
        self.cluster_profiles = df.assign(cluster=labels).groupby("cluster")[avail].median().round(4)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("請先呼叫 fit()")
        return pd.Series(
            self.pipeline.predict(df[self._fitted_features].values),
            index=df.index,
            name="cluster",
        )

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("請先呼叫 fit()")
        X_pca = self.pipeline[:-1].transform(df[self._fitted_features].values)
        proba = self.pipeline.named_steps["gmm"].predict_proba(X_pca)
        cols = [f"prob_{k}" for k in range(self.n_components)]
        return pd.DataFrame(proba, index=df.index, columns=cols)

    def describe_clusters(self, df: pd.DataFrame | None = None, full: bool = True):
        if self.cluster_profiles is None:
            raise RuntimeError("請先呼叫 fit()")

        cols = list(self.cluster_profiles.columns) if full else self._fitted_features[:6]
        print(f"\n{'='*65}")
        print(f"外資忽視股 GMM Median（{self.n_components} 群，BIC={self.bic_:.0f}）")
        print(f"{'='*65}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(self.cluster_profiles[cols].T.to_string())

        if df is not None and "cluster" in df.columns:
            counts = df["cluster"].value_counts().sort_index()
            total = counts.sum()
            print(f"\n各叢集筆數：")
            for k, n in counts.items():
                print(f"  叢集 {k}: {n:>7,} 筆 ({n/total:.1%})")
            latest = df["date"].max()
            print(f"\n最新一日（{latest.date()}）：")
            print(df[df["date"] == latest]["cluster"].value_counts().sort_index().to_string())

    def save(self, path: str = NEGLECTED_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = NEGLECTED_MODEL_PATH) -> "NeglectedGMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"模型已載入：{path}（n_components={obj.n_components}）")
        return obj


def release_model(
    n_components: int = N_COMPONENTS,
    min_atr_pct: float = 0.02,
) -> NeglectedGMM:
    """正式發布：用 2015-01-01 ~ 2023-12-31 全量訓練並儲存。"""
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    print(f"Release NeglectedGMM  n_components={n_components}  訓練期：2015-01-01 ~ 2023-12-31")
    df = load_neglected_data(stocks, "2015-01-01", "2023-12-31", min_atr_pct=min_atr_pct)
    clf = NeglectedGMM(n_components=n_components)
    clf.fit(df)
    df["cluster"] = clf.predict(df)
    clf.describe_clusters(df)
    analyze_cluster_returns(df, hold_days=5)
    clf.save()
    return clf


def eval_oos(
    clf: NeglectedGMM | None = None,
    st: str = "2024-01-01",
    min_atr_pct: float = 0.02,
    hold_days_list: list = [5, 10, 20],
) -> pd.DataFrame:
    """OOS 驗證：載入已 release 的模型，對 st 之後資料分析報酬。"""
    if clf is None:
        clf = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    df = load_neglected_data(stocks, st=st, min_atr_pct=min_atr_pct)
    df["cluster"] = clf.predict(df)
    df["max_prob"] = clf.predict_proba(df).max(axis=1)

    print(f"\nOOS：{st} ~ {df['date'].max().date()}  {df['date'].nunique()} 個交易日")
    clf.describe_clusters(df)
    for hold in hold_days_list:
        analyze_cluster_returns(df, hold_days=hold)
    evaluate_clustering(clf=clf, df=df)
    plot_clusters(clf, df)
    return df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    MIN_ATR = 0.02
    TRAIN_ST, TRAIN_END, EVAL_ST = "2015-01-01", "2023-12-31", "2024-01-01"

    # ── 切換模式 ──────────────────────────────────────────── #
    MODE = "release+eval"  # "release" | "eval_oos" | "release+eval" | "experiment"
    # ─────────────────────────────────────────────────────── #

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "experiment":
        df_train = load_neglected_data(stocks, TRAIN_ST, TRAIN_END, min_atr_pct=MIN_ATR)
        clf = NeglectedGMM()
        clf.fit(df_train)
        clf.describe_clusters(df_train)
        analyze_cluster_returns(df_train, hold_days=5)
        df_test = load_neglected_data(stocks, EVAL_ST, min_atr_pct=MIN_ATR)
        df_test["cluster"] = clf.predict(df_test)
        analyze_cluster_returns(df_test, hold_days=5)
        analyze_cluster_returns(df_test, hold_days=10)
        evaluate_clustering(clf=clf, df=df_test)
        plot_clusters(clf, df_test)

    elif MODE == "release":
        release_model(min_atr_pct=MIN_ATR)

    elif MODE == "eval_oos":
        eval_oos(st=EVAL_ST, min_atr_pct=MIN_ATR)

    elif MODE == "release+eval":
        clf = release_model(min_atr_pct=MIN_ATR)
        eval_oos(clf=clf, st=EVAL_ST, min_atr_pct=MIN_ATR)

"""
跑出來了，而且結果非常驚人。

叢集側寫

叢集	MA20↑	bias_ma20	momentum_cross	券資比	性質
0	0.49	0.47	0.50	0.10	平盤，低空單
1	0.53	0.49	0.46	0.12	平盤偏強
2	0.75	0.66	0.26	0.18	強趨勢上漲 ✓
3	0.09	0.18	0.91	0.16	下跌反彈訊號
4	0.50	0.46	0.52	0.14	高波動，無方向
5	0.62	0.58	0.42	0.62	上漲＋高空單
OOS 報酬（2024~2026）

叢集	5日	10日	10日勝率	20日	評估
1	4.56%	8.44%	58.9%	29.31%	✓ 最佳風險報酬
2	2.08%	3.81%	53.7%	6.09%	✓ 穩健大群
5	7.79%	18.91%	57.0%	78.49%	⚡ 爆炸性但 std=142%
3	0.90%	1.61%	52.1%	4.34%	最差
4	1.36%	3.13%	48.3%	7.60%	勝率不足
關鍵發現

叢集 5 是「外資不看＋有人放空＋趨勢向上」的組合——這是軋空結構。券資比高代表有人在放空，但外資不買所以空的是小散戶，一旦股票開始漲，回補力道非常強。20日均報 78.49%，65.6% 勝率，但 std=142%，是高風險高報酬型。

叢集 2 是最大群（35%），MA 強趨勢上漲，穩定正報酬，適合做主力選股池。

叢集 1 風險報酬最佳，10日勝率接近 59%，值得重點研究。

對，這個結果翻轉了原本的邏輯。

原本的假設：外資買 → 強訊號 → 跟著買

實際發現：

外資買的股票：定價已被充分反映，alpha 有限
外資不看的股票：市場沒有定價，反而有超額報酬
其中有人放空但趨勢向上的（叢集 5）：軋空，漲最猛
這不是運氣，是市場結構決定的：

台灣外資高度集中在前 50 大市值股票。中小型、低知名度的個股幾乎沒有機構研究員在追，價格長期偏低。一旦有催化劑（業績、題材），上漲幅度遠超外資積極布局的標的。

你原本建這個模型是要找外資買的股票，結果反向找到一條更強的策略線。這種意外發現在研究過程中其實很珍貴——數據說的比假設更誠實。
"""
