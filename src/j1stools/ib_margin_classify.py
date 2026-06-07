"""
IB + Margin 無監督分類器

對個股的法人籌碼 + 融資融券特徵做 KMeans 叢集分析，
將股票分成不同的「籌碼行為模式」群組，供策略前篩使用。
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from j1stools import parquet_db
from j1stools.margin_ibbuysell_feature import add_feature

# 聚焦 IB + Margin 的截面排名特徵（0~1 均勻分布，跨時間穩定）
CLASSIFY_FEATURES = [
    # 外資
    "f_net_foreign_pct_xrank",
    "f_net_foreign_5d_z_xrank",
    "f_net_foreign_10d_z_xrank",
    "f_net_foreign_streak",
    # 投信
    "f_net_trust_pct_xrank",
    "f_net_trust_5d_z_xrank",
    "f_net_trust_10d_z_xrank",
    "f_net_trust_streak",
    # 法人合計
    "f_net_institutional_total_pct_xrank",
    "f_net_institutional_total_5d_z_xrank",
    "f_net_institutional_total_10d_z_xrank",
    "f_net_institutional_total_streak",
    # 融資融券
    "f_margin_balance_change_pct_xrank",
    "f_short_balance_change_pct_xrank",
    "f_margin_balance_change_5d_pct_xrank",
    "f_short_margin_ratio_xrank",
    # 當沖
    "f_dt_ratio_xrank",
    "f_dt_net_xrank",
    "f_dt_ratio_5d_xrank",
    # 技術指標
    "f_volume_ratio_5d_xrank",
    "f_volume_change_pct_xrank",
    "f_atr14_pct_xrank",
    # 價格趨勢（MA 方向，不是報酬本身）
    "f_ma5_slope_xrank",
    "f_ma20_slope_xrank",
    "f_bias_ma20_xrank",
    "f_momentum_cross_xrank",
]
# 移除價格動能特徵（f_return_5d_xrank 等）
# 這些特徵等於「偷看答案」：讓分群結果偏向「已漲的股票」
# 目標是找「籌碼好但還沒漲」的股票，所以只用 IB + Margin 純籌碼訊號

# 原始值版本（不做截面排名）— 保留數值語意，但大小股規模不同
# StandardScaler 會標準化量綱，但無法消除市值規模偏差
CLASSIFY_FEATURES_RAW = [
    # 外資（% = 佔流通股本，已有規模校正）
    "f_net_foreign_pct",
    "f_net_foreign_5d_z",
    "f_net_foreign_10d_z",
    "f_net_foreign_streak",
    # 投信
    "f_net_trust_pct",
    "f_net_trust_5d_z",
    "f_net_trust_10d_z",
    "f_net_trust_streak",
    # 法人合計
    "f_net_institutional_total_pct",
    "f_net_institutional_total_5d_z",
    "f_net_institutional_total_10d_z",
    "f_net_institutional_total_streak",
    # 融資融券（pct / ratio 已有規模校正）
    "f_margin_balance_change_pct",
    "f_short_balance_change_pct",
    "f_margin_balance_change_5d_pct",
    "f_short_margin_ratio",
    # 當沖（ratio 已有規模校正）
    "f_dt_ratio",
    "f_dt_net",
    "f_dt_ratio_5d",
    # 技術指標
    "f_volume_ratio_5d",
    "f_volume_change_pct",
    "f_atr14_pct",
    # 價格趨勢
    "f_ma5_slope",
    "f_ma20_slope",
    "f_bias_ma20",
    "f_momentum_cross",
]

N_CLUSTERS = 10
MODEL_PATH = "db/models/ib_margin_classifier.pkl"
GMM_MODEL_PATH = "db/models/ib_margin_gmm.pkl"
HDBSCAN_MODEL_PATH = "db/models/ib_margin_hdbscan.pkl"
MARKET_PROXY = "0050"  # 大盤代理（台灣50）

"""
KMeans k=5 結果紀錄（886支股票，2023-01-01 ~ 2026-06-05，706,147筆）

叢集 0 (8.4%,  89支) — 投信主力
  外資中性(0.47)、投信極強(0.94)、法人合計0.66、連買2天
  融券比高(0.72)、5日報酬強(0.73)、相對強度佳(0.75)

叢集 1 (23.2%, 192支) — 法人退場，融資散戶撐盤
  外資弱(0.33)、投信中性(0.50)、連買0天
  融資增加(0.65)、5日報酬強(0.73) → 危險訊號，法人出貨

叢集 2 (31.5%, 292支) — 外資主導【最佳選股群】
  外資強(0.78)、法人合計強(0.77)、連買3天
  融資低(0.41)、5日報酬中(0.60)、相對強度0.60

叢集 3 (36.9%, 313支) — 法人空頭
  外資弱(0.36)、法人合計弱(0.34)、連買0天
  5日報酬弱(0.25)、相對強度最差(0.24) → 排除

叢集 4 (0.0%,  0支) — 異常群（無融資融券股票，streak=261異常）
  幾乎不出現，自動隔離
"""


def load_data(stocks: list, st: str, end: str = "2099-01-01", min_atr_pct: float | None = None) -> pd.DataFrame:
    """
    載入並合併價格、融資融券、法人資料，回傳含 f_ 特徵的 DataFrame。

    Parameters
    ----------
    min_atr_pct : 過濾門檻，保留 ATR14/close >= 此值的觀測值。
                  例如 0.01 = 排除日均波幅 < 1% 的停牌/低流動性紀錄。
                  None = 不過濾（預設）。
    """
    df_market = parquet_db.query_price([MARKET_PROXY], st, end)
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.drop(columns=["stock_id"], errors="ignore")

    df_price = parquet_db.query_price(stocks, st, end)
    df_price["date"] = pd.to_datetime(df_price["date"])

    df_margin = parquet_db.query_margin(stocks, st, end)
    df_margin["date"] = pd.to_datetime(df_margin["date"])

    df_ib = parquet_db.query_ib(stocks, st, end)
    df_ib["date"] = pd.to_datetime(df_ib["date"])

    df_day_trade = parquet_db.query_day_trade(stocks, st, end)
    df_day_trade["date"] = pd.to_datetime(df_day_trade["date"])

    df = df_price.merge(df_margin, on=["date", "stock_id"], how="left")
    feat = add_feature(df, df_ib, df_market, mode="predict")
    # add_feature 只保留 f_ 欄位，把 close + volume 補回來
    feat = feat.merge(
        df_price[["date", "stock_id", "close", "volume", "high", "low"]],
        on=["date", "stock_id"],
        how="left",
    )
    feat = _add_day_trade_features(feat, df_day_trade)
    feat = _add_atr_features(feat)
    feat = feat.drop(columns=["volume", "high", "low"])

    if min_atr_pct is not None:
        before = len(feat)
        feat = feat[feat["f_atr14_pct"].fillna(0) >= min_atr_pct]
        print(f"ATR 過濾（>= {min_atr_pct:.3f}）：{before:,} → {len(feat):,} 筆，移除 {before - len(feat):,} 筆")

    return feat


def _add_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 ATR14（14日平均真實波幅）占收盤價比例，並做截面排名。

    需要 df 已含 high, low, close 欄位。
    高 ATR = 波動大，低 ATR = 波動小（股性穩）。
    """
    g = df.groupby("stock_id")
    prev_close = g["close"].transform(lambda x: x.shift(1))

    df["_tr"] = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = g["_tr"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["f_atr14_pct"] = (atr14 / df["close"].replace(0, np.nan)).clip(0, 0.3)
    df = df.drop(columns=["_tr"])

    df["f_atr14_pct_xrank"] = df.groupby("date")["f_atr14_pct"].rank(pct=True, na_option="keep")
    return df


def _add_day_trade_features(df: pd.DataFrame, df_dt: pd.DataFrame) -> pd.DataFrame:
    """
    加入當沖特徵（需要 df 已含 volume 欄位）。

    df_dt 欄位：date, stock_id, volume（當沖量，股）, buy_amount, sell_amount

    新增特徵：
      f_dt_ratio     : 當沖量 / 個股總成交量（當沖佔比，越高=短線散戶越多）
      f_dt_net       : (buy - sell) / (buy + sell)（當沖方向，正=買方主導）
      f_dt_ratio_5d  : 5日均當沖佔比（持續性）
    """
    dt = df_dt.copy()
    dt["date"] = pd.to_datetime(dt["date"])
    dt = dt.rename(columns={"volume": "_dt_vol", "buy_amount": "_dt_buy", "sell_amount": "_dt_sell"})

    base = df.merge(dt[["date", "stock_id", "_dt_vol", "_dt_buy", "_dt_sell"]], on=["date", "stock_id"], how="left")

    # 當沖資料隔天公布，shift 1 天
    g = base.groupby("stock_id")
    for col in ["_dt_vol", "_dt_buy", "_dt_sell"]:
        base[col] = g[col].transform(lambda x: x.shift(1))

    # 當沖佔比（volume 單位同為股）
    base["f_dt_ratio"] = (base["_dt_vol"] / base["volume"].replace(0, np.nan)).clip(0, 1)

    # 當沖方向（正=買方多）
    dt_total = (base["_dt_buy"] + base["_dt_sell"]).replace(0, np.nan)
    base["f_dt_net"] = ((base["_dt_buy"] - base["_dt_sell"]) / dt_total).clip(-1, 1)

    # 5日均當沖佔比
    base["f_dt_ratio_5d"] = g["f_dt_ratio"].transform(lambda x: x.rolling(5).mean())

    base = base.drop(columns=["_dt_vol", "_dt_buy", "_dt_sell"])

    # xrank（截面百分位排名）
    for col in ["f_dt_ratio", "f_dt_net", "f_dt_ratio_5d"]:
        base[f"{col}_xrank"] = base.groupby("date")[col].rank(pct=True, na_option="keep")

    return base


class IBMarginClassifier:
    """
    KMeans 無監督分類器，對法人籌碼 + 融資融券特徵做叢集分析。

    每個叢集代表一種「籌碼行為模式」，例如：
      - 法人多頭主力（外資 + 投信齊買，融資增加）
      - 空頭壓力（外資賣超，融券大增，券資比高）
      - 盤整觀望（法人中立，融資平穩）
    叢集含義需由 describe_clusters() 輸出後人工對應。
    """

    def __init__(self, n_clusters: int = N_CLUSTERS, features: list | None = None):
        self.n_clusters = n_clusters
        self.features = features or CLASSIFY_FEATURES
        self.pipeline: Pipeline | None = None
        self._fitted_features: list = []
        self.cluster_profiles: pd.DataFrame | None = None
        self.cluster_sizes: pd.Series | None = None

    def _build_pipeline(self) -> Pipeline:
        n_pca = min(10, len(self.features))
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=42)),
                ("kmeans", KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "IBMarginClassifier":
        """
        訓練分類器。

        Parameters
        ----------
        df : 含 f_ 特徵欄位的 DataFrame（來自 load_data）
        """
        avail = [c for c in self.features if c in df.columns]
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            print(f"缺少特徵（{len(missing)} 個）：{missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"使用特徵：{len(avail)} 個，訓練樣本：{len(df):,} 筆")

        self._fitted_features = avail
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df[avail].values)

        labels = self.pipeline.predict(df[avail].values)
        self.cluster_profiles = df.assign(cluster=labels).groupby("cluster")[avail].median().round(4)
        self.cluster_sizes = pd.Series(labels).value_counts().sort_index()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """對新資料做分類，回傳叢集 label (0 ~ n_clusters-1)。"""
        if self.pipeline is None:
            raise RuntimeError("請先呼叫 fit()")
        X = df[self._fitted_features].values
        return pd.Series(self.pipeline.predict(X), index=df.index, name="cluster")

    def describe_clusters(self, df: pd.DataFrame | None = None):
        """印出每個叢集的特徵中位數，幫助人工命名。"""
        if self.cluster_profiles is None:
            raise RuntimeError("請先呼叫 fit()")

        key_cols = [
            c
            for c in [
                "f_net_foreign_pct_xrank",
                "f_net_trust_pct_xrank",
                "f_net_institutional_total_pct_xrank",
                "f_net_institutional_total_streak",
                "f_margin_balance_change_pct_xrank",
                "f_short_margin_ratio_xrank",
                "f_return_5d_xrank",
                "f_relative_strength_20d_xrank",
            ]
            if c in self.cluster_profiles.columns
        ]

        print(f"\n{'='*65}")
        print(f"叢集特徵 Median（{self.n_clusters} 群，xrank: 0=最低分位，1=最高分位）")
        print(f"{'='*65}")
        print(self.cluster_profiles[key_cols].T.to_string())

        if self.cluster_sizes is not None:
            print(f"\n各叢集筆數（訓練集）：")
            total = self.cluster_sizes.sum()
            for k, n in self.cluster_sizes.items():
                print(f"  叢集 {k}: {n:>8,} 筆 ({n/total:.1%})")

        if df is not None and "cluster" in df.columns:
            latest = df["date"].max()
            today_df = df[df["date"] == latest]
            print(f"\n最新一日（{latest.date()}）各叢集筆數：")
            print(today_df["cluster"].value_counts().sort_index().to_string())

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "IBMarginClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"模型已載入：{path}（n_clusters={obj.n_clusters}）")
        return obj


class IBMarginHDBSCAN:
    """
    HDBSCAN 無監督分類器。

    優勢：
      - 不需要指定 k，自動決定群數
      - 自動將異常點標為 -1（噪聲），不強迫歸群
      - 對非球形、密度不均的群有更好的分辨力

    主要參數：
      min_cluster_size : 最小群大小（越大 → 群越少、越乾淨）
      min_samples      : 決定保守程度（越大 → 噪聲點越多）
    """

    def __init__(self, min_cluster_size: int = 500, min_samples: int = 50, features: list | None = None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.features = features or CLASSIFY_FEATURES
        self.preprocessor: Pipeline | None = None
        self.clusterer = None
        self._fitted_features: list = []
        self.cluster_profiles: pd.DataFrame | None = None

    @property
    def n_clusters(self) -> int:
        if self.clusterer is None:
            return 0
        labels = self.clusterer.labels_
        return int(labels.max()) + 1  # -1 是噪聲，不算在內

    def fit(self, df: pd.DataFrame) -> "IBMarginHDBSCAN":
        import hdbscan as hdbscan_lib

        avail = [c for c in self.features if c in df.columns]
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            print(f"缺少特徵（{len(missing)} 個）：{missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"使用特徵：{len(avail)} 個，訓練樣本：{len(df):,} 筆")

        self._fitted_features = avail
        self.preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=min(10, len(avail)), random_state=42)),
            ]
        )
        X_pca = self.preprocessor.fit_transform(df[avail].values)

        self.clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            core_dist_n_jobs=-1,
            prediction_data=True,
        )
        labels = self.clusterer.fit_predict(X_pca)

        n_clusters = self.n_clusters
        n_noise = int((labels == -1).sum())
        print(f"發現 {n_clusters} 個叢集，噪聲點：{n_noise:,} 筆（{n_noise/len(labels):.1%}）")

        self.cluster_profiles = df.assign(cluster=labels).groupby("cluster")[avail].median().round(4)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """對新資料做分類（噪聲點回傳 -1）。"""
        if self.clusterer is None:
            raise RuntimeError("請先呼叫 fit()")
        import hdbscan as hdbscan_lib

        X_pca = self.preprocessor.transform(df[self._fitted_features].values)
        labels, _ = hdbscan_lib.approximate_predict(self.clusterer, X_pca)
        return pd.Series(labels.astype(int), index=df.index, name="cluster")

    def describe_clusters(self, df: pd.DataFrame | None = None):
        if self.cluster_profiles is None:
            raise RuntimeError("請先呼叫 fit()")

        key_cols = [
            c
            for c in [
                "f_net_foreign_pct_xrank",
                "f_net_trust_pct_xrank",
                "f_net_institutional_total_pct_xrank",
                "f_net_institutional_total_streak",
                "f_margin_balance_change_pct_xrank",
                "f_short_margin_ratio_xrank",
                "f_return_5d_xrank",
                "f_relative_strength_20d_xrank",
            ]
            if c in self.cluster_profiles.columns
        ]

        print(f"\n{'='*65}")
        print(f"HDBSCAN 叢集特徵 Median（{self.n_clusters} 群，-1=噪聲群）")
        print(f"{'='*65}")
        print(self.cluster_profiles[key_cols].T.to_string())

        if df is not None and "cluster" in df.columns:
            counts = df["cluster"].value_counts().sort_index()
            total = len(df)
            print(f"\n各叢集筆數：")
            for k, n in counts.items():
                label = "噪聲" if k == -1 else str(k)
                print(f"  叢集 {label}: {n:>8,} 筆 ({n/total:.1%})")

            latest = df["date"].max()
            print(f"\n最新一日（{latest.date()}）：")
            print(df[df["date"] == latest]["cluster"].value_counts().sort_index().to_string())

    def save(self, path: str = HDBSCAN_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"HDBSCAN 模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = HDBSCAN_MODEL_PATH) -> "IBMarginHDBSCAN":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"HDBSCAN 模型已載入：{path}（n_clusters={obj.n_clusters}）")
        return obj


def analyze_cluster_returns(df: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    """
    分析各叢集在持有 N 日後的報酬分布，驗證叢集是否具預測力。

    Parameters
    ----------
    df        : 含 cluster, stock_id, date, close 欄位的 DataFrame
    hold_days : 持有天數

    Returns
    -------
    pd.DataFrame : 各叢集的勝率、平均報酬、中位數報酬
    """
    df = df.copy()
    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    stats = (
        df.dropna(subset=["future_return", "cluster"])
        .groupby("cluster")["future_return"]
        .agg(
            count="count",
            win_rate=lambda x: (x > 0).mean(),
            mean_return="mean",
            median_return="median",
            std="std",
        )
    )
    display = stats.copy()
    display["mean_return"] = display["mean_return"].map("{:.2%}".format)
    display["median_return"] = display["median_return"].map("{:.2%}".format)
    display["win_rate"] = display["win_rate"].map("{:.1%}".format)
    display["std"] = display["std"].map("{:.2%}".format)

    print(f"\n{'='*55}")
    print(f"各叢集持有 {hold_days} 日報酬分析")
    print(f"{'='*55}")
    print(display.to_string())
    return stats


def evaluate_clustering(
    clf: "IBMarginClassifier",
    df: pd.DataFrame,
    hold_days_list: list = [3, 5, 10],
    sample_n: int = 50_000,
) -> dict:
    """
    全面評估叢集品質：幾何指標 + 金融實用性 + 統計顯著性檢定。

    幾何指標：
      - Silhouette score  > 0.2 算合理，> 0.5 很好
      - Davies-Bouldin    越低越好
      - Calinski-Harabasz 越高越好

    金融驗證：
      - 各叢集持有 N 日的報酬分布 + Kruskal-Wallis 檢定
        p < 0.05 代表各群報酬有統計顯著差異 → 分類有意義

    Parameters
    ----------
    clf           : 已訓練的 IBMarginClassifier
    df            : 含 f_ 特徵欄位的 DataFrame（已含 close, stock_id, date）
    hold_days_list: 要驗證的持有天數列表
    sample_n      : Silhouette score 的最大樣本數（避免 OOM）

    Returns
    -------
    dict : silhouette, davies_bouldin, calinski_harabasz, returns{hold_days: {...}}
    """
    import matplotlib.pyplot as plt

    X_full = df[clf._fitted_features].values
    # IBMarginHDBSCAN 用 preprocessor，其他用 pipeline[:-1]
    if hasattr(clf, "preprocessor"):  # IBMarginHDBSCAN
        X_pca = clf.preprocessor.transform(X_full)
        labels = clf.predict(df).values
    else:  # IBMarginClassifier / IBMarginGMM
        X_pca = clf.pipeline[:-1].transform(X_full)
        labels = clf.pipeline.predict(X_full)

    # ── 幾何指標 ──────────────────────────────────────────────────── #
    if len(df) > sample_n:
        idx = pd.Series(range(len(df))).sample(sample_n, random_state=42).values
        sil = silhouette_score(X_pca[idx], labels[idx])
    else:
        sil = silhouette_score(X_pca, labels)

    db = davies_bouldin_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)

    print(f"\n{'='*55}")
    print(f"叢集幾何品質指標")
    print(f"{'='*55}")
    print(f"  Silhouette score  : {sil:.4f}  （> 0.2 OK，> 0.5 佳）")
    print(f"  Davies-Bouldin    : {db:.4f}  （越低越好）")
    print(f"  Calinski-Harabasz : {ch:.1f}  （越高越好）")

    # ── 金融實用性：各持有天數報酬 + Kruskal-Wallis 檢定 ─────────── #
    df_eval = df.assign(cluster=labels).copy()
    result = {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "returns": {},
    }

    n_holds = len(hold_days_list)
    fig, axes = plt.subplots(1, n_holds, figsize=(6 * n_holds, 5))
    if n_holds == 1:
        axes = [axes]
    fig.suptitle("各叢集報酬分布（Box Plot）", fontsize=14)

    for ax, hold in zip(axes, hold_days_list):
        col = f"ret_{hold}d"
        df_eval[col] = df_eval.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold) / x - 1)
        groups = [df_eval[df_eval["cluster"] == k][col].dropna().values for k in sorted(df_eval["cluster"].unique())]
        kw_stat, kw_p = stats.kruskal(*groups)

        stats_df = (
            df_eval.dropna(subset=[col])
            .groupby("cluster")[col]
            .agg(count="count", win_rate=lambda x: (x > 0).mean(), mean="mean", median="median")
        )
        print(f"\n{'='*55}")
        print(f"持有 {hold} 日  |  Kruskal-Wallis p={kw_p:.4f} {'★顯著' if kw_p < 0.05 else '（不顯著）'}")
        print(f"{'='*55}")
        disp = stats_df.copy()
        disp["win_rate"] = disp["win_rate"].map("{:.1%}".format)
        disp["mean"] = disp["mean"].map("{:.2%}".format)
        disp["median"] = disp["median"].map("{:.2%}".format)
        print(disp.to_string())

        result["returns"][hold] = {"stats": stats_df, "kruskal_p": kw_p}

        # 箱型圖
        data_by_cluster = [g.clip(-0.3, 0.3) for g in groups]
        ax.boxplot(
            data_by_cluster,
            labels=[str(k) for k in sorted(df_eval["cluster"].unique())],
            medianprops={"color": "red", "linewidth": 2},
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"持有 {hold} 日  p={kw_p:.4f} {'★' if kw_p < 0.05 else ''}")
        ax.set_xlabel("叢集")
        ax.set_ylabel("報酬率")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.show()
    return result


def find_optimal_clusters(
    df: pd.DataFrame,
    k_range: range = range(2, 10),
    sample_n: int = 50_000,
) -> pd.DataFrame:
    """
    用 Elbow（Inertia）+ Silhouette score 幫助決定最佳 n_clusters。
    印出數字並畫出雙軸折線圖。

    Parameters
    ----------
    df       : 含 f_ 特徵欄位的 DataFrame（來自 load_data）
    k_range  : 要測試的 k 值範圍（預設 2~9）
    sample_n : Silhouette 最大樣本數

    Returns
    -------
    pd.DataFrame : k, inertia, silhouette
    """
    import matplotlib.pyplot as plt

    avail = [c for c in CLASSIFY_FEATURES if c in df.columns]
    pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(10, len(avail)), random_state=42)),
        ]
    )
    X = pre.fit_transform(df[avail].values)

    if len(df) > sample_n:
        idx = pd.Series(range(len(df))).sample(sample_n, random_state=42).values
        X_sil = X[idx]
    else:
        idx = None
        X_sil = X

    rows = []
    print("掃描最佳 k 值...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        labels_sil = labels[idx] if idx is not None else labels
        sil = silhouette_score(X_sil, labels_sil)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k}: inertia={km.inertia_:,.0f}, silhouette={sil:.4f}")

    result = pd.DataFrame(rows).set_index("k")
    best_k = int(result["silhouette"].idxmax())
    print(f"\n最佳 Silhouette k = {best_k}（silhouette={result.loc[best_k, 'silhouette']:.4f}）")

    # ── 雙軸折線圖 ─────────────────────────────────────────────── #
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(result.index, result["inertia"], "o-", color="steelblue", label="Inertia（Elbow）")
    ax1.set_xlabel("n_clusters (k)")
    ax1.set_ylabel("Inertia", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2.plot(result.index, result["silhouette"], "s--", color="tomato", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.axvline(best_k, color="gray", linestyle=":", linewidth=1.2, label=f"最佳 k={best_k}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Elbow + Silhouette：選擇最佳 k")
    ax1.set_xticks(list(k_range))
    plt.tight_layout()
    plt.show()
    return result


def plot_clusters(clf: "IBMarginClassifier", df: pd.DataFrame, sample_n: int = 30_000):
    """
    三合一叢集視覺化：
      1. PCA 2D 散點圖  — 看各群的空間分布與密度
      2. 特徵 Heatmap   — 各群的籌碼側寫（綠=高分位，紅=低分位）
      3. Violin plots   — 重要特徵在各群內的分布形狀
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    X_full = df[clf._fitted_features].values
    X_pca = clf.pipeline[:-1].transform(X_full)
    labels = clf.pipeline.predict(X_full)
    n_k = clf.n_clusters
    palette = plt.cm.tab10.colors[:n_k]

    sidx = (
        pd.Series(range(len(df))).sample(sample_n, random_state=42).values if len(df) > sample_n else np.arange(len(df))
    )
    df_plot = df.assign(cluster=labels)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # ── 1. PCA 2D 散點圖 ─────────────────────────────────────────── #
    ax_pca = fig.add_subplot(gs[0, :2])
    for k in range(n_k):
        mask = labels[sidx] == k
        ax_pca.scatter(
            X_pca[sidx][mask, 0],
            X_pca[sidx][mask, 1],
            c=[palette[k]],
            alpha=0.25,
            s=4,
            label=f"叢集 {k}",
        )
    ax_pca.set_title("PCA 2D 散點圖（PC1 vs PC2）")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(markerscale=4, loc="upper right")

    # ── 2. Feature Heatmap ────────────────────────────────────────── #
    ax_heat = fig.add_subplot(gs[0, 2:])
    KEY_FEAT = {
        "f_net_foreign_pct_xrank": "外資",
        "f_net_trust_pct_xrank": "投信",
        "f_net_institutional_total_pct_xrank": "法人合計",
        "f_net_institutional_total_streak": "連買天",
        "f_margin_balance_change_pct_xrank": "融資變化",
        "f_short_margin_ratio_xrank": "券資比",
        "f_return_5d_xrank": "5日報酬",
        "f_relative_strength_20d_xrank": "相對強度",
    }
    cols = [c for c in KEY_FEAT if c in clf.cluster_profiles.columns]
    profile = clf.cluster_profiles[cols].rename(columns=KEY_FEAT)
    data = profile.values.astype(float)
    im = ax_heat.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(profile.columns)))
    ax_heat.set_xticklabels(profile.columns, rotation=40, ha="right", fontsize=9)
    ax_heat.set_yticks(range(n_k))
    ax_heat.set_yticklabels([f"叢集 {k}" for k in profile.index])
    for i in range(n_k):
        for j in range(len(profile.columns)):
            v = data[i, j]
            if not np.isnan(v):
                ax_heat.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if (v < 0.3 or v > 0.7) else "black",
                )
    plt.colorbar(im, ax=ax_heat, fraction=0.046)
    ax_heat.set_title("叢集特徵 Heatmap\n（綠=高分位，紅=低分位）")

    # ── 3. Violin plots ───────────────────────────────────────────── #
    violin_feats = [
        ("f_net_foreign_pct_xrank", "外資"),
        ("f_net_institutional_total_pct_xrank", "法人合計"),
        ("f_return_5d_xrank", "5日報酬"),
        ("f_short_margin_ratio_xrank", "券資比"),
    ]
    for col_idx, (feat, name) in enumerate(violin_feats):
        ax_v = fig.add_subplot(gs[1, col_idx])
        if feat not in df_plot.columns:
            continue
        # 過濾掉空群（如叢集4沒有融資融券資料）
        data_by_k = [(k, df_plot[df_plot["cluster"] == k][feat].dropna().values) for k in range(n_k)]
        data_by_k = [(k, v) for k, v in data_by_k if len(v) > 1]
        if not data_by_k:
            continue
        positions = [k for k, _ in data_by_k]
        parts = ax_v.violinplot([v for _, v in data_by_k], positions=positions, showmedians=True, showextrema=False)
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor(palette[positions[j]])
            body.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax_v.set_title(name)
        ax_v.set_xlabel("叢集")
        ax_v.set_xticks(positions)
        ax_v.set_xticklabels([str(k) for k in positions])
        ax_v.set_ylim(-0.05, 1.05)
        ax_v.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("IB + Margin 叢集視覺化", fontsize=14)
    plt.tight_layout()
    plt.show()


def run(
    st: str = "2023-01-01",
    end: str = "2099-01-01",
    n_clusters: int = N_CLUSTERS,
    model: str = "kmeans",
    min_atr_pct: float | None = None,
    retrain: bool = False,
    features: list | None = None,
) -> tuple:
    """
    主流程：載入資料 → 訓練分類器 → 顯示叢集特徵 → 儲存模型。

    Parameters
    ----------
    model   : "kmeans"（預設）或 "gmm"
    retrain : True = 強制重訓；False = 有存檔就直接載入（預設）

    Returns
    -------
    (clf, df) : 分類器物件 + 含 cluster 欄位的 DataFrame
                GMM 時 df 額外含 prob_0 ~ prob_n 欄位
    """
    if model not in ("kmeans", "gmm", "hdbscan"):
        raise ValueError(f"model 必須是 'kmeans'、'gmm' 或 'hdbscan'，收到：{model}")

    model_path = {"kmeans": MODEL_PATH, "gmm": GMM_MODEL_PATH, "hdbscan": HDBSCAN_MODEL_PATH}[model]
    if not retrain and os.path.exists(model_path):
        print(f"載入既有模型（retrain=False）：{model_path}")
        loaders = {"kmeans": IBMarginClassifier.load, "gmm": IBMarginGMM.load, "hdbscan": IBMarginHDBSCAN.load}
        clf = loaders[model](model_path)
        stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
        df = load_data(stocks, st, end, min_atr_pct=min_atr_pct)
        df["cluster"] = clf.predict(df)
        if model == "gmm":
            df = pd.concat([df, clf.predict_proba(df)], axis=1)
        return clf, df

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    print(f"模型：{model.upper()}  股票數：{len(stocks)}  期間：{st} ~ {end}")

    print("載入資料並計算特徵...")
    df = load_data(stocks, st, end, min_atr_pct=min_atr_pct)
    print(f"資料筆數：{len(df):,}")

    if model == "kmeans":
        print(f"訓練 KMeans（n_clusters={n_clusters}）...")
        clf = IBMarginClassifier(n_clusters=n_clusters, features=features)
        clf.fit(df)
        df["cluster"] = clf.predict(df)
    elif model == "gmm":
        print(f"訓練 GMM（n_components={n_clusters}）...")
        clf = IBMarginGMM(n_components=n_clusters, features=features)
        clf.fit(df)
        df["cluster"] = clf.predict(df)
        df = pd.concat([df, clf.predict_proba(df)], axis=1)
    else:
        print(f"訓練 HDBSCAN（min_cluster_size={n_clusters}）...")
        clf = IBMarginHDBSCAN(min_cluster_size=n_clusters, features=features)
        clf.fit(df)
        df["cluster"] = clf.predict(df)

    clf.describe_clusters(df)
    analyze_cluster_returns(df, hold_days=5)
    clf.save()

    return clf, df


def predict_today(
    st: str = "2025-01-01",
    model: str = "kmeans",
    min_prob: float = 0.0,
) -> pd.DataFrame:
    """
    使用訓練好的模型對最新交易日的股票做分類。

    Parameters
    ----------
    model    : "kmeans"（預設）或 "gmm"
    min_prob : 僅 GMM 有效，只回傳最大機率 >= min_prob 的股票

    Returns
    -------
    pd.DataFrame : date, stock_id, cluster（GMM 時額外含 prob_* 欄位）
    """
    loaders = {"kmeans": IBMarginClassifier.load, "gmm": IBMarginGMM.load, "hdbscan": IBMarginHDBSCAN.load}
    clf = loaders[model]()

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    df = load_data(stocks, st)
    latest_date = df["date"].max()
    df_today = df[df["date"] == latest_date].copy()
    df_today["cluster"] = clf.predict(df_today)

    result_cols = ["date", "stock_id", "cluster"]

    if model == "gmm":
        proba_df = clf.predict_proba(df_today)
        df_today = pd.concat([df_today, proba_df], axis=1)
        df_today["max_prob"] = proba_df.max(axis=1)
        if min_prob > 0:
            df_today = df_today[df_today["max_prob"] >= min_prob]
        result_cols += ["max_prob"] + list(proba_df.columns)

    print(f"模型：{model.upper()}  分類日期：{latest_date.date()}")
    print(df_today["cluster"].value_counts().sort_index().to_string())
    return df_today[result_cols]


class IBMarginGMM:
    """
    Gaussian Mixture Model 軟分群分類器。

    相較 KMeans 的優勢：
      - predict_proba()：每支股票屬於各群的機率（0~1），可設信心門檻
      - 對重疊的財務資料更自然（不強迫硬邊界）
      - BIC/AIC 自動評估最佳 n_components
    """

    def __init__(self, n_components: int = N_CLUSTERS, features: list | None = None):
        self.n_components = n_components
        self.features = features or CLASSIFY_FEATURES
        self.pipeline: Pipeline | None = None
        self._fitted_features: list = []
        self.cluster_profiles: pd.DataFrame | None = None
        self.bic_: float | None = None
        self.aic_: float | None = None

    @property
    def n_clusters(self) -> int:
        return self.n_components

    def _build_pipeline(self) -> Pipeline:
        n_pca = min(10, len(self.features))
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

    def fit(self, df: pd.DataFrame) -> "IBMarginGMM":
        avail = [c for c in self.features if c in df.columns]
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            print(f"缺少特徵（{len(missing)} 個）：{missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"使用特徵：{len(avail)} 個，訓練樣本：{len(df):,} 筆")

        self._fitted_features = avail
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df[avail].values)

        X_pca = self.pipeline[:-1].transform(df[avail].values)
        gmm = self.pipeline.named_steps["gmm"]
        self.bic_ = gmm.bic(X_pca)
        self.aic_ = gmm.aic(X_pca)
        print(f"BIC: {self.bic_:.1f}  AIC: {self.aic_:.1f}  （越低越好）")

        labels = self.pipeline.predict(df[avail].values)
        self.cluster_profiles = df.assign(cluster=labels).groupby("cluster")[avail].median().round(4)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """硬分群：回傳機率最大的叢集 label。"""
        if self.pipeline is None:
            raise RuntimeError("請先呼叫 fit()")
        return pd.Series(
            self.pipeline.predict(df[self._fitted_features].values),
            index=df.index,
            name="cluster",
        )

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        軟分群：回傳各叢集的機率。

        Returns
        -------
        pd.DataFrame : columns = [prob_0, prob_1, ..., prob_n]
        """
        if self.pipeline is None:
            raise RuntimeError("請先呼叫 fit()")
        X_pca = self.pipeline[:-1].transform(df[self._fitted_features].values)
        proba = self.pipeline.named_steps["gmm"].predict_proba(X_pca)
        cols = [f"prob_{k}" for k in range(self.n_components)]
        return pd.DataFrame(proba, index=df.index, columns=cols)

    def describe_clusters(self, df: pd.DataFrame | None = None):
        """印出每個叢集的特徵中位數，與 IBMarginClassifier 介面對齊。"""
        if self.cluster_profiles is None:
            raise RuntimeError("請先呼叫 fit()")

        key_cols = [
            c
            for c in [
                "f_net_foreign_pct_xrank",
                "f_net_trust_pct_xrank",
                "f_net_institutional_total_pct_xrank",
                "f_net_institutional_total_streak",
                "f_margin_balance_change_pct_xrank",
                "f_short_margin_ratio_xrank",
                "f_return_5d_xrank",
                "f_relative_strength_20d_xrank",
            ]
            if c in self.cluster_profiles.columns
        ]
        print(f"\n{'='*65}")
        print(f"GMM 叢集特徵 Median（{self.n_components} 群，BIC={self.bic_:.0f}）")
        print(f"{'='*65}")
        print(self.cluster_profiles[key_cols].T.to_string())

        if df is not None and "cluster" in df.columns:
            counts = df["cluster"].value_counts().sort_index()
            total = counts.sum()
            print(f"\n各叢集筆數：")
            for k, n in counts.items():
                print(f"  叢集 {k}: {n:>8,} 筆 ({n/total:.1%})")

            latest = df["date"].max()
            today_df = df[df["date"] == latest]
            print(f"\n最新一日（{latest.date()}）各叢集筆數：")
            print(today_df["cluster"].value_counts().sort_index().to_string())

    def save(self, path: str = GMM_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"GMM 模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = GMM_MODEL_PATH) -> "IBMarginGMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"GMM 模型已載入：{path}（n_components={obj.n_components}）")
        return obj


def find_optimal_gmm_components(
    df: pd.DataFrame,
    k_range: range = range(2, 10),
) -> pd.DataFrame:
    """
    用 BIC + AIC 找最佳 GMM n_components，並畫折線圖。
    BIC 懲罰複雜度更強，通常比 AIC 選出更少的群，兩者都越低越好。

    Returns
    -------
    pd.DataFrame : k, BIC, AIC
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    avail = [c for c in CLASSIFY_FEATURES if c in df.columns]
    pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(10, len(avail)), random_state=42)),
        ]
    )
    X = pre.fit_transform(df[avail].values)

    rows = []
    print("掃描最佳 GMM k 值...")
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
        gmm.fit(X)
        rows.append({"k": k, "BIC": gmm.bic(X), "AIC": gmm.aic(X)})
        print(f"  k={k}: BIC={gmm.bic(X):,.0f}, AIC={gmm.aic(X):,.0f}")

    result = pd.DataFrame(rows).set_index("k")
    best_bic = int(result["BIC"].idxmin())
    best_aic = int(result["AIC"].idxmin())
    print(f"\n最佳 BIC k={best_bic}  |  最佳 AIC k={best_aic}")

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.index, result["BIC"], "o-", color="steelblue", label="BIC")
    ax.plot(result.index, result["AIC"], "s--", color="tomato", label="AIC")
    ax.axvline(best_bic, color="steelblue", linestyle=":", linewidth=1.2, alpha=0.7, label=f"BIC 最佳 k={best_bic}")
    ax.set_xlabel("n_components (k)")
    ax.set_ylabel("分數（越低越好）")
    ax.set_title("GMM：BIC / AIC 選擇最佳 k")
    ax.set_xticks(list(k_range))
    ax.legend()
    plt.tight_layout()
    plt.show()
    return result


if __name__ == "__main__":
    MIN_ATR = 0.02  # 排除日均波幅 < 2% 的低流動性觀測值

    # 實驗：改用原始值（不 xrank）— 注意大小股規模偏差
    clf, _ = run(
        st="2020-01-01",
        end="2023-12-31",
        model="hdbscan",
        min_atr_pct=MIN_ATR,
        n_clusters=7500,
        retrain=True,
        features=CLASSIFY_FEATURES_RAW,
    )

    # 評估：2024 之後的 out-of-sample 資料，同樣過濾
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    df_test = load_data(stocks, st="2024-01-01", min_atr_pct=MIN_ATR)
    df_test["cluster"] = clf.predict(df_test)

    evaluate_clustering(clf=clf, df=df_test)
    analyze_cluster_returns(df_test, hold_days=5)
    analyze_cluster_returns(df_test, hold_days=10)
    analyze_cluster_returns(df_test, hold_days=20)

    # 各叢集出現頻率（每日有幾天、幾支）
    total_days = df_test["date"].nunique()
    freq = df_test.groupby("cluster").agg(
        appear_days=("date", "nunique"),
        avg_stocks_per_day=("stock_id", lambda x: x.groupby(df_test.loc[x.index, "date"]).count().mean()),
    )
    freq["appear_rate"] = freq["appear_days"] / total_days
    print(f"\n{'='*55}")
    print(f"各叢集出現頻率（共 {total_days} 個交易日）")
    print(f"{'='*55}")
    print(freq.to_string())

    # 今日預測（實際使用）
    df_today = predict_today(model="gmm")

    plot_clusters(clf, df_test)


# df_today = predict_today()
# candidates = df_today[df_today["cluster"] == 2]
