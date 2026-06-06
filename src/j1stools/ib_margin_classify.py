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
    # 價格動能（相對大盤）
    "f_return_5d_xrank",
    "f_return_20d_xrank",
    "f_relative_strength_5d_xrank",
    "f_relative_strength_20d_xrank",
]

N_CLUSTERS = 5
MODEL_PATH = "db/models/ib_margin_classifier.pkl"
MARKET_PROXY = "0050"  # 大盤代理（台灣50）


def load_data(stocks: list, st: str, end: str = "2099-01-01") -> pd.DataFrame:
    """
    載入並合併價格、融資融券、法人資料，回傳含 f_ 特徵的 DataFrame。
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

    df = df_price.merge(df_margin, on=["date", "stock_id"], how="left")
    feat = add_feature(df, df_ib, df_market, mode="predict")
    # add_feature 只保留 f_ 欄位，把 close 補回來供報酬計算使用
    feat = feat.merge(df_price[["date", "stock_id", "close"]], on=["date", "stock_id"], how="left")
    return feat


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
    pipe_pre = clf.pipeline[:-1]  # imputer + scaler + pca
    X_pca = pipe_pre.transform(X_full)
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
        data_by_k = [(k, df_plot[df_plot["cluster"] == k][feat].dropna().values)
                     for k in range(n_k)]
        data_by_k = [(k, v) for k, v in data_by_k if len(v) > 1]
        if not data_by_k:
            continue
        positions = [k for k, _ in data_by_k]
        parts = ax_v.violinplot([v for _, v in data_by_k], positions=positions,
                                showmedians=True, showextrema=False)
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


def run(st: str = "2023-01-01", end: str = "2099-01-01", n_clusters: int = N_CLUSTERS) -> tuple:
    """
    主流程：載入資料 → 訓練分類器 → 顯示叢集特徵 → 儲存模型。

    Returns
    -------
    (clf, df) : 分類器物件 + 含 cluster 欄位的 DataFrame
    """
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    print(f"股票數：{len(stocks)}，期間：{st} ~ {end}")

    print("載入資料並計算特徵...")
    df = load_data(stocks, st, end)
    print(f"資料筆數：{len(df):,}")

    print(f"訓練 KMeans（n_clusters={n_clusters}）...")
    clf = IBMarginClassifier(n_clusters=n_clusters)
    clf.fit(df)
    df["cluster"] = clf.predict(df)

    clf.describe_clusters(df)
    analyze_cluster_returns(df, hold_days=5)
    clf.save()

    return clf, df


def predict_today(st: str = "2025-01-01") -> pd.DataFrame:
    """
    使用訓練好的模型對最新交易日的股票做分類。

    Returns
    -------
    pd.DataFrame : 含 date, stock_id, cluster 欄位
    """
    clf = IBMarginClassifier.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    df = load_data(stocks, st)
    latest_date = df["date"].max()
    df_today = df[df["date"] == latest_date].copy()
    df_today["cluster"] = clf.predict(df_today)

    print(f"分類日期：{latest_date.date()}")
    print(df_today["cluster"].value_counts().sort_index().to_string())
    return df_today[["date", "stock_id", "cluster"]]


if __name__ == "__main__":
    clf, df = run(st="2023-01-01", n_clusters=N_CLUSTERS)
    plot_clusters(clf, df)

    # find_optimal_clusters(df)
