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
NEGLECTED_MODEL_PATH = "models/neglected_gmm.joblib"


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
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = NEGLECTED_MODEL_PATH) -> "NeglectedGMM":
        import sys
        import joblib

        # joblib/pickle 存檔時若從 __main__ 執行，類別記為 __main__.NeglectedGMM
        # 從其他 script 載入時需注入到 __main__，避免 AttributeError
        main = sys.modules.get("__main__")
        if main is not None and not hasattr(main, "NeglectedGMM"):
            setattr(main, "NeglectedGMM", cls)

        obj = joblib.load(path)
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


def predict_today(
    clusters: list[int] | None = None,
    st: str = "2025-01-01",
    min_atr_pct: float = 0.02,
    min_prob: float = 0.0,
) -> pd.DataFrame:
    """
    用已 release 的模型對最新交易日的外資忽視股做分群，回傳候選股票。

    Parameters
    ----------
    clusters  : 要保留的叢集編號，預設 [2, 3]（穩健群）
    st        : 載入資料的起始日（只需最近幾個月給特徵足夠的滾動窗口）
    min_prob  : GMM 最大歸屬機率的門檻，0.0 = 不過濾
    """
    if clusters is None:
        clusters = [1, 2, 3]

    clf = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    df = load_neglected_data(stocks, st=st, min_atr_pct=min_atr_pct)

    latest = df["date"].max()
    df_today = df[df["date"] == latest].copy()
    df_today["cluster"] = clf.predict(df_today)

    proba = clf.predict_proba(df_today)
    df_today["max_prob"] = proba.max(axis=1)

    result = df_today[df_today["cluster"].isin(clusters)].copy()
    if min_prob > 0:
        result = result[result["max_prob"] >= min_prob]

    print(f"分類日期：{latest.date()}  外資忽視股總數：{len(df_today)}  候選（叢集 {clusters}）：{len(result)} 支")
    print(result[["stock_id", "cluster", "max_prob"]].sort_values("cluster").to_string(index=False))
    return result[["date", "stock_id", "cluster", "max_prob"]]


def deep_analyze(
    clf: NeglectedGMM | None = None,
    cluster_id: int = 5,
    oos_st: str = "2024-01-01",
    bear_st: str = "2022-01-01",
    bear_end: str = "2022-12-31",
    min_atr_pct: float = 0.02,
    hold_days: int = 20,
):
    """
    對特定叢集做深度分析：
    1. 股票清單 — 確認組成，排除誤解
    2. Alpha 分析 — 扣除大盤報酬，看真正的超額報酬
    3. 空頭壓力測試 — 用 2022 跌勢驗證是否只在多頭有效
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    if clf is None:
        clf = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    # ── 1. 股票清單 ──────────────────────────────────────── #
    df_oos = load_neglected_data(stocks, oos_st, min_atr_pct=min_atr_pct)
    df_oos["cluster"] = clf.predict(df_oos)
    c = df_oos[df_oos["cluster"] == cluster_id]

    print(f"\n{'='*60}")
    print(f"叢集 {cluster_id} 股票清單（OOS {oos_st}~）")
    print(f"{'='*60}")
    print(f"唯一股票數：{c['stock_id'].nunique()} 支")
    print(f"每日平均出現：{len(c) / df_oos['date'].nunique():.1f} 支")
    print(f"\n出現頻率最高的 20 支：")
    print(c["stock_id"].value_counts().head(20).to_string())
    print(f"\n原始特徵統計（非 xrank）：")
    raw_cols = [
        col
        for col in ["f_short_margin_ratio", "f_margin_balance_change_pct", "f_net_foreign_pct", "f_atr14_pct"]
        if col in c.columns
    ]
    if raw_cols:
        print(c[raw_cols].describe().round(4).to_string())

    # ── 2. Alpha 分析（扣除大盤）──────────────────────────── #
    print(f"\n{'='*60}")
    print(f"Alpha 分析（持有 {hold_days} 日，扣除 0050 報酬）")
    print(f"期間：{oos_st} ~ {df_oos['date'].max().date()}")
    print(f"{'='*60}")
    from j1stools import parquet_db as pdb

    df_market = pdb.query_price(["0050"], oos_st)
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date")
    df_market["market_ret"] = df_market["close"].shift(-hold_days) / df_market["close"] - 1
    df_market = df_market[["date", "market_ret"]]

    df_oos["future_ret"] = df_oos.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    df_oos = df_oos.merge(df_market, on="date", how="left")
    df_oos["alpha"] = df_oos["future_ret"] - df_oos["market_ret"]

    alpha_by_cluster = (
        df_oos.dropna(subset=["alpha", "cluster"])
        .groupby("cluster")["alpha"]
        .agg(count="count", win_rate=lambda x: (x > 0).mean(), mean="mean", median="median", std="std")
    )
    disp = alpha_by_cluster.copy()
    disp["mean"] = disp["mean"].map("{:.2%}".format)
    disp["median"] = disp["median"].map("{:.2%}".format)
    disp["win_rate"] = disp["win_rate"].map("{:.1%}".format)
    disp["std"] = disp["std"].map("{:.2%}".format)
    print(disp.to_string())

    c_alpha = df_oos[df_oos["cluster"] == cluster_id]["alpha"].dropna()
    print(
        f"\n叢集 {cluster_id} Alpha：mean={c_alpha.mean():.2%}  median={c_alpha.median():.2%}  勝率={( c_alpha > 0).mean():.1%}"
    )

    # ── 3. 空頭壓力測試（2022）────────────────────────────── #
    print(f"\n{'='*60}")
    print(f"空頭壓力測試（{bear_st} ~ {bear_end}）")
    print(f"{'='*60}")
    df_mkt_bear = parquet_db.query_price(["0050"], bear_st, bear_end)
    df_mkt_bear["date"] = pd.to_datetime(df_mkt_bear["date"])
    df_mkt_bear = df_mkt_bear.sort_values("date")
    mkt_total = df_mkt_bear["close"].iloc[-1] / df_mkt_bear["close"].iloc[0] - 1
    print(f"0050 全期報酬：{mkt_total:.2%}  （{bear_st} ~ {df_mkt_bear['date'].max().date()}）")

    df_bear = load_neglected_data(stocks, bear_st, bear_end, min_atr_pct=min_atr_pct)
    if len(df_bear) == 0:
        print("此期間無外資忽視股資料")
        return df_oos

    df_bear["cluster"] = clf.predict(df_bear)
    df_bear["future_ret"] = df_bear.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    bear_stats = (
        df_bear.dropna(subset=["future_ret", "cluster"])
        .groupby("cluster")["future_ret"]
        .agg(count="count", win_rate=lambda x: (x > 0).mean(), mean="mean", median="median")
    )
    disp2 = bear_stats.copy()
    disp2["mean"] = disp2["mean"].map("{:.2%}".format)
    disp2["median"] = disp2["median"].map("{:.2%}".format)
    disp2["win_rate"] = disp2["win_rate"].map("{:.1%}".format)
    print(disp2.to_string())

    # 空頭 vs 多頭 對照圖
    c_bear = df_bear[df_bear["cluster"] == cluster_id]["future_ret"].dropna().clip(-0.5, 0.5)
    c_bull = df_oos[df_oos["cluster"] == cluster_id]["future_ret"].dropna().clip(-0.5, 0.5)
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(c_bull, bins=50, alpha=0.6, label=f"多頭 OOS（{oos_st}~）", color="steelblue")
    ax.hist(c_bear, bins=50, alpha=0.6, label=f"空頭（{bear_st}~{bear_end}）", color="tomato")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"叢集 {cluster_id}  {hold_days}日報酬分布：多頭 vs 空頭")
    ax.set_xlabel("報酬率")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.legend()
    plt.tight_layout()
    plt.show()

    return df_oos


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    MIN_ATR = 0.02
    TRAIN_ST, TRAIN_END, EVAL_ST = "2015-01-01", "2023-12-31", "2024-01-01"
    CLUSTER_ID = 5  # 要深度分析的叢集

    # ── 切換模式 ──────────────────────────────────────────── #
    #  release       : 訓練 2015~2023，存 pkl
    #  eval_oos      : 載入 pkl，驗證 2024~ OOS 報酬
    #  release+eval  : 一鍵訓練 + 驗證
    #  experiment    : 快速實驗（不覆蓋 pkl）
    #  deep_analyze  : 對 CLUSTER_ID 做深度分析（股票清單 + Alpha + 空頭壓測）
    #  predict_today : 今日候選股票（叢集 2、3）
    MODE = "release"
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

    elif MODE == "deep_analyze":
        deep_analyze(cluster_id=CLUSTER_ID, oos_st=EVAL_ST, min_atr_pct=MIN_ATR)

    elif MODE == "predict_today":
        predict_today(clusters=[1, 2, 3])

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

現在三個問題的答案都完整了：

叢集 5 vs 大盤（2022 空頭）

20日均報	勝率
0050（大盤）	-21.70%	—
叢集 5	-24.10%	17.6%
叢集 3	-0.22%	47.0%
叢集 2	-0.83%	41.9%
叢集 5 比大盤更差——不是防禦，是放大跌幅。

研究結論總覽

叢集	多頭 20日	空頭 20日	性質
5	+78.49%	-24.10%	極高 Beta，順勢放大
2	+6.09%	-0.83%	穩健，空頭防禦
3	+4.34%	-0.22%	最保守，空頭最抗跌
叢集 5 不是「Alpha 來源」，是純 Beta 放大器——多頭時要持，空頭時要離。叢集 2、3 才是全天候可用的穩定選股群。
"""
