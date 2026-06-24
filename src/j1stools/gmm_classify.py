"""
IB + Margin 無監督分類器（放量突破策略用）

設計：
  1. load_breakout_stocks  : 放量上漲宇宙（量>=2x + 當日上漲）
  2. load_transition_stocks: 轉折股宇宙（投信買 + 外資未跟進）
  3. IBMarginGMM           : GMM 軟分群（法人籌碼 + 融資融券 25 特徵）
  4. __main__ breakout 模式: 訓練並儲存 BREAKOUT_GMM_MODEL_PATH

MODE:
  breakout : 訓練放量宇宙 GMM，分析叢集報酬，儲存模型
  pca      : 觀察特徵在 PCA 2D/3D 的分布（調參輔助）
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from j1stools import parquet_db
from j1stools.margin_ibbuysell_feature import add_feature

# 聚焦 IB + Margin 的截面排名特徵（0~1 均勻分布，跨時間穩定）
CLASSIFY_FEATURES = [
    # ── 外資籌碼 ──────────────────────────────────────────
    "f_net_foreign_pct_xrank",  # 當日外資買超佔流通股本（截面排名）
    "f_net_foreign_5d_z_xrank",  # 外資 5 日買超 z-score：高 = 近期買超異常強
    "f_net_foreign_10d_z_xrank",  # 外資 10 日 z-score：中期趨勢方向
    "f_net_foreign_streak",  # 連續買賣天數：正=連買、負=連賣
    # ── 投信籌碼（轉折核心訊號）──────────────────────────
    "f_net_trust_pct_xrank",  # 當日投信買超強度（截面排名）
    "f_net_trust_5d_z_xrank",  # 投信 5 日 z-score：高 = 投信正在加碼
    "f_net_trust_10d_z_xrank",  # 投信 10 日 z-score：中期佈局強度
    "f_net_trust_streak",  # 投信連買天數：>=2 是「外資還沒發現」的早期訊號
    # ── 法人合計（外資+投信+自營商）──────────────────────
    "f_net_institutional_total_pct_xrank",  # 整體法人當日買超強度
    "f_net_institutional_total_5d_z_xrank",  # 整體法人 5 日 z-score
    "f_net_institutional_total_10d_z_xrank",  # 整體法人 10 日 z-score
    "f_net_institutional_total_streak",  # 整體法人連買天數
    # ── 融資融券 ──────────────────────────────────────────
    "f_margin_balance_change_pct_xrank",  # 融資餘額增減（增加=散戶跟進）
    "f_short_balance_change_pct_xrank",  # 融券餘額增減（增加=空方施壓）
    "f_margin_balance_change_5d_pct_xrank",  # 5 日融資趨勢
    "f_short_margin_ratio_xrank",  # 融券/融資比：高 = 多空對立激烈
    # ── 當沖 ──────────────────────────────────────────────
    "f_dt_ratio_xrank",  # 當沖佔比：高 = 短線散戶多
    "f_dt_net_xrank",  # 當沖方向（正=買方主導）
    "f_dt_ratio_5d_xrank",  # 5 日當沖佔比趨勢
    # ── 量能 ──────────────────────────────────────────────
    "f_volume_ratio_5d_xrank",  # 今日成交量 / 5 日均量：高 = 放量
    "f_volume_change_pct_xrank",  # 成交量日增率：正 = 量能擴張
    "f_atr14_pct_xrank",  # 14 日 ATR 佔收盤價比：股票波動性
    # ── 價格趨勢（方向，非報酬）────────────────────────────
    "f_ma5_slope_xrank",  # MA5 斜率：短期趨勢方向
    "f_ma20_slope_xrank",  # MA20 斜率：中期趨勢方向
    "f_ma60_slope_xrank",  # MA60 斜率：中長期趨勢方向
    "f_bias_ma20_xrank",  # 偏離 MA20 程度：高 = 超漲，低 = 超跌
    "f_bias_ma60_xrank",  # 偏離 MA60 程度：突破時在中長期均線上下
    "f_momentum_cross_xrank",  # MA5/MA20 黃金/死亡交叉訊號
]
# 移除價格動能特徵（f_return_5d_xrank 等）
# 目標是找「籌碼好但還沒漲」的股票，避免分群偏向已漲股

N_CLUSTERS = 10
BREAKOUT_GMM_MODEL_PATH = "db/models/ib_margin_breakout_gmm.joblib"
MARKET_PROXY = "0050"  # 大盤代理（台灣50）
MIN_ATR_PCT = 0.02  # ATR 過濾門檻（GMM 訓練與 LGBM 資料共用）
VOLUME_RATIO_MIN = 1  # 放量門檻（None = 關閉）
EXCLUDE_CLUSTERS: list[int] = [1, 8]  # 達標率最差的叢集，cluster_inspect 後更新

# 分類器額外特徵：量能梯度 + GMM 軟分群信心（GMM 分群本身不用這些）
CLASSIFIER_FEATURES = CLASSIFY_FEATURES + ["f_volume_ratio_20d_xrank", "gmm_max_prob"]


def load_transition_stocks(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    min_atr_pct: float = 0.02,
    foreign_pct_max: float = 0.25,
    trust_streak_min: int = 2,
) -> pd.DataFrame:
    """
    轉折股宇宙：外資仍低活躍 + 投信開始連續買進。

    條件：
      - f_net_foreign_pct_xrank < foreign_pct_max（或 NaN）
      - f_net_trust_streak >= trust_streak_min
      - 排除外資連買 >= 8 天（漲幅已反映）
    """
    df = load_data(stocks, st, end, min_atr_pct=min_atr_pct, require_complete=False)

    foreign_low = df["f_net_foreign_pct_xrank"].isna() | (df["f_net_foreign_pct_xrank"] < foreign_pct_max)
    trust_buying = df["f_net_trust_streak"] >= trust_streak_min
    not_overbought = df["f_net_foreign_streak"].fillna(0) < 8

    mask = foreign_low & trust_buying & not_overbought
    result = df[mask].copy()

    total = len(df)
    n = len(result)
    print(
        f"轉折股過濾（外資<{foreign_pct_max:.0%} & 投信streak>={trust_streak_min}）："
        f"{total:,} → {n:,} 筆（{n/total:.1%}），唯一股票：{result['stock_id'].nunique()} 支"
    )
    return result


def load_breakout_stocks(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    min_atr_pct: float = 0.02,
    volume_ratio_min: float | None = VOLUME_RATIO_MIN,  # None = 關閉放量過濾
    min_daily_return: float | None = 0.0,  # None = 關閉上漲過濾；0.0 = 上漲即可
    require_complete: bool = False,
) -> pd.DataFrame:
    """
    放量上漲宇宙：今日量 >= 20日均量的 volume_ratio_min 倍，且今日收盤上漲。

    假設：量能放大 + 價格上漲 = 有人在買且市場認可。
    搭配 GMM 分群，找出「哪種籌碼型態下的放量最可靠」。

    volume_ratio_min=None：關閉放量過濾。
    min_daily_return=None：關閉上漲過濾（保留所有日）。
    """
    df = load_data(stocks, st, end, min_atr_pct=min_atr_pct, require_complete=require_complete)
    total = len(df)

    # ── 上漲過濾 ─────────────────────────────────────────────────
    if min_daily_return is not None:
        before = len(df)
        df = df[df["f_daily_return"] > min_daily_return].copy()
        print(f"上漲過濾（daily_return>{min_daily_return:.1%}）：{before:,} → {len(df):,} 筆（{len(df)/before:.1%}），唯一股票：{df['stock_id'].nunique()} 支")

    # ── 放量過濾 ─────────────────────────────────────────────────
    if volume_ratio_min is not None:
        before = len(df)
        df = df[df["f_volume_ratio_20d"] >= volume_ratio_min].copy()
        print(f"放量過濾（量>={volume_ratio_min:.1f}x均量）：{before:,} → {len(df):,} 筆（{len(df)/before:.1%}），唯一股票：{df['stock_id'].nunique()} 支")

    print(f"宇宙總計：{total:,} → {len(df):,} 筆（{len(df)/total:.1%}）")
    return df


def load_data(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    min_atr_pct: float | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    載入並合併價格、融資融券、法人資料，回傳含 f_ 特徵的 DataFrame。

    Parameters
    ----------
    min_atr_pct      : ATR14/close 門檻，低波動股過濾掉
    require_complete : True = 過濾缺關鍵籌碼特徵的觀測值（外資忽視股）
                       False = 保留全部（含外資忽視股）
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
    feat = feat.merge(
        df_price[["date", "stock_id", "close", "volume", "high", "low"]],
        on=["date", "stock_id"],
        how="left",
    )
    feat = _add_day_trade_features(feat, df_day_trade)
    feat = _add_atr_features(feat)

    feat["f_volume_ratio_20d"] = feat.groupby("stock_id")["volume"].transform(
        lambda x: x / x.rolling(20, min_periods=5).mean()
    )
    feat["f_volume_ratio_20d_xrank"] = feat.groupby("date")["f_volume_ratio_20d"].rank(pct=True)
    feat["f_daily_return"] = feat.groupby("stock_id")["close"].transform(lambda x: x.pct_change(fill_method=None))
    feat = feat.drop(columns=["volume", "high", "low"])

    if min_atr_pct is not None:
        before = len(feat)
        feat = feat[feat["f_atr14_pct"].fillna(0) >= min_atr_pct]
        print(f"ATR 過濾（>= {min_atr_pct:.3f}）：{before:,} → {len(feat):,} 筆，移除 {before - len(feat):,} 筆")

    if require_complete:
        # 過濾外資忽視股（z-score/dt 缺失），避免 GMM 誤聚成假高報酬群
        before = len(feat)
        feat = feat.dropna(subset=["f_net_foreign_5d_z_xrank", "f_dt_ratio_xrank"])
        print(f"籌碼完整性過濾：{before:,} → {len(feat):,} 筆，移除 {before - len(feat):,} 筆")

    return feat


def _add_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """ATR14 佔收盤價比例 + 截面排名。需要 df 含 high, low, close。"""
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
    加入當沖特徵（需要 df 含 volume）。

    新增：f_dt_ratio（當沖佔比）、f_dt_net（方向）、f_dt_ratio_5d（5日趨勢）及各自的 xrank。
    當沖資料隔天公布，shift 1 天對齊。
    """
    dt = df_dt.copy()
    dt["date"] = pd.to_datetime(dt["date"])
    dt = dt.rename(columns={"volume": "_dt_vol", "buy_amount": "_dt_buy", "sell_amount": "_dt_sell"})

    base = df.merge(dt[["date", "stock_id", "_dt_vol", "_dt_buy", "_dt_sell"]], on=["date", "stock_id"], how="left")

    g = base.groupby("stock_id")
    for col in ["_dt_vol", "_dt_buy", "_dt_sell"]:
        base[col] = g[col].transform(lambda x: x.shift(1))

    base["f_dt_ratio"] = (base["_dt_vol"] / base["volume"].replace(0, np.nan)).clip(0, 1)
    dt_total = (base["_dt_buy"] + base["_dt_sell"]).replace(0, np.nan)
    base["f_dt_net"] = ((base["_dt_buy"] - base["_dt_sell"]) / dt_total).clip(-1, 1)
    base["f_dt_ratio_5d"] = g["f_dt_ratio"].transform(lambda x: x.rolling(5).mean())
    base = base.drop(columns=["_dt_vol", "_dt_buy", "_dt_sell"])

    for col in ["f_dt_ratio", "f_dt_net", "f_dt_ratio_5d"]:
        base[f"{col}_xrank"] = base.groupby("date")[col].rank(pct=True, na_option="keep")

    return base


class IBMarginGMM:
    """
    Gaussian Mixture Model 軟分群分類器（放量突破策略用）。

    輸入：CLASSIFY_FEATURES（25 個 xrank 截面特徵）
    輸出：10 個叢集，代表不同籌碼行為模式
    最佳叢集：2（外資0.83+連買5天）、7（外資0.94+連買5天）、9（OOS報酬最佳）

    相較 KMeans 優勢：predict_proba() 可設信心門檻、對重疊財務資料更自然。
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
        """軟分群：回傳各叢集機率，columns = [prob_0, ..., prob_n]。"""
        if self.pipeline is None:
            raise RuntimeError("請先呼叫 fit()")
        X_pca = self.pipeline[:-1].transform(df[self._fitted_features].values)
        proba = self.pipeline.named_steps["gmm"].predict_proba(X_pca)
        cols = [f"prob_{k}" for k in range(self.n_components)]
        return pd.DataFrame(proba, index=df.index, columns=cols)

    def describe_clusters(self, df: pd.DataFrame | None = None, full: bool = True):
        """印出每個叢集的特徵中位數。full=True 顯示所有訓練特徵。"""
        if self.cluster_profiles is None:
            raise RuntimeError("請先呼叫 fit()")

        if full:
            cols = list(self.cluster_profiles.columns)
        else:
            cols = [
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
        print(
            f"GMM 叢集特徵 Median（{self.n_components} 群，BIC={self.bic_:.0f}，{'全部' if full else '摘要'} {len(cols)} 欄）"
        )
        print(f"{'='*65}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(self.cluster_profiles[cols].T.to_string())

        if df is not None and "cluster" in df.columns:
            counts = df["cluster"].value_counts().sort_index()
            total = counts.sum()
            print(f"\n各叢集筆數：")
            for k, n in counts.items():
                print(f"  叢集 {k}: {n:>8,} 筆 ({n/total:.1%})")

    def save(self, path: str = BREAKOUT_GMM_MODEL_PATH):
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"GMM 模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = BREAKOUT_GMM_MODEL_PATH) -> "IBMarginGMM":
        import sys, joblib

        main = sys.modules.get("__main__")
        if main is not None and not hasattr(main, "IBMarginGMM"):
            setattr(main, "IBMarginGMM", cls)
        obj = joblib.load(path)
        print(f"GMM 模型已載入：{path}（n_components={obj.n_components}）")
        return obj


def find_optimal_gmm_components(
    df: pd.DataFrame,
    k_range: range = range(2, 21),
) -> pd.DataFrame:
    """
    用 BIC + AIC 找最佳 GMM n_components，畫折線圖。

    BIC 懲罰複雜度更強，通常比 AIC 選出更少的群，兩者都越低越好。
    用訓練期資料掃，不能用 OOS（否則等於用未來資料調參）。
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
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=10, reg_covar=1e-3)
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
    ax.set_title("GMM：BIC / AIC 選擇最佳 n_components")
    ax.set_xticks(list(k_range))
    ax.legend()
    plt.tight_layout()
    plt.show()
    return result


def analyze_cluster_returns(df: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    """各叢集持有 N 日後的報酬分布：勝率、平均報酬、中位數。"""
    df = df.copy()
    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    result = (
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
    display = result.copy()
    display["mean_return"] = display["mean_return"].map("{:.2%}".format)
    display["median_return"] = display["median_return"].map("{:.2%}".format)
    display["win_rate"] = display["win_rate"].map("{:.1%}".format)
    display["std"] = display["std"].map("{:.2%}".format)

    print(f"\n{'='*55}")
    print(f"各叢集持有 {hold_days} 日報酬分析")
    print(f"{'='*55}")
    print(display.to_string())
    return result


def evaluate_clustering(
    clf: "IBMarginGMM",
    df: pd.DataFrame,
    hold_days_list: list = [3, 5, 10],
    sample_n: int = 50_000,
) -> dict:
    """
    全面評估叢集品質：幾何指標 + 金融實用性 + Kruskal-Wallis 檢定。

    幾何指標：Silhouette(>0.2 合理) / Davies-Bouldin(越低越好) / Calinski-Harabasz(越高越好)
    金融驗證：各叢集持有 N 日報酬分布，p<0.05 代表各群報酬有統計顯著差異
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    X_full = df[clf._fitted_features].values
    X_pca = clf.pipeline[:-1].transform(X_full)
    labels = clf.pipeline.predict(X_full)

    if len(df) > sample_n:
        idx = pd.Series(range(len(df))).sample(sample_n, random_state=42).values
        labels_sil, X_sil = labels[idx], X_pca[idx]
    else:
        labels_sil, X_sil = labels, X_pca

    n_unique = len(set(labels_sil))
    if n_unique >= 2:
        sil = silhouette_score(X_sil, labels_sil)
        db = davies_bouldin_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)
    else:
        sil = db = ch = float("nan")

    print(f"\n{'='*55}\n叢集幾何品質指標\n{'='*55}")
    print(f"  Silhouette score  : {sil:.4f}  （> 0.2 OK，> 0.5 佳）")
    print(f"  Davies-Bouldin    : {db:.4f}  （越低越好）")
    print(f"  Calinski-Harabasz : {ch:.1f}  （越高越好）")

    df_eval = df.assign(cluster=labels).copy()
    result = {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "returns": {}}

    n_holds = len(hold_days_list)
    fig, axes = plt.subplots(1, n_holds, figsize=(6 * n_holds, 5))
    if n_holds == 1:
        axes = [axes]
    fig.suptitle("各叢集報酬分布（Box Plot）", fontsize=14)

    for ax, hold in zip(axes, hold_days_list):
        col = f"ret_{hold}d"
        df_eval[col] = df_eval.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold) / x - 1)
        groups = [df_eval[df_eval["cluster"] == k][col].dropna().values for k in sorted(df_eval["cluster"].unique())]
        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups)
            kw_label = f"p={kw_p:.4f} {'★顯著' if kw_p < 0.05 else '（不顯著）'}"
        else:
            kw_p = float("nan")
            kw_label = "N/A"

        stats_df = (
            df_eval.dropna(subset=[col])
            .groupby("cluster")[col]
            .agg(count="count", win_rate=lambda x: (x > 0).mean(), mean="mean", median="median")
        )
        print(f"\n{'='*55}")
        print(f"持有 {hold} 日  |  Kruskal-Wallis {kw_label}")
        print(f"{'='*55}")
        disp = stats_df.copy()
        disp["win_rate"] = disp["win_rate"].map("{:.1%}".format)
        disp["mean"] = disp["mean"].map("{:.2%}".format)
        disp["median"] = disp["median"].map("{:.2%}".format)
        print(disp.to_string())
        result["returns"][hold] = {"stats": stats_df, "kruskal_p": kw_p}

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


def plot_clusters(clf: "IBMarginGMM", df: pd.DataFrame, sample_n: int = 30_000):
    """
    三合一叢集視覺化：
      1. PCA 2D 散點圖  — 各群空間分布
      2. 特徵 Heatmap   — 各群籌碼側寫（綠=高分位，紅=低分位）
      3. Violin plots   — 重要特徵在各群內的分布形狀
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    X_full = df[clf._fitted_features].values
    X_pca = clf.pipeline[:-1].transform(X_full)
    labels = clf.pipeline.predict(X_full)
    n_k = clf.n_clusters
    palette = plt.cm.tab20.colors[: max(n_k, 1)]

    sidx = (
        pd.Series(range(len(df))).sample(sample_n, random_state=42).values if len(df) > sample_n else np.arange(len(df))
    )
    df_plot = df.assign(cluster=labels)
    unique_labels = sorted(df_plot["cluster"].unique())

    def _cluster_color(k):
        return palette[k % len(palette)]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # ── PCA 2D 散點圖 ─────────────────────────────────────────── #
    ax_pca = fig.add_subplot(gs[0, :2])
    for k in unique_labels:
        mask = labels[sidx] == k
        ax_pca.scatter(
            X_pca[sidx][mask, 0], X_pca[sidx][mask, 1], c=[_cluster_color(k)], alpha=0.25, s=4, label=f"叢集 {k}"
        )
    ax_pca.set_title("PCA 2D 散點圖（PC1 vs PC2）")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(markerscale=4, loc="upper right")

    # ── Feature Heatmap ────────────────────────────────────────── #
    ax_heat = fig.add_subplot(gs[0, 2:])
    _KEY_CANDIDATES = [
        ("f_net_foreign_pct_xrank", "外資"),
        ("f_net_trust_pct_xrank", "投信"),
        ("f_net_institutional_total_pct_xrank", "法人合計"),
        ("f_net_institutional_total_streak", "連買天"),
        ("f_margin_balance_change_pct_xrank", "融資變化"),
        ("f_short_margin_ratio_xrank", "券資比"),
    ]
    KEY_FEAT = {col: lbl for col, lbl in _KEY_CANDIDATES if col in clf.cluster_profiles.columns}
    cols = list(KEY_FEAT.keys())
    if cols:
        profile = clf.cluster_profiles[cols].rename(columns=KEY_FEAT)
        data = profile.values.astype(float)
        # 各欄尺度不一（xrank 為 0~1 排名，連買天為原始天數），故逐欄獨立 min-max
        # 正規化決定顏色，避免單一離群尺度的欄位拉垮其餘欄位的色階對比
        col_min = np.nanmin(data, axis=0)
        col_max = np.nanmax(data, axis=0)
        col_span = np.where(col_max > col_min, col_max - col_min, 1)
        data_norm = (data - col_min) / col_span
        im = ax_heat.imshow(data_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax_heat.set_xticks(range(len(profile.columns)))
        ax_heat.set_xticklabels(profile.columns, rotation=40, ha="right", fontsize=9)
        ax_heat.set_yticks(range(len(profile)))
        ax_heat.set_yticklabels([f"叢集 {k}" for k in profile.index])
        for i in range(len(profile)):
            for j in range(len(profile.columns)):
                v = data[i, j]
                if not np.isnan(v):
                    ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046)
        cbar.set_label("列內相對高低（0=該欄最低，1=該欄最高）", fontsize=8)
    ax_heat.set_title("叢集特徵 Heatmap（格內數字為實際 median 值）")

    # ── Violin plots ───────────────────────────────────────────── #
    _VIOLIN_CANDIDATES = [
        ("f_net_foreign_pct_xrank", "外資"),
        ("f_net_institutional_total_pct_xrank", "法人合計"),
        ("f_short_margin_ratio_xrank", "券資比"),
        ("f_dt_ratio_xrank", "當沖佔比"),
    ]
    violin_feats = [(col, lbl) for col, lbl in _VIOLIN_CANDIDATES if col in df_plot.columns]

    for col_idx, (feat, name) in enumerate(violin_feats):
        ax_v = fig.add_subplot(gs[1, col_idx])
        data_by_k = [(k, df_plot[df_plot["cluster"] == k][feat].dropna().values) for k in unique_labels]
        data_by_k = [(k, v) for k, v in data_by_k if len(v) > 1]
        if not data_by_k:
            ax_v.set_title(name)
            continue
        positions = [k for k, _ in data_by_k]
        parts = ax_v.violinplot([v for _, v in data_by_k], positions=positions, showmedians=True, showextrema=False)
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor(_cluster_color(positions[j]))
            body.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax_v.set_title(name)
        ax_v.set_xlabel("叢集")
        ax_v.set_xticks(positions)
        ax_v.set_xticklabels([str(k) for k in positions])

    fig.suptitle("IBMarginGMM 叢集視覺化", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_pca(
    df: pd.DataFrame,
    features: list | None = None,
    sample_n: int = 50_000,
    use_robust: bool = False,
):
    """
    訓練前觀察資料結構：PCA 降到 2D/3D，看有沒有自然群落。

    有明顯分離的球團 → HDBSCAN 可用。
    一大坨均勻雲     → GMM/KMeans 即可。

    use_robust : True = RobustScaler（適合 raw 特徵）
                 False = StandardScaler（適合 xrank，預設）
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    feats = features or CLASSIFY_FEATURES
    avail = [c for c in feats if c in df.columns]

    scaler = RobustScaler() if use_robust else StandardScaler()
    pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
            ("pca", PCA(n_components=3, random_state=42)),
        ]
    )
    X3 = pre.fit_transform(df[avail].values)

    if len(df) > sample_n:
        idx = pd.Series(range(len(df))).sample(sample_n, random_state=42).values
        X3 = X3[idx]

    var = pre.named_steps["pca"].explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(X3[:, 0], X3[:, 1], alpha=0.05, s=2, color="steelblue")
    axes[0].set_xlabel(f"PC1 ({var[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({var[1]:.1%})")
    axes[0].set_title("PC1 vs PC2")

    axes[1].scatter(X3[:, 0], X3[:, 2], alpha=0.05, s=2, color="tomato")
    axes[1].set_xlabel(f"PC1 ({var[0]:.1%})")
    axes[1].set_ylabel(f"PC3 ({var[2]:.1%})")
    axes[1].set_title("PC1 vs PC3")

    scaler_name = "RobustScaler" if use_robust else "StandardScaler"
    fig.suptitle(f"PCA 資料結構觀察（{scaler_name}，{len(X3):,} 筆）", fontsize=13)
    plt.tight_layout()
    plt.show()
    print(f"PC1~3 累積解釋變異：{sum(var):.1%}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    MIN_ATR = 0.02
    TRAIN_ST, TRAIN_END, EVAL_ST = "2020-01-01", "2023-12-31", "2024-01-01"

    # ── 切換模式 ──────────────────────────────────────────────────────────── #
    #
    #  scan     : 【選 k 用】BIC/AIC 折線圖，決定 n_components 最佳值
    #             → 用訓練期資料掃，不能用 OOS（否則等於用未來資料調參）
    #
    #  pca      : 【選演算法用】PCA 2D/3D 分布，確認群落結構
    #             → 一大坨橢圓形 = GMM 適合；明顯球團 = HDBSCAN 也適合
    #
    #  breakout : 【訓練用】GMM 分群，分析叢集報酬，儲存模型
    #             → 儲存至 BREAKOUT_GMM_MODEL_PATH，供 gmm_model_plus.py 使用
    #
    MODE = "breakout"  # "scan" | "pca" | "breakout"
    # ─────────────────────────────────────────────────────────────────────── #

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "breakout":
        print("── 訓練集（放量上漲）──")
        df_train = load_breakout_stocks(stocks, TRAIN_ST, TRAIN_END, min_atr_pct=MIN_ATR)

        print("\n── 訓練 IBMarginGMM ──")
        clf = IBMarginGMM(n_components=N_CLUSTERS)
        clf.fit(df_train)

        print("\n── 測試集（放量上漲）──")
        df_test = load_breakout_stocks(stocks, EVAL_ST, min_atr_pct=MIN_ATR)
        df_test["cluster"] = clf.predict(df_test)

        print("\n── 叢集輪廓 ──")
        clf.describe_clusters(df_test)

        print("\n── 叢集 OOS 報酬 ──")
        analyze_cluster_returns(df_test, hold_days=5)
        analyze_cluster_returns(df_test, hold_days=10)

        print("\n── 叢集品質指標 ──")
        evaluate_clustering(clf=clf, df=df_test)

        print("\n── 叢集視覺化 ──")
        plot_clusters(clf, df_test)

        print("\n── 儲存模型 ──")
        clf.save(BREAKOUT_GMM_MODEL_PATH)

    elif MODE == "scan":
        # 用訓練期資料掃最佳 k，不能用 OOS（否則等於用未來資料調參）
        df = load_breakout_stocks(stocks, TRAIN_ST, TRAIN_END, min_atr_pct=MIN_ATR)
        find_optimal_gmm_components(df, k_range=range(2, 21))

    elif MODE == "pca":
        df = load_breakout_stocks(stocks, TRAIN_ST, TRAIN_END, min_atr_pct=MIN_ATR)
        plot_pca(df, features=CLASSIFY_FEATURES, use_robust=False)
