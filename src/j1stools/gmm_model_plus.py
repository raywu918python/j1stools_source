"""
GMM 過濾 + RFC 信心度選股

設計：
  1. NeglectedGMM 縮小宇宙（外資忽視股）
  2. RFC 以 FEATURES 打達標信心分（class 2）
  3. 依信心門檻動態決定每期進場股數
  4. profit_label 三分類標籤：0=盤整, 1=停損, 2=達標
"""

from j1stools.neglected_stock_classify import NEGLECTED_FEATURES
from j1stools.ib_margin_classify import CLASSIFY_FEATURES

FEATURES = NEGLECTED_FEATURES + [
    "cluster",
    "f_return_5d_xrank",
    "f_return_10d_xrank",
    "f_return_20d_xrank",
    "f_relative_strength_5d_xrank",
    "f_relative_strength_20d_xrank",
    "f_margin_balance_change_10d_pct_xrank",
    "f_short_balance_change_pct_xrank",
    "f_volatility_vs_market_xrank",
    "f_market_volatility_20d",
]

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from j1stools import j1s_chart, parquet_db
from j1stools.ib_margin_classify import (
    BREAKOUT_GMM_MODEL_PATH,
    CLASSIFY_FEATURES,
    IBMarginGMM,
    load_breakout_stocks,
    load_transition_stocks,
)
from j1stools.label_builder import profit_label
from j1stools.neglected_stock_classify import NeglectedGMM, load_neglected_data

HOLD_DAYS = 20
PROFIT_TARGET = 0.10
STOP_LOSS = -0.10


def build_dataset(
    clf_gmm: NeglectedGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """GMM 過濾 → profit_label 標籤。clusters=None 使用全叢集。"""
    df = load_neglected_data(stocks, st, end, min_atr_pct=min_atr_pct)
    df["cluster"] = clf_gmm.predict(df)

    if clusters is not None:
        df = df[df["cluster"].isin(clusters)].copy()

    # profit_label 需要 high / low
    df_hl = parquet_db.query_price(df["stock_id"].unique().tolist(), st, end)
    df_hl["date"] = pd.to_datetime(df_hl["date"])
    df = df.merge(df_hl[["date", "stock_id", "high", "low"]], on=["date", "stock_id"], how="left")

    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)

    df = profit_label(df, hold_days=hold_days, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
    df["Y"] = df["target"].astype(int)
    df = df.dropna(subset=["future_return"]).drop(columns=["target", "high", "low"], errors="ignore")

    cluster_str = f"叢集 {clusters}" if clusters is not None else "全叢集"
    print(f"資料集：{len(df):,} 筆  達標(2)：{(df['Y']==2).mean():.1%}  {cluster_str}  持有 {hold_days} 日")
    return df


def _fit(model, df_train: pd.DataFrame, features: list | None = None):
    feat_list = features if features is not None else FEATURES
    avail = [c for c in feat_list if c in df_train.columns]
    X = df_train[avail].fillna(0.5)
    y = df_train["Y"].values
    model.fit(X, y)
    model._fitted_features = avail
    auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr", average="macro")
    print(f"訓練集 AUC：{auc:.4f}（in-sample，{len(avail)} 特徵）")
    return model


def build_dataset_transition(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = 0.02,
    foreign_pct_max: float = 0.25,
    trust_streak_min: int = 2,
) -> pd.DataFrame:
    """投信開始買 + 外資還沒大買 的轉折股宇宙 → profit_label 標籤。"""
    df = load_transition_stocks(
        stocks,
        st,
        end,
        min_atr_pct=min_atr_pct,
        foreign_pct_max=foreign_pct_max,
        trust_streak_min=trust_streak_min,
    )

    df_hl = parquet_db.query_price(df["stock_id"].unique().tolist(), st, end)
    df_hl["date"] = pd.to_datetime(df_hl["date"])
    df = df.merge(df_hl[["date", "stock_id", "high", "low"]], on=["date", "stock_id"], how="left")

    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    df = profit_label(df, hold_days=hold_days, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
    df["Y"] = df["target"].astype(int)
    df = df.dropna(subset=["future_return"]).drop(columns=["target", "high", "low"], errors="ignore")

    print(f"資料集：{len(df):,} 筆  達標(2)：{(df['Y']==2).mean():.1%}  持有 {hold_days} 日")
    return df


def build_dataset_breakout(
    clf_gmm: IBMarginGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = 0.02,
    volume_ratio_min: float = 2.0,
) -> pd.DataFrame:
    """放量上漲宇宙 → GMM 叢集過濾 → profit_label 標籤。"""
    df = load_breakout_stocks(stocks, st, end, min_atr_pct=min_atr_pct, volume_ratio_min=volume_ratio_min)
    df["cluster"] = clf_gmm.predict(df)

    if clusters is not None:
        df = df[df["cluster"].isin(clusters)].copy()

    df_hl = parquet_db.query_price(df["stock_id"].unique().tolist(), st, end)
    df_hl["date"] = pd.to_datetime(df_hl["date"])
    df = df.merge(df_hl[["date", "stock_id", "high", "low"]], on=["date", "stock_id"], how="left")

    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    df = profit_label(df, hold_days=hold_days, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
    df["Y"] = df["target"].astype(int)
    df = df.dropna(subset=["future_return"]).drop(columns=["target", "high", "low"], errors="ignore")

    cluster_str = f"叢集 {clusters}" if clusters is not None else "全叢集"
    print(f"資料集：{len(df):,} 筆  達標(2)：{(df['Y']==2).mean():.1%}  {cluster_str}  持有 {hold_days} 日")
    return df


def make_signal_breakout(
    model: RandomForestClassifier,
    clf_gmm: IBMarginGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    min_atr_pct: float = 0.02,
    volume_ratio_min: float = 2.0,
) -> pd.DataFrame:
    """放量上漲過濾 → GMM 叢集 → RFC 信心分，產生 backtest_platform 格式訊號。"""
    df = load_breakout_stocks(stocks, st, end, min_atr_pct=min_atr_pct, volume_ratio_min=volume_ratio_min)
    df["cluster"] = clf_gmm.predict(df)
    if clusters is not None:
        df = df[df["cluster"].isin(clusters)].copy()

    avail = model._fitted_features
    proba = model.predict_proba(df[avail].fillna(0.5))
    df["2"] = proba[:, 2]

    signal = df[["date", "stock_id", "2"]].copy()
    signal["date"] = pd.to_datetime(signal["date"])
    print(f"訊號筆數：{len(signal):,}  日期：{signal['date'].min().date()} ~ {signal['date'].max().date()}")
    return signal


def train_rfc(df_train: pd.DataFrame, features: list | None = None) -> RandomForestClassifier:
    return _fit(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        df_train,
        features,
    )


def train_xgb(df_train: pd.DataFrame, features: list | None = None) -> XGBClassifier:
    return _fit(
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        df_train,
        features,
    )


def train_lgbm(df_train: pd.DataFrame, features: list | None = None) -> LGBMClassifier:
    return _fit(
        LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        df_train,
        features,
    )


class EnsembleModel:
    """RFC + XGB + LGBM 加權平均，相容 eval_signal / make_signal / backtest 介面。"""

    def __init__(self, models: list, weights: list | None = None):
        w = weights if weights is not None else [1.0] * len(models)
        total = sum(w)
        self.models = models
        self.weights = [x / total for x in w]
        self._fitted_features = models[0]._fitted_features

    def predict_proba(self, X):
        import warnings

        proba = np.zeros((X.shape[0], 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for m, w in zip(self.models, self.weights):
                proba += w * m.predict_proba(X)
        return proba


def train_ensemble(df_train: pd.DataFrame, df_val: pd.DataFrame) -> EnsembleModel:
    """
    訓練三模型並以 df_val 的 OOS AUC 加權組成 Ensemble。
    df_val 必須與 df_train 不重疊（時間上在後）。
    """
    rfc = train_rfc(df_train)
    xgb = train_xgb(df_train)
    lgbm = train_lgbm(df_train)

    def oos_auc(m):
        avail = m._fitted_features
        X = df_val[avail].fillna(0.5)
        y = df_val["Y"].values
        if len(set(y)) < 3:
            print(f"  警告：驗證集只有 {sorted(set(y))} 兩個 class，AUC 無法計算，改用等權")
            return 1.0
        return roc_auc_score(y, m.predict_proba(X), multi_class="ovr", average="macro")

    aucs = {m.__class__.__name__: oos_auc(m) for m in [rfc, xgb, lgbm]}
    for name, auc in aucs.items():
        print(f"  {name:25s} Val AUC: {auc:.4f}")

    ensemble = EnsembleModel([rfc, xgb, lgbm], weights=list(aucs.values()))
    print(f"  Ensemble 權重: {[f'{w:.3f}' for w in ensemble.weights]}")
    return ensemble


def eval_signal(
    model: RandomForestClassifier,
    df_test: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    純訊號品質分析（不做投組回測，留給量化框架）。

    用全部 OOS 資料點計算各信心門檻的達標率與日均訊號數，
    回傳帶 prob 欄位的 df，可直接接量化回測。
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    if thresholds is None:
        thresholds = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45]

    avail = model._fitted_features
    X = df_test[avail].fillna(0.5)
    proba_all = model.predict_proba(X)
    prob = proba_all[:, 2]

    auc = roc_auc_score(df_test["Y"].values, proba_all, multi_class="ovr", average="macro")
    print(f"\nOOS AUC：{auc:.4f}")
    print(
        classification_report(
            df_test["Y"].values,
            proba_all.argmax(axis=1),
            target_names=["盤整", "停損", "達標"],
        )
    )

    df_out = df_test.copy()
    df_out["prob"] = prob
    df_out["future_return_clip"] = df_out["future_return"].clip(-0.5, 1.0)

    n_dates = df_out["date"].nunique()
    date_range = f"{df_out['date'].min().date()} ~ {df_out['date'].max().date()}"
    base_rate = (df_out["Y"] == 2).mean()

    rows = []
    for thr in thresholds:
        subset = df_out[df_out["prob"] >= thr]
        if len(subset) == 0:
            rows.append(
                {
                    "門檻": f">={thr:.0%}",
                    "總筆數": 0,
                    "日均訊號": 0.0,
                    "盤整": 0,
                    "停損": 0,
                    "達標": 0,
                    "達標率": float("nan"),
                    "平均報酬": float("nan"),
                }
            )
            continue
        vc = subset["Y"].value_counts()
        rows.append(
            {
                "門檻": f">={thr:.0%}",
                "總筆數": len(subset),
                "日均訊號": round(len(subset) / n_dates, 1),
                "盤整": vc.get(0, 0),
                "停損": vc.get(1, 0),
                "達標": vc.get(2, 0),
                "達標率": (subset["Y"] == 2).mean(),
                "平均報酬": subset["future_return_clip"].mean(),
            }
        )

    result = pd.DataFrame(rows)
    print(f"\n{'='*68}")
    print(f"GMM 過濾 + RFC 訊號品質  {date_range}  共 {n_dates} 個交易日")
    print(f"基準達標率（全宇宙）：{base_rate:.1%}")
    print(f"{'='*68}")
    print(
        result.to_string(
            index=False,
            formatters={
                "達標率": "{:.1%}".format,
                "平均報酬": "{:.2%}".format,
                "日均訊號": "{:.1f}".format,
            },
        )
    )

    # prob 分布 + 校準曲線
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(prob, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_title("class 2 prob 分布（全 OOS）")
    axes[0].set_xlabel("prob")
    axes[0].set_ylabel("筆數")

    df_out["prob_bin"] = pd.cut(prob, bins=10)
    cal = (
        df_out.groupby("prob_bin", observed=True)
        .agg(
            達標率=("Y", lambda x: (x == 2).mean()),
        )
        .reset_index()
    )
    bin_mid = cal["prob_bin"].apply(lambda b: b.mid)
    bin_width = cal["prob_bin"].apply(lambda b: b.length).iloc[0] * 0.85
    axes[1].bar(bin_mid, cal["達標率"], width=bin_width, color="seagreen", alpha=0.8, label="實際達標率")
    axes[1].axhline(base_rate, color="gray", linestyle="--", linewidth=0.8, label=f"基準 {base_rate:.1%}")
    axes[1].set_title("prob vs 實際達標率（校準曲線）")
    axes[1].set_xlabel("prob")
    axes[1].set_ylabel("達標率")
    axes[1].legend()

    plt.suptitle("GMM 過濾 + RFC 訊號品質分析")
    plt.tight_layout()
    plt.show()
    return df_out


def add_market_filter(signal: pd.DataFrame, st: str, ma_period: int = 120) -> pd.DataFrame:
    """
    市場擇時過濾：0050 > MA(ma_period) 的多頭日，訊號清零（自然空倉）。
    0050 < MA → 空頭/橫盤 → 保留忽視股訊號。
    """
    mkt = parquet_db.query_price(["0050"], st)
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").set_index("date")["close"]
    ma = mkt.rolling(ma_period, min_periods=1).mean()
    bearish = (mkt < ma).rename("bearish").reset_index()  # True = 空頭/橫盤，保留訊號

    signal = signal.copy()
    signal["date"] = pd.to_datetime(signal["date"])
    signal = signal.merge(bearish, on="date", how="left")
    signal["2"] = signal["2"] * signal["bearish"].fillna(True).astype(float)
    signal = signal.drop(columns=["bearish"])

    active_days = bearish[bearish["bearish"]]["date"]
    pct = len(active_days) / len(bearish)
    print(f"市場過濾（MA{ma_period}）：策略進場日佔 {pct:.1%}（0050 < MA 的天數）")
    return signal


def eval_cluster_perf(clf_gmm: NeglectedGMM, df: pd.DataFrame) -> pd.DataFrame:
    """各 GMM 叢集的 OOS 達標率 + 平均報酬，用於選出最佳 target_cluster。"""
    df = df.copy()
    df["cluster"] = clf_gmm.predict(df)
    stats = (
        df.groupby("cluster")
        .agg(
            筆數=("Y", "count"),
            達標率=("Y", lambda x: (x == 2).mean()),
            平均報酬=("future_return", "mean"),
            中位報酬=("future_return", "median"),
            停損率=("Y", lambda x: (x == 1).mean()),
        )
        .sort_values("中位報酬", ascending=False)
    )
    print("\nGMM 叢集 OOS 績效：")
    print(stats.to_string(float_format="{:.2%}".format))
    return stats


def make_signal_gmm(
    clf_gmm: NeglectedGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    target_cluster: int | None = None,
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """
    純 GMM 訊號，不用 RFC。
    target_cluster=None → 買全部外資忽視股（訊號=1.0）
    target_cluster=N   → 用 GMM 軟機率 prob_N 當訊號分數
    """
    df = load_neglected_data(stocks, st, end, min_atr_pct=min_atr_pct)
    proba = clf_gmm.predict_proba(df)
    df = pd.concat([df, proba], axis=1)

    if target_cluster is None:
        df["2"] = 1.0
    else:
        df["2"] = df[f"prob_{target_cluster}"]

    signal = df[["date", "stock_id", "2"]].copy()
    signal["date"] = pd.to_datetime(signal["date"])
    print(f"GMM 訊號筆數：{len(signal):,}  target_cluster={target_cluster}")
    return signal


def make_signal(
    model: RandomForestClassifier,
    clf_gmm: NeglectedGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """
    產生 backtest_platform 需要的 signal DataFrame。
    欄位：date, stock_id, "2"（class 2 prob）。
    """
    df = load_neglected_data(stocks, st, end, min_atr_pct=min_atr_pct)
    df["cluster"] = clf_gmm.predict(df)
    if clusters is not None:
        df = df[df["cluster"].isin(clusters)].copy()

    avail = model._fitted_features
    proba = model.predict_proba(df[avail].fillna(0.5))
    df["2"] = proba[:, 2]

    signal = df[["date", "stock_id", "2"]].copy()
    signal["date"] = pd.to_datetime(signal["date"])
    print(f"訊號筆數：{len(signal):,}  日期：{signal['date'].min().date()} ~ {signal['date'].max().date()}")
    return signal


def backtest(
    model: RandomForestClassifier,
    clf_gmm: NeglectedGMM,
    stocks: list,
    st: str = "2024-01-01",
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    top_n: int = 5,
    threshold: float = 0.35,
    max_positions: int = 5,
    use_fixed_sl: bool = True,
    sl_stop: float = 0.10,
    use_fixed_tp: bool = True,
    tp_stop: float = 0.10,
    use_hold_days: bool = True,
    hold_days: int = 10,
    group_limit: int = 2,
    min_volume: int = 200,
):
    """
    GMM 過濾 + 模型訊號 → backtest_platform 回測。
    回傳 portfolio_value, trades_df, positions, close_df, market_df。
    """
    from j1stools import backtest_platform

    signal = make_signal(model, clf_gmm, stocks, st, end, clusters=clusters)

    portfolio_value, trades_df, positions, close_df = backtest_platform.prepare_data_backtest(
        signal,
        top_n=top_n,
        threshold=threshold,
        max_positions=max_positions,
        use_sl_trail=False,
        use_fixed_sl=use_fixed_sl,
        sl_stop=sl_stop,
        use_fixed_tp=use_fixed_tp,
        tp_stop=tp_stop,
        use_hold_days=use_hold_days,
        hold_days=hold_days,
        group_limit=group_limit,
        min_volume=min_volume,
    )

    # 大盤對比（0050）
    eq_dates = pd.to_datetime(portfolio_value["date"]).dt.normalize()
    mkt = parquet_db.query_price(["0050"], st, end)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
    mkt = mkt[mkt["date"].isin(eq_dates)].reset_index(drop=True)
    if mkt.empty:
        mkt = parquet_db.query_price(["0050"], st, end)
        mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
        mkt = mkt[mkt["date"] >= eq_dates.iloc[0]].reset_index(drop=True)
    start_val = portfolio_value["total"].iloc[0]
    mkt["total"] = (mkt["close"] / mkt["close"].iloc[0] * start_val).round(2)
    market_df = mkt[["date", "total"]]

    j1s_chart.plot_performance(
        portfolio_value=portfolio_value,
        trades_df=trades_df,
        is_web=False,
    )
    return portfolio_value, trades_df, positions, close_df, market_df


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_ST = "2015-01-01"
    VAL_ST = "2022-01-01"  # 驗證集起點（AUC 加權用）
    VAL_END = "2023-12-31"  # 訓練結束 / 驗證結束
    EVAL_ST = "2024-01-01"  # OOS 測試起點
    CLUSTERS = None  # NeglectedGMM 叢集過濾（None=全部）

    # ── 切換模式 ──────────────────────────────────────────────────────────── #
    #
    #  【外資忽視股策略（NeglectedGMM）】
    #  signal       : Ensemble 訊號品質分析（prob 分布 + 校準曲線）
    #  backtest     : Ensemble 回測
    #  gmm_backtest : 純 GMM 訊號回測（不用 RFC）
    #
    #  【轉折股策略（IB Margin：投信買超 + 外資未跟進）】
    #  transition_signal   : Ensemble 訊號品質分析
    #  transition_backtest : 全買回測（不用 RFC，驗證策略本身）
    #
    #  【放量突破策略（IBMarginGMM：量 >= 2x + 當日上漲）】
    #  breakout_signal   : Ensemble 訊號品質分析
    #  breakout_backtest : Ensemble 回測
    #
    MODE = "breakout_backtest"
    # ─────────────────────────────────────────────────────────────────────── #

    GMM_TARGET_CLUSTER = 5  # gmm_backtest 用：None=全部忽視股 | int=指定叢集機率當訊號

    # ── Breakout GMM 設定 ─────────────────────────────────────────────────── #
    BREAKOUT_CLUSTERS = [2, 7, 9]  # 叢集 2（外資強）、7（外資主力）、9（最佳報酬）
    VOLUME_RATIO_MIN = 2.0  # 放量門檻：今日量 >= N 倍 20日均量
    # ─────────────────────────────────────────────────────────────────────── #

    clf_gmm = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE in ("breakout_signal", "breakout_backtest"):
        clf_breakout = IBMarginGMM.load(BREAKOUT_GMM_MODEL_PATH)

        print("── 建立 Breakout 訓練集 ──")
        df_train = build_dataset_breakout(
            clf_breakout, stocks, TRAIN_ST, VAL_ST, clusters=BREAKOUT_CLUSTERS, volume_ratio_min=VOLUME_RATIO_MIN
        )

        print("\n── 建立 Breakout 驗證集 ──")
        df_val = build_dataset_breakout(
            clf_breakout, stocks, VAL_ST, VAL_END, clusters=BREAKOUT_CLUSTERS, volume_ratio_min=VOLUME_RATIO_MIN
        )

        print("\n── 訓練 Ensemble（RFC + XGB + LGBM）──")
        ensemble = train_ensemble(df_train, df_val)

        print("\n── 建立 Breakout 測試集 ──")
        df_test = build_dataset_breakout(
            clf_breakout, stocks, EVAL_ST, clusters=BREAKOUT_CLUSTERS, volume_ratio_min=VOLUME_RATIO_MIN
        )

        if MODE == "breakout_signal":
            eval_signal(ensemble, df_test)

        elif MODE == "breakout_backtest":
            from j1stools import backtest_platform, j1s_chart

            lgbm = train_lgbm(df_train, features=CLASSIFY_FEATURES)

            def _run_backtest(label, m, clusters):
                print(f"\n{'='*60}\n【{label}】\n{'='*60}")
                sig = make_signal_breakout(
                    m,
                    clf_breakout,
                    stocks,
                    st=EVAL_ST,
                    clusters=clusters,
                    volume_ratio_min=VOLUME_RATIO_MIN,
                )
                pv, td, _, _ = backtest_platform.prepare_data_backtest(
                    sig,
                    top_n=5,
                    threshold=0.50,
                    max_positions=5,
                    use_sl_trail=False,
                    use_fixed_sl=True,
                    sl_stop=0.10,
                    use_fixed_tp=True,
                    tp_stop=0.10,
                    use_hold_days=True,
                    hold_days=HOLD_DAYS,
                    group_limit=2,
                    min_volume=200,
                )
                j1s_chart.plot_performance(pv, td)

            # ── GMM 過濾（叢集 2/7/9）──
            _run_backtest("LGBM + GMM 叢集 2/7/9", lgbm, BREAKOUT_CLUSTERS)

            # ── Baseline：全叢集（驗證 GMM 是否有效）──
            print("\n── 建立 Baseline 訓練集（全叢集）──")
            df_train_all = build_dataset_breakout(
                clf_breakout,
                stocks,
                TRAIN_ST,
                VAL_ST,
                clusters=None,
                volume_ratio_min=VOLUME_RATIO_MIN,
            )
            lgbm_all = train_lgbm(df_train_all, features=CLASSIFY_FEATURES)
            _run_backtest("LGBM Baseline（全叢集，無 GMM 過濾）", lgbm_all, None)

    else:
        print("── 建立訓練集 ──")
        df_train = build_dataset(clf_gmm, stocks, TRAIN_ST, VAL_ST, clusters=CLUSTERS)

        print("\n── 建立驗證集（AUC 加權用）──")
        df_val = build_dataset(clf_gmm, stocks, VAL_ST, VAL_END, clusters=CLUSTERS)

        print("\n── 訓練 Ensemble（RFC + XGB + LGBM，AUC 加權）──")
        ensemble = train_ensemble(df_train, df_val)

        print("\n── 建立測試集 ──")
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=CLUSTERS)

        if MODE == "signal":
            rfc = train_rfc(df_train)
            xgb = train_xgb(df_train)
            lgbm = train_lgbm(df_train)
            for name, m in [("RFC", rfc), ("XGB", xgb), ("LGBM", lgbm), ("Ensemble", ensemble)]:
                print(f"\n{'='*60}\n【{name}】\n{'='*60}")
                eval_signal(m, df_test)

        elif MODE == "backtest":
            print(f"\n── 回測（Ensemble）──")
            portfolio_value, trades_df, positions, close_df, market_df = backtest(
                ensemble,
                clf_gmm,
                stocks,
                st=EVAL_ST,
                top_n=5,
                threshold=0.5,
                max_positions=5,
                use_fixed_sl=True,
                sl_stop=0.10,
                use_fixed_tp=True,
                tp_stop=0.10,
                use_hold_days=True,
                hold_days=10,
            )

        elif MODE == "transition_signal":
            print("\n── 建立轉折股訓練集 ──")
            df_tr_train = build_dataset_transition(stocks, TRAIN_ST, VAL_ST)
            print("\n── 建立轉折股驗證集 ──")
            df_tr_val = build_dataset_transition(stocks, VAL_ST, VAL_END)
            print("\n── 建立轉折股測試集 ──")
            df_tr_test = build_dataset_transition(stocks, EVAL_ST)

            print("\n── 訓練（使用完整 IB 特徵）──")
            rfc = train_rfc(df_tr_train, features=CLASSIFY_FEATURES)
            xgb = train_xgb(df_tr_train, features=CLASSIFY_FEATURES)
            lgbm = train_lgbm(df_tr_train, features=CLASSIFY_FEATURES)

            def _oos(m):
                avail = m._fitted_features
                X = df_tr_val[avail].fillna(0.5).values
                return roc_auc_score(df_tr_val["Y"].values, m.predict_proba(X), multi_class="ovr", average="macro")

            aucs = {n: _oos(m) for n, m in [("RFC", rfc), ("XGB", xgb), ("LGBM", lgbm)]}
            for n, a in aucs.items():
                print(f"  {n:6s} Val AUC: {a:.4f}")
            tr_ensemble = EnsembleModel([rfc, xgb, lgbm], weights=list(aucs.values()))

            for name, m in [("RFC", rfc), ("XGB", xgb), ("LGBM", lgbm), ("Ensemble", tr_ensemble)]:
                print(f"\n{'='*60}\n【{name}】\n{'='*60}")
                eval_signal(m, df_tr_test)

        elif MODE == "transition_backtest":
            from j1stools import backtest_platform

            print("\n── 轉折股回測（全買，不用 model）──")
            df_tr = load_transition_stocks(stocks, st=EVAL_ST, min_atr_pct=0.02)
            df_tr["date"] = pd.to_datetime(df_tr["date"])
            signal = df_tr[["date", "stock_id"]].copy()
            signal["2"] = 1.0

            portfolio_value, trades_df, positions, close_df = backtest_platform.prepare_data_backtest(
                signal,
                top_n=5,
                threshold=0.0,
                max_positions=5,
                use_sl_trail=False,
                use_fixed_sl=True,
                sl_stop=0.10,
                use_fixed_tp=False,
                use_hold_days=True,
                hold_days=10,
                group_limit=2,
                min_volume=50,
            )
            j1s_chart.plot_performance(portfolio_value, trades_df)

        elif MODE == "gmm_backtest":
            from j1stools import backtest_platform

            print("\n── GMM 叢集 OOS 達標率 ──")
            eval_cluster_perf(clf_gmm, df_test)

            print(f"\n── GMM 回測（target_cluster={GMM_TARGET_CLUSTER}）──")
            signal = make_signal_gmm(clf_gmm, stocks, st=EVAL_ST, target_cluster=GMM_TARGET_CLUSTER)

            portfolio_value, trades_df, positions, close_df = backtest_platform.prepare_data_backtest(
                signal,
                top_n=3,
                threshold=0.0,
                max_positions=5,
                use_sl_trail=False,
                use_fixed_sl=True,
                sl_stop=0.10,
                use_fixed_tp=False,
                use_hold_days=True,
                hold_days=10,
                group_limit=2,
                min_volume=200,
            )
            j1s_chart.plot_performance(portfolio_value, trades_df)
