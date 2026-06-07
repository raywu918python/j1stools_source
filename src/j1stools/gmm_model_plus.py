"""
GMM 過濾 + RFC 信心度選股

設計：
  1. NeglectedGMM 縮小宇宙（外資忽視股）
  2. RFC 以 FEATURES 打達標信心分（class 2）
  3. 依信心門檻動態決定每期進場股數
  4. profit_label 三分類標籤：0=盤整, 1=停損, 2=達標
"""

from j1stools.neglected_stock_classify import NEGLECTED_FEATURES

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

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from j1stools import j1s_chart, parquet_db
from j1stools.label_builder import profit_label
from j1stools.neglected_stock_classify import NeglectedGMM, load_neglected_data

HOLD_DAYS = 10
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


def _fit(model, df_train: pd.DataFrame):
    avail = [c for c in FEATURES if c in df_train.columns]
    X = df_train[avail].fillna(0.5).values
    y = df_train["Y"].values
    model.fit(X, y)
    model._fitted_features = avail
    auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr", average="macro")
    print(f"訓練集 AUC：{auc:.4f}（in-sample）")
    return model


def train_rfc(df_train: pd.DataFrame) -> RandomForestClassifier:
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
    )


def train_xgb(df_train: pd.DataFrame) -> XGBClassifier:
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
    )


def train_lgbm(df_train: pd.DataFrame) -> LGBMClassifier:
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
    )


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
    X = df_test[avail].fillna(0.5).values
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
    proba = model.predict_proba(df[avail].fillna(0.5).values)
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

    TRAIN_ST, TRAIN_END = "2015-01-01", "2023-12-31"
    EVAL_ST = "2024-01-01"
    CLUSTERS = None
    MODE = "backtest"  # "signal" | "backtest"
    BACKTEST_MODEL = "lgbm"  # "rfc" | "xgb" | "lgbm"

    clf_gmm = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    print("── 建立訓練集 ──")
    df_train = build_dataset(clf_gmm, stocks, TRAIN_ST, TRAIN_END, clusters=CLUSTERS)

    print("\n── 訓練三模型 ──")
    rfc = train_rfc(df_train)
    xgb = train_xgb(df_train)
    lgbm = train_lgbm(df_train)

    if MODE == "signal":
        print("\n── 建立測試集 ──")
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=CLUSTERS)

        for name, m in [("RFC", rfc), ("XGB", xgb), ("LGBM", lgbm)]:
            print(f"\n{'='*60}")
            print(f"【{name}】")
            print(f"{'='*60}")
            eval_signal(m, df_test)

    elif MODE == "backtest":
        m = {"rfc": rfc, "xgb": xgb, "lgbm": lgbm}[BACKTEST_MODEL]
        print(f"\n── 回測（{BACKTEST_MODEL.upper()}）──")
        portfolio_value, trades_df, positions, close_df, market_df = backtest(
            m,
            clf_gmm,
            stocks,
            st=EVAL_ST,
            top_n=5,
            threshold=0.40,
            max_positions=5,
            use_fixed_sl=True,
            sl_stop=0.10,
            use_fixed_tp=True,
            tp_stop=0.10,
            use_hold_days=True,
            hold_days=10,
        )
