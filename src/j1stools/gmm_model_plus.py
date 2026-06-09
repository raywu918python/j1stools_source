"""
放量突破策略：IBMarginGMM + LGBM 信心選股

流程：
  全市場 → ATR 過濾 → 放量上漲（量>=2x + 當日上漲）
        → IBMarginGMM 分群 → 選叢集 2/7/9
        → LGBM 打達標信心分（class 2）
        → prob >= 0.5 進場

profit_label 三分類：0=盤整, 1=停損(-10%), 2=達標(+10%)，持有期 HOLD_DAYS 日

MODE:
  breakout_signal   : 訊號品質分析（prob 分布 + 校準曲線）
  breakout_backtest : 回測（LGBM+GMM 叢集 2/7/9 vs 全叢集 baseline 對照）
"""

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
)
from j1stools.label_builder import profit_label

HOLD_DAYS = 20
PROFIT_TARGET = 0.10
STOP_LOSS = -0.10


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
    """放量上漲宇宙 → GMM 叢集過濾 → profit_label 標籤。clusters=None 使用全叢集。"""
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
    model,
    clf_gmm: IBMarginGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    min_atr_pct: float = 0.02,
    volume_ratio_min: float = 2.0,
) -> pd.DataFrame:
    """放量上漲過濾 → GMM 叢集 → LGBM 信心分，產生 backtest_platform 格式訊號。"""
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


def _fit(model, df_train: pd.DataFrame, features: list | None = None):
    feat_list = features if features is not None else CLASSIFY_FEATURES
    avail = [c for c in feat_list if c in df_train.columns]
    X = df_train[avail].fillna(0.5)
    y = df_train["Y"].values
    model.fit(X, y)
    model._fitted_features = avail
    auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr", average="macro")
    print(f"訓練集 AUC：{auc:.4f}（in-sample，{len(avail)} 特徵）")
    return model


def train_rfc(df_train: pd.DataFrame, features: list | None = None) -> RandomForestClassifier:
    """RFC：n_estimators=200, max_depth=6, min_samples_leaf=50, class_weight=balanced"""
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
    """XGB：n_estimators=200, max_depth=6, lr=0.05"""
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
    """LGBM：n_estimators=200, max_depth=6, lr=0.05, class_weight=balanced"""
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
    """RFC + XGB + LGBM 加權平均，相容 eval_signal / make_signal_breakout 介面。"""

    def __init__(self, models: list, weights: list | None = None):
        w = weights if weights is not None else [1.0] * len(models)
        total = sum(w)
        self.models = models
        self.weights = [x / total for x in w]
        self._fitted_features = models[0]._fitted_features

    def predict_proba(self, X):
        import warnings

        proba = np.zeros((len(X), 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for m, w in zip(self.models, self.weights):
                proba += w * m.predict_proba(X)
        return proba


def train_ensemble(df_train: pd.DataFrame, df_val: pd.DataFrame) -> EnsembleModel:
    """
    訓練 RFC + XGB + LGBM 並以 df_val 的 OOS AUC 加權組成 Ensemble。
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
    model,
    df_test: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    訊號品質分析（不做投組回測）。

    各信心門檻的達標率 + 日均訊號數，輸出 prob 分布直方圖 + 校準曲線。
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    if thresholds is None:
        thresholds = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    avail = model._fitted_features
    X = df_test[avail].fillna(0.5)
    proba_all = model.predict_proba(X)
    prob = proba_all[:, 2]

    y_true = df_test["Y"].values
    if len(set(y_true)) >= 3:
        auc = roc_auc_score(y_true, proba_all, multi_class="ovr", average="macro")
        print(f"\nOOS AUC：{auc:.4f}")
    else:
        auc = float("nan")
        print(f"\nOOS AUC：N/A（測試集只有 {sorted(set(y_true))} 兩個 class）")
    print(
        classification_report(
            y_true,
            proba_all.argmax(axis=1),
            target_names=["盤整", "停損", "達標"],
            zero_division=0,
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
    print(f"Breakout 訊號品質  {date_range}  共 {n_dates} 個交易日")
    print(f"基準達標率（叢集 2/7/9 全宇宙）：{base_rate:.1%}")
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

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(prob, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_title("class 2 prob 分布（全 OOS）")
    axes[0].set_xlabel("prob")
    axes[0].set_ylabel("筆數")

    df_out["prob_bin"] = pd.cut(prob, bins=10)
    cal = df_out.groupby("prob_bin", observed=True).agg(達標率=("Y", lambda x: (x == 2).mean())).reset_index()
    bin_mid = cal["prob_bin"].apply(lambda b: b.mid)
    bin_width = cal["prob_bin"].apply(lambda b: b.length).iloc[0] * 0.85
    axes[1].bar(bin_mid, cal["達標率"], width=bin_width, color="seagreen", alpha=0.8, label="實際達標率")
    axes[1].axhline(base_rate, color="gray", linestyle="--", linewidth=0.8, label=f"基準 {base_rate:.1%}")
    axes[1].set_title("prob vs 實際達標率（校準曲線）")
    axes[1].set_xlabel("prob")
    axes[1].set_ylabel("達標率")
    axes[1].legend()

    plt.suptitle("Breakout 訊號品質分析（GMM 叢集 2/7/9）")
    plt.tight_layout()
    plt.show()
    return df_out


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_ST = "2015-01-01"
    VAL_ST = "2022-01-01"  # 驗證集起點（Ensemble AUC 加權用）
    VAL_END = "2023-12-31"  # 訓練 / 驗證結束
    EVAL_ST = "2024-01-01"  # OOS 測試起點

    BREAKOUT_CLUSTERS = [2, 7, 9]  # 叢集 2（外資強）、7（外資主力）、9（最佳報酬）
    VOLUME_RATIO_MIN = 2.0  # 放量門檻：今日量 >= N 倍 20日均量

    # ── 切換模式 ──────────────────────────────────────────────────────────── #
    #
    #  breakout_compare  : 【模型選型】RFC / XGB / LGBM 三個校準曲線對比
    #                      → 找出信心分最準的模型
    #
    #  breakout_signal   : 【門檻確認】單一模型校準曲線 + 達標率表
    #                      → 決定 backtest 的 prob threshold
    #
    #  breakout_backtest : 【策略驗證】LGBM+GMM 叢集 2/7/9 vs 全叢集 baseline
    #                      → 確認 GMM 過濾是否有效
    #
    MODE = "breakout_backtest"  # "breakout_compare" | "breakout_signal" | "breakout_backtest"
    # ─────────────────────────────────────────────────────────────────────── #

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    clf_breakout = IBMarginGMM.load(BREAKOUT_GMM_MODEL_PATH)

    print("── 建立 Breakout 訓練集 ──")
    df_train = build_dataset_breakout(
        clf_breakout,
        stocks,
        TRAIN_ST,
        VAL_ST,
        clusters=BREAKOUT_CLUSTERS,
        volume_ratio_min=VOLUME_RATIO_MIN,
    )

    print("\n── 建立 Breakout 驗證集 ──")
    df_val = build_dataset_breakout(
        clf_breakout,
        stocks,
        VAL_ST,
        VAL_END,
        clusters=BREAKOUT_CLUSTERS,
        volume_ratio_min=VOLUME_RATIO_MIN,
    )

    print("\n── 訓練 Ensemble（RFC + XGB + LGBM）──")
    ensemble = train_ensemble(df_train, df_val)

    print("\n── 建立 Breakout 測試集 ──")
    df_test = build_dataset_breakout(
        clf_breakout,
        stocks,
        EVAL_ST,
        clusters=BREAKOUT_CLUSTERS,
        volume_ratio_min=VOLUME_RATIO_MIN,
    )

    if MODE == "breakout_compare":
        rfc = train_rfc(df_train, features=CLASSIFY_FEATURES)
        xgb = train_xgb(df_train, features=CLASSIFY_FEATURES)
        lgbm = train_lgbm(df_train, features=CLASSIFY_FEATURES)
        for name, m in [("RFC", rfc), ("XGB", xgb), ("LGBM", lgbm)]:
            print(f"\n{'='*60}\n【{name} 訊號品質】\n{'='*60}")
            eval_signal(m, df_test)

    elif MODE == "breakout_signal":
        lgbm = train_lgbm(df_train, features=CLASSIFY_FEATURES)
        eval_signal(lgbm, df_test)

    elif MODE == "breakout_backtest":
        from j1stools import backtest_platform

        rfc = train_rfc(df_train, features=CLASSIFY_FEATURES)
        xgb = train_xgb(df_train, features=CLASSIFY_FEATURES)
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
            # j1s_chart.plot_performance(pv, td)

        # ── 第一步：三模型比較（GMM 叢集 2/7/9）──
        for name, m in [("RFC", rfc), ("XGB", xgb), ("LGBM", lgbm)]:
            _run_backtest(f"{name} + GMM 叢集 2/7/9", m, BREAKOUT_CLUSTERS)

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
