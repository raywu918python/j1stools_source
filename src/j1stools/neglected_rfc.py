"""
外資忽視股 RFC 選股模型

流程：
  1. GMM 篩選叢集 2/3（穩健技術面群）
  2. RFC 在這個宇宙裡學「哪些股票接下來會漲」
  3. OOS 回測驗證效果

不是偷看答案：
  - GMM 只用當下特徵分群（無未來資訊）
  - RFC 的 Y = 未來 N 日漲跌（正確的監督學習目標）
  - 訓練用 2015~2023，測試用 2024~
"""

import os
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from j1stools import parquet_db
from j1stools.label_builder import profit_label
from j1stools.neglected_stock_classify import (
    NeglectedGMM,
    load_neglected_data,
    NEGLECTED_FEATURES,
)

RFC_MODEL_PATH = "db/models/neglected_rfc.pkl"
PROFIT_TARGET = 0.10  # 與 profit_label 預設一致

# RFC 特徵：NEGLECTED_FEATURES（GMM 用的技術面）+ GMM 沒用的維度
# prob_k 是 NEGLECTED_FEATURES 的非線性組合，移除以避免冗餘
RFC_FEATURES = NEGLECTED_FEATURES + [
    # GMM 叢集編號：讓 RFC 自己學要不要用叢集資訊
    "cluster",
    # 價格動能：GMM 只有 slope 代理，這裡加直接報酬
    "f_return_5d_xrank",
    "f_return_10d_xrank",
    "f_return_20d_xrank",
    # 相對強度 vs 大盤：GMM 完全沒用
    "f_relative_strength_5d_xrank",
    "f_relative_strength_20d_xrank",
    # 更長週期融資：GMM 只有 5d
    "f_margin_balance_change_10d_pct_xrank",
    # 融券絕對變化：GMM 只用比率
    "f_short_balance_change_pct_xrank",
    # 個股 vs 大盤波動：GMM 沒用
    "f_volatility_vs_market_xrank",
    # 大盤環境（raw，截面 xrank 無意義）
    "f_market_volatility_20d",
]


def build_dataset(
    clf_gmm: NeglectedGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    hold_days: int = 10,
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """
    載入外資忽視股 → GMM 分群 → 計算未來報酬標籤。

    Parameters
    ----------
    clusters  : 預測時要保留的叢集；None = 不過濾（訓練模式，學全部叢集）
    hold_days : 持有天數
    """
    df = load_neglected_data(stocks, st, end, min_atr_pct=min_atr_pct)
    df["cluster"] = clf_gmm.predict(df)

    # GMM 軟機率（保留欄位但不加入 RFC_FEATURES，供後續分析用）
    proba = clf_gmm.predict_proba(df)
    df = pd.concat([df, proba], axis=1)

    # 叢集過濾：None = 訓練時用全部叢集，非 None = 預測/eval 時過濾
    if clusters is not None:
        df = df[df["cluster"].isin(clusters)].copy()

    # 補回 high/low（profit_label 需要，load_data 已 drop）
    df_price_hl = parquet_db.query_price(df["stock_id"].unique().tolist(), st, end)
    df_price_hl["date"] = pd.to_datetime(df_price_hl["date"])
    df = df.merge(df_price_hl[["date", "stock_id", "high", "low"]], on=["date", "stock_id"], how="left")

    # 未來收盤報酬（累積報酬用）
    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)

    # profit_label：0=盤整, 1=停損, 2=達標（三類，保留原始語意）
    # 注意：profit_label 內部會計算並 drop 自己的 max_ret，所以必須在它之後再算我們的
    df = profit_label(df, hold_days=hold_days, profit_target=PROFIT_TARGET)
    df["Y"] = df["target"].astype(int)  # 三類：0/1/2

    # 持有期間最高報酬（win_rate 指標，與 profit_label 對齊；在 drop high 前計算）
    future_max = df.groupby("stock_id")["high"].transform(
        lambda x: x.shift(-hold_days).rolling(window=hold_days, min_periods=1).max()
    )
    df["max_ret"] = (future_max - df["close"]) / df["close"]

    # dropna 用 future_return 而非 target（profit_label 對末端列不會輸出 NaN）
    df = df.dropna(subset=["future_return"]).drop(columns=["target", "high", "low"], errors="ignore")

    n2 = (df["Y"] == 2).mean()
    cluster_info = f"叢集 {clusters}" if clusters is not None else "全叢集"
    print(f"資料集：{len(df):,} 筆  達標(2)：{n2:.1%}  {cluster_info}  持有 {hold_days} 日")
    return df


def train_rfc(
    df_train: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 6,
    features: list | None = None,
) -> RandomForestClassifier:
    """在訓練集上訓練 RFC。features=None 時使用全域 RFC_FEATURES。"""
    feat_list = features if features is not None else RFC_FEATURES
    avail = [c for c in feat_list if c in df_train.columns]
    X = df_train[avail].fillna(0.5).values
    y = df_train["Y"].values

    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rfc.fit(X, y)
    rfc._fitted_features = avail

    auc = roc_auc_score(y, rfc.predict_proba(X), multi_class="ovr", average="macro")
    print(f"訓練集 AUC（macro OvR）：{auc:.4f}（in-sample，供參考）")
    return rfc


def train_xgb(
    df_train: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 6,
    features: list | None = None,
) -> XGBClassifier:
    feat_list = features if features is not None else RFC_FEATURES
    avail = [c for c in feat_list if c in df_train.columns]
    X = df_train[avail].fillna(0.5).values
    y = df_train["Y"].values

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb.fit(X, y)
    xgb._fitted_features = avail

    auc = roc_auc_score(y, xgb.predict_proba(X), multi_class="ovr", average="macro")
    print(f"訓練集 AUC（macro OvR）：{auc:.4f}（in-sample，供參考）")
    return xgb


def tune_xgb(
    df_train: pd.DataFrame,
    n_trials: int = 50,
    val_st: str = "2022-01-01",
) -> dict:
    """
    用 Optuna 搜尋 XGBoost 最佳超參數。
    時序切割：val_st 前訓練，之後驗證，避免 data leak。
    需要 pip install optuna
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    avail = [c for c in RFC_FEATURES if c in df_train.columns]
    df_fit = df_train[df_train["date"] < val_st]
    df_val = df_train[df_train["date"] >= val_st]

    X_fit = df_fit[avail].fillna(0.5).values
    y_fit = df_fit["Y"].values
    X_val = df_val[avail].fillna(0.5).values
    y_val = df_val["Y"].values

    print(f"調參切割：訓練 {len(df_fit):,} 筆 / 驗證 {len(df_val):,} 筆（{val_st} 起）")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0),
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        m = XGBClassifier(**params)
        m.fit(X_fit, y_fit)
        return roc_auc_score(y_val, m.predict_proba(X_val), multi_class="ovr", average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n最佳 Val AUC：{study.best_value:.4f}")
    print("最佳超參數：")
    for k, v in study.best_params.items():
        print(f"  {k:25s}: {v}")
    return study.best_params


def train_lgbm(
    df_train: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 6,
    features: list | None = None,
) -> LGBMClassifier:
    feat_list = features if features is not None else RFC_FEATURES
    avail = [c for c in feat_list if c in df_train.columns]
    X = df_train[avail].fillna(0.5).values
    y = df_train["Y"].values

    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgbm.fit(X, y)
    lgbm._fitted_features = avail

    auc = roc_auc_score(y, lgbm.predict_proba(X), multi_class="ovr", average="macro")
    print(f"訓練集 AUC（macro OvR）：{auc:.4f}（in-sample，供參考）")
    return lgbm


class EnsembleModel:
    """多模型加權平均 wrapper，相容 eval_rfc 介面。"""

    def __init__(self, models: list, weights: list | None = None):
        import numpy as np

        w = weights if weights is not None else [1.0] * len(models)
        total = sum(w)
        self.models = models
        self.weights = [x / total for x in w]
        self._fitted_features = models[0]._fitted_features
        self.feature_importances_ = np.zeros(len(self._fitted_features))

    def predict_proba(self, X):
        import numpy as np

        # 三類加權平均：(N, 3) 格式
        proba = np.zeros((X.shape[0], 3))
        for model, w in zip(self.models, self.weights):
            proba += w * model.predict_proba(X)
        return proba


def eval_rfc(
    rfc: RandomForestClassifier,
    df_test: pd.DataFrame,
    top_n: int = 10,
    hold_days: int = 10,
    min_prob: float = 0.0,
    top_pct: float | None = None,
) -> pd.DataFrame:
    """
    OOS 評估：
      - AUC / 分類報告
      - 每日選 top_n 高機率股票，計算組合報酬

    min_prob : 絕對門檻（固定數值，跨模型不公平）
    top_pct  : 分位數門檻，例如 0.2 = 只取該模型 prob 分佈前 20% 的觀測
               設定後會覆蓋 min_prob，讓各模型用自己的尺度比較
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    import numpy as np

    avail = rfc._fitted_features
    X_test = df_test[avail].fillna(0.5).values
    y_test = df_test["Y"].values
    proba_all = rfc.predict_proba(X_test)  # (N, 3) for 3-class
    prob = proba_all[:, 2]  # class 2 = 達標機率（選股訊號）

    # 分位數門檻：用該模型自己的 prob 分佈決定，跨模型公平比較
    effective_min_prob = min_prob
    if top_pct is not None:
        effective_min_prob = float(np.quantile(prob, 1.0 - top_pct))
        print(f"top_pct={top_pct:.0%} → 有效門檻 prob >= {effective_min_prob:.4f}")

    # Multi-class AUC（OvR）
    auc = roc_auc_score(y_test, proba_all, multi_class="ovr", average="macro")
    print(f"\n{'='*55}")
    print(f"OOS AUC（macro OvR）：{auc:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_test, proba_all.argmax(axis=1), target_names=["盤整", "停損", "達標"]))

    # 特徵重要性
    fi = pd.Series(rfc.feature_importances_, index=avail).sort_values(ascending=False)
    print(f"\n特徵重要性 Top 10：")
    print(fi.head(10).map("{:.4f}".format).to_string())

    # 信心度分層勝率
    df_cal = pd.DataFrame({"prob": prob, "Y": y_test, "future_return": df_test["future_return"].values})
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"]
    df_cal["bucket"] = pd.cut(df_cal["prob"], bins=bins, labels=labels, right=False)
    cal = df_cal.groupby("bucket", observed=True).agg(
        筆數=("Y", "count"),
        平均信心=("prob", "mean"),
        達標率=("Y", lambda x: (x == 2).mean()),  # class 2 = 達標
        平均報酬=("future_return", "mean"),
    )
    print(f"\n{'='*60}")
    print("信心度分層勝率")
    print(f"{'='*60}")
    print(cal.to_string(float_format=lambda x: f"{x:.2%}"))

    # 每日 top_n 組合回測（非重疊：每 hold_days 天換一次倉）
    df_eval = df_test.copy()
    df_eval["prob"] = prob
    # clip 極端值，避免單一股票暴漲/暴跌炸掉累積計算
    df_eval["future_return_clip"] = df_eval["future_return"].clip(-0.5, 1.0)

    all_dates = sorted(df_eval["date"].unique())
    # 每 hold_days 天取一個換倉日（非重疊窗口）
    rebal_dates = all_dates[::hold_days]

    rfc_returns, base_returns = [], []
    for date in rebal_dates:
        g = df_eval[df_eval["date"] == date]
        if len(g) == 0:
            continue
        # RFC top_n（信心門檻過濾後再選）
        candidates = g[g["prob"] >= effective_min_prob]
        if len(candidates) == 0:
            continue  # 當天無股票達門檻，空倉
        top_rfc = candidates.nlargest(top_n, "prob")
        rfc_returns.append(
            {
                "date": date,
                "mean_return": top_rfc["future_return_clip"].mean(),
                "win_rate": (top_rfc["max_ret"] >= PROFIT_TARGET).mean(),
            }
        )
        # Baseline：隨機選 top_n
        top_base = g.sample(min(top_n, len(g)), random_state=42)
        base_returns.append(
            {
                "date": date,
                "mean_return": top_base["future_return_clip"].mean(),
                "win_rate": (top_base["max_ret"] >= PROFIT_TARGET).mean(),
            }
        )

    df_perf = pd.DataFrame(rfc_returns).set_index("date")
    df_base = pd.DataFrame(base_returns).set_index("date")
    cum_rfc = (1 + df_perf["mean_return"]).cumprod()
    cum_base = (1 + df_base["mean_return"]).cumprod()

    threshold_str = f"  門檻 prob>={effective_min_prob:.4f}" if effective_min_prob > 0 else ""
    date_range = f"{df_eval['date'].min().date()} ~ {df_eval['date'].max().date()}"
    print(f"\n{'='*60}")
    print(f"非重疊換倉 Top-{top_n} 組合（每 {hold_days} 日換一次{threshold_str}）  共 {len(rfc_returns)} 期進場")
    print(f"期間：{date_range}")
    print(f"{'='*60}")
    print(f"{'':15} {'RFC':>10} {'隨機 Baseline':>15}")
    print(f"  平均報酬   {df_perf['mean_return'].mean():>9.2%} {df_base['mean_return'].mean():>14.2%}")
    print(f"  勝率       {df_perf['win_rate'].mean():>9.1%} {df_base['win_rate'].mean():>14.1%}")
    print(f"  累積報酬   {cum_rfc.iloc[-1]-1:>9.2%} {cum_base.iloc[-1]-1:>14.2%}")

    # 累積報酬比較圖
    _, axes = plt.subplots(2, 1, figsize=(12, 8))
    cum_rfc.plot(ax=axes[0], color="steelblue", label=f"RFC Top-{top_n}")
    cum_base.plot(ax=axes[0], color="tomato", linestyle="--", label=f"隨機 Baseline")
    axes[0].axhline(1, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"外資忽視股 RFC vs 隨機 Baseline — Top-{top_n} 每 {hold_days} 日換倉")
    axes[0].set_ylabel("累積報酬倍數")
    axes[0].legend()

    df_perf["win_rate"].rolling(5).mean().plot(ax=axes[1], color="steelblue", label="RFC")
    df_base["win_rate"].rolling(5).mean().plot(ax=axes[1], color="tomato", linestyle="--", label="隨機")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("滾動 5 期勝率")
    axes[1].set_ylabel("勝率")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return df_perf


def save_rfc(rfc: RandomForestClassifier, path: str = RFC_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rfc, f)
    print(f"RFC 模型已儲存：{path}")


def load_rfc(path: str = RFC_MODEL_PATH) -> RandomForestClassifier:
    with open(path, "rb") as f:
        rfc = pickle.load(f)
    print(f"RFC 模型已載入：{path}")
    return rfc


XGB_MODEL_PATH = "models/neglected_xgb.joblib"


def save_xgb(xgb: XGBClassifier, path: str = XGB_MODEL_PATH):
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(xgb, path)
    print(f"XGB 模型已儲存：{path}")


def load_xgb(path: str = XGB_MODEL_PATH) -> XGBClassifier:
    import joblib

    xgb = joblib.load(path)
    print(f"XGB 模型已載入：{path}")
    return xgb


ENSEMBLE_MODEL_PATH = "models/neglected_ensemble.joblib"


def save_ensemble(ensemble: EnsembleModel, path: str = ENSEMBLE_MODEL_PATH):
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(ensemble, path)
    print(f"Ensemble 模型已儲存：{path}")


def load_ensemble(path: str = ENSEMBLE_MODEL_PATH) -> EnsembleModel:
    import sys
    import joblib

    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "EnsembleModel"):
        setattr(main, "EnsembleModel", EnsembleModel)

    ensemble = joblib.load(path)
    print(f"Ensemble 模型已載入：{path}")
    return ensemble


def predict_today_rfc(
    top_n: int = 10,
    clusters: list[int] | None = None,
    st: str = "2025-01-01",
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """今日候選：GMM 叢集 2/3 → RFC 打分 → 回傳 top_n。"""
    if clusters is None:
        clusters = [1, 2, 3]

    clf_gmm = NeglectedGMM.load()
    rfc = load_rfc()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    df = load_neglected_data(stocks, st=st, min_atr_pct=min_atr_pct)
    df["cluster"] = clf_gmm.predict(df)
    proba_gmm = clf_gmm.predict_proba(df)
    df = pd.concat([df, proba_gmm], axis=1)

    latest = df["date"].max()
    today = df[(df["date"] == latest) & df["cluster"].isin(clusters)].copy()

    avail = rfc._fitted_features
    today["rfc_prob"] = rfc.predict_proba(today[avail].fillna(0.5).values)[:, 1]

    result = today.nlargest(top_n, "rfc_prob")[["date", "stock_id", "cluster", "rfc_prob"]]
    print(f"\n分類日期：{latest.date()}  GMM 叢集 {clusters} 共 {len(today)} 支 → RFC Top-{top_n}：")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_ST, TRAIN_END = "2015-01-01", "2023-12-31"
    EVAL_ST = "2024-01-01"
    HOLD_DAYS = 10
    CLUSTERS = [1, 2, 3]
    TOP_N = 10
    MIN_ATR = 0.02
    MIN_PROB = 0.0  # 絕對門檻，0.0 = 不過濾
    TOP_PCT = 0.2  # 分位數門檻：各模型自己前 20%，None = 不用

    # ── 切換模式 ─────────────────────────────────────────── #
    MODE = "train+eval"  # release | tune_xgb | "train+eval" | "eval_only" | "predict_today"
    # ────────────────────────────────────────────────────── #

    clf_gmm = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train+eval":
        print("建立訓練集...")
        df_train = build_dataset(
            clf_gmm, stocks, TRAIN_ST, TRAIN_END, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR
        )
        print("\n建立測試集...")
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)

        # 三模型 PK：同一訓練集、同一測試集
        print("\n" + "=" * 60)
        print("【RFC】")
        print("=" * 60)
        rfc = train_rfc(df_train)
        eval_rfc(rfc, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, min_prob=MIN_PROB, top_pct=TOP_PCT)

        print("\n" + "=" * 60)
        print("【XGBoost】")
        print("=" * 60)
        xgb = train_xgb(df_train)
        eval_rfc(xgb, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, min_prob=MIN_PROB, top_pct=TOP_PCT)

        print("\n" + "=" * 60)
        print("【LightGBM】")
        print("=" * 60)
        lgbm = train_lgbm(df_train)
        eval_rfc(lgbm, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, min_prob=MIN_PROB, top_pct=TOP_PCT)

        # OOS AUC 計算（用於加權）
        from sklearn.metrics import roc_auc_score as _auc

        def _oos_auc(model):
            avail = model._fitted_features
            X = df_test[avail].fillna(0.5).values
            return _auc(df_test["Y"].values, model.predict_proba(X), multi_class="ovr", average="macro")

        auc_xgb = _oos_auc(xgb)
        auc_lgbm = _oos_auc(lgbm)

        print("\n" + "=" * 60)
        print(f"【Ensemble XGB + LGBM  (AUC 加權 {auc_xgb:.4f} / {auc_lgbm:.4f})】")
        print("=" * 60)
        ensemble = EnsembleModel([xgb, lgbm], weights=[auc_xgb, auc_lgbm])
        eval_rfc(ensemble, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, min_prob=MIN_PROB, top_pct=TOP_PCT)

        save_rfc(rfc)

    elif MODE == "tune_xgb":
        print("建立訓練集...")
        df_train = build_dataset(
            clf_gmm, stocks, TRAIN_ST, TRAIN_END, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR
        )
        best_params = tune_xgb(df_train, n_trials=50, val_st="2022-01-01")

        print("\n用最佳參數在全訓練集重新訓練並 OOS 評估...")
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)
        avail = [c for c in RFC_FEATURES if c in df_train.columns]
        X_all = df_train[avail].fillna(0.5).values
        y_all = df_train["Y"].values
        xgb_tuned = XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        xgb_tuned.fit(X_all, y_all)
        xgb_tuned._fitted_features = avail
        auc_in = roc_auc_score(y_all, xgb_tuned.predict_proba(X_all), multi_class="ovr", average="macro")
        print(f"訓練集 AUC（macro OvR）：{auc_in:.4f}（in-sample）")

        print("\n" + "=" * 60)
        print("【XGBoost 調參版】")
        print("=" * 60)
        eval_rfc(xgb_tuned, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, top_pct=TOP_PCT)

        # 調參後的 XGB + 預設 LGBM 的 Ensemble
        print("\n建立 LGBM（預設）...")
        lgbm_default = train_lgbm(df_train)
        from sklearn.metrics import roc_auc_score as _auc

        def _oas(m):
            X = df_test[m._fitted_features].fillna(0.5).values
            return _auc(df_test["Y"].values, m.predict_proba(X), multi_class="ovr", average="macro")

        auc_xgb_t = _oas(xgb_tuned)
        auc_lgbm = _oas(lgbm_default)
        print("\n" + "=" * 60)
        print(f"【Ensemble 調參XGB + 預設LGBM  (AUC {auc_xgb_t:.4f} / {auc_lgbm:.4f})】")
        print("=" * 60)
        ensemble_tuned = EnsembleModel([xgb_tuned, lgbm_default], weights=[auc_xgb_t, auc_lgbm])
        eval_rfc(ensemble_tuned, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, top_pct=TOP_PCT)

    elif MODE == "release":
        # 正式 release：全訓練集（2015-2023）訓練 XGB + LGBM，組成 Ensemble 儲存
        print("建立 Release 訓練集（2015-2023）...")
        df_train = build_dataset(
            clf_gmm, stocks, TRAIN_ST, TRAIN_END, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR
        )
        print("\n訓練 XGBoost...")
        xgb_rel = train_xgb(df_train)
        print("\n訓練 LightGBM...")
        lgbm_rel = train_lgbm(df_train)

        from sklearn.metrics import roc_auc_score as _auc2

        auc_x = _auc2(
            df_train["Y"],
            xgb_rel.predict_proba(df_train[xgb_rel._fitted_features].fillna(0.5).values),
            multi_class="ovr",
            average="macro",
        )
        auc_l = _auc2(
            df_train["Y"],
            lgbm_rel.predict_proba(df_train[lgbm_rel._fitted_features].fillna(0.5).values),
            multi_class="ovr",
            average="macro",
        )

        ensemble_rel = EnsembleModel([xgb_rel, lgbm_rel], weights=[auc_x, auc_l])
        save_ensemble(ensemble_rel)
        print(f"\nRelease 完成：{ENSEMBLE_MODEL_PATH}")
        print(f"  XGB  in-sample AUC：{auc_x:.4f}  weight：{ensemble_rel.weights[0]:.4f}")
        print(f"  LGBM in-sample AUC：{auc_l:.4f}  weight：{ensemble_rel.weights[1]:.4f}")
        print(f"特徵：{ensemble_rel._fitted_features}")

    elif MODE == "eval_only":
        ensemble = load_ensemble()
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)
        eval_rfc(ensemble, df_test, top_n=TOP_N, hold_days=HOLD_DAYS, top_pct=TOP_PCT)

    elif MODE == "predict_today":
        predict_today_rfc(top_n=TOP_N, clusters=CLUSTERS)


"""
勝率 52-53% 的問題有幾個角度可以想：

方向一：信心門檻過濾
目前是每次換倉固定選 Top-10，不管信心高不高。改成「prob > 0.6 才進場，不夠就空倉或減倉」——LightGBM 60-70% 的勝率已經有 34.85%，對應到 future_return > 0 的勝率也會更高。

方向二：三模型 Ensemble
把三個模型的 prob 平均（或加權），投票一致才選。可以減少各個模型的雜訊，對 Top-N 的品質提升比較直接。

方向三：Y 標籤改法
現在 win rate 是量 future_return > 0，但 Y 學的是「10日內碰到 +10%」，兩件事不一樣。可以把 win rate 改成量「持有期間最高點 > 入場價 + N%」，讓回測指標和訓練目標對齊。

方向四：縮短持有天數
10 日持有的雜訊多，5 日動能更強。但 profit_label 的 +10% 門檻要一起調。



討論一下，那這個例子，cnn有沒有可以用的上的地方
可以用，但用途和現在的樹模型完全不同。

現在的模型（RFC/XGB/LGBM）：

輸入：當天的截面 scalar 特徵（xrank、slope、ratio 等）
學的是：這一天的「籌碼 + 技術狀態」→ 達標機率
CNN 能做的事：

1. 1D CNN on time series（最實用）
把過去 N 天的 OHLCV 直接當序列輸入：


[close_t-19, close_t-18, ... close_t-0]  ← 20天收盤
[volume_t-19, ...]
→ 1D CNN → 特徵萃取 → 分類
優點：不需要手工算 ma_slope、bias 這些，CNN 自己學時序模式。

2. 2D CNN on OHLCV matrix（K線圖）
把 K 線畫成圖像，CNN 識別視覺形態（頭肩、突破、旗形）：


[n_days × n_features] → 2D CNN → 分類
這個場景能不能用？

外資忽視股的特性是「靜悄悄地在動」——外資不買、不當沖，但股價可能在緩步收斂。這種型態恰好是 1D CNN 或圖形識別能捕捉的。

問題：

訓練樣本 110K 筆，對 CNN 來說剛好在邊界，不算多
訓練時間比樹模型長很多
解釋性差（不知道為什麼選這支）
最合適的定位： CNN 作為第 4 個模型加入 ensemble，提供樹模型看不到的時序形態資訊，四模型投票加權。

有興趣試嗎？還是先把目前三模型的門檻回測跑完？

目前樹模型的弱點，依重要性排：

1. 沒有做 ensemble（最直接的提升）
三個模型各自跑，沒有合起來。XGBoost 最強（AUC 0.567, 626%），但加上 LGBM 的高勝率（45.3%）做加權平均，理論上比單一模型更穩。

2. 超參數沒調過
目前三個模型都用預設值（n_estimators=200, max_depth=6, lr=0.05），XGBoost 特別值得花時間調，因為它是主力。

3. f_market_volatility_20d 佔 32% 讓人不安
幾乎所有選股能力都靠這個市場擇時特徵，代表模型本質上是「市場平靜就買忽視股」。這在 2024-2026 牛市成立，換個環境可能失效。需要驗證這個特徵是不是真的穩定，還是只是 OOS 期間的巧合。

4. 沒有 walk-forward 驗證
現在是單一訓練/測試切割（2015-2023 / 2024-2026），不知道 2024-2025 訓練是否對 2025-2026 OOS 仍有效。

最有效益的下一步：先做 ensemble（XGB + LGBM 加權平均），成本低、效果直接可量。要試嗎？
"""
