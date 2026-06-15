"""
BollingLGBMBreak：BB上軌盤整後突破預測（LGBM Binary Classification）

進場過濾條件（時序，依序發生）：
    1. 近期 CROSS_LOOKBACK 根K內，low 碰到 bb_lower（觸及下軌）
    2. 近期 ZONE_DAYS 根K內，high 碰到 bb_mid（從下軌反彈至中軌）
    3. 近 ZONE_DAYS 天收盤漲幅 <= ZONE_RANGE_PCT（碰中軌後蓄勢，不急漲）

標籤（profit_label 二分類）：
    Y=1：持有期內先觸及 +PROFIT_TARGET 且未先觸及 -STOP_LOSS（真突破）
    Y=0：觸及止損 或 持有期滿未達標（失敗）
    回測時取 up_prob >= THRESHOLD 的信號進場

特徵（信號當天，不需時序窗口）：
    bb_daily_return, bb_hl_range, bb_close_pos, bb_pos_in_band,
    bb_width_pct, bb_atr_pct, vol_ratio, ret_5d, ret_20d,
    bb_squeeze, dist_to_upper
"""

import numpy as np
import pandas as pd

BB_FEATURES = [
    # ── 原始 BB 特徵 ──
    "bb_daily_return",  # 當日報酬率
    "bb_hl_range",  # (high-low)/close
    "bb_close_pos",  # (close-low)/(high-low)
    "bb_pos_in_band",  # (close-lower)/width
    "bb_width_pct",  # (upper-lower)/mid
    "bb_atr_pct",  # ATR14/close
    # ── 新增：信號宇宙內仍有變異的特徵 ──
    "vol_ratio",  # 今日量 / 20日均量（縮量 vs 放量盤整）
    "ret_5d",  # 前5日漲幅（盤整前走勢強弱）
    "ret_20d",  # 前20日漲幅（中期趨勢）
    "bb_squeeze",  # bb_width / bb_width 20日均（帶寬收縮程度）
    "dist_to_upper",  # (upper-close)/close（距上軌還有多遠）
]

# ── 超參數 ────────────────────────────────────────────────────────────────── #
HOLD_DAYS = 10  # 持有天數：進場後持有幾日（標籤窗口 & 回測出場）
BB_PERIOD = 20  # BB 計算週期（20日移動平均）
BB_STD = 2.0  # BB 標準差倍數（上軌 = 中軌 + 2σ）
MIN_ATR_PCT = 0.02  # 最低 ATR 過濾：排除流動性差的股票（ATR14/close < 2% 不進場）
ZONE_DAYS = 3  # 盤整觀察天數：條件1&3 的滾動窗口大小
ZONE_RANGE_PCT = 0.06  # 盤整振幅上限：ZONE_DAYS 根K的 max_high-min_low / close <= 6%
CROSS_LOOKBACK = 10  # 條件1：往回幾根K內要有觸及下軌（lower touch lookback）
PROFIT_TARGET = 0.1  # 止盈門檻：持有期內漲幅達 8% 視為真突破（profit_label class 2）
STOP_LOSS = 0.1  # 止損門檻：持有期內跌幅達 5% 視為失敗（profit_label class 1）
THRESHOLD = 0.5 # 回測進場門檻：up_prob >= THRESHOLD 才發出信號（約 top 25%，p75）

LGBM_PARAMS = {
    "objective": "binary",         # 二分類：預測突破成功機率
    "metric": "auc",               # 驗證集用 AUC 做 early stopping
    "num_leaves": 31,              # 樹的葉節點數（63→31，降低複雜度）
    "learning_rate": 0.01,         # 學習率（0.03→0.01，更小步伐配合 early stopping）
    "min_child_samples": 80,       # 每葉最少樣本數（30→80，最直接防過擬合）
    "subsample": 0.8,              # 每棵樹隨機抽 80% 樣本（列採樣）
    "subsample_freq": 1,           # 每棵樹都做 subsample
    "colsample_bytree": 0.6,       # 每棵樹隨機抽特徵（0.7→0.6，增加多樣性）
    "reg_alpha": 0.1,              # L1 正則化
    "reg_lambda": 5.0,             # L2 正則化（1.0→5.0，更強正則）
    "n_estimators": 2000,          # 最大樹數（配合更小 LR）
    "n_jobs": -1,                  # 使用全部 CPU
    "verbose": -1,                 # 關閉 LightGBM 預設輸出
}

# ── 原始特徵計算（與 bolling_cnn_break 完全相同）─────────────────────────── #


def build_bb_daily_features(stocks: list, st: str, end: str = "2099-01-01"):
    """
    計算 BB_FEATURES 及 ATR，並標記 bb_consolidation_signal。
    回傳含 date, stock_id, high, low, close, *BB_FEATURES,
    atr14_pct, bb_upper, bb_consolidation_signal 的 DataFrame。
    """
    import time as _time
    from j1stools import parquet_db

    _t0 = _time.time()
    print(f"  [BollingBreak] 載入 price（{len(stocks)} 檔，{st}~{end}）...")

    df = parquet_db.query_price(stocks, st, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"])

    df["bb_daily_return"] = df.groupby("stock_id")["close"].pct_change(fill_method=None)

    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["bb_hl_range"] = hl / df["close"].replace(0, np.nan)
    df["bb_close_pos"] = (df["close"] - df["low"]) / hl

    def _atr14(grp):
        tr = pd.concat(
            [
                grp["high"] - grp["low"],
                (grp["high"] - grp["close"].shift(1)).abs(),
                (grp["low"] - grp["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(14, min_periods=5).mean() / grp["close"].replace(0, np.nan)

    df["atr14_pct"] = df.groupby("stock_id", group_keys=False).apply(_atr14, include_groups=False)
    df["bb_atr_pct"] = df["atr14_pct"]

    _pv = df.pivot(index="date", columns="stock_id", values="close")
    _bb_mid = _pv.rolling(BB_PERIOD, min_periods=10).mean()
    _bb_std = _pv.rolling(BB_PERIOD, min_periods=10).std(ddof=0)
    _bb_upper = _bb_mid + BB_STD * _bb_std
    _bb_lower = _bb_mid - BB_STD * _bb_std
    _bb_width = (_bb_upper - _bb_lower).replace(0, np.nan)

    def _stack_col(df_wide, name):
        return df_wide.stack(future_stack=True).rename(name).reset_index()

    df = df.merge(_stack_col(_bb_lower, "bb_lower"), on=["date", "stock_id"], how="left")
    df = df.merge(_stack_col(_bb_mid, "bb_mid"), on=["date", "stock_id"], how="left")
    df = df.merge(_stack_col(_bb_upper, "bb_upper"), on=["date", "stock_id"], how="left")
    df = df.merge(_stack_col(_bb_width, "bb_width"), on=["date", "stock_id"], how="left")

    df["bb_pos_in_band"] = (df["close"] - df["bb_lower"]) / df["bb_width"]
    df["bb_width_pct"] = df["bb_width"] / df["bb_mid"].replace(0, np.nan)

    df["bb_daily_return"] = df["bb_daily_return"].clip(-0.3, 0.3)
    df["bb_pos_in_band"] = df["bb_pos_in_band"].clip(-0.5, 2.5)
    df["bb_width_pct"] = df["bb_width_pct"].clip(0, 0.5)
    df["bb_atr_pct"] = df["bb_atr_pct"].clip(0, 0.3)

    # ── 新增特徵 ──
    vol_ma20 = df.groupby("stock_id")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["vol_ratio"] = (df["volume"] / vol_ma20.replace(0, np.nan)).clip(0, 10)

    df["ret_5d"] = (
        df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(5, fill_method=None)).clip(-0.5, 0.5)
    )

    df["ret_20d"] = (
        df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(20, fill_method=None)).clip(-0.8, 0.8)
    )

    bb_width_ma20 = df.groupby("stock_id")["bb_width"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["bb_squeeze"] = (df["bb_width"] / bb_width_ma20.replace(0, np.nan)).clip(0, 3)

    df["dist_to_upper"] = ((df["bb_upper"] - df["close"]) / df["close"].replace(0, np.nan)).clip(-0.1, 0.2)

    def _consolidation_signal(grp):
        # 條件1：近期 CROSS_LOOKBACK 根K內，low 碰到下軌
        recent_lower = (grp["low"] <= grp["bb_lower"]).rolling(CROSS_LOOKBACK, min_periods=1).max() >= 1
        # 條件2：近期 ZONE_DAYS 根K內，high 碰到中軌（從下軌反彈至中軌）
        any_touch_mid = (grp["high"] >= grp["bb_mid"]).rolling(ZONE_DAYS, min_periods=ZONE_DAYS).max() >= 1
        # 條件3：近 ZONE_DAYS 天收盤漲幅 <= ZONE_RANGE_PCT（碰中軌後蓄勢不急漲）
        gain_3d = grp["close"].pct_change(ZONE_DAYS, fill_method=None)
        small_gain = gain_3d <= ZONE_RANGE_PCT
        return (recent_lower & any_touch_mid & small_gain).astype(np.int8)

    df["bb_consolidation_signal"] = df.groupby("stock_id", group_keys=False).apply(
        _consolidation_signal, include_groups=False
    )

    cols = [
        "date",
        "stock_id",
        "high",
        "low",
        "close",
        "atr14_pct",
        "bb_upper",
        "bb_consolidation_signal",
    ] + BB_FEATURES
    print(f"  [BollingBreak] 特徵完成（{_time.time()-_t0:.1f}s），共 {len(df):,} 行")
    return df[cols].sort_values(["stock_id", "date"]).reset_index(drop=True)


# ── 資料集（信號當天特徵，無需時序窗口）─────────────────────────────────── #


def build_break_dataset(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = MIN_ATR_PCT,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    回傳 (X, y, df_meta)：
        X      : (n, n_features) float32，信號當天的 BB_FEATURES
        y      : (n,) int，0=未突破, 1=突破
        df_meta: 對應 DataFrame（含 date, stock_id, Y）
    """
    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=60)).strftime("%Y-%m-%d")
    end_buf = (pd.Timestamp(end) + pd.DateOffset(days=hold_days * 2)).strftime("%Y-%m-%d")

    print(f"── 載入 BB 原始特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_bb_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    from j1stools.label_builder import profit_label as _profit_label

    df_all = _profit_label(df_all, hold_days=hold_days, profit_target=PROFIT_TARGET, stop_loss=-STOP_LOSS)
    # Y=1：真突破（hit TP 未先 hit SL），Y=0：失敗（止損或盤整）
    df_all["Y"] = (df_all["target"] == 2).astype(np.int8)
    df_all = df_all.drop(columns=["target"])

    df_signal = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["date"] < pd.Timestamp(end))
            & (df_all["bb_consolidation_signal"] == 1)
            & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
        ]
        .copy()
        .reset_index(drop=True)
    )

    n_pos = df_signal["Y"].sum()
    pos_rate = df_signal["Y"].mean()
    print(f"盤整信號：{len(df_signal):,}  Y=1（突破）：{n_pos:,} 筆  正例率：{pos_rate:.1%}")

    X = df_signal[BB_FEATURES].fillna(0).values.astype(np.float32)
    y = df_signal["Y"].values.astype(np.int8)
    return X, y, df_signal


# ── LGBM 分類器 ──────────────────────────────────────────────────────────── #


class BollingBreakLGBM:
    """BB上軌盤整後突破 LGBM 二分類器（預測突破成功機率）。"""

    def __init__(self, params: dict | None = None):
        self._params = {**LGBM_PARAMS, **(params or {})}
        self._model = None
        self._best_iter: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BollingBreakLGBM":
        import lightgbm as lgb

        # 自動計算 scale_pos_weight 平衡正負例
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        params = {**self._params, "scale_pos_weight": n_neg / max(n_pos, 1)}

        self._model = lgb.LGBMClassifier(**params)

        X_df = pd.DataFrame(X, columns=BB_FEATURES)
        X_val_df = pd.DataFrame(X_val, columns=BB_FEATURES) if X_val is not None else None

        fit_kwargs = {}
        if X_val_df is not None:
            fit_kwargs["eval_set"] = [(X_val_df, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

        self._model.fit(X_df, y, **fit_kwargs)
        self._best_iter = getattr(self._model, "best_iteration_", self._params["n_estimators"])

        imp = sorted(zip(BB_FEATURES, self._model.feature_importances_), key=lambda x: -x[1])
        print("  特徵重要度：" + "  ".join(f"{n}={v}" for n, v in imp))

        if X_val_df is not None:
            from sklearn.metrics import roc_auc_score

            up_prob_v = self._model.predict_proba(X_val_df)[:, 1]
            auc = roc_auc_score(y_val, up_prob_v)
            print(
                f"  val AUC={auc:.4f}  up_prob 分布：min={up_prob_v.min():.3f}  "
                f"p50={np.median(up_prob_v):.3f}  max={up_prob_v.max():.3f}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳突破成功機率（up_prob）。"""
        assert self._model is not None
        X_df = pd.DataFrame(X, columns=BB_FEATURES)
        return self._model.predict_proba(X_df)[:, 1]


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_bolling_break_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> BollingBreakLGBM:
    from sklearn.metrics import roc_auc_score

    print(f"\n── 訓練 BollingBreakLGBM (Binary) ──")
    clf = BollingBreakLGBM()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_auc = roc_auc_score(y_train, clf.predict(X_train))
    print(f"訓練集 AUC：{train_auc:.4f}（in-sample）")
    return clf


# ── 存取模型 ──────────────────────────────────────────────────────────────── #

BOLLING_BREAK_LGBM_PATH = "db/models/bolling_break_lgbm.joblib"


def save_bolling_break_lgbm(
    clf: BollingBreakLGBM, path: str = BOLLING_BREAK_LGBM_PATH, hold_days: int = HOLD_DAYS
) -> None:
    import joblib, os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": clf._model, "hold_days": hold_days}, path)
    print(f"BollingBreakLGBM 已存：{path}（hold_days={hold_days}）")


def load_bolling_break_lgbm(path: str = BOLLING_BREAK_LGBM_PATH) -> tuple[BollingBreakLGBM, int]:
    import joblib

    state = joblib.load(path)
    clf = BollingBreakLGBM()
    clf._model = state["model"]
    hold_days = state.get("hold_days", HOLD_DAYS)
    print(f"BollingBreakLGBM 已載入：{path}（hold_days={hold_days}）")
    return clf, hold_days


# ── 信號生成（回測用）────────────────────────────────────────────────────── #


def make_signal_bolling_break_lgbm(
    clf: BollingBreakLGBM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    min_atr_pct: float = MIN_ATR_PCT,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=60)).strftime("%Y-%m-%d")
    df_all = build_bb_daily_features(stocks, st_buf, end)
    df_all["date"] = pd.to_datetime(df_all["date"])

    df_scan = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["bb_consolidation_signal"] == 1)
            & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
        ]
        .copy()
        .reset_index(drop=True)
    )
    print(f"  盤整信號：{len(df_scan):,} 筆")

    if df_scan.empty:
        return pd.DataFrame(columns=["date", "stock_id", "up_prob"])

    X = df_scan[BB_FEATURES].fillna(0).values.astype(np.float32)
    up_prob = clf.predict(X)

    signal = df_scan[["date", "stock_id"]].copy()
    signal["up_prob"] = up_prob
    signal["2"] = up_prob  # backtest_platform 排序用
    signal["date"] = pd.to_datetime(signal["date"])

    print(
        f"  up_prob 分布：mean={up_prob.mean():.3f}  p50={np.median(up_prob):.3f}  p90={np.percentile(up_prob,90):.3f}"
    )
    signal = signal[signal["up_prob"] >= threshold].reset_index(drop=True)
    print(f"  信號筆數（up_prob>={threshold}）：{len(signal):,}", end="")
    if len(signal) > 0:
        print(f"  日期：{signal['date'].min().date()} ~ {signal['date'].max().date()}")
    else:
        print()
    return signal


# ── Main ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import matplotlib

    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    from j1stools import backtest_platform, j1s_chart, parquet_db

    TRAIN_ST = "2015-01-01"
    EVAL_ST = "2024-01-01"
    MODE = "eval"  # "train" | "eval" | "backtest"

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train":
        print("\n══ 建立訓練集 ══")
        X_train, y_train, df_tr = build_break_dataset(stocks, TRAIN_ST, EVAL_ST, hold_days=HOLD_DAYS)
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_break_dataset(stocks, EVAL_ST, hold_days=HOLD_DAYS)
        print(f"訓練集：{X_train.shape}  測試集：{X_test.shape}")

        clf = train_bolling_break_lgbm(X_train, y_train, X_val=X_test, y_val=y_test)
        save_bolling_break_lgbm(clf, hold_days=HOLD_DAYS)

    elif MODE == "eval":
        from sklearn.metrics import roc_auc_score

        clf, hold_days_loaded = load_bolling_break_lgbm()
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_break_dataset(stocks, EVAL_ST, hold_days=hold_days_loaded)

        up_prob = clf.predict(X_test)
        auc = roc_auc_score(y_test, up_prob)
        base_rate = y_test.mean()
        print(f"\nOOS AUC：{auc:.4f}  （n={len(y_test):,}，正例率={base_rate:.1%}）")
        print(
            f"up_prob 分布：min={up_prob.min():.3f}  p25={np.percentile(up_prob,25):.3f}  "
            f"p50={np.median(up_prob):.3f}  p75={np.percentile(up_prob,75):.3f}  max={up_prob.max():.3f}"
        )

        # Threshold 分析：不同 threshold 對應的信號數和精度
        print(f"\n{'─'*54}")
        print(f"{'THRESHOLD':>10}  {'信號數':>8}  {'精度':>8}  {'提升':>6}")
        print(f"{'─'*54}")
        pcts = np.arange(50, 96, 5)  # p50, p55, ..., p95
        thr_vals = np.unique(np.round(np.percentile(up_prob, pcts), 3))
        for thr in thr_vals:
            mask = up_prob >= thr
            n = int(mask.sum())
            if n == 0:
                continue
            prec = y_test[mask].mean()
            print(f"{thr:>10.3f}  {n:>8,}  {prec:>8.1%}  {prec/base_rate:>6.2f}x")
        print(f"{'─'*54}")

        # 圖：左＝分布，右＝threshold vs 精度曲線
        plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(up_prob[y_test == 0], bins=40, alpha=0.6, color="steelblue", label="Y=0（失敗）", density=True)
        axes[0].hist(up_prob[y_test == 1], bins=40, alpha=0.6, color="tomato", label="Y=1（突破）", density=True)
        axes[0].set_xlabel("up_prob")
        axes[0].set_ylabel("density")
        axes[0].set_title(f"up_prob 分布（AUC={auc:.4f}）")
        axes[0].legend()

        # 右圖：threshold sweep（從 p30 到 p97，每 1%）
        sweep_pcts = np.arange(30, 98, 1)
        sweep_thrs = np.percentile(up_prob, sweep_pcts)
        sweep_prec = []
        sweep_n = []
        for t in sweep_thrs:
            mask = up_prob >= t
            n = mask.sum()
            sweep_prec.append(y_test[mask].mean() if n >= 30 else np.nan)
            sweep_n.append(n)

        ax1 = axes[1]
        ax2 = ax1.twinx()
        ax1.plot(sweep_thrs, sweep_prec, color="seagreen", linewidth=2, label="精度")
        ax1.axhline(base_rate, color="gray", linestyle="--", linewidth=0.8, label=f"基準 {base_rate:.1%}")
        ax2.fill_between(sweep_thrs, sweep_n, alpha=0.15, color="steelblue")
        ax2.set_ylabel("信號數", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_xlabel("THRESHOLD")
        ax1.set_ylabel("精度（突破率）")
        ax1.set_title("Precision vs THRESHOLD (OOS)")
        ax1.legend(loc="upper left")

        plt.suptitle(f"BollingBreakLGBM | {hold_days_loaded}d | BB上軌盤整突破")
        plt.tight_layout()
        plt.show()

    elif MODE == "backtest":
        clf, hold_days_loaded = load_bolling_break_lgbm()
        print("\n══ 獨立回測 ══")
        backtest_platform.IS_USE_CACHE = True
        sig = make_signal_bolling_break_lgbm(clf, stocks, st=EVAL_ST, threshold=THRESHOLD)

        print(f"\n{'='*60}\n【BollingBreakLGBM 高預測報酬 + 持有{hold_days_loaded}日】\n{'='*60}")
        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=5,
            threshold=THRESHOLD,
            max_positions=3,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=0.05,
            use_fixed_tp=True,
            tp_stop=0.08,
            use_hold_days=True,
            hold_days=hold_days_loaded,
            group_limit=99,
            min_volume=200,
            use_fixed_sl_tp=False,
        )
        count = 20
        if td is not None and len(td) > 0:
            print(f"\n最近 {count} 筆交易（共 {len(td):,} 筆）：")
            print(td.tail(count).to_string(index=False))
        j1s_chart.plot_performance(pv, td)

    else:
        print(f"未知 MODE：{MODE!r}，請選 train / eval / backtest")
