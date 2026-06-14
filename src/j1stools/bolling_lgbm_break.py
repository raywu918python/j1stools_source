"""
BollingLGBMBreak：BB上軌盤整後突破預測（LGBM Regression）

進場過濾條件（同時成立）：
    1. ZONE_DAYS 天的 close 都在 bb_mid ~ bb_upper 之間
    2. 其中最低1根K的 low 有碰到 bb_mid（下影線測中軌）
    3. ZONE_DAYS 根K的 max_high - min_low <= ZONE_RANGE_PCT

標籤（連續）：
    Y = 持有期（HOLD_DAYS）內最大收盤漲幅（fwd_max_return）
    回測時按預測值排序，取 pred_return >= THRESHOLD 的信號進場

特徵（信號當天，不需時序窗口）：
    bb_daily_return, bb_hl_range, bb_close_pos, bb_pos_in_band,
    bb_width_pct, bb_atr_pct, vol_ratio, ret_5d, ret_20d,
    bb_squeeze, dist_to_upper
"""

import numpy as np
import pandas as pd

BB_FEATURES = [
    # ── 原始 BB 特徵 ──
    "bb_daily_return",   # 當日報酬率
    "bb_hl_range",       # (high-low)/close
    "bb_close_pos",      # (close-low)/(high-low)
    "bb_pos_in_band",    # (close-lower)/width
    "bb_width_pct",      # (upper-lower)/mid
    "bb_atr_pct",        # ATR14/close
    # ── 新增：信號宇宙內仍有變異的特徵 ──
    "vol_ratio",         # 今日量 / 20日均量（縮量 vs 放量盤整）
    "ret_5d",            # 前5日漲幅（盤整前走勢強弱）
    "ret_20d",           # 前20日漲幅（中期趨勢）
    "bb_squeeze",        # bb_width / bb_width 20日均（帶寬收縮程度）
    "dist_to_upper",     # (upper-close)/close（距上軌還有多遠）
]

# ── 超參數 ────────────────────────────────────────────────────────────────── #
HOLD_DAYS = 10          # 持有天數：進場後持有幾日（標籤窗口 & 回測出場）
BB_PERIOD = 20          # BB 計算週期（20日移動平均）
BB_STD = 2.0            # BB 標準差倍數（上軌 = 中軌 + 2σ）
MIN_ATR_PCT = 0.02      # 最低 ATR 過濾：排除流動性差的股票（ATR14/close < 2% 不進場）
ZONE_DAYS = 3           # 盤整觀察天數：條件1&3 的滾動窗口大小
ZONE_RANGE_PCT = 0.06   # 盤整振幅上限：ZONE_DAYS 根K的 max_high-min_low / close <= 6%
THRESHOLD = 0.05        # 回測進場門檻：pred_return >= THRESHOLD 才發出信號

LGBM_PARAMS = {
    "objective": "regression",   # 回歸預測未來最大漲幅
    "metric": "rmse",            # 驗證集用 RMSE 做 early stopping
    "num_leaves": 63,            # 樹的葉節點數（複雜度控制，63 ≈ 深度6）
    "learning_rate": 0.03,       # 學習率（配合 n_estimators=1000）
    "min_child_samples": 30,     # 每個葉節點最少樣本數（防過擬合）
    "subsample": 0.8,            # 每棵樹隨機抽 80% 樣本（列採樣）
    "subsample_freq": 1,         # 每棵樹都做 subsample
    "colsample_bytree": 0.7,     # 每棵樹隨機抽 70% 特徵（欄採樣）
    "reg_alpha": 0.1,            # L1 正則化
    "reg_lambda": 1.0,           # L2 正則化
    "n_estimators": 1000,        # 最大樹數（early stopping 會提前停）
    "n_jobs": -1,                # 使用全部 CPU
    "verbose": -1,               # 關閉 LightGBM 預設輸出
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

    df["ret_5d"] = df.groupby("stock_id")["close"].transform(
        lambda x: x.pct_change(5, fill_method=None)
    ).clip(-0.5, 0.5)

    df["ret_20d"] = df.groupby("stock_id")["close"].transform(
        lambda x: x.pct_change(20, fill_method=None)
    ).clip(-0.8, 0.8)

    bb_width_ma20 = df.groupby("stock_id")["bb_width"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df["bb_squeeze"] = (df["bb_width"] / bb_width_ma20.replace(0, np.nan)).clip(0, 3)

    df["dist_to_upper"] = ((df["bb_upper"] - df["close"]) / df["close"].replace(0, np.nan)).clip(-0.1, 0.2)

    def _consolidation_signal(grp):
        in_zone = (grp["close"] >= grp["bb_mid"]) & (grp["close"] <= grp["bb_upper"])
        all_in_zone = in_zone.rolling(ZONE_DAYS, min_periods=ZONE_DAYS).min() >= 1
        touched_mid = (grp["low"] <= grp["bb_mid"]).astype(int)
        any_touch_mid = touched_mid.rolling(ZONE_DAYS, min_periods=ZONE_DAYS).max() >= 1
        roll_high = grp["high"].rolling(ZONE_DAYS, min_periods=ZONE_DAYS).max()
        roll_low = grp["low"].rolling(ZONE_DAYS, min_periods=ZONE_DAYS).min()
        hl_range = (roll_high - roll_low) / grp["close"].replace(0, np.nan)
        return (all_in_zone & any_touch_mid & (hl_range <= ZONE_RANGE_PCT)).astype(np.int8)

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

    def _break_label(grp):
        close = grp["close"].values.astype(np.float32)
        n = len(close)
        padded = np.concatenate([close, np.full(hold_days, np.nan, dtype=np.float32)])
        stacked = np.stack([padded[h : h + n] for h in range(1, hold_days + 1)], axis=1)
        with np.errstate(all="ignore"):
            max_future = np.nanmax(stacked, axis=1)
        fwd_ret = (max_future - close) / np.clip(close, 1e-6, None)
        fwd_ret[-hold_days:] = np.nan
        return pd.Series(fwd_ret.astype(np.float32), index=grp.index)

    df_all["Y"] = df_all.groupby("stock_id", group_keys=False).apply(_break_label, include_groups=False)

    df_signal = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["date"] < pd.Timestamp(end))
            & (df_all["bb_consolidation_signal"] == 1)
            & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
            & df_all["Y"].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_signal["Y"] = df_signal["Y"].astype(np.float32)

    y_arr = df_signal["Y"].values
    print(
        f"盤整信號：{len(df_signal):,}  "
        f"Y mean={y_arr.mean():.3f}  p25={np.percentile(y_arr,25):.3f}  "
        f"p50={np.median(y_arr):.3f}  p75={np.percentile(y_arr,75):.3f}  max={y_arr.max():.3f}"
    )

    X = df_signal[BB_FEATURES].fillna(0).values.astype(np.float32)
    y = df_signal["Y"].values
    return X, y, df_signal


# ── LGBM 回歸器 ──────────────────────────────────────────────────────────── #


class BollingBreakLGBM:
    """BB上軌盤整後突破 LGBM 回歸器（預測未來最大漲幅）。"""

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

        self._model = lgb.LGBMRegressor(**self._params)

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
            pred_v = self._model.predict(X_val_df)
            ic = np.corrcoef(pred_v, y_val)[0, 1]
            print(f"  val IC={ic:.4f}  pred 分布：min={pred_v.min():.3f}  p50={np.median(pred_v):.3f}  max={pred_v.max():.3f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳預測的未來最大漲幅（連續值）。"""
        assert self._model is not None
        X_df = pd.DataFrame(X, columns=BB_FEATURES)
        return self._model.predict(X_df)


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_bolling_break_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> BollingBreakLGBM:
    print(f"\n── 訓練 BollingBreakLGBM (Regression) ──")
    clf = BollingBreakLGBM()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_ic = np.corrcoef(clf.predict(X_train), y_train)[0, 1]
    print(f"訓練集 IC：{train_ic:.4f}（in-sample）")
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
        return pd.DataFrame(columns=["date", "stock_id", "pred_return"])

    X = df_scan[BB_FEATURES].fillna(0).values.astype(np.float32)
    pred_return = clf.predict(X)

    signal = df_scan[["date", "stock_id"]].copy()
    signal["pred_return"] = pred_return
    signal["2"] = pred_return
    signal["date"] = pd.to_datetime(signal["date"])

    print(
        f"  pred_return 分布：mean={pred_return.mean():.3f}  p50={np.median(pred_return):.3f}  p90={np.percentile(pred_return,90):.3f}"
    )
    signal = signal[signal["pred_return"] >= threshold].reset_index(drop=True)
    print(f"  信號筆數（pred_return>={threshold}）：{len(signal):,}", end="")
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
    MODE = "train"  # "train" | "eval" | "backtest"

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
        clf, hold_days_loaded = load_bolling_break_lgbm()
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_break_dataset(stocks, EVAL_ST, hold_days=hold_days_loaded)

        pred = clf.predict(X_test)
        ic = np.corrcoef(pred, y_test)[0, 1]
        print(f"\nOOS IC：{ic:.4f}  （n={len(y_test):,}）")
        print(f"pred 分布：min={pred.min():.3f}  p25={np.percentile(pred,25):.3f}  "
              f"p50={np.median(pred):.3f}  p75={np.percentile(pred,75):.3f}  max={pred.max():.3f}")

        # Decile 分析：按預測值分10組，看各組真實報酬均值
        print(f"\n{'─'*52}")
        print(f"{'pred_return decile':>20}  {'真實平均漲幅':>12}  {'筆數':>7}")
        print(f"{'─'*52}")
        df_eval = pd.DataFrame({"pred": pred, "Y": y_test})
        df_eval["decile"] = pd.qcut(pred, q=10, labels=False, duplicates="drop")
        for d, grp in df_eval.groupby("decile"):
            print(f"{d:>20}  {grp['Y'].mean():>12.3f}  {len(grp):>7,}")
        print(f"{'─'*52}")
        print(f"全體均值：{y_test.mean():.3f}")

        plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(pred, y_test, alpha=0.1, s=5, color="steelblue")
        axes[0].set_xlabel("pred_return")
        axes[0].set_ylabel("actual max_return")
        axes[0].set_title(f"預測 vs 實際（IC={ic:.4f}）")

        mean_by_decile = df_eval.groupby("decile")["Y"].mean()
        axes[1].bar(mean_by_decile.index, mean_by_decile.values, color="seagreen", alpha=0.8)
        axes[1].axhline(y_test.mean(), color="gray", linestyle="--", linewidth=0.8, label="全體均值")
        axes[1].set_xlabel("pred_return decile (0=lowest)")
        axes[1].set_ylabel("真實平均漲幅")
        axes[1].set_title("Mean actual return by decile (OOS)")
        axes[1].legend()
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
