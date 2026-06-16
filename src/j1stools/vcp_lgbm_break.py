"""
VcpLGBMBreak：波動壓縮後放量突破預測（LGBM Binary Classification）

進場過濾條件（VCP - Volatility Contraction Pattern）：
    1. ATR 壓縮：短期 ATR(5) / 長期 ATR(20) < ATR_RATIO（波動收縮）
    2. 橫盤整理：前 CONSOL_DAYS 根K高低點範圍 <= RANGE_PCT（價格緊縮）
    3. 價格突破：收盤 > 前 CONSOL_DAYS 根K最高高點（突破整理區上軌）
    4. 放量確認：今日成交量 > 前 CONSOL_DAYS 均量 × VOL_RATIO（帶量突破）

標籤（profit_label 二分類）：
    Y=1：持有期內先觸及 +PROFIT_TARGET 且未先觸及 -STOP_LOSS（真突破）
    Y=0：觸及止損 或 持有期滿未達標（失敗）
    回測時取 up_prob >= THRESHOLD 的信號進場

特徵（12 個）：
    atr_ratio, range_pct, vol_ratio, vol_dry_trend, break_dist,
    consol_ret, atr_pct, ma60_dist, ret_20d, ret_60d, up_day_ratio, close_pos
"""

import numpy as np
import pandas as pd

VCP_FEATURES = [
    # ── 價格/波動 ──
    "atr_ratio",      # 短ATR / 長ATR（壓縮程度）
    "range_pct",      # 整理區高低點範圍
    "vol_dry_trend",  # 整理期量能遞減趨勢
    "break_dist",     # (close - consol_high) / consol_high
    "consol_ret",     # 整理期間收益率
    "atr_pct",        # ATR14 / close
    "ma60_dist",      # (close - MA60) / MA60
    "ret_20d",        # 前20日報酬率
    "ret_60d",        # 前60日報酬率
    "up_day_ratio",   # 整理期上漲日比例
    "close_pos",      # 今日K線強度
    "high_level_pct", # 整理頂部 / 前120日最高點
    "days_since_big_move", # 距上次20日漲幅>20%有幾天（近=過熱，30~60天=VCP甜蜜點）
    # ── 籌碼 ──
    "f_ib_net_pct",   # 外資淨買超 / 20日均量（正=外資買，負=外資賣）
    "f_ib_5d",        # 5日外資累積淨買超 / 20日均量
    "f_margin_chg",   # 融資餘額日變化%（正=散戶加碼）
    "f_short_ratio",  # 券資比（高=空頭壓力大）
    "f_dt_ratio",     # 當沖占成交量比例
    "f_dt_net",       # 當沖方向（正=買當>賣當）
    # ── 大盤（0050）──
    "mkt_ret_5d",     # 大盤5日報酬
    "mkt_ret_20d",    # 大盤20日報酬（中期趨勢）
    "mkt_ma60_dist",  # 大盤 vs MA60 距離（多/空環境）
    "mkt_atr_ratio",  # 大盤波動收縮程度
    "mkt_high_pct",   # 大盤 vs 120日最高（強弱位置）
]

# ── 超參數 ────────────────────────────────────────────────────────────────── #
HOLD_DAYS     = 20    # 持有天數
CONSOL_DAYS   = 10    # 整理觀察窗口（10個交易日，約2週）
ATR_SHORT     = 5     # 短期ATR週期（僅用於特徵，不作為信號條件）
ATR_LONG      = 20    # 長期ATR週期（僅用於特徵）
RANGE_PCT     = 0.08  # 整理區高低點差距上限（8%以內，更嚴格橫盤）
VOL_RATIO     = 1.0   # 突破日放量門檻（均量的1.5倍，確保放量）
MIN_ATR_PCT   = 0.02  # 最低ATR過濾（排除低波動股）
PROFIT_TARGET = 0.10  # 止盈門檻
STOP_LOSS     = 0.10  # 止損門檻（break-even = 10/(10+10) = 50%）
HIGH_LEVEL    = 0.90  # 整理頂部需達前120日最高的90%以上（更靠近高點）
THRESHOLD     = 0.6  # 回測進場門檻（VCP基準46%，找模型能提升的區間）
MODEL_TYPE    = "xgb"  # "lgbm" | "xgb" | "ensemble"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.01,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.01,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}

# ── 特徵計算 ──────────────────────────────────────────────────────────────── #


def _compute_vcp(grp: pd.DataFrame) -> pd.DataFrame:
    """Per-stock VCP 信號與特徵（整合在同一次 groupby apply）。"""
    close  = grp["close"]
    high   = grp["high"]
    low    = grp["low"]
    volume = grp["volume"]
    open_p = grp["open"] if "open" in grp.columns else close.shift(1)

    # ── ATR ──
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr5  = tr.rolling(ATR_SHORT, min_periods=3).mean()
    atr20 = tr.rolling(ATR_LONG,  min_periods=10).mean()
    atr14 = tr.rolling(14,        min_periods=5).mean()

    # ── 整理期指標（shift(1)：不含今天，純看前 CONSOL_DAYS 天）──
    half = max(CONSOL_DAYS // 2, 3)
    min_p = max(CONSOL_DAYS // 2, 5)

    consol_high = high.shift(1).rolling(CONSOL_DAYS, min_periods=min_p).max()
    consol_low  = low.shift(1).rolling(CONSOL_DAYS,  min_periods=min_p).min()
    vol_ma_c    = volume.shift(1).rolling(CONSOL_DAYS, min_periods=min_p).mean()

    # ── 整理在高點判斷（需提前算，信號條件4用到）──
    high_120      = high.shift(1).rolling(120, min_periods=60).max()
    cond_high_lvl = consol_high / high_120.replace(0, np.nan) >= HIGH_LEVEL

    # ── 信號條件（4個：橫盤 + 突破前高 + 放量 + 在高點盤整）──
    cond_range = (consol_high - consol_low) / consol_low.replace(0, np.nan) <= RANGE_PCT
    cond_break = close > consol_high
    cond_vol   = volume > vol_ma_c * VOL_RATIO

    signal = (cond_range & cond_break & cond_vol & cond_high_lvl).astype(np.int8)

    # ── 特徵 ──
    atr_pct    = (atr14 / close.replace(0, np.nan)).clip(0, 0.3)
    atr_ratio  = (atr5.shift(1) / atr20.shift(1).replace(0, np.nan)).clip(0, 2)
    range_pct  = ((consol_high - consol_low) / consol_low.replace(0, np.nan)).clip(0, 0.5)
    vol_ratio  = (volume / vol_ma_c.replace(0, np.nan)).clip(0, 10)
    break_dist = ((close - consol_high) / consol_high.replace(0, np.nan)).clip(-0.05, 0.15)

    # 整理期間漲幅：整理起點 close vs 整理終點 close（shift(1)）
    consol_start = close.shift(CONSOL_DAYS)
    consol_end   = close.shift(1)
    consol_ret   = ((consol_end - consol_start) / consol_start.replace(0, np.nan)).clip(-0.5, 0.5)

    # 距上次「20日漲幅超過20%」有幾天（近=過熱，遠=有段時間了，適合VCP）
    ret20 = close.pct_change(20, fill_method=None)
    _big_move = (ret20 >= 0.20).values.astype(float)
    _n = len(_big_move)
    _pos = np.where(_big_move == 1, np.arange(_n, dtype=float), np.nan)
    _last = pd.Series(_pos).ffill().values
    _days = np.where(np.isnan(_last), 200.0, np.arange(_n, dtype=float) - _last)
    days_since_big_move = pd.Series(_days.clip(0, 200), index=close.index).shift(1).fillna(200)

    # MA60 距離
    ma60     = close.rolling(60, min_periods=30).mean()
    ma60_dist = ((close - ma60) / ma60.replace(0, np.nan)).clip(-0.5, 0.5)

    # 動能
    ret_20d = close.pct_change(20, fill_method=None).clip(-0.8, 0.8)
    ret_60d = close.pct_change(60, fill_method=None).clip(-1.0, 1.0)

    # 整理期上漲日比例（close > open）
    up_day       = (close > open_p).astype(float)
    up_day_ratio = up_day.shift(1).rolling(CONSOL_DAYS, min_periods=min_p).mean()

    # 量能遞減趨勢：前半段均量 vs 後半段均量（正值 = 量在縮，好信號）
    vol_first     = volume.shift(half + 1).rolling(half, min_periods=half // 2).mean()
    vol_second    = volume.shift(1).rolling(half, min_periods=half // 2).mean()
    vol_sum       = (vol_first + vol_second).replace(0, np.nan)
    vol_dry_trend = ((vol_first - vol_second) / vol_sum).clip(-1, 1)

    # 今日K線強度
    hl        = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / hl).clip(0, 1)

    # 整理頂部 vs 前120日最高（接近1.0 = 在高點盤整，遠低於1.0 = 低點反彈）
    high_level_pct = (consol_high / high_120.replace(0, np.nan)).clip(0.5, 1.1)

    return pd.DataFrame(
        {
            "vcp_signal":    signal,
            "atr_pct":       atr_pct,
            "atr_ratio":     atr_ratio,
            "range_pct":     range_pct,
            "vol_ratio":     vol_ratio,
            "break_dist":    break_dist,
            "consol_ret":    consol_ret,
            "ma60_dist":     ma60_dist,
            "ret_20d":       ret_20d,
            "ret_60d":       ret_60d,
            "up_day_ratio":  up_day_ratio,
            "vol_dry_trend": vol_dry_trend,
            "close_pos":           close_pos,
            "high_level_pct":      high_level_pct,
            "days_since_big_move": days_since_big_move,
        },
        index=grp.index,
    )


def _compute_chips(df: pd.DataFrame, stocks: list, st: str, end: str) -> pd.DataFrame:
    """載入並計算外資/融資/當沖籌碼特徵，合併回 df（按 date+stock_id）。"""
    from j1stools import parquet_db

    # ── 外資 ──
    df_ib = parquet_db.query_ib(stocks, st, end)
    df_ib["date"] = pd.to_datetime(df_ib["date"])
    df_ib["net"] = df_ib["buy"] - df_ib["sell"]
    ib_piv = (
        df_ib[df_ib["name"] == "Foreign_Investor"]
        .pivot_table(index=["date", "stock_id"], values="net", aggfunc="sum")
        .reset_index()
        .rename(columns={"net": "_ib_net"})
    )

    # ── 融資融券 ──
    df_mg = parquet_db.query_margin(stocks, st, end)
    df_mg["date"] = pd.to_datetime(df_mg["date"])
    mg_cols = ["date", "stock_id",
               "margin_purchase_today_balance", "margin_purchase_yesterday_balance",
               "short_sale_today_balance"]

    # ── 當沖 ──
    df_dt = parquet_db.query_day_trade(stocks, st, end)
    df_dt["date"] = pd.to_datetime(df_dt["date"])
    df_dt = df_dt.rename(columns={"volume": "_dt_vol", "buy_amount": "_dt_buy", "sell_amount": "_dt_sell"})

    # ── 合併 ──
    base = df[["date", "stock_id", "volume"]].copy()
    base = base.merge(ib_piv, on=["date", "stock_id"], how="left")
    base = base.merge(df_mg[mg_cols], on=["date", "stock_id"], how="left")
    base = base.merge(df_dt[["date", "stock_id", "_dt_vol", "_dt_buy", "_dt_sell"]],
                      on=["date", "stock_id"], how="left")

    g = base.groupby("stock_id")

    # 所有籌碼資料 shift=1（收盤後才公布，實際交易用前一天的資料）
    for col in ["_ib_net", "margin_purchase_today_balance", "margin_purchase_yesterday_balance",
                "short_sale_today_balance", "_dt_vol", "_dt_buy", "_dt_sell"]:
        if col in base.columns:
            base[col] = g[col].transform(lambda x: x.shift(1))

    # 外資特徵
    vol20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    base["f_ib_net_pct"] = (base["_ib_net"] / vol20).clip(-5, 5)
    ib5 = g["_ib_net"].transform(lambda x: x.rolling(5, min_periods=3).sum())
    base["f_ib_5d"] = (ib5 / vol20).clip(-10, 10)

    # 融資融券特徵
    mg_prev = base["margin_purchase_yesterday_balance"].replace(0, np.nan)
    base["f_margin_chg"] = (
        (base["margin_purchase_today_balance"] - base["margin_purchase_yesterday_balance"]) / mg_prev
    ).clip(-0.5, 0.5)
    base["f_short_ratio"] = (
        base["short_sale_today_balance"] / base["margin_purchase_today_balance"].replace(0, np.nan)
    ).clip(0, 5)

    # 當沖特徵
    base["f_dt_ratio"] = (base["_dt_vol"] / base["volume"].replace(0, np.nan)).clip(0, 1)
    dt_total = (base["_dt_buy"] + base["_dt_sell"]).replace(0, np.nan)
    base["f_dt_net"] = ((base["_dt_buy"] - base["_dt_sell"]) / dt_total).clip(-1, 1)

    chips_cols = ["f_ib_net_pct", "f_ib_5d", "f_margin_chg", "f_short_ratio", "f_dt_ratio", "f_dt_net"]
    return base[["date", "stock_id"] + chips_cols]


def _compute_market_features(st: str, end: str) -> pd.DataFrame:
    """用 0050 當大盤代理，計算市場環境特徵（按 date join）。"""
    from j1stools import parquet_db

    df = parquet_db.query_price(["0050"], st, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    c = df["close"]
    h = df["high"]
    l = df["low"]

    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr5  = tr.rolling(5,  min_periods=3).mean()
    atr20 = tr.rolling(20, min_periods=10).mean()

    ma60  = c.rolling(60, min_periods=30).mean()
    h120  = h.rolling(120, min_periods=60).max()

    mkt = pd.DataFrame({
        "date":            df["date"],
        "mkt_ret_5d":      c.pct_change(5,  fill_method=None).clip(-0.3, 0.3),
        "mkt_ret_20d":     c.pct_change(20, fill_method=None).clip(-0.5, 0.5),
        "mkt_ma60_dist":   ((c - ma60) / ma60.replace(0, np.nan)).clip(-0.3, 0.3),
        "mkt_atr_ratio":   (atr5 / atr20.replace(0, np.nan)).clip(0, 2),
        "mkt_high_pct":    (c / h120.replace(0, np.nan)).clip(0.5, 1.1),
    })
    # 所有大盤特徵 shift(1)，確保不用到當天收盤
    for col in ["mkt_ret_5d", "mkt_ret_20d", "mkt_ma60_dist", "mkt_atr_ratio", "mkt_high_pct"]:
        mkt[col] = mkt[col].shift(1)

    return mkt.dropna(subset=["mkt_ret_20d"])


def build_vcp_daily_features(stocks: list, st: str, end: str = "2099-01-01") -> pd.DataFrame:
    """
    計算 VCP_FEATURES（含籌碼）及信號欄位。
    回傳含 date, stock_id, close, atr_pct, vcp_signal, *VCP_FEATURES 的 DataFrame。
    """
    import time as _time
    from j1stools import parquet_db

    _t0 = _time.time()
    print(f"  [VcpBreak] 載入 price（{len(stocks)} 檔，{st}~{end}）...")

    df = parquet_db.query_price(stocks, st, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    feat = df.groupby("stock_id", group_keys=False).apply(_compute_vcp, include_groups=False)
    for col in feat.columns:
        df[col] = feat[col].values

    # 籌碼特徵
    chips = _compute_chips(df, stocks, st, end)
    df = df.merge(chips, on=["date", "stock_id"], how="left")
    chips_cols = ["f_ib_net_pct", "f_ib_5d", "f_margin_chg", "f_short_ratio", "f_dt_ratio", "f_dt_net"]
    for col in chips_cols:
        df[col] = df[col].fillna(0)

    # 大盤特徵（0050）
    mkt = _compute_market_features(st, end)
    df = df.merge(mkt, on="date", how="left")
    mkt_cols = ["mkt_ret_5d", "mkt_ret_20d", "mkt_ma60_dist", "mkt_atr_ratio", "mkt_high_pct"]
    for col in mkt_cols:
        df[col] = df[col].fillna(0)

    cols = ["date", "stock_id", "high", "low", "close", "vcp_signal"] + VCP_FEATURES
    print(f"  [VcpBreak] 特徵完成（{_time.time()-_t0:.1f}s），共 {len(df):,} 行")
    return df[cols].sort_values(["stock_id", "date"]).reset_index(drop=True)


# ── 資料集 ────────────────────────────────────────────────────────────────── #


def build_vcp_dataset(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = MIN_ATR_PCT,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    回傳 (X, y, df_meta)：
        X       : (n, n_features) float32
        y       : (n,) int，0=失敗, 1=突破成功
        df_meta : 含 date, stock_id, Y 的 DataFrame
    """
    buf_days = max(CONSOL_DAYS, ATR_LONG, 60) + 10
    st_buf  = (pd.Timestamp(st) - pd.DateOffset(days=buf_days * 2)).strftime("%Y-%m-%d")
    end_buf = (pd.Timestamp(end) + pd.DateOffset(days=hold_days * 2)).strftime("%Y-%m-%d")

    print(f"── 載入 VCP 原始特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_vcp_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    from j1stools.label_builder import profit_label as _profit_label

    df_all = _profit_label(df_all, hold_days=hold_days, profit_target=PROFIT_TARGET, stop_loss=-STOP_LOSS)
    df_all["Y"] = (df_all["target"] == 2).astype(np.int8)
    df_all = df_all.drop(columns=["target"])

    df_signal = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["date"] < pd.Timestamp(end))
            & (df_all["atr_pct"].fillna(0) >= min_atr_pct)
        ]
        .copy()
        .reset_index(drop=True)
    )

    n_pos    = df_signal["Y"].sum()
    pos_rate = df_signal["Y"].mean()
    print(f"全市場樣本：{len(df_signal):,}  Y=1：{n_pos:,} 筆  正例率：{pos_rate:.1%}")

    X = df_signal[VCP_FEATURES].fillna(0).values.astype(np.float32)
    y = df_signal["Y"].values.astype(np.int8)
    return X, y, df_signal


# ── LGBM 分類器 ──────────────────────────────────────────────────────────── #


class VcpBreakLGBM:
    """VCP 放量突破 LGBM 二分類器（預測突破成功機率）。"""

    def __init__(self, params: dict | None = None):
        self._params = {**LGBM_PARAMS, **(params or {})}
        self._model  = None
        self._best_iter: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "VcpBreakLGBM":
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        n_neg  = int((y == 0).sum())
        n_pos  = int((y == 1).sum())
        params = {**self._params, "scale_pos_weight": n_neg / max(n_pos, 1)}

        self._model = lgb.LGBMClassifier(**params)

        X_df     = pd.DataFrame(X, columns=VCP_FEATURES)
        X_val_df = pd.DataFrame(X_val, columns=VCP_FEATURES) if X_val is not None else None

        fit_kwargs = {}
        if X_val_df is not None:
            fit_kwargs["eval_set"]  = [(X_val_df, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

        self._model.fit(X_df, y, **fit_kwargs)
        self._best_iter = getattr(self._model, "best_iteration_", self._params["n_estimators"])

        imp = sorted(zip(VCP_FEATURES, self._model.feature_importances_), key=lambda x: -x[1])
        print("  特徵重要度：" + "  ".join(f"{n}={v}" for n, v in imp))

        if X_val_df is not None:
            up_prob_v = self._model.predict_proba(X_val_df)[:, 1]
            auc       = roc_auc_score(y_val, up_prob_v)
            print(
                f"  val AUC={auc:.4f}  up_prob 分布：min={up_prob_v.min():.3f}  "
                f"p50={np.median(up_prob_v):.3f}  max={up_prob_v.max():.3f}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(pd.DataFrame(X, columns=VCP_FEATURES))[:, 1]


class VcpBreakXGB:
    """VCP 放量突破 XGBoost 二分類器。"""

    def __init__(self, params: dict | None = None):
        self._params = {**XGB_PARAMS, **(params or {})}
        self._model  = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "VcpBreakXGB":
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        params = {**self._params, "scale_pos_weight": n_neg / max(n_pos, 1)}

        if X_val is not None:
            params["early_stopping_rounds"] = 50

        self._model = xgb.XGBClassifier(**params)

        fit_kwargs = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"]  = 100

        self._model.fit(X, y, **fit_kwargs)

        imp = sorted(zip(VCP_FEATURES, self._model.feature_importances_), key=lambda x: -x[1])
        print("  特徵重要度：" + "  ".join(f"{n}={v:.0f}" for n, v in imp))

        if X_val is not None:
            up_prob_v = self._model.predict_proba(X_val)[:, 1]
            auc       = roc_auc_score(y_val, up_prob_v)
            print(
                f"  val AUC={auc:.4f}  up_prob 分布：min={up_prob_v.min():.3f}  "
                f"p50={np.median(up_prob_v):.3f}  max={up_prob_v.max():.3f}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_vcp_break_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> VcpBreakLGBM:
    from sklearn.metrics import roc_auc_score

    print("\n── 訓練 VcpBreakLGBM (Binary) ──")
    clf       = VcpBreakLGBM()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_auc = roc_auc_score(y_train, clf.predict(X_train))
    print(f"訓練集 AUC：{train_auc:.4f}（in-sample）")
    return clf


def train_vcp_break_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> VcpBreakXGB:
    from sklearn.metrics import roc_auc_score

    print("\n── 訓練 VcpBreakXGB (Binary) ──")
    clf       = VcpBreakXGB()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_auc = roc_auc_score(y_train, clf.predict(X_train))
    print(f"訓練集 AUC：{train_auc:.4f}（in-sample）")
    return clf


# ── 存取模型 ──────────────────────────────────────────────────────────────── #

VCP_BREAK_LGBM_PATH = "db/models/vcp_break_lgbm.joblib"


def save_vcp_break_lgbm(
    clf: VcpBreakLGBM, path: str = VCP_BREAK_LGBM_PATH, hold_days: int = HOLD_DAYS
) -> None:
    import joblib, os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": clf._model, "hold_days": hold_days}, path)
    print(f"VcpBreakLGBM 已存：{path}（hold_days={hold_days}）")


def load_vcp_break_lgbm(path: str = VCP_BREAK_LGBM_PATH) -> tuple["VcpBreakLGBM", int]:
    import joblib

    state       = joblib.load(path)
    clf         = VcpBreakLGBM()
    clf._model  = state["model"]
    hold_days   = state.get("hold_days", HOLD_DAYS)
    print(f"VcpBreakLGBM 已載入：{path}（hold_days={hold_days}）")
    return clf, hold_days


VCP_BREAK_XGB_PATH = "db/models/vcp_break_xgb.joblib"


def save_vcp_break_xgb(
    clf: VcpBreakXGB, path: str = VCP_BREAK_XGB_PATH, hold_days: int = HOLD_DAYS
) -> None:
    import joblib, os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": clf._model, "hold_days": hold_days}, path)
    print(f"VcpBreakXGB 已存：{path}（hold_days={hold_days}）")


def load_vcp_break_xgb(path: str = VCP_BREAK_XGB_PATH) -> tuple["VcpBreakXGB", int]:
    import joblib

    state       = joblib.load(path)
    clf         = VcpBreakXGB()
    clf._model  = state["model"]
    hold_days   = state.get("hold_days", HOLD_DAYS)
    print(f"VcpBreakXGB 已載入：{path}（hold_days={hold_days}）")
    return clf, hold_days


def _load_model(model_type: str):
    """MODEL_TYPE 切換載入。回傳 (clf, hold_days)。"""
    if model_type == "lgbm":
        return load_vcp_break_lgbm()
    if model_type == "xgb":
        return load_vcp_break_xgb()
    if model_type == "ensemble":
        clf_l, hd = load_vcp_break_lgbm()
        clf_x, _  = load_vcp_break_xgb()

        class _Ensemble:
            def predict(self, X):
                return (clf_l.predict(X) + clf_x.predict(X)) / 2

        return _Ensemble(), hd
    raise ValueError(f"未知 MODEL_TYPE：{model_type!r}")


# ── 信號生成（回測用）────────────────────────────────────────────────────── #


def make_signal_vcp_break_lgbm(
    clf: VcpBreakLGBM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    min_atr_pct: float = MIN_ATR_PCT,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    buf_days = max(CONSOL_DAYS, ATR_LONG, 60) + 10
    st_buf   = (pd.Timestamp(st) - pd.DateOffset(days=buf_days * 2)).strftime("%Y-%m-%d")

    df_all  = build_vcp_daily_features(stocks, st_buf, end)
    df_all["date"] = pd.to_datetime(df_all["date"])

    df_scan = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["atr_pct"].fillna(0) >= min_atr_pct)
        ]
        .copy()
        .reset_index(drop=True)
    )
    print(f"  全市場掃描：{len(df_scan):,} 筆")

    if df_scan.empty:
        return pd.DataFrame(columns=["date", "stock_id", "up_prob"])

    X       = df_scan[VCP_FEATURES].fillna(0).values.astype(np.float32)
    up_prob = clf.predict(X)

    signal          = df_scan[["date", "stock_id"]].copy()
    signal["up_prob"] = up_prob
    signal["2"]       = up_prob
    signal["date"]    = pd.to_datetime(signal["date"])

    print(
        f"  up_prob 分布：mean={up_prob.mean():.3f}  "
        f"p50={np.median(up_prob):.3f}  p90={np.percentile(up_prob,90):.3f}"
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
    EVAL_ST  = "2024-01-01"
    MODE     = "backtest"  # "train" | "eval" | "backtest"

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train":
        print("\n══ 建立訓練集 ══")
        X_train, y_train, df_tr = build_vcp_dataset(stocks, TRAIN_ST, EVAL_ST, hold_days=HOLD_DAYS)
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te   = build_vcp_dataset(stocks, EVAL_ST, hold_days=HOLD_DAYS)
        print(f"訓練集：{X_train.shape}  測試集：{X_test.shape}")

        if MODEL_TYPE == "lgbm":
            clf = train_vcp_break_lgbm(X_train, y_train, X_val=X_test, y_val=y_test)
            save_vcp_break_lgbm(clf, hold_days=HOLD_DAYS)
        elif MODEL_TYPE == "xgb":
            clf = train_vcp_break_xgb(X_train, y_train, X_val=X_test, y_val=y_test)
            save_vcp_break_xgb(clf, hold_days=HOLD_DAYS)
        elif MODEL_TYPE == "ensemble":
            clf_l = train_vcp_break_lgbm(X_train, y_train, X_val=X_test, y_val=y_test)
            save_vcp_break_lgbm(clf_l, hold_days=HOLD_DAYS)
            clf_x = train_vcp_break_xgb(X_train, y_train, X_val=X_test, y_val=y_test)
            save_vcp_break_xgb(clf_x, hold_days=HOLD_DAYS)

    elif MODE == "eval":
        clf, hold_days_loaded = _load_model(MODEL_TYPE)
        plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]

        from sklearn.metrics import roc_auc_score
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_vcp_dataset(stocks, EVAL_ST, hold_days=hold_days_loaded)

        up_prob   = clf.predict(X_test)
        auc       = roc_auc_score(y_test, up_prob)
        base_rate = y_test.mean()
        print(f"\nOOS AUC：{auc:.4f}  （n={len(y_test):,}，正例率={base_rate:.1%}）")
        print(
            f"up_prob 分布：min={up_prob.min():.3f}  p25={np.percentile(up_prob,25):.3f}  "
            f"p50={np.median(up_prob):.3f}  p75={np.percentile(up_prob,75):.3f}  max={up_prob.max():.3f}"
        )

        print(f"\n{'─'*54}")
        print(f"{'THRESHOLD':>10}  {'信號數':>8}  {'精度':>8}  {'提升':>6}")
        print(f"{'─'*54}")
        pcts     = np.arange(50, 96, 5)
        thr_vals = np.unique(np.round(np.percentile(up_prob, pcts), 3))
        for thr in thr_vals:
            mask = up_prob >= thr
            n    = int(mask.sum())
            if n == 0:
                continue
            prec = y_test[mask].mean()
            print(f"{thr:>10.3f}  {n:>8,}  {prec:>8.1%}  {prec/base_rate:>6.2f}x")
        print(f"{'─'*54}")

        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(up_prob[y_test == 0], bins=40, alpha=0.6, color="steelblue", label="Y=0（失敗）", density=True)
        axes[0].hist(up_prob[y_test == 1], bins=40, alpha=0.6, color="tomato",    label="Y=1（突破）", density=True)
        axes[0].set_xlabel("up_prob")
        axes[0].set_ylabel("density")
        axes[0].set_title(f"up_prob 分布（AUC={auc:.4f}）")
        axes[0].legend()

        sweep_pcts = np.arange(30, 98, 1)
        sweep_thrs = np.percentile(up_prob, sweep_pcts)
        sweep_prec, sweep_n = [], []
        for t in sweep_thrs:
            mask = up_prob >= t
            n    = mask.sum()
            sweep_prec.append(y_test[mask].mean() if n >= 30 else np.nan)
            sweep_n.append(n)

        ax1 = axes[1]
        ax2 = ax1.twinx()
        ax1.plot(sweep_thrs, sweep_prec, color="seagreen",  linewidth=2, label="精度")
        ax1.axhline(base_rate, color="gray", linestyle="--", linewidth=0.8, label=f"基準 {base_rate:.1%}")
        ax1.axhline(STOP_LOSS / (PROFIT_TARGET + STOP_LOSS), color="red", linestyle=":", linewidth=0.8,
                    label=f"break-even {STOP_LOSS/(PROFIT_TARGET+STOP_LOSS):.1%}")
        ax2.fill_between(sweep_thrs, sweep_n, alpha=0.15, color="steelblue")
        ax2.set_ylabel("信號數", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_xlabel("THRESHOLD")
        ax1.set_ylabel("精度（突破率）")
        ax1.set_title("Precision vs THRESHOLD (OOS)")
        ax1.legend(loc="upper left")

        plt.suptitle(f"VcpBreak[{MODEL_TYPE.upper()}] | {hold_days_loaded}d | VCP 波動壓縮放量突破")
        plt.tight_layout()
        plt.show()

    elif MODE == "backtest":
        clf, hold_days_loaded = _load_model(MODEL_TYPE)
        print("\n══ 獨立回測 ══")
        backtest_platform.IS_USE_CACHE = True
        sig = make_signal_vcp_break_lgbm(clf, stocks, st=EVAL_ST, threshold=THRESHOLD)

        if sig.empty:
            print(f"⚠️  無訊號（THRESHOLD={THRESHOLD}，VCP條件太嚴或threshold太高），請調整後重試")
            import sys; sys.exit(0)

        print(f"\n{'='*60}\n【VcpBreak[{MODEL_TYPE.upper()}] 高預測報酬 + 持有{hold_days_loaded}日】\n{'='*60}")
        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=30,
            threshold=THRESHOLD,
            max_positions=3,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=STOP_LOSS,
            use_fixed_tp=True,
            tp_stop=PROFIT_TARGET,
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
