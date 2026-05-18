"""
B-Bull MVP — v3
===============
新增：RSI、MACD 柱狀體、開盤缺口、相對大盤強弱
新增：特徵診斷（移除 f_dev_ma60 後的 AUC，確認其他特徵有無獨立貢獻）

使用方式：
    from b_bull_mvp import run
    results, data = run(price_df, market_df)

price_df  欄位：date, stock_id, close（+ open/volume 選用）
market_df 欄位：date, close（0050 ETF）
"""

import warnings
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# 超參數
# ============================================================
MA_SHORT = 5
MA_MID = 10
MA_LONG = 20
MA_REGIME = 60

REGIME_BUF = 0.05
REGIME_DAYS = 3

LABEL_DAYS = 5
LABEL_THR = 0.02

N_SPLITS = 5
MIN_TRAIN = 252

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# 原始均線特徵
BASE_FEATURES = [
    "f_dev_ma20",  # 昨收偏離 MA20
    "f_dev_ma60",  # 昨收偏離 MA60（現在是特徵，不是守門員）
    "f_ma60_slope",  # MA60 五日斜率（多頭/空頭方向）← 新增
    "f_ma5_slope",  # MA5 三日斜率
    "f_vol_ratio",  # 量比
    "f_ma_align",  # 均線多頭排列
]

# 新增技術指標特徵
NEW_FEATURES = [
    "f_rsi14",  # RSI 14
    "f_macd_hist",  # MACD 柱狀體
    "f_gap",  # 開盤缺口
    "f_rel_mkt",  # 相對大盤 5 日強弱
]

FEATURE_COLS = BASE_FEATURES + NEW_FEATURES


# ============================================================
# 資料清理
# ============================================================
def _clean_price(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.strip().str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d["stock_id"] = d["stock_id"].astype(str).str.strip()

    for col in ["date", "stock_id", "close"]:
        if col not in d.columns:
            raise ValueError(f"price_df 缺少必要欄位：{col}")

    if "volume" not in d.columns:
        print("⚠️  price_df 沒有 volume 欄位，f_vol_ratio 將略過")
        d["volume"] = np.nan

    if "open" not in d.columns:
        print("⚠️  price_df 沒有 open 欄位，f_gap 將略過")
        d["open"] = np.nan

    before = len(d)
    d = d[d["close"] > 0].dropna(subset=["close"])
    if len(d) < before:
        print(f"⚠️  移除 {before - len(d)} 筆收盤價異常")

    return d.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _clean_market(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.strip().str.lower()
    d["date"] = pd.to_datetime(d["date"])

    if "close" not in d.columns:
        raise ValueError("market_df 需要有 close 欄位")

    return (
        d[["date", "close"]]
        .rename(columns={"close": "market_close"})
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


# ============================================================
# 特徵計算（per stock）
# ============================================================
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-6)
    return 100 - 100 / (1 + rs)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_features(grp: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)
    d = d.merge(market, on="date", how="left")

    c = d["close"].shift(1)  # 昨收（13:00 可見）

    ma5 = c.rolling(MA_SHORT).mean()
    ma10 = c.rolling(MA_MID).mean()
    ma20 = c.rolling(MA_LONG).mean()
    ma60 = c.rolling(MA_REGIME).mean()

    # --- 原始均線特徵 ---
    d["f_dev_ma20"] = (c - ma20) / ma20.clip(lower=1e-6)
    d["f_dev_ma60"] = (c - ma60) / ma60.clip(lower=1e-6)
    d["f_ma60_slope"] = (ma60 - ma60.shift(5)) / ma60.shift(5).clip(lower=1e-6)  # MA60 五日斜率
    d["f_ma5_slope"] = (ma5 - ma5.shift(3)) / ma5.shift(3).clip(lower=1e-6)
    d["f_ma_align"] = (ma5 - ma10) / ma10.clip(lower=1e-6) + (ma10 - ma20) / ma20.clip(lower=1e-6)

    if d["volume"].notna().any():
        v = d["volume"].shift(1)
        d["f_vol_ratio"] = v.rolling(5).mean() / v.rolling(20).mean().clip(lower=1e-6)
    else:
        d["f_vol_ratio"] = np.nan

    # --- 新特徵 1：RSI 14（用昨收序列）---
    d["f_rsi14"] = _rsi(c, 14)

    # --- 新特徵 2：MACD 柱狀體 ---
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    d["f_macd_hist"] = macd_line - signal  # 正 = 動能加速，負 = 動能減速

    # --- 新特徵 3：開盤缺口（今開 vs 昨收）---
    if d["open"].notna().any():
        # 今日 open 是 13:00 前已發生的資訊，可以用
        d["f_gap"] = (d["open"] - d["close"].shift(1)) / d["close"].shift(1).clip(lower=1e-6)
    else:
        d["f_gap"] = np.nan

    # --- 新特徵 4：個股 vs 大盤 5 日相對強弱 ---
    stock_ret5 = d["close"].shift(1).pct_change(5)
    market_ret5 = d["market_close"].shift(1).pct_change(5)
    d["f_rel_mkt"] = stock_ret5 - market_ret5

    return d


# ============================================================
# MA60 機制標記
# ============================================================
def _label_regime(grp: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)
    ma60 = d["close"].rolling(MA_REGIME).mean()

    regimes, above_streak, current = [], 0, "buffer"

    for close, m60 in zip(d["close"], ma60):
        if pd.isna(m60):
            regimes.append("buffer")
            continue

        ratio = close / m60

        if ratio > 1 + REGIME_BUF:
            above_streak += 1
            if above_streak >= REGIME_DAYS:
                current = "bull"
        elif ratio < 1.0:
            if current == "bull":
                current = "buffer"
            above_streak = 0
        else:
            if current != "bull":
                above_streak = 0

        regimes.append(current)

    d["regime"] = regimes
    return d


# ============================================================
# 目標變數
# ============================================================
def _compute_label(grp: pd.DataFrame) -> pd.DataFrame:
    """market_close 已在 _compute_features 裡 merge 進來"""
    d = grp.copy()
    ret_s = d["close"].shift(-LABEL_DAYS) / d["close"] - 1
    ret_m = d["market_close"].shift(-LABEL_DAYS) / d["market_close"] - 1
    d["excess_ret"] = ret_s - ret_m
    d["label"] = (d["excess_ret"] > LABEL_THR).astype(int)
    return d


# ============================================================
# 建構全資料集
# ============================================================
def _build_dataset(price_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    min_rows = MA_REGIME + LABEL_DAYS + 30
    results, skipped = [], 0

    for sid, grp in price_df.groupby("stock_id"):
        if len(grp) < min_rows:
            skipped += 1
            continue
        g = _compute_features(grp, market_df)
        g = _label_regime(g)
        g = _compute_label(g)
        results.append(g)

    if skipped:
        print(f"⚠️  跳過 {skipped} 支資料不足的股票（需 ≥ {min_rows} 筆）")

    data = pd.concat(results, ignore_index=True)
    return data.dropna(subset=["label", "excess_ret"])


# ============================================================
# 訓練輔助
# ============================================================
def _get_valid_feats(bull: pd.DataFrame, feats: list) -> list:
    return [f for f in feats if f in bull.columns and bull[f].notna().sum() > 100]


def _train_xgb(train: pd.DataFrame, feats: list) -> XGBClassifier:
    neg, pos = (train["label"] == 0).sum(), (train["label"] == 1).sum()
    spw = round(neg / max(pos, 1), 2)
    cut = int(len(train) * 0.8)
    tr2, v2 = train.iloc[:cut], train.iloc[cut:]

    clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    clf.fit(
        tr2[feats],
        tr2["label"],
        eval_set=[(v2[feats], v2["label"])],
        verbose=False,
    )
    return clf


# ============================================================
# Walk-Forward
# ============================================================
# Walk-Forward（按年切）
# ============================================================
def _walk_forward(data: pd.DataFrame, feats: list, label: str = "") -> pd.DataFrame:
    """
    Test 固定是完整一年，Train 是該年之前的所有資料。
    不再強制過濾 regime == bull，由呼叫端決定傳入哪些樣本。
    """
    d = data.dropna(subset=feats + ["label"]).copy()

    valid_feats = _get_valid_feats(d, feats)
    dropped = set(feats) - set(valid_feats)
    if dropped:
        print(f"  ⚠️  略過特徵（資料不足）：{dropped}")

    if len(d) == 0:
        print("  ❌ 沒有樣本")
        return pd.DataFrame()

    min_year = d["date"].dt.year.min()
    max_year = d["date"].dt.year.max()
    test_years = range(min_year + 1, max_year + 1)

    tag = f" [{label}]" if label else ""
    print(f"樣本：{len(d):,} 筆 | {min_year} ~ {max_year}{tag}")

    rows = []
    for test_year in test_years:
        train = d[d["date"].dt.year < test_year]
        test = d[d["date"].dt.year == test_year]

        train_days = train["date"].nunique()
        if train_days < MIN_TRAIN:
            print(f"  {test_year} 跳過：訓練天數不足 ({train_days} < {MIN_TRAIN})")
            continue

        if len(test) < 30 or test["label"].nunique() < 2:
            print(f"  {test_year} 跳過：測試集不足或 label 單一 ({len(test)} 筆)")
            continue

        clf = _train_xgb(train, valid_feats)
        y_pred = clf.predict(test[valid_feats])
        y_prob = clf.predict_proba(test[valid_feats])[:, 1]

        prec = precision_score(test["label"], y_pred, zero_division=0)
        auc = roc_auc_score(test["label"], y_prob)

        rows.append(
            {
                "test_year": test_year,
                "train_start": str(train["date"].min())[:10],
                "train_end": str(train["date"].max())[:10],
                "train_days": train_days,
                "train_n": len(train),
                "test_n": len(test),
                "label_rate": round(float(train["label"].mean()), 3),
                "precision": round(prec, 3),
                "auc": round(auc, 3),
                "n_signals": int(y_pred.sum()),
            }
        )

        print(
            f"  {test_year} | "
            f"Train {str(train['date'].min())[:10]}~{str(train['date'].max())[:10]} "
            f"({len(train):,}) → "
            f"Test {test_year} ({len(test):,}) | "
            f"Precision={prec:.3f}  AUC={auc:.3f}  訊號={int(y_pred.sum())}"
        )

    return pd.DataFrame(rows)


# ============================================================
# 特徵重要性
# ============================================================
def _feature_importance(data: pd.DataFrame, feats: list) -> pd.Series:
    d = data.dropna(subset=feats + ["label"])
    valid_feats = _get_valid_feats(d, feats)

    clf = _train_xgb(d, valid_feats)
    imp = pd.Series(clf.feature_importances_, index=valid_feats).sort_values(ascending=False)

    print("\n--- 特徵重要性 ---")
    for feat, score in imp.items():
        bar = "█" * int(score * 50)
        print(f"  {feat:<22} {bar}  {score:.4f}")
    return imp


# ============================================================
# 特徵診斷：移除 f_dev_ma60 後的 AUC
# ============================================================
def _diagnose(data: pd.DataFrame, full_results: pd.DataFrame, feats: list) -> None:
    print("\n" + "=" * 60)
    print("特徵診斷：移除 f_dev_ma60 後的 AUC")
    print("（確認其他特徵是否有獨立貢獻）")
    print("=" * 60)

    feats_no_ma60 = [f for f in feats if f != "f_dev_ma60"]
    results_no_ma60 = _walk_forward(data, feats_no_ma60, label="無 f_dev_ma60")

    if len(results_no_ma60) == 0:
        return

    full_auc = full_results["auc"].mean()
    drop_auc = results_no_ma60["auc"].mean()
    delta = full_auc - drop_auc

    print(f"\n  完整特徵 AUC   : {full_auc:.3f}")
    print(f"  移除後 AUC     : {drop_auc:.3f}")
    print(f"  AUC 差距       : {delta:+.3f}")

    if drop_auc >= 0.53:
        print("  ✅ 其他特徵有獨立貢獻（AUC ≥ 0.53），值得保留")
    else:
        print("  ⚠️  其他特徵貢獻有限（AUC < 0.53），考慮換更強的特徵")

    if delta < 0.01:
        print("  ℹ️  f_dev_ma60 貢獻極小（< 0.01），可能與其他特徵高度相關")
    elif delta < 0.03:
        print("  ℹ️  f_dev_ma60 有適度貢獻，保留")
    else:
        print("  ℹ️  f_dev_ma60 貢獻顯著（> 0.03），是核心特徵")


# ============================================================
# 收鍊突破分析（內部）
# ============================================================
BREAKOUT_WINDOW = 10  # 確認收鍊後幾個交易日內必須突破


def _clean_triangle(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.astype(str).str.strip().str.lower()

    for col in ["stock_id", "h2_date", "high"]:
        if col not in d.columns:
            raise ValueError(f"triangle_df 缺少欄位：{col}")

    d["stock_id"] = d["stock_id"].astype(str).str.strip()
    d["h2_date"] = pd.to_datetime(d["h2_date"])
    d["h2_high"] = pd.to_numeric(d["high"], errors="coerce")
    d["pattern_date"] = pd.to_datetime(d["date"]) if "date" in d.columns else d["h2_date"]

    return d.dropna(subset=["h2_high", "h2_date"]).reset_index(drop=True)


def _find_entries(triangle_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    從 date（確認收鍊當天）之後算 BREAKOUT_WINDOW 個交易日
    第一個收盤 > h2_date 的 high → 進場
    days_to_entry 用交易日數計算
    """
    tri = _clean_triangle(triangle_df)
    prc = price_df.copy()
    prc.columns = prc.columns.str.strip().str.lower()
    prc["date"] = pd.to_datetime(prc["date"])
    prc["stock_id"] = prc["stock_id"].astype(str).str.strip()
    prc["close"] = pd.to_numeric(prc["close"], errors="coerce")
    prc = prc[prc["close"] > 0].sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 確認 stock_id 格式一致
    tri_ids = set(tri["stock_id"].unique())
    prc_ids = set(prc["stock_id"].unique())
    overlap = tri_ids & prc_ids
    missing = tri_ids - prc_ids
    if missing:
        print(f"  ⚠️  {len(missing)} 支股票在 price_df 找不到（stock_id 格式可能不一致）")
        print(f"      triangle 範例：{list(tri_ids)[:3]}")
        print(f"      price    範例：{list(prc_ids)[:3]}")
    print(f"  ✓ 有效匹配股票：{len(overlap)} 支")

    # 建立 per-stock 索引加速查詢
    prc_grouped = {sid: grp.reset_index(drop=True) for sid, grp in prc.groupby("stock_id")}

    records = []
    for _, row in tri.iterrows():
        sid = row["stock_id"]
        h2_date = row["h2_date"]
        breakout_high = row["h2_high"]
        pattern_date = row["pattern_date"]

        if sid not in prc_grouped:
            continue

        stock_prc = prc_grouped[sid]

        # 取 pattern_date 之後的交易日（head 確保只取前 N 個交易日）
        future = stock_prc[stock_prc["date"] > pattern_date].head(BREAKOUT_WINDOW)

        if len(future) == 0:
            continue

        hit = future[future["close"] > breakout_high]

        if len(hit) == 0:
            records.append(
                {
                    "stock_id": sid,
                    "pattern_date": pattern_date,
                    "h2_date": h2_date,
                    "breakout_high": round(breakout_high, 4),
                    "entry_date": pd.NaT,
                    "entry_close": np.nan,
                    "trading_days_to_entry": np.nan,
                    "triggered": False,
                }
            )
        else:
            entry = hit.iloc[0]
            # 用交易日數（在 future 裡的位置 + 1）
            trading_days = future.index.get_loc(entry.name) + 1

            records.append(
                {
                    "stock_id": sid,
                    "pattern_date": pattern_date,
                    "h2_date": h2_date,
                    "breakout_high": round(breakout_high, 4),
                    "entry_date": entry["date"],
                    "entry_close": round(entry["close"], 4),
                    "trading_days_to_entry": trading_days,
                    "triggered": True,
                }
            )

    return pd.DataFrame(records)


def _find_ma_cross(data: pd.DataFrame) -> pd.DataFrame:
    """
    偵測三種 MA 黃金交叉：
      MA5  上穿 MA10
      MA5  上穿 MA20
      MA10 上穿 MA20
    只在 regime = bull 且特徵完整的樣本中偵測
    """
    records = []

    for sid, grp in data.groupby("stock_id"):
        d = grp.copy().sort_values("date").reset_index(drop=True)

        # 用昨收計算的均線（已在 _compute_features 算好，這裡重算以確保一致）
        c = d["close"].shift(1)
        ma5 = c.rolling(MA_SHORT).mean()
        ma10 = c.rolling(MA_MID).mean()
        ma20 = c.rolling(MA_LONG).mean()

        # 交叉：昨日 A < B，今日 A > B
        cross_5_10 = (ma5.shift(1) < ma10.shift(1)) & (ma5 > ma10)
        cross_5_20 = (ma5.shift(1) < ma20.shift(1)) & (ma5 > ma20)
        cross_10_20 = (ma10.shift(1) < ma20.shift(1)) & (ma10 > ma20)

        is_cross = cross_5_10 | cross_5_20 | cross_10_20

        # 只取 regime = bull 的交叉日
        cross_days = d[(is_cross) & (d["regime"] == "bull")]

        for _, row in cross_days.iterrows():
            cross_type = []
            idx = row.name
            if cross_5_10.iloc[idx]:
                cross_type.append("MA5xMA10")
            if cross_5_20.iloc[idx]:
                cross_type.append("MA5xMA20")
            if cross_10_20.iloc[idx]:
                cross_type.append("MA10xMA20")

            records.append(
                {
                    "stock_id": sid,
                    "entry_date": row["date"],
                    "entry_type": "+".join(cross_type),
                    "entry_close": row["close"],
                }
            )

    return pd.DataFrame(records)


def _print_ma_cross_stats(ma_signals: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"MA 交叉訊號統計")
    print(f"{'='*60}")
    print(f"訊號總數      : {len(ma_signals):,}")

    type_counts = ma_signals["entry_type"].value_counts()
    print(f"\n交叉類型分佈：")
    for t, cnt in type_counts.items():
        bar = "█" * int(cnt / type_counts.max() * 20)
        print(f"  {t:<20} {bar}  {cnt:,}")
    print(f"{'='*60}")


def _print_breakout_stats(signals: pd.DataFrame) -> None:
    n_total = len(signals)
    n_triggered = signals["triggered"].sum()
    n_miss = n_total - n_triggered

    print(f"\n{'='*60}")
    print(f"收鍊突破統計")
    print(f"{'='*60}")
    print(f"型態總數      : {n_total}")
    print(f"10日內突破    : {n_triggered}  ({n_triggered/max(n_total,1):.1%})")
    print(f"未突破        : {n_miss}  ({n_miss/max(n_total,1):.1%})")

    triggered = signals[signals["triggered"]]
    if len(triggered) > 0:
        print(f"\n突破後進場速度（交易日）：")
        print(f"  平均 {triggered['trading_days_to_entry'].mean():.1f} 天")
        print(f"  最快 {triggered['trading_days_to_entry'].min():.0f} 天")
        print(f"  最慢 {triggered['trading_days_to_entry'].max():.0f} 天")

        bins = [0, 2, 5, 7, 10]
        labels = ["1-2天", "3-5天", "6-7天", "8-10天"]
        triggered = triggered.copy()
        triggered["speed"] = pd.cut(triggered["trading_days_to_entry"], bins=bins, labels=labels)
        dist = triggered["speed"].value_counts().sort_index()
        print(f"\n  速度分佈：")
        for lbl, cnt in dist.items():
            bar = "█" * int(cnt / max(dist) * 20)
            print(f"    {lbl}  {bar}  {cnt}")
    print(f"{'='*60}")


# ============================================================
# 主函式
# ============================================================
def run(
    price_df: pd.DataFrame, market_df: pd.DataFrame, triangle_df: pd.DataFrame = None, wedge_df: pd.DataFrame = None
):
    """
    Parameters
    ----------
    price_df     : 欄位 date / stock_id / close（+ open / volume 選用）
    market_df    : 欄位 date / close（0050 ETF）
    triangle_df  : 收鍊輸出（選用），欄位 date / stock_id / h2_date / high
    wedge_df     : 下降楔形輸出（選用），格式同 triangle_df

    Returns
    -------
    results    : Walk-Forward 各 Fold 結果
    data       : 完整標記資料集
    signals    : 收鍊突破訊號
    wedge_signals : 下降楔形突破訊號
    ma_signals : MA 交叉訊號
    """
    print("=" * 60)
    print("B-Bull MVP  v3  (XGBoost + RSI + MACD + 診斷)")
    print("=" * 60)

    price_df = _clean_price(price_df)
    market_df = _clean_market(market_df)

    s = max(price_df["date"].min(), market_df["date"].min())
    e = min(price_df["date"].max(), market_df["date"].max())
    price_df = price_df[(price_df["date"] >= s) & (price_df["date"] <= e)]
    market_df = market_df[(market_df["date"] >= s) & (market_df["date"] <= e)]

    print(f"股票：{price_df['stock_id'].nunique()} 支 | " f"日期：{s.date()} ~ {e.date()}")

    print("\n計算特徵與標記...")
    data = _build_dataset(price_df, market_df)

    print(f"\n總樣本         : {len(data):,}")
    print(f"Label=1 比例   : {data['label'].mean():.1%}  ← 全樣本基準線")
    print(f"（MA60 位置分佈）")
    print(f"  多頭區間     : {(data['regime']=='bull').sum():,}  ({(data['regime']=='bull').mean():.1%})")
    print(f"  緩衝帶       : {(data['regime']=='buffer').sum():,}  ({(data['regime']=='buffer').mean():.1%})")

    # 有效特徵
    valid_all = _get_valid_feats(data, FEATURE_COLS)
    print(f"\n使用特徵（{len(valid_all)} 個）：{valid_all}")

    # Walk-Forward（全樣本，MA60 改為特徵）
    print("\n--- Walk-Forward（全樣本，MA60 為特徵）---")
    results = _walk_forward(data, valid_all)

    if len(results) > 0:
        avg_prec = results["precision"].mean()
        avg_auc = results["auc"].mean()
        label_rate = data["label"].mean()

        print(f"\n{'='*60}")
        print(f"平均 Precision  : {avg_prec:.3f}  （基準：{label_rate:.3f}）")
        print(f"平均 AUC        : {avg_auc:.3f}  （基準：0.500）")
        print(f"Precision 標準差: {results['precision'].std():.3f}  " f"（共 {len(results)} 個測試年）")
        print("\n判讀：")
        if avg_auc > 0.58 and avg_prec > label_rate:
            print("  ✅ 特徵有效 → 可繼續加特徵或加入 1D-CNN")
        elif avg_auc > 0.54:
            print("  ⚠️  訊號微弱 → 建議再加更強特徵")
        else:
            print("  ❌ 無預測力 → 重新檢視 target 定義或特徵設計")
        print(f"{'='*60}")
        print(f"\n{results.to_string(index=False)}")

        # 特徵重要性（全樣本）
        _feature_importance(data, valid_all)

        # 診斷：移除 f_dev_ma60
        if "f_dev_ma60" in valid_all:
            _diagnose(data, results, valid_all)

    # 收鍊/楔形/MA交叉 訊號
    signals = pd.DataFrame()
    wedge_signals = pd.DataFrame()
    ma_signals = pd.DataFrame()

    # ── MA 交叉訊號（永遠跑）──
    print("\n--- MA 交叉訊號 ---")
    ma_signals = _find_ma_cross(data)
    _print_ma_cross_stats(ma_signals)

    if triangle_df is not None:
        print("\n--- 收鍊突破分析 ---")
        signals = _find_entries(triangle_df, price_df)
        _print_breakout_stats(signals)

    if wedge_df is not None:
        print("\n--- 下降楔形突破分析 ---")
        wedge_signals = _find_entries(wedge_df, price_df)
        _print_breakout_stats(wedge_signals)

    # ── 合併所有候選訊號 ──
    keep_rows = []

    if len(ma_signals) > 0:
        ma_keep = ma_signals[["stock_id", "entry_date"]].rename(columns={"entry_date": "date"})
        ma_keep["entry_source"] = "ma_cross"
        keep_rows.append(ma_keep)

    if len(signals) > 0:
        tri_triggered = signals[signals["triggered"]][["stock_id", "entry_date"]].rename(columns={"entry_date": "date"})
        tri_triggered["entry_source"] = "triangle"
        keep_rows.append(tri_triggered)

    if len(wedge_signals) > 0:
        wedge_triggered = wedge_signals[wedge_signals["triggered"]][["stock_id", "entry_date"]].rename(
            columns={"entry_date": "date"}
        )
        wedge_triggered["entry_source"] = "wedge"
        keep_rows.append(wedge_triggered)

    # ── 各訊號來源獨立 Walk-Forward ──
    sources = []
    if len(ma_signals) > 0:
        ma_only = ma_signals[["stock_id", "entry_date"]].rename(columns={"entry_date": "date"})
        sources.append(("MA 交叉", ma_only))

    if len(signals) > 0:
        tri_only = signals[signals["triggered"]][["stock_id", "entry_date"]].rename(columns={"entry_date": "date"})
        sources.append(("收鍊突破", tri_only))

    if len(wedge_signals) > 0:
        wedge_only = wedge_signals[wedge_signals["triggered"]][["stock_id", "entry_date"]].rename(
            columns={"entry_date": "date"}
        )
        sources.append(("楔形突破", wedge_only))

    summary_rows = []
    for src_name, src_df in sources:
        src_data = data.merge(src_df, on=["stock_id", "date"], how="inner")
        if len(src_data) == 0:
            continue

        label_rate_src = src_data["label"].mean() if len(src_data) > 0 else 0

        print(f"\n--- Walk-Forward：{src_name} ---")
        print(f"樣本數         : {len(src_data):,}")
        print(f"Label=1 比例   : {label_rate_src:.1%}  ← 基準線")

        valid_src = _get_valid_feats(src_data, FEATURE_COLS)
        src_results = _walk_forward(src_data, valid_src, label=src_name)

        if len(src_results) > 0:
            src_prec = src_results["precision"].mean()
            src_auc = src_results["auc"].mean()
            beat_base = (src_prec - label_rate_src) / label_rate_src * 100

            print(f"\n{'='*60}")
            print(
                f"[{src_name}] 平均 Precision : {src_prec:.3f}  "
                f"（基準：{label_rate_src:.3f}，超越 {beat_base:+.1f}%）"
            )
            print(f"[{src_name}] 平均 AUC       : {src_auc:.3f}")
            print(f"{'='*60}")
            print(f"\n{src_results.to_string(index=False)}")

            summary_rows.append(
                {
                    "source": src_name,
                    "n_signals": len(src_data),
                    "label_rate": round(label_rate_src, 3),
                    "precision": round(src_prec, 3),
                    "auc": round(src_auc, 3),
                    "beat_base_%": round(beat_base, 1),
                }
            )

    # 彙總對比表
    if summary_rows:
        summary_rows.append(
            {
                "source": "全樣本（基準）",
                "n_signals": len(data),
                "label_rate": round(float(data["label"].mean()), 3),
                "precision": round(results["precision"].mean(), 3),
                "auc": round(results["auc"].mean(), 3),
                "beat_base_%": round(
                    (results["precision"].mean() - data["label"].mean()) / data["label"].mean() * 100, 1
                ),
            }
        )
        summary_df = pd.DataFrame(summary_rows)
        print(f"\n{'='*60}")
        print(f"各訊號來源彙總對比")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))

    return results, data, signals, wedge_signals, ma_signals


def _filter_by_triangle(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    只保留「在收鍊確認後 BREAKOUT_WINDOW 個交易日窗口內」的樣本
    用 merge 取代 apply，速度快很多
    """
    if len(signals) == 0:
        return pd.DataFrame()

    all_dates_sorted = sorted(data["date"].unique())
    date_rank = {d: i for i, d in enumerate(all_dates_sorted)}

    # 建立白名單 DataFrame
    rows = []
    for _, row in signals.iterrows():
        sid = row["stock_id"]
        pattern_date = row["pattern_date"]

        if pattern_date not in date_rank:
            continue

        start_rank = date_rank[pattern_date] + 1
        end_rank = start_rank + BREAKOUT_WINDOW
        window_dates = all_dates_sorted[start_rank:end_rank]

        for d in window_dates:
            rows.append({"stock_id": sid, "date": d})

    if not rows:
        return pd.DataFrame()

    whitelist = pd.DataFrame(rows).drop_duplicates()
    filtered = data.merge(whitelist, on=["stock_id", "date"], how="inner")

    return filtered
