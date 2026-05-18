"""
MA5 穿 MA10 + XGBoost 過濾器
============================
Step 1：找所有 MA5 上穿 MA10（股價在 MA60 以上）的訊號
Step 2：XGBoost 學「哪些交叉當天特徵會在 10 天內漲 10%」
Step 3：Walk-Forward 驗證

使用方式：
  from ma_cross_xgb import run
  results, signals = run(price_df, market_df)
"""

import warnings
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# 參數
# ============================================================
MA_SHORT = 5
MA_MID = 10
MA_LONG = 20
MA_REGIME = 60

TARGET_DAYS = 10
TARGET_RET = 0.10

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

# 特徵說明
FEATURE_COLS = [
    "f_dev_ma20",  # 昨收偏離 MA20
    "f_dev_ma60",  # 昨收偏離 MA60
    "f_ma60_slope",  # MA60 五日斜率
    "f_ma5_slope",  # MA5 三日斜率
    "f_ma_align",  # 均線多頭排列程度
    "f_cross_gap",  # 交叉幅度（MA5 超過 MA10 的幅度）
    "f_vol_ratio",  # 量比
    "f_rsi14",  # RSI 14
    "f_macd_hist",  # MACD 柱狀體
    "f_gap",  # 開盤缺口
    "f_rel_mkt",  # 相對大盤 5 日強弱
]


# ============================================================
# 資料清理
# ============================================================
def _clean_price(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.strip().str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d["stock_id"] = d["stock_id"].astype(str).str.strip()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    if "high" not in d.columns:
        d["high"] = d["close"]
    if "volume" not in d.columns:
        d["volume"] = np.nan
    if "open" not in d.columns:
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
    return (
        d[["date", "close"]]
        .rename(columns={"close": "market_close"})
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


# ============================================================
# 特徵計算 + 訊號偵測（per stock）
# ============================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-6)
    return 100 - 100 / (1 + rs)


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _process_stock(grp: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)
    d = d.merge(market, on="date", how="left")

    c = d["close"].shift(1)  # 昨收（13:00 可見）
    ma5 = c.rolling(MA_SHORT).mean()
    ma10 = c.rolling(MA_MID).mean()
    ma20 = c.rolling(MA_LONG).mean()
    ma60 = c.rolling(MA_REGIME).mean()

    # 進場條件
    above_ma60 = d["close"] > ma60  # 股價在 MA60 以上
    cross_5_10 = (ma5.shift(1) < ma10.shift(1)) & (ma5 > ma10)  # 黃金交叉
    d["signal"] = above_ma60 & cross_5_10

    # 特徵（用昨收，13:00 可見）
    d["f_dev_ma20"] = (c - ma20) / ma20.clip(lower=1e-6)
    d["f_dev_ma60"] = (c - ma60) / ma60.clip(lower=1e-6)
    d["f_ma60_slope"] = (ma60 - ma60.shift(5)) / ma60.shift(5).clip(lower=1e-6)
    d["f_ma5_slope"] = (ma5 - ma5.shift(3)) / ma5.shift(3).clip(lower=1e-6)
    d["f_ma_align"] = (ma5 - ma10) / ma10.clip(lower=1e-6) + (ma10 - ma20) / ma20.clip(lower=1e-6)
    d["f_cross_gap"] = (ma5 - ma10) / ma10.clip(lower=1e-6)  # 交叉幅度

    if d["volume"].notna().any():
        v = d["volume"].shift(1)
        d["f_vol_ratio"] = v.rolling(5).mean() / v.rolling(20).mean().clip(lower=1e-6)
    else:
        d["f_vol_ratio"] = np.nan

    d["f_rsi14"] = _rsi(c, 14)
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd = ema12 - ema26
    d["f_macd_hist"] = macd - _ema(macd, 9)

    if d["open"].notna().any():
        d["f_gap"] = (d["open"] - d["close"].shift(1)) / d["close"].shift(1).clip(lower=1e-6)
    else:
        d["f_gap"] = np.nan

    ret5_stock = d["close"].shift(1).pct_change(5)
    ret5_market = d["market_close"].shift(1).pct_change(5)
    d["f_rel_mkt"] = ret5_stock - ret5_market

    # Label：10 天內最高價 >= 進場收盤 × 1.10
    future_max = pd.concat([d["high"].shift(-i) for i in range(1, TARGET_DAYS + 1)], axis=1).max(axis=1)
    d["future_max_high"] = future_max
    d["label"] = (future_max >= d["close"] * (1 + TARGET_RET)).astype(int)

    # 只回傳有訊號且 label 有效的行
    sig = d[d["signal"] & d["future_max_high"].notna()].copy()
    return sig


# ============================================================
# 訓練輔助
# ============================================================
def _get_valid_feats(df, feats):
    return [f for f in feats if f in df.columns and df[f].notna().sum() > 50]


def _train_xgb(train, feats):
    neg = (train["label"] == 0).sum()
    pos = (train["label"] == 1).sum()
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
# Walk-Forward（按年切）
# ============================================================
def _walk_forward(signals: pd.DataFrame) -> pd.DataFrame:
    valid_feats = _get_valid_feats(signals, FEATURE_COLS)
    d = signals.dropna(subset=valid_feats + ["label"]).copy()

    min_year = d["date"].dt.year.min()
    max_year = d["date"].dt.year.max()
    baseline = d["label"].mean()

    print(f"訊號樣本：{len(d):,} 筆 | {min_year} ~ {max_year}")
    print(f"基準勝率：{baseline:.1%}（隨機進場）")

    rows = []
    for test_year in range(min_year + 1, max_year + 1):
        train = d[d["date"].dt.year < test_year]
        test = d[d["date"].dt.year == test_year]

        train_days = train["date"].nunique()
        if train_days < MIN_TRAIN:
            print(f"  {test_year} 跳過：訓練天數不足 ({train_days} < {MIN_TRAIN})")
            continue
        if len(test) < 30 or test["label"].nunique() < 2:
            print(f"  {test_year} 跳過：測試集不足")
            continue

        clf = _train_xgb(train, valid_feats)
        y_pred = clf.predict(test[valid_feats])
        y_prob = clf.predict_proba(test[valid_feats])[:, 1]

        prec = precision_score(test["label"], y_pred, zero_division=0)
        auc = roc_auc_score(test["label"], y_prob)
        n_sig = int(y_pred.sum())

        rows.append(
            {
                "test_year": test_year,
                "train_n": len(train),
                "test_n": len(test),
                "baseline": round(float(test["label"].mean()), 3),
                "precision": round(prec, 3),
                "auc": round(auc, 3),
                "n_signals": n_sig,
            }
        )

        print(
            f"  {test_year} | Train ({len(train):,}) → Test ({len(test):,}) | "
            f"基準 {test['label'].mean():.1%}  "
            f"Precision={prec:.3f}  AUC={auc:.3f}  訊號={n_sig}"
        )

    return pd.DataFrame(rows)


# ============================================================
# 特徵重要性
# ============================================================
def _feature_importance(signals: pd.DataFrame) -> None:
    valid_feats = _get_valid_feats(signals, FEATURE_COLS)
    d = signals.dropna(subset=valid_feats + ["label"])
    clf = _train_xgb(d, valid_feats)
    imp = pd.Series(clf.feature_importances_, index=valid_feats).sort_values(ascending=False)
    print("\n--- 特徵重要性 ---")
    for feat, score in imp.items():
        bar = "█" * int(score * 50)
        print(f"  {feat:<20} {bar}  {score:.4f}")


# ============================================================
# 主函式
# ============================================================
def run(price_df: pd.DataFrame, market_df: pd.DataFrame) -> tuple:
    print("=" * 60)
    print("MA5 穿 MA10 + XGBoost 過濾器")
    print("=" * 60)

    price_df = _clean_price(price_df)
    market_df = _clean_market(market_df)

    s = max(price_df["date"].min(), market_df["date"].min())
    e = min(price_df["date"].max(), market_df["date"].max())
    price_df = price_df[(price_df["date"] >= s) & (price_df["date"] <= e)]
    market_df = market_df[(market_df["date"] >= s) & (market_df["date"] <= e)]
    print(f"股票：{price_df['stock_id'].nunique()} 支 | 日期：{s.date()} ~ {e.date()}")

    # 計算訊號 + 特徵
    print("\n計算訊號與特徵...")
    all_signals = []
    skipped = 0
    min_rows = MA_REGIME + TARGET_DAYS + 30

    for sid, grp in price_df.groupby("stock_id"):
        if len(grp) < min_rows:
            skipped += 1
            continue
        try:
            sig = _process_stock(grp, market_df)
            if len(sig) > 0:
                all_signals.append(sig)
        except Exception:
            skipped += 1

    if skipped:
        print(f"⚠️  跳過 {skipped} 支股票")

    signals = pd.concat(all_signals, ignore_index=True)
    signals = signals.sort_values(["date", "stock_id"]).reset_index(drop=True)

    total = len(signals)
    baseline = signals["label"].mean()
    print(f"\n總訊號數   : {total:,}")
    print(f"基準勝率   : {baseline:.1%}  （10天內漲10%的比例）")

    # Walk-Forward
    print("\n--- Walk-Forward ---")
    results = _walk_forward(signals)

    if len(results) > 0:
        avg_prec = results["precision"].mean()
        avg_auc = results["auc"].mean()
        avg_base = results["baseline"].mean()
        beat = (avg_prec - avg_base) / avg_base * 100

        print(f"\n{'='*60}")
        print(f"平均基準勝率   : {avg_base:.1%}")
        print(f"平均 Precision : {avg_prec:.3f}  （超越基準 {beat:+.1f}%）")
        print(f"平均 AUC       : {avg_auc:.3f}")
        print(f"Precision 標準差: {results['precision'].std():.3f}")
        print("\n判讀：")
        if avg_auc > 0.58 and avg_prec > avg_base:
            print("  ✅ XGBoost 有效過濾，Precision 提升")
        elif avg_prec > avg_base:
            print("  ⚠️  Precision 有提升但 AUC 不足，訊號方向對但區分力有限")
        else:
            print("  ❌ XGBoost 未能提升勝率，考慮調整特徵或 target")
        print(f"{'='*60}")
        print(f"\n{results.to_string(index=False)}")

    _feature_importance(signals)

    return results, signals
