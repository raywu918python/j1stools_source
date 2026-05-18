"""
B-Bull MVP
==========
直接傳入 DataFrame，不讀 CSV。

使用方式：
    from b_bull_mvp import run

    results, data = run(price_df, market_df)

參數：
    price_df  : 欄位 date, stock_id, close（+ volume 選用，open/high/low 忽略）
    market_df : 欄位 date, close（0050 ETF）
"""

import random
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score

from j1stools import parquet_db

warnings.filterwarnings("ignore")

# ============================================================
# 超參數（全部集中在這，改這裡就好）
# ============================================================
MA_SHORT = 5
MA_MID = 10
MA_LONG = 20
MA_REGIME = 60

REGIME_BUF = 0.05  # 緩衝帶 +5%
REGIME_DAYS = 3  # 站上 MA60+5% 需持續天數

LABEL_DAYS = 5  # 進場後觀察幾天
LABEL_THR = 0.02  # 超額報酬門檻（vs 0050）

N_SPLITS = 5
MIN_TRAIN = 252  # 最少訓練樣本天數

RFC_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

FEATURE_COLS = [
    "f_dev_ma20",  # 昨收偏離 MA20
    "f_dev_ma60",  # 昨收偏離 MA60
    "f_ma5_slope",  # MA5 三日斜率
    "f_vol_ratio",  # 量比（無 volume 時略過）
    "f_ma_align",  # 均線多頭排列程度
]


# ============================================================
# 內部：資料清理
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
# 特徵計算（per stock，shift(1) 不看今日收盤）
# ============================================================
def _compute_features(grp: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)
    c = d["close"].shift(1)  # 昨收

    ma5 = c.rolling(MA_SHORT).mean()
    ma10 = c.rolling(MA_MID).mean()
    ma20 = c.rolling(MA_LONG).mean()
    ma60 = c.rolling(MA_REGIME).mean()

    d["f_dev_ma20"] = (c - ma20) / ma20.clip(lower=1e-6)
    d["f_dev_ma60"] = (c - ma60) / ma60.clip(lower=1e-6)
    d["f_ma5_slope"] = (ma5 - ma5.shift(3)) / ma5.shift(3).clip(lower=1e-6)
    d["f_ma_align"] = (ma5 - ma10) / ma10.clip(lower=1e-6) + (ma10 - ma20) / ma20.clip(lower=1e-6)

    if d["volume"].notna().any():
        v = d["volume"].shift(1)
        d["f_vol_ratio"] = v.rolling(5).mean() / v.rolling(20).mean().clip(lower=1e-6)
    else:
        d["f_vol_ratio"] = np.nan

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
def _compute_label(grp: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    d = grp.merge(market, on="date", how="left")
    ret_s = d["close"].shift(-LABEL_DAYS) / d["close"] - 1
    ret_m = d["market_close"].shift(-LABEL_DAYS) / d["market_close"] - 1
    d["excess_ret"] = ret_s - ret_m
    d["label"] = (d["excess_ret"] > LABEL_THR).astype(int)
    return d


# ============================================================
# 建構全資料集
# ============================================================
def _build_dataset(price_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    min_rows = MA_REGIME + LABEL_DAYS + 10
    results, skipped = [], 0

    for sid, grp in price_df.groupby("stock_id"):
        if len(grp) < min_rows:
            skipped += 1
            continue
        g = _compute_features(grp)
        g = _label_regime(g)
        g = _compute_label(g, market_df)
        results.append(g)

    if skipped:
        print(f"⚠️  跳過 {skipped} 支資料不足的股票（需 ≥ {min_rows} 筆）")

    data = pd.concat(results, ignore_index=True)
    return data.dropna(subset=["label", "excess_ret"])


# ============================================================
# Walk-Forward
# ============================================================
def _walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    bull = data[data["regime"] == "bull"].dropna(subset=FEATURE_COLS + ["label"]).copy()

    valid_feats = [f for f in FEATURE_COLS if bull[f].notna().sum() > 100]
    dropped = set(FEATURE_COLS) - set(valid_feats)
    if dropped:
        print(f"⚠️  略過特徵（資料不足）：{dropped}")

    if len(bull) == 0:
        print("❌ 沒有多頭樣本，請檢查資料或 REGIME_BUF 參數")
        return pd.DataFrame()

    dates = sorted(bull["date"].unique())
    fold_size = max(1, len(dates) // N_SPLITS)
    rows = []

    print(f"多頭樣本：{len(bull):,} 筆，{len(dates)} 個交易日")

    for fold in range(2, N_SPLITS + 1):
        tr_dates = dates[: fold_size * (fold - 1)]
        te_dates = dates[fold_size * (fold - 1) : fold_size * fold]

        if len(tr_dates) < MIN_TRAIN:
            print(f"  Fold {fold} 跳過：訓練天數不足 ({len(tr_dates)} < {MIN_TRAIN})")
            continue

        train = bull[bull["date"].isin(tr_dates)]
        test = bull[bull["date"].isin(te_dates)]

        if len(test) < 30 or test["label"].nunique() < 2:
            print(f"  Fold {fold} 跳過：測試集不足或 label 單一")
            continue

        clf = RandomForestClassifier(**RFC_PARAMS)
        clf.fit(train[valid_feats], train["label"])

        y_pred = clf.predict(test[valid_feats])
        y_prob = clf.predict_proba(test[valid_feats])[:, 1]

        prec = precision_score(test["label"], y_pred, zero_division=0)
        auc = roc_auc_score(test["label"], y_prob)

        rows.append(
            {
                "fold": fold,
                "train_start": str(tr_dates[0])[:10],
                "train_end": str(tr_dates[-1])[:10],
                "test_start": str(te_dates[0])[:10],
                "test_end": str(te_dates[-1])[:10],
                "train_n": len(train),
                "test_n": len(test),
                "label_rate": round(float(train["label"].mean()), 3),
                "precision": round(prec, 3),
                "auc": round(auc, 3),
                "n_signals": int(y_pred.sum()),
            }
        )

        print(
            f"  Fold {fold} | "
            f"{str(tr_dates[0])[:10]}~{str(tr_dates[-1])[:10]} ({len(train):,}) → "
            f"{str(te_dates[0])[:10]}~{str(te_dates[-1])[:10]} | "
            f"Precision={prec:.3f}  AUC={auc:.3f}  訊號={int(y_pred.sum())}"
        )

    return pd.DataFrame(rows)


# ============================================================
# 特徵重要性
# ============================================================
def _feature_importance(data: pd.DataFrame) -> pd.Series:
    bull = data[data["regime"] == "bull"].dropna(subset=FEATURE_COLS + ["label"])
    valid_feats = [f for f in FEATURE_COLS if bull[f].notna().sum() > 100]

    clf = RandomForestClassifier(**RFC_PARAMS)
    clf.fit(bull[valid_feats], bull["label"])

    imp = pd.Series(clf.feature_importances_, index=valid_feats).sort_values(ascending=False)
    print("\n--- 特徵重要性 ---")
    for feat, score in imp.items():
        bar = "█" * int(score * 50)
        print(f"  {feat:<20} {bar}  {score:.4f}")
    return imp


# ============================================================
# 主函式（對外接口）
# ============================================================
def run(price_df: pd.DataFrame, market_df: pd.DataFrame):
    """
    Parameters
    ----------
    price_df  : 個股長表格，欄位 date / stock_id / close（volume 選用）
    market_df : 0050 日頻資料，欄位 date / close

    Returns
    -------
    results : Walk-Forward 各 Fold 結果 DataFrame
    data    : 完整標記後的資料集（含特徵、regime、label）
    """
    print("=" * 60)
    print("B-Bull MVP")
    print("=" * 60)

    # 清理
    price_df = _clean_price(price_df)
    market_df = _clean_market(market_df)

    # 日期交集
    s = max(price_df["date"].min(), market_df["date"].min())
    e = min(price_df["date"].max(), market_df["date"].max())
    price_df = price_df[(price_df["date"] >= s) & (price_df["date"] <= e)]
    market_df = market_df[(market_df["date"] >= s) & (market_df["date"] <= e)]

    print(f"股票：{price_df['stock_id'].nunique()} 支 | " f"日期：{s.date()} ~ {e.date()}")

    # 建構資料集
    print("\n計算特徵與標記...")
    data = _build_dataset(price_df, market_df)

    bull = data[data["regime"] == "bull"]
    print(f"\n總樣本         : {len(data):,}")
    print(f"多頭樣本       : {len(bull):,} ({len(bull)/max(len(data),1):.1%})")
    print(f"緩衝帶（丟棄） : {(data['regime']=='buffer').sum():,}")
    print(f"Label=1 比例   : {bull['label'].mean():.1%}  ← 基準線")

    # Walk-Forward
    print("\n--- Walk-Forward ---")
    results = _walk_forward(data)

    if len(results) > 0:
        avg_prec = results["precision"].mean()
        avg_auc = results["auc"].mean()
        label_rate = bull["label"].mean()

        print(f"\n{'='*60}")
        print(f"平均 Precision  : {avg_prec:.3f}  （基準：{label_rate:.3f}）")
        print(f"平均 AUC        : {avg_auc:.3f}  （基準：0.500）")
        print(f"Precision 標準差: {results['precision'].std():.3f}")
        print("\n判讀：")
        if avg_auc > 0.55 and avg_prec > label_rate:
            print("  ✅ 特徵有效 → 可繼續加特徵或升級 XGBoost")
        elif avg_auc > 0.52:
            print("  ⚠️  訊號微弱 → 建議加更多特徵（RSI、MACD、籌碼）")
        else:
            print("  ❌ 無預測力 → 重新檢視 target 定義或特徵設計")
        print(f"{'='*60}")
        print(f"\n{results.to_string(index=False)}")

    _feature_importance(data)

    return results, data


stocks = random.sample(parquet_db.query_stocks_ids_list(), 100)
price_df = parquet_db.query_price(stocks, "2015-01-01", "2024-01-01")
market_df = parquet_db.query_price(["0050"], "2015-01-01", "2024-01-01")
results, data = run(price_df, market_df)
