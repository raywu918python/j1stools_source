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

# 原始 5 個均線特徵
BASE_FEATURES = [
    "f_dev_ma20",  # 昨收偏離 MA20
    "f_dev_ma60",  # 昨收偏離 MA60  ← 診斷時會移除這個
    "f_ma5_slope",  # MA5 三日斜率
    "f_vol_ratio",  # 量比
    "f_ma_align",  # 均線多頭排列
]

# 新增特徵
NEW_FEATURES = [
    "f_rsi14",  # RSI 14
    "f_macd_hist",  # MACD 柱狀體（動能加速/減速）
    "f_gap",  # 開盤缺口（今開 vs 昨收）
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
def _walk_forward(data: pd.DataFrame, feats: list, label: str = "") -> pd.DataFrame:
    bull = data[data["regime"] == "bull"].dropna(subset=feats + ["label"]).copy()

    valid_feats = _get_valid_feats(bull, feats)
    dropped = set(feats) - set(valid_feats)
    if dropped:
        print(f"  ⚠️  略過特徵（資料不足）：{dropped}")

    if len(bull) == 0:
        print("  ❌ 沒有多頭樣本")
        return pd.DataFrame()

    dates = sorted(bull["date"].unique())
    fold_size = max(1, len(dates) // N_SPLITS)
    rows = []

    tag = f" [{label}]" if label else ""
    print(f"多頭樣本：{len(bull):,} 筆，{len(dates)} 個交易日{tag}")

    for fold in range(2, N_SPLITS + 1):
        tr_dates = dates[: fold_size * (fold - 1)]
        te_dates = dates[fold_size * (fold - 1) : fold_size * fold]

        if len(tr_dates) < MIN_TRAIN:
            continue

        train = bull[bull["date"].isin(tr_dates)]
        test = bull[bull["date"].isin(te_dates)]

        if len(test) < 30 or test["label"].nunique() < 2:
            continue

        clf = _train_xgb(train, valid_feats)
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
def _feature_importance(data: pd.DataFrame, feats: list) -> pd.Series:
    bull = data[data["regime"] == "bull"].dropna(subset=feats + ["label"])
    valid_feats = _get_valid_feats(bull, feats)

    clf = _train_xgb(bull, valid_feats)
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
# 主函式
# ============================================================
def run(price_df: pd.DataFrame, market_df: pd.DataFrame):
    """
    Parameters
    ----------
    price_df  : 欄位 date / stock_id / close（+ open / volume 選用）
    market_df : 欄位 date / close（0050 ETF）

    Returns
    -------
    results : Walk-Forward 各 Fold 結果
    data    : 完整標記資料集
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

    bull = data[data["regime"] == "bull"]
    print(f"\n總樣本         : {len(data):,}")
    print(f"多頭樣本       : {len(bull):,} ({len(bull)/max(len(data),1):.1%})")
    print(f"緩衝帶（丟棄） : {(data['regime']=='buffer').sum():,}")
    print(f"Label=1 比例   : {bull['label'].mean():.1%}  ← 基準線")

    # 有效特徵（去除資料不足的）
    valid_all = _get_valid_feats(bull, FEATURE_COLS)
    print(f"\n使用特徵（{len(valid_all)} 個）：{valid_all}")

    # Walk-Forward（完整特徵）
    print("\n--- Walk-Forward（完整特徵）---")
    results = _walk_forward(data, valid_all)

    if len(results) > 0:
        avg_prec = results["precision"].mean()
        avg_auc = results["auc"].mean()
        label_rate = bull["label"].mean()

        print(f"\n{'='*60}")
        print(f"平均 Precision  : {avg_prec:.3f}  （基準：{label_rate:.3f}）")
        print(f"平均 AUC        : {avg_auc:.3f}  （基準：0.500）")
        print(f"Precision 標準差: {results['precision'].std():.3f}")
        print("\n判讀：")
        if avg_auc > 0.58 and avg_prec > label_rate:
            print("  ✅ 特徵有效 → 可繼續加特徵或加入 1D-CNN")
        elif avg_auc > 0.54:
            print("  ⚠️  訊號微弱 → 建議再加更強特徵")
        else:
            print("  ❌ 無預測力 → 重新檢視 target 定義或特徵設計")
        print(f"{'='*60}")
        print(f"\n{results.to_string(index=False)}")

        # 特徵重要性
        _feature_importance(data, valid_all)

        # 診斷：移除 f_dev_ma60
        if "f_dev_ma60" in valid_all:
            _diagnose(data, results, valid_all)

    return results, data
