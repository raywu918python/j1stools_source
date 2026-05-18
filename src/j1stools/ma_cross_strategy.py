"""
MA5 上穿 MA10 策略統計
======================
進場條件：
  1. 股價在 MA60 以上（收盤 > MA60）
  2. MA5 上穿 MA10（黃金交叉當天）

目標：
  進場後 10 天內，最高價 >= 進場收盤價 × 1.10

統計：
  整體勝率、各年勝率、各種市況下的勝率

使用方式：
  from ma_cross_strategy import run
  results = run(price_df, market_df)

price_df  欄位：date, stock_id, close, high（+ open/volume 選用）
market_df 欄位：date, close（0050 ETF，用來標記多空市場背景）
"""

import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 參數
# ============================================================
MA_SHORT = 5
MA_MID = 10
MA_LONG = 20
MA_REGIME = 60

TARGET_DAYS = 10  # 觀察天數
TARGET_RET = 0.10  # 目標報酬 10%

REGIME_BUF = 0.05  # MA60 緩衝帶
REGIME_DAYS = 3  # 站上 MA60+5% 需持續天數


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
        print("⚠️  price_df 沒有 high 欄位，用 close 代替")
        d["high"] = d["close"]

    before = len(d)
    d = d[d["close"] > 0].dropna(subset=["close"])
    if len(d) < before:
        print(f"⚠️  移除 {before - len(d)} 筆收盤價異常")

    return d.sort_values(["stock_id", "date"]).reset_index(drop=True)


# ============================================================
# 單支股票：計算訊號 + 結果
# ============================================================
def _process_stock(grp: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)

    # 均線
    d["ma5"] = d["close"].rolling(MA_SHORT).mean()
    d["ma10"] = d["close"].rolling(MA_MID).mean()
    d["ma20"] = d["close"].rolling(MA_LONG).mean()
    d["ma60"] = d["close"].rolling(MA_REGIME).mean()

    # MA60 機制標記（簡化版：收盤 > MA60 就算多頭）
    d["above_ma60"] = d["close"] > d["ma60"]

    # MA5 上穿 MA10：昨日 MA5 < MA10，今日 MA5 > MA10
    d["cross_5_10"] = (d["ma5"].shift(1) < d["ma10"].shift(1)) & (d["ma5"] > d["ma10"])

    # 進場條件：在 MA60 以上 + 黃金交叉
    d["signal"] = d["above_ma60"] & d["cross_5_10"]

    # 建立未來 10 天最高價（不含進場當天）
    # 用 shift(-1) 到 shift(-10) 的 high 取最大值
    future_highs = pd.concat([d["high"].shift(-i) for i in range(1, TARGET_DAYS + 1)], axis=1).max(axis=1)

    d["future_max_high"] = future_highs
    d["target_price"] = d["close"] * (1 + TARGET_RET)
    d["win"] = d["future_max_high"] >= d["target_price"]

    # 只回傳有訊號的行
    signals = d[d["signal"] & d["future_max_high"].notna()].copy()
    return signals[
        ["date", "stock_id", "close", "ma5", "ma10", "ma60", "above_ma60", "future_max_high", "target_price", "win"]
    ]


# ============================================================
# 主函式
# ============================================================
def run(price_df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    price_df  : 欄位 date / stock_id / close / high
    market_df : 欄位 date / close（0050，選用，用來標記大盤多空背景）

    Returns
    -------
    signals : 所有進場訊號 DataFrame，含 win 欄位
    """
    print("=" * 60)
    print("MA5 上穿 MA10 策略（股價在 MA60 以上）")
    print("=" * 60)

    price_df = _clean_price(price_df)
    print(
        f"股票：{price_df['stock_id'].nunique()} 支 | "
        f"日期：{price_df['date'].min().date()} ~ {price_df['date'].max().date()}"
    )

    # 大盤 MA60（用來標記大盤多空背景，選用）
    if market_df is not None:
        mkt = market_df.copy()
        mkt.columns = mkt.columns.str.strip().str.lower()
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt["mkt_ma60"] = mkt["close"].rolling(MA_REGIME).mean()
        mkt["mkt_above_ma60"] = mkt["close"] > mkt["mkt_ma60"]
        mkt = mkt[["date", "close", "mkt_ma60", "mkt_above_ma60"]].rename(columns={"close": "mkt_close"})

    # 逐支股票計算訊號
    all_signals = []
    skipped = 0
    min_rows = MA_REGIME + TARGET_DAYS + 5

    for sid, grp in price_df.groupby("stock_id"):
        if len(grp) < min_rows:
            skipped += 1
            continue
        try:
            sig = _process_stock(grp)
            if len(sig) > 0:
                all_signals.append(sig)
        except Exception:
            skipped += 1

    if skipped:
        print(f"⚠️  跳過 {skipped} 支資料不足的股票")

    if not all_signals:
        print("❌ 沒有找到任何訊號")
        return pd.DataFrame()

    signals = pd.concat(all_signals, ignore_index=True)
    signals = signals.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # 合併大盤背景
    if market_df is not None:
        signals = signals.merge(mkt, on="date", how="left")

    _print_stats(signals)
    return signals


# ============================================================
# 統計輸出
# ============================================================
def _print_stats(signals: pd.DataFrame) -> None:
    total = len(signals)
    wins = signals["win"].sum()
    win_rate = wins / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"整體統計")
    print(f"{'='*60}")
    print(f"總訊號數   : {total:,}")
    print(f"達標（win）: {wins:,}")
    print(f"勝率       : {win_rate:.1%}")
    print(f"目標       : 10天內最高價 ≥ 進場收盤 × 1.10")

    # 各年勝率
    signals["year"] = signals["date"].dt.year
    yearly = signals.groupby("year")["win"].agg(["sum", "count"]).rename(columns={"sum": "wins", "count": "total"})
    yearly["win_rate"] = yearly["wins"] / yearly["total"]

    print(f"\n--- 各年勝率 ---")
    for yr, row in yearly.iterrows():
        bar = "█" * int(row["win_rate"] * 30)
        print(f"  {yr}  {bar:<30}  {row['win_rate']:.1%}  " f"({int(row['wins'])}/{int(row['total'])})")

    # 大盤多空背景（若有）
    if "mkt_above_ma60" in signals.columns:
        print(f"\n--- 大盤多空背景下的勝率 ---")
        for is_bull, label in [(True, "大盤多頭（0050 > MA60）"), (False, "大盤空頭（0050 < MA60）")]:
            sub = signals[signals["mkt_above_ma60"] == is_bull]
            if len(sub) == 0:
                continue
            wr = sub["win"].mean()
            print(f"  {label}")
            print(f"    訊號數 {len(sub):,}  勝率 {wr:.1%}")

    # 月份分佈
    signals["month"] = signals["date"].dt.month
    monthly = signals.groupby("month")["win"].agg(["mean", "count"])
    print(f"\n--- 月份勝率 ---")
    for m, row in monthly.iterrows():
        bar = "█" * int(row["mean"] * 20)
        print(f"  {m:2d}月  {bar:<20}  {row['mean']:.1%}  ({int(row['count'])} 筆)")

    # 進場價距離 MA60 分佈
    signals["dev_ma60"] = (signals["close"] - signals["ma60"]) / signals["ma60"]
    bins = [0, 0.03, 0.07, 0.15, 0.30, float("inf")]
    labels = ["0-3%", "3-7%", "7-15%", "15-30%", ">30%"]
    signals["ma60_band"] = pd.cut(signals["dev_ma60"], bins=bins, labels=labels)
    band_stats = signals.groupby("ma60_band", observed=True)["win"].agg(["mean", "count"])
    print(f"\n--- 進場時距離 MA60 幅度 vs 勝率 ---")
    for band, row in band_stats.iterrows():
        bar = "█" * int(row["mean"] * 30)
        print(f"  {str(band):<8}  {bar:<30}  {row['mean']:.1%}  ({int(row['count']):,} 筆)")

    print(f"{'='*60}")
