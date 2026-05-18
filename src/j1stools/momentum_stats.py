"""
動能突破訊號統計
================
條件：
  1. 收盤 > MA60（大趨勢向上）
  2. 當日 ATR > 昨日 ATR（波動擴張）
  3. 當日 K 棒振幅（high-low）> 前10日平均振幅
  4. MA5 > MA10（短期多頭排列）

統計：
  進場後 10 天的最高漲幅分佈
  不設固定目標，看實際能漲多少

使用方式：
  from momentum_stats import run
  results = run(price_df, market_df)
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

MA_SHORT = 5
MA_MID = 10
MA_LONG = 20
MA_REGIME = 60
ATR_PERIOD = 14
LOOK_FORWARD = 10  # 進場後觀察幾天


def _calc_atr(df: pd.DataFrame) -> pd.Series:
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


def _process_stock(grp: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)

    c = d["close"]
    ma5 = c.rolling(MA_SHORT).mean()
    ma10 = c.rolling(MA_MID).mean()
    ma60 = c.rolling(MA_REGIME).mean()

    atr = _calc_atr(d)
    candle = d["high"] - d["low"]  # 當日振幅
    avg_candle = candle.rolling(10).mean().shift(1)  # 前10日平均振幅

    # 4 個條件（用當日資料，不 shift，模擬收盤後確認）
    cond1 = c > ma60  # MA60 以上
    cond2 = atr > atr.shift(1)  # ATR 上升
    cond3 = candle > avg_candle  # 當日振幅 > 前10日均
    cond4 = ma5 > ma10  # MA5 > MA10

    d["signal"] = cond1 & cond2 & cond3 & cond4

    # 進場後 LOOK_FORWARD 天的最高漲幅
    future_max = pd.concat([d["high"].shift(-i) for i in range(1, LOOK_FORWARD + 1)], axis=1).max(axis=1)

    future_close = d["close"].shift(-LOOK_FORWARD)

    d["future_max_ret"] = (future_max - c) / c * 100
    d["future_close_ret"] = (future_close - c) / c * 100

    sig = d[d["signal"] & d["future_max_ret"].notna()].copy()
    return sig[["date", "stock_id", "close", "future_max_ret", "future_close_ret"]]


def run(price_df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:

    print("=" * 60)
    print("動能突破訊號統計")
    print("=" * 60)
    print("條件：MA60以上 + ATR上升 + 大K棒 + MA5>MA10")

    prc = price_df.copy()
    prc.columns = prc.columns.str.strip().str.lower()
    prc["date"] = pd.to_datetime(prc["date"])
    prc["stock_id"] = prc["stock_id"].astype(str).str.strip()

    if "high" not in prc.columns:
        prc["high"] = prc["close"]
    if "low" not in prc.columns:
        prc["low"] = prc["close"]

    all_signals = []
    skipped = 0
    min_rows = MA_REGIME + LOOK_FORWARD + 15

    for sid, grp in prc.groupby("stock_id"):
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
        print(f"⚠️  跳過 {skipped} 支股票")

    signals = pd.concat(all_signals, ignore_index=True)
    signals = signals.sort_values(["date", "stock_id"]).reset_index(drop=True)

    _print_stats(signals)
    return signals


def _print_stats(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n總訊號數      : {n:,}")
    print(f"股票數        : {df['stock_id'].nunique()}")
    print(f"日期範圍      : {df['date'].min().date()} ~ {df['date'].max().date()}")

    mr = df["future_max_ret"]
    cr = df["future_close_ret"]

    print(f"\n{'='*60}")
    print(f"進場後 {LOOK_FORWARD} 天最高漲幅分佈")
    print(f"{'='*60}")
    print(f"  平均最高漲幅   : {mr.mean():.2f}%")
    print(f"  中位數         : {mr.median():.2f}%")
    print(f"  25 分位        : {mr.quantile(0.25):.2f}%")
    print(f"  75 分位        : {mr.quantile(0.75):.2f}%")

    print(f"\n  達到 +5%  的比例  : {(mr >= 5).mean():.1%}")
    print(f"  達到 +10% 的比例  : {(mr >= 10).mean():.1%}")
    print(f"  達到 +15% 的比例  : {(mr >= 15).mean():.1%}")
    print(f"  達到 +20% 的比例  : {(mr >= 20).mean():.1%}")

    print(f"\n{'='*60}")
    print(f"進場後 {LOOK_FORWARD} 天收盤漲幅分佈")
    print(f"{'='*60}")
    print(f"  平均收盤漲幅   : {cr.mean():.2f}%")
    print(f"  正報酬比例     : {(cr > 0).mean():.1%}")
    print(f"  負報酬比例     : {(cr < 0).mean():.1%}")
    print(f"  > +5%  比例    : {(cr >= 5).mean():.1%}")
    print(f"  > +10% 比例    : {(cr >= 10).mean():.1%}")
    print(f"  < -5%  比例    : {(cr <= -5).mean():.1%}")
    print(f"  < -10% 比例    : {(cr <= -10).mean():.1%}")

    # 各年統計
    df = df.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year")["future_max_ret"].agg(["mean", "median", "count"])
    print(f"\n{'='*60}")
    print(f"各年平均最高漲幅")
    print(f"{'='*60}")
    for yr, row in yearly.iterrows():
        bar = "█" * int(max(row["mean"], 0) * 2)
        print(f"  {yr}  {bar:<20}  均 {row['mean']:.1f}%  中位 {row['median']:.1f}%  ({int(row['count'])} 筆)")

    print(f"{'='*60}")
