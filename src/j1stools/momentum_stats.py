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

# ============================================================
# 條件開關（True = 啟用，False = 關閉）
# ============================================================
USE_COND1 = True  # 收盤 > MA60
USE_COND2 = True  # ATR > 昨日 ATR（波動擴張）
USE_COND3 = True  # 當日振幅 > 前10日平均振幅（大K棒）
USE_COND4 = True  # MA5 > MA10
USE_COND5 = True  # 連3根上漲且都在 MA60 以上
USE_COND6 = True  # MA60 過去10天斜率 > 0.5%
USE_MAX_DEV_MA60 = True  # 過濾距離 MA60 超過 50% 的訊號
MAX_DEV_MA60 = 0.50  # 距離上限（預設 50%）
USE_DEDUP = True  # 同支股票間隔 LOOK_FORWARD 天去重


def _calc_atr(df: pd.DataFrame) -> pd.Series:
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


def _calc_ma60_turn_days(df: pd.DataFrame) -> pd.Series:
    """
    計算每天距離「MA60 斜率從負轉正」的天數
    負轉正 = 昨日 MA60 斜率 < 0，今日 MA60 斜率 >= 0
    回傳：每個交易日距離最近一次轉正的天數
    """
    ma60 = df["close"].rolling(60).mean()
    ma60_slope = ma60 - ma60.shift(5)  # 5日斜率

    # 找轉正點：昨日斜率 < 0，今日斜率 >= 0
    turn_positive = (ma60_slope.shift(1) < 0) & (ma60_slope >= 0)

    # 每天距離最近轉正點的天數
    days_since_turn = pd.Series(np.nan, index=df.index)
    last_turn = None
    for i, (idx, is_turn) in enumerate(turn_positive.items()):
        if is_turn:
            last_turn = i
        if last_turn is not None:
            days_since_turn[idx] = i - last_turn

    return days_since_turn


def _process_stock(grp: pd.DataFrame) -> pd.DataFrame:
    d = grp.copy().sort_values("date").reset_index(drop=True)

    c = d["close"]
    ma5 = c.rolling(MA_SHORT).mean()
    ma10 = c.rolling(MA_MID).mean()
    ma60 = c.rolling(MA_REGIME).mean()

    atr = _calc_atr(d)
    candle = d["high"] - d["low"]
    avg_candle = candle.rolling(10).mean().shift(1)

    # 4 個條件
    # 各條件（可透過開關控制）
    true_series = pd.Series(True, index=d.index)

    cond1 = (c > ma60) if USE_COND1 else true_series
    cond2 = (atr > atr.shift(1)) if USE_COND2 else true_series
    cond3 = (candle > avg_candle) if USE_COND3 else true_series
    cond4 = (ma5 > ma10) if USE_COND4 else true_series
    cond5 = (
        (
            (c > c.shift(1))
            & (c.shift(1) > c.shift(2))
            & (c.shift(2) > c.shift(3))
            & (c.shift(1) > ma60.shift(1))
            & (c.shift(2) > ma60.shift(2))
        )
        if USE_COND5
        else true_series
    )
    ma60_slope = (ma60 - ma60.shift(10)) / ma60.shift(10)
    cond6 = (ma60_slope > 0.005) if USE_COND6 else true_series

    # 距離 MA60 上限過濾
    dev_ma60 = (c - ma60) / ma60.clip(lower=1e-6)
    cond_max_dev = (dev_ma60 <= MAX_DEV_MA60) if USE_MAX_DEV_MA60 else true_series

    d["signal"] = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond_max_dev

    # 去重：同一支股票，兩個訊號之間至少間隔 LOOK_FORWARD 天
    if USE_DEDUP:
        signal_rows = d[d["signal"]].copy()
        if len(signal_rows) > 0:
            valid = []
            last_date = pd.Timestamp("2000-01-01")
            for idx, row in signal_rows.iterrows():
                if (row["date"] - last_date).days >= LOOK_FORWARD:
                    valid.append(idx)
                    last_date = row["date"]
            mask = pd.Series(False, index=d.index)
            if valid:
                mask[valid] = True
            d["signal"] = mask

    # 進場後 LOOK_FORWARD 天的最高漲幅
    future_max = pd.concat([d["high"].shift(-i) for i in range(1, LOOK_FORWARD + 1)], axis=1).max(axis=1)

    future_close = d["close"].shift(-LOOK_FORWARD)

    d["future_max_ret"] = (future_max - c) / c * 100
    d["future_close_ret"] = (future_close - c) / c * 100

    # 只保留 future_close_ret 也有值的樣本（確保 10 天窗口完整）
    d["days_since_ma60_turn"] = _calc_ma60_turn_days(d)
    d["dev_ma60"] = (d["close"] - ma60) / ma60  # 收盤距離 MA60 的幅度

    sig = d[d["signal"] & d["future_max_ret"].notna() & d["future_close_ret"].notna()].copy()
    return sig[["date", "stock_id", "close", "future_max_ret", "future_close_ret", "days_since_ma60_turn", "dev_ma60"]]


def run(price_df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:

    print("=" * 60)
    print("動能突破訊號統計")
    print("=" * 60)
    active = []
    if USE_COND1:
        active.append("MA60以上")
    if USE_COND2:
        active.append("ATR上升")
    if USE_COND3:
        active.append("大K棒")
    if USE_COND4:
        active.append("MA5>MA10")
    if USE_COND5:
        active.append("連3根上漲(MA60以上)")
    if USE_COND6:
        active.append("MA60斜率>0.5%")
    if USE_MAX_DEV_MA60:
        active.append(f"距MA60<{MAX_DEV_MA60:.0%}")
    dedup_str = f"（同支股票間隔{LOOK_FORWARD}天去重）" if USE_DEDUP else "（不去重）"
    print(f"條件：{' + '.join(active)}{dedup_str}")

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
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  股票 {sid} 錯誤：{e}")

    if skipped:
        print(f"⚠️  跳過 {skipped} 支股票")

    if not all_signals:
        print("❌ 沒有找到任何訊號，請檢查資料欄位（需要 high / low）")
        return pd.DataFrame()

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

    # 失敗案例：最高漲幅 < 0
    failures = df[df["future_max_ret"] < 0].copy()
    if len(failures) > 0:
        print(f"\n{'='*60}")
        print(f"失敗案例（10天最高點仍低於進場價）")
        print(f"{'='*60}")
        print(f"失敗筆數      : {len(failures):,}  ({len(failures)/len(df):.1%})")
        print(f"平均下跌幅度  : {failures['future_max_ret'].mean():.2f}%")
        print(f"平均收盤跌幅  : {failures['future_close_ret'].mean():.2f}%")

        # 各年失敗率
        yearly_fail = df.groupby("year").apply(lambda x: (x["future_max_ret"] < 0).mean())
        print(f"\n--- 各年失敗率 ---")
        for yr, rate in yearly_fail.items():
            bar = "█" * int(rate * 20)
            print(f"  {yr}  {bar:<20}  {rate:.1%}")

        # 最近 20 筆失敗
        print(f"\n--- 最近 20 筆失敗樣本 ---")
        recent = failures.nlargest(20, "date")[
            ["date", "stock_id", "close", "future_max_ret", "future_close_ret"]
        ].reset_index(drop=True)
        print(recent.to_string(index=False))

    # 成功案例：最高漲幅 >= 10%
    success = df[df["future_max_ret"] >= 10].copy()
    if len(success) > 0:
        print(f"\n{'='*60}")
        print(f"成功樣本（10天最高漲幅 ≥ 10%，最近20筆）")
        print(f"{'='*60}")
        recent_success = success.nlargest(20, "date")[
            ["date", "stock_id", "close", "future_max_ret", "future_close_ret"]
        ].reset_index(drop=True)
        print(recent_success.to_string(index=False))

    # MA60 轉正天數分佈
    if "days_since_ma60_turn" in df.columns:
        d2 = df.dropna(subset=["days_since_ma60_turn"]).copy()
        print(f"\n{'='*60}")
        print(f"距離 MA60 斜率轉正的天數 vs 平均最高漲幅")
        print(f"{'='*60}")
        print(f"  平均 {d2['days_since_ma60_turn'].mean():.1f} 天  中位數 {d2['days_since_ma60_turn'].median():.0f} 天")

        bins = [0, 5, 15, 30, 60, float("inf")]
        labels = ["1-5天", "6-15天", "16-30天", "31-60天", ">60天"]
        d2["turn_band"] = pd.cut(d2["days_since_ma60_turn"], bins=bins, labels=labels)

        for band, grp in d2.groupby("turn_band", observed=True):
            avg = grp["future_max_ret"].mean()
            cnt = len(grp)
            win_rate = (grp["future_max_ret"] >= 10).mean()
            bar = "█" * int(avg * 2)
            print(f"  {str(band):<10}  {bar:<20}  均 {avg:.1f}%  達+10% {win_rate:.1%}  ({cnt:,} 筆)")

    print(f"{'='*60}")
