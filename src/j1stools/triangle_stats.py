"""
收鍊突破統計（寬鬆版）
======================
條件：
  收鍊確認後 20 日內
  收盤突破 h1_date 當天的 high
  不需要黃金交叉

使用方式：
  from triangle_stats import run
  results = run(triangle_df, price_df)

triangle_df : 欄位 date / stock_id / h1_date / h1_high（或從 price_df 查）
price_df    : 欄位 date / stock_id / close / high
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

BREAKOUT_WINDOW = 20  # 確認後幾個交易日內突破
TARGET_DAYS = 10  # 進場後觀察幾天
TARGET_RET = 0.10  # 目標報酬 10%


def run(triangle_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:

    print("=" * 60)
    print("收鍊突破統計（寬鬆版，突破 h1 高點，20 日窗口）")
    print("=" * 60)

    # 清理 triangle_df
    tri = triangle_df.copy()
    tri.columns = tri.columns.str.strip().str.lower()
    tri["date"] = pd.to_datetime(tri["date"])
    tri["h1_date"] = pd.to_datetime(tri["h1_date"])
    tri["stock_id"] = tri["stock_id"].astype(str).str.strip()

    # 清理 price_df
    prc = price_df.copy()
    prc.columns = prc.columns.str.strip().str.lower()
    prc["date"] = pd.to_datetime(prc["date"])
    prc["stock_id"] = prc["stock_id"].astype(str).str.strip()
    prc["close"] = pd.to_numeric(prc["close"], errors="coerce")
    prc["high"] = pd.to_numeric(prc["high"], errors="coerce")
    prc = prc.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 從 price_df 查 h1_date 當天的 high
    h1_price = prc[["date", "stock_id", "high"]].rename(columns={"date": "h1_date", "high": "h1_high"})
    tri = tri.merge(h1_price, on=["stock_id", "h1_date"], how="left")

    missing = tri["h1_high"].isna().sum()
    if missing:
        print(f"⚠️  {missing} 筆找不到 h1_date 的 high，略過")
    tri = tri.dropna(subset=["h1_high"])

    print(f"型態總數      : {len(tri):,}")
    print(f"有效匹配股票  : {tri['stock_id'].nunique()} 支")

    # 建立 per-stock 索引
    prc_grouped = {sid: grp.reset_index(drop=True) for sid, grp in prc.groupby("stock_id")}

    records = []
    for _, row in tri.iterrows():
        sid = row["stock_id"]
        pattern_date = row["date"]
        h1_high = row["h1_high"]

        if sid not in prc_grouped:
            continue

        stock = prc_grouped[sid]
        window = stock[stock["date"] > pattern_date].head(BREAKOUT_WINDOW)

        if len(window) == 0:
            continue

        # 找第一個收盤突破 h1_high 的日子
        hit = window[window["close"] > h1_high]

        if len(hit) == 0:
            records.append(
                {
                    "stock_id": sid,
                    "pattern_date": pattern_date,
                    "h1_date": row["h1_date"],
                    "h1_high": round(h1_high, 4),
                    "entry_date": pd.NaT,
                    "entry_close": np.nan,
                    "days_to_entry": np.nan,
                    "triggered": False,
                    "win": np.nan,
                    "future_ret": np.nan,
                }
            )
        else:
            entry = hit.iloc[0]
            entry_date = entry["date"]
            entry_close = entry["close"]
            days = int((window["date"] <= entry_date).sum())

            # 進場後 TARGET_DAYS 天的最高價
            future = stock[stock["date"] > entry_date].head(TARGET_DAYS)
            if len(future) == 0:
                win = np.nan
                future_ret = np.nan
            else:
                max_high = future["high"].max()
                future_ret = (max_high - entry_close) / entry_close * 100
                win = int(max_high >= entry_close * (1 + TARGET_RET))

            records.append(
                {
                    "stock_id": sid,
                    "pattern_date": pattern_date,
                    "h1_date": row["h1_date"],
                    "h1_high": round(h1_high, 4),
                    "entry_date": entry_date,
                    "entry_close": round(entry_close, 4),
                    "days_to_entry": days,
                    "triggered": True,
                    "win": win,
                    "future_ret": round(future_ret, 2) if not np.isnan(future_ret) else np.nan,
                }
            )

    signals = pd.DataFrame(records)
    _print_stats(signals)
    return signals


def _print_stats(signals: pd.DataFrame) -> None:
    n_total = len(signals)
    n_triggered = signals["triggered"].sum()
    n_miss = n_total - n_triggered
    trig_rate = n_triggered / max(n_total, 1)

    print(f"\n{'='*60}")
    print(f"突破統計")
    print(f"{'='*60}")
    print(f"型態總數          : {n_total:,}")
    print(f"20日內突破 h1high : {n_triggered:,}  ({trig_rate:.1%})")
    print(f"未突破            : {n_miss:,}  ({1-trig_rate:.1%})")

    triggered = signals[signals["triggered"] & signals["win"].notna()]
    if len(triggered) == 0:
        return

    win_rate = triggered["win"].mean()
    print(f"\n進場後 10 天內漲 10% 勝率：{win_rate:.1%}")
    print(f"平均最高報酬      : {triggered['future_ret'].mean():.2f}%")
    print(f"進場樣本數        : {len(triggered):,}")

    print(f"\n--- 進場速度（交易日）---")
    print(f"  平均 {triggered['days_to_entry'].mean():.1f} 天")
    print(f"  最快 {triggered['days_to_entry'].min():.0f} 天")
    print(f"  最慢 {triggered['days_to_entry'].max():.0f} 天")

    bins = [0, 3, 7, 14, 20]
    labels = ["1-3天", "4-7天", "8-14天", "15-20天"]
    t2 = triggered.copy()
    t2["speed"] = pd.cut(t2["days_to_entry"], bins=bins, labels=labels)
    dist = t2["speed"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        bar = "█" * int(cnt / max(dist) * 20)
        print(f"  {lbl}  {bar:<20}  {cnt:,}")

    # 各年勝率
    triggered = triggered.copy()
    triggered["year"] = pd.to_datetime(triggered["pattern_date"]).dt.year
    yearly = triggered.groupby("year")["win"].agg(["mean", "count"])
    print(f"\n--- 各年勝率 ---")
    for yr, row in yearly.iterrows():
        bar = "█" * int(row["mean"] * 20)
        print(f"  {yr}  {bar:<20}  {row['mean']:.1%}  ({int(row['count'])} 筆)")

    print(f"{'='*60}")
