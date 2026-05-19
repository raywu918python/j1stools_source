"""
動能突破策略回測
================
進場：momentum_stats.run() 產生的訊號
排序：MA60 轉正天數最短優先
出場：固定百分比 或 移動停損

使用方式：
  from momentum_backtest import run_backtest
  equity, trades = run_backtest(signals, price_df)

  # 固定停利停損
  equity, trades = run_backtest(signals, price_df,
                                exit_mode='fixed',
                                stop_pct=0.08, target_pct=0.10)

  # 移動停損
  equity, trades = run_backtest(signals, price_df,
                                exit_mode='trailing',
                                stop_pct=0.08,
                                trail_pct=0.05,
                                target_pct=0.15)
"""

import pandas as pd
import numpy as np
import warnings
from enum import Enum

warnings.filterwarnings("ignore")

LIMIT_UP_THR = 0.095


class SortBy(Enum):
    """
    排序欄位定義，預設升降序如下：
      MA60_TURN_DAYS : 天數小→大（最新轉正優先）
      DEV_MA60       : 距離大→小（最遠優先）
      PRED_SCORE     : 分數大→小（最高優先）
      NONE           : 不排序
    """

    MA60_TURN_DAYS = ("days_since_ma60_turn", True)
    DEV_MA60 = ("dev_ma60", False)
    PRED_SCORE = ("pred_score", False)
    NONE = (None, True)

    def __init__(self, col, ascending):
        self.col = col
        self.ascending = ascending

    def asc(self):
        """強制升序"""
        return _SortKey(self.col, True)

    def desc(self):
        """強制降序"""
        return _SortKey(self.col, False)


class _SortKey:
    """SortBy.asc() / SortBy.desc() 回傳的自訂排序物件"""

    def __init__(self, col, ascending):
        self.col = col
        self.ascending = ascending


def _prepare_price(price_df):
    d = price_df.copy()
    d.columns = d.columns.str.strip().str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d["stock_id"] = d["stock_id"].astype(str).str.strip()
    if "high" not in d.columns:
        d["high"] = d["close"]
    if "low" not in d.columns:
        d["low"] = d["close"]
    price_map = {}
    for sid, grp in d.groupby("stock_id"):
        price_map[sid] = grp.sort_values("date").reset_index(drop=True)
    return price_map


def _is_limit_up(sid, date, price_map):
    if sid not in price_map:
        return False
    stock = price_map[sid]
    row = stock[stock["date"] == date]
    prev = stock[stock["date"] < date]
    if len(row) == 0 or len(prev) == 0:
        return False
    return ((row.iloc[0]["close"] - prev.iloc[-1]["close"]) / prev.iloc[-1]["close"]) >= LIMIT_UP_THR


def run_backtest(
    signals,
    price_df,
    init_capital=1_000_000,
    stop_pct=0.08,
    target_pct=0.10,
    trail_pct=0.05,
    max_hold=10,
    position_pct=0.20,
    max_positions=5,
    exit_mode="fixed",
    sort_by=SortBy.MA60_TURN_DAYS,
    df_market=None,  # 大盤 ETF（date / close）
):

    print("=" * 60)
    print("動能突破策略回測")
    print("=" * 60)
    print(f"初始資金     : {init_capital:,.0f}")
    print(f"每次倉位     : {position_pct:.0%}  最多 {max_positions} 支")
    print(f"出場模式     : {'移動停損' if exit_mode == 'trailing' else '固定停利停損'}")
    print(f"初始停損     : -{stop_pct:.0%}")
    if exit_mode == "trailing":
        print(f"移動停損     : 最高點回落 -{trail_pct:.0%}")
    print(f"目標         : +{target_pct:.0%}")
    print(f"強制出場     : 第 {max_hold} 個交易日")
    print(f"排序         : MA60 轉正天數最短優先")

    price_map = _prepare_price(price_df)

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    sig["stock_id"] = sig["stock_id"].astype(str).str.strip()
    sig = sig.sort_values("date").reset_index(drop=True)

    # 解析排序設定
    sort_keys = sort_by if isinstance(sort_by, list) else [sort_by]
    sort_cols = [s.col for s in sort_keys if s.col and s.col in sig.columns]
    sort_ascs = [s.ascending for s in sort_keys if s.col and s.col in sig.columns]

    if sort_cols:
        sort_desc = " → ".join(
            f"{s.col}({'↑' if s.ascending else '↓'})" for s in sort_keys if s.col and s.col in sig.columns
        )
        print(f"排序         : {sort_desc}")
    else:
        print(f"排序         : 無")

    print(f"訊號總數     : {len(sig):,}")

    capital, positions, trades, equity_curve = init_capital, {}, [], []

    for date in sorted(sig["date"].unique()):

        # 出場
        to_close = []
        for sid, pos in positions.items():
            if sid not in price_map:
                continue
            stock = price_map[sid]
            today = stock[stock["date"] == date]
            if len(today) == 0:
                continue
            row = today.iloc[0]

            if exit_mode == "trailing" and row["high"] > pos["highest"]:
                pos["highest"] = row["high"]
                trail = pos["highest"] * (1 - trail_pct)
                if trail > pos["stop"]:
                    pos["stop"] = trail

            if row["low"] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = (
                    "trail_stop" if exit_mode == "trailing" and pos["highest"] > pos["entry_price"] else "stop_loss"
                )
            elif row["high"] >= pos["target"]:
                exit_price, exit_reason = pos["target"], "target"
            elif (
                stock[stock["date"] <= date].shape[0] - stock[stock["date"] <= pos["entry_date"]].shape[0]
            ) >= max_hold:
                exit_price, exit_reason = row["close"], "timeout"
            else:
                continue

            pnl = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["cost"]
            capital += pos["cost"] + pnl
            trades.append(
                {
                    "stock_id": sid,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": date,
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "highest": round(pos["highest"], 4),
                    "cost": pos["cost"],
                    "pnl": round(pnl, 2),
                    "return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 3),
                    "hold_days": (
                        stock[stock["date"] <= date].shape[0] - stock[stock["date"] <= pos["entry_date"]].shape[0]
                    ),
                }
            )
            to_close.append(sid)

        for sid in to_close:
            del positions[sid]

        # 進場（依排序欄位）
        today_sig = sig[sig["date"] == date].copy()
        if sort_cols:
            today_sig = today_sig.sort_values(sort_cols, ascending=sort_ascs, na_position="last")

        for _, row in today_sig.iterrows():
            sid = row["stock_id"]
            if sid in positions:
                continue
            if len(positions) >= max_positions:
                break
            if _is_limit_up(sid, date, price_map):
                continue
            if sid not in price_map:
                continue

            entry_price = row["close"]
            size = (capital * position_pct) / entry_price
            cost = size * entry_price
            if cost > capital:
                continue

            capital -= cost
            positions[sid] = {
                "entry_date": date,
                "entry_price": entry_price,
                "stop": entry_price * (1 - stop_pct),
                "target": entry_price * (1 + target_pct),
                "highest": entry_price,
                "size": size,
                "cost": cost,
            }

        # 每日資產
        market_value = sum(
            pos["size"] * price_map[sid][price_map[sid]["date"] == date].iloc[0]["close"]
            for sid, pos in positions.items()
            if sid in price_map and len(price_map[sid][price_map[sid]["date"] == date]) > 0
        )
        equity_curve.append(
            {
                "date": date,
                "cash": round(capital, 2),
                "market_value": round(market_value, 2),
                "total": round(capital + market_value, 2),
                "n_positions": len(positions),
            }
        )

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    _print_results(equity_df, trades_df, init_capital, df_market)
    return equity_df, trades_df


def _print_results(equity, trades, init_capital, df_market=None):
    if len(trades) == 0:
        print("❌ 沒有成交任何交易")
        return

    final = equity["total"].iloc[-1]
    total_ret = (final - init_capital) / init_capital * 100
    years = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365
    cagr = ((final / init_capital) ** (1 / years) - 1) * 100
    max_dd = ((equity["total"] - equity["total"].cummax()) / equity["total"].cummax() * 100).min()

    n_trades = len(trades)
    n_win = (trades["return_pct"] > 0).sum()
    win_rate = n_win / n_trades
    avg_win = trades[trades["return_pct"] > 0]["return_pct"].mean()
    avg_loss = trades[trades["return_pct"] <= 0]["return_pct"].mean()

    print(f"\n{'='*60}")
    print(f"回測結果")
    print(f"{'='*60}")
    print(f"初始資金       : {init_capital:>12,.0f}")
    print(f"最終資產       : {final:>12,.0f}")
    print(f"總報酬         : {total_ret:>+11.2f}%")
    print(f"年化報酬(CAGR) : {cagr:>+8.2f}%")
    print(f"最大回撤       : {max_dd:>+11.2f}%")

    # 大盤比較
    if df_market is not None:
        mkt = df_market.copy()
        mkt.columns = mkt.columns.str.strip().str.lower()
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt = mkt.sort_values("date")
        start_date = equity["date"].iloc[0]
        end_date = equity["date"].iloc[-1]
        mkt = mkt[(mkt["date"] >= start_date) & (mkt["date"] <= end_date)]

        if len(mkt) >= 2:
            mkt_start = mkt["close"].iloc[0]
            mkt_end = mkt["close"].iloc[-1]
            mkt_ret = (mkt_end - mkt_start) / mkt_start * 100
            mkt_cagr = ((mkt_end / mkt_start) ** (1 / years) - 1) * 100
            mkt_dd = ((mkt["close"] - mkt["close"].cummax()) / mkt["close"].cummax() * 100).min()

            print(f"\n--- 大盤比較 ---")
            print(f"{'':20}  {'策略':>10}  {'大盤':>10}  {'超額':>10}")
            print(f"{'總報酬':20}  {total_ret:>+9.2f}%  {mkt_ret:>+9.2f}%  {total_ret-mkt_ret:>+9.2f}%")
            print(f"{'年化報酬(CAGR)':20}  {cagr:>+9.2f}%  {mkt_cagr:>+9.2f}%  {cagr-mkt_cagr:>+9.2f}%")
            print(f"{'最大回撤':20}  {max_dd:>+9.2f}%  {mkt_dd:>+9.2f}%  {max_dd-mkt_dd:>+9.2f}%")

    print(f"\n--- 交易統計 ---")
    print(f"總交易次數     : {n_trades:,}")
    print(f"勝             : {n_win:,}  ({win_rate:.1%})")
    print(f"敗             : {n_trades-n_win:,}  ({1-win_rate:.1%})")
    print(f"平均獲利       : {avg_win:>+.2f}%")
    print(f"平均虧損       : {avg_loss:>+.2f}%")
    print(f"獲利因子       : {abs(avg_win/avg_loss):.2f}x")

    print(f"\n--- 出場原因 ---")
    for reason, cnt in trades["exit_reason"].value_counts().items():
        bar = "█" * int(cnt / len(trades) * 30)
        print(f"  {reason:<12} {bar:<30}  {cnt:,} ({cnt/len(trades):.1%})")

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    yearly = trades.groupby("year").agg(
        n=("pnl", "count"), pnl=("pnl", "sum"), wr=("return_pct", lambda x: (x > 0).mean())
    )
    print(f"\n--- 各年統計 ---")
    for yr, row in yearly.iterrows():
        bar = "█" * int(row["wr"] * 20)
        sign = "+" if row["pnl"] >= 0 else ""
        print(f"  {yr}  {bar:<20}  勝率 {row['wr']:.1%}  " f"PnL {sign}{row['pnl']:,.0f}  ({int(row['n'])} 筆)")

    print(f"\n--- 最近 20 筆交易記錄 ---")
    recent = trades.nlargest(20, "exit_date")[
        ["entry_date", "stock_id", "entry_price", "exit_date", "exit_price", "exit_reason", "return_pct", "hold_days"]
    ].reset_index(drop=True)
    print(recent.to_string(index=False))
    print(f"{'='*60}")
