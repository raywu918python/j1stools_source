"""
回測引擎
========
策略規則：
  進場：訊號當天收盤價（集合競價）
  停損：進場價 - ATR14 × 2
  目標：進場價 + ATR14 × 3
  強制出場：第 10 個交易日收盤
  資金：每次用總資金 10%，最多同時持倉 5 支
  漲停板：若進場當天漲停（無法集合競價買入），跳過該訊號

使用方式：
  from backtest import run_backtest
  equity, trades = run_backtest(signals, price_df, init_capital=1_000_000)

signals   : ma_cross_xgb.run() 回傳的 signals_model DataFrame
price_df  : 原始日頻價格資料
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 參數
# ============================================================
ATR_PERIOD = 14  # ATR 計算天數
ATR_STOP = 2.0  # 初始停損 = 進場價 - ATR × 2
ATR_TRAIL = 1.5  # 追蹤停損 = 最高點 - ATR × 1.5
ATR_TARGET = 3.0  # 目標     = 進場價 + ATR × 3
MAX_HOLD = 10  # 最長持倉天數
POSITION_PCT = 0.20  # 每次用總資金 20%（5支 = 100%）
MAX_POSITIONS = 5  # 最多同時持倉 5 支
LIMIT_UP_THR = 0.095  # 台股漲停判斷門檻


# ============================================================
# ATR 計算
# ============================================================
def _calc_atr(stock_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = TR 的 period 天移動平均
    """
    h = stock_df["high"]
    l = stock_df["low"]
    c = stock_df["close"].shift(1)

    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)

    return tr.rolling(period).mean()


# ============================================================
# 資料準備
# ============================================================
def _prepare_price(price_df: pd.DataFrame) -> dict:
    """建立 per-stock 價格字典（含 ATR），加速查詢"""
    d = price_df.copy()
    d.columns = d.columns.str.strip().str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d["stock_id"] = d["stock_id"].astype(str).str.strip()

    if "high" not in d.columns:
        d["high"] = d["close"]
    if "low" not in d.columns:
        d["low"] = d["close"]
    if "open" not in d.columns:
        d["open"] = d["close"]

    price_map = {}
    for sid, grp in d.groupby("stock_id"):
        g = grp.sort_values("date").reset_index(drop=True).copy()
        g["atr"] = _calc_atr(g)
        price_map[sid] = g
    return price_map


# ============================================================
# 單筆交易模擬
# ============================================================
def _simulate_trade(sid: str, entry_date: pd.Timestamp, entry_price: float, price_map: dict) -> dict:
    """
    模擬一筆交易的完整生命週期
    回傳交易結果字典
    """
    if sid not in price_map:
        return None

    stock = price_map[sid]
    future = stock[stock["date"] > entry_date].head(MAX_HOLD)

    if len(future) == 0:
        return None

    stop_price = entry_price * (1 - STOP_LOSS)
    target_price = entry_price * (1 + TARGET_RET)

    exit_date = None
    exit_price = None
    exit_reason = None

    for i, (_, row) in enumerate(future.iterrows()):
        day_low = row["low"]
        day_high = row["high"]
        day_close = row["close"]
        day_date = row["date"]

        # 先判斷停損（用當日最低價）
        if day_low <= stop_price:
            exit_date = day_date
            exit_price = stop_price  # 假設以停損價成交
            exit_reason = "stop_loss"
            break

        # 再判斷達標（用當日最高價）
        if day_high >= target_price:
            exit_date = day_date
            exit_price = target_price  # 假設以目標價成交
            exit_reason = "target"
            break

        # 最後一天強制出場
        if i == len(future) - 1:
            exit_date = day_date
            exit_price = day_close
            exit_reason = "timeout"

    if exit_date is None:
        return None

    ret = (exit_price - entry_price) / entry_price
    hold_days = (future[future["date"] <= exit_date]).shape[0]

    return {
        "stock_id": sid,
        "entry_date": entry_date,
        "entry_price": round(entry_price, 4),
        "exit_date": exit_date,
        "exit_price": round(exit_price, 4),
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "return_pct": round(ret * 100, 4),
    }


# ============================================================
# 漲停板判斷
# ============================================================
def _is_limit_up(sid: str, date: pd.Timestamp, price_map: dict) -> bool:
    """判斷當天是否漲停（漲幅 ≥ 9.5%）"""
    if sid not in price_map:
        return False
    stock = price_map[sid]
    row = stock[stock["date"] == date]
    if len(row) == 0:
        return False
    prev = stock[stock["date"] < date]
    if len(prev) == 0:
        return False
    prev_close = prev.iloc[-1]["close"]
    today_close = row.iloc[0]["close"]
    return (today_close - prev_close) / prev_close >= LIMIT_UP_THR


# ============================================================
# 主回測函式
# ============================================================
def run_backtest(signals: pd.DataFrame, price_df: pd.DataFrame, init_capital: float = 1_000_000) -> tuple:
    """
    Parameters
    ----------
    signals      : ma_cross_xgb.run() 的 signals，需有 date / stock_id / close / label
    price_df     : 原始日頻價格（date / stock_id / open / high / low / close）
    init_capital : 初始資金（預設 100 萬）

    Returns
    -------
    equity_df : 每日資金曲線
    trades_df : 所有交易明細
    """
    print("=" * 60)
    print("回測開始")
    print("=" * 60)
    print(f"初始資金     : {init_capital:,.0f}")
    print(f"每次倉位     : {POSITION_PCT:.0%}  最多 {MAX_POSITIONS} 支（滿倉 {POSITION_PCT*MAX_POSITIONS:.0%}）")
    print(f"初始停損     : ATR{ATR_PERIOD} × {ATR_STOP}（動態）")
    print(f"追蹤停損     : 最高點 - ATR{ATR_PERIOD} × {ATR_TRAIL}（動態）")
    print(f"目標         : ATR{ATR_PERIOD} × {ATR_TARGET}（動態）")
    print(f"強制出場     : 第 {MAX_HOLD} 個交易日")

    price_map = _prepare_price(price_df)

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.sort_values("date").reset_index(drop=True)

    capital = init_capital
    positions = {}
    trades = []
    equity_curve = []

    all_dates = sorted(sig["date"].unique())

    for date in all_dates:
        # ── 先處理今天到期 / 停損的持倉 ──
        to_close = []
        for sid, pos in positions.items():
            if sid not in price_map:
                continue
            stock = price_map[sid]
            today = stock[stock["date"] == date]
            if len(today) == 0:
                continue
            row = today.iloc[0]
            atr = pos["atr"]

            # 更新追蹤最高點，並上移追蹤停損
            if row["high"] > pos["highest"]:
                pos["highest"] = row["high"]
                trail = pos["highest"] - ATR_TRAIL * atr
                if trail > pos["stop"]:
                    pos["stop"] = trail

            # 判斷出場
            if row["low"] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "stop_loss" if pos["highest"] <= pos["entry_price"] else "trail_stop"
            elif row["high"] >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "target"
            elif (
                stock[stock["date"] <= date].shape[0] - stock[stock["date"] <= pos["entry_date"]].shape[0]
            ) >= MAX_HOLD:
                exit_price = row["close"]
                exit_reason = "timeout"
            else:
                continue

            pnl = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["cost"]
            capital += pos["cost"] + pnl

            trades.append(
                {
                    "stock_id": sid,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "highest": round(pos["highest"], 4),
                    "exit_date": date,
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "atr": round(atr, 4),
                    "stop_pct": round((pos["entry_price"] - pos["init_stop"]) / pos["entry_price"] * 100, 2),
                    "target_pct": round((pos["target"] - pos["entry_price"]) / pos["entry_price"] * 100, 2),
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

        # ── 今天的新訊號（依信心分數排序）──
        today_signals = sig[sig["date"] == date]
        if "xgb_prob" in today_signals.columns:
            today_signals = today_signals.sort_values("xgb_prob", ascending=False)

        for _, row in today_signals.iterrows():
            sid = row["stock_id"]

            if sid in positions:
                continue
            if len(positions) >= MAX_POSITIONS:
                continue
            if _is_limit_up(sid, date, price_map):
                continue

            entry_price = row["close"]

            if sid not in price_map:
                continue
            stock_data = price_map[sid]
            entry_row = stock_data[stock_data["date"] == date]
            if len(entry_row) == 0 or pd.isna(entry_row.iloc[0]["atr"]):
                continue
            atr = entry_row.iloc[0]["atr"]

            init_stop = entry_price - ATR_STOP * atr
            target_price = entry_price + ATR_TARGET * atr

            size = (capital * POSITION_PCT) / entry_price
            cost = size * entry_price

            if cost > capital:
                continue

            capital -= cost
            positions[sid] = {
                "entry_date": date,
                "entry_price": entry_price,
                "init_stop": init_stop,
                "stop": init_stop,
                "target": target_price,
                "highest": entry_price,
                "atr": atr,
                "size": size,
                "cost": cost,
            }

        # ── 記錄今日總資產 ──
        market_value = 0
        for sid, pos in positions.items():
            if sid not in price_map:
                continue
            stock = price_map[sid]
            today = stock[stock["date"] == date]
            if len(today) > 0:
                market_value += pos["size"] * today.iloc[0]["close"]

        total_assets = capital + market_value
        equity_curve.append(
            {
                "date": date,
                "cash": round(capital, 2),
                "market_value": round(market_value, 2),
                "total": round(total_assets, 2),
                "n_positions": len(positions),
            }
        )

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    _print_results(equity_df, trades_df, init_capital)

    return equity_df, trades_df


# ============================================================
# 結果統計
# ============================================================
def _print_results(equity: pd.DataFrame, trades: pd.DataFrame, init_capital: float) -> None:
    if len(trades) == 0:
        print("❌ 沒有成交任何交易")
        return

    final = equity["total"].iloc[-1]
    total_ret = (final - init_capital) / init_capital * 100
    years = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365
    cagr = ((final / init_capital) ** (1 / years) - 1) * 100

    # 最大回撤
    roll_max = equity["total"].cummax()
    drawdown = (equity["total"] - roll_max) / roll_max * 100
    max_dd = drawdown.min()

    # 交易統計
    n_trades = len(trades)
    n_win = (trades["return_pct"] > 0).sum()
    n_loss = (trades["return_pct"] <= 0).sum()
    win_rate = n_win / n_trades
    avg_win = trades[trades["return_pct"] > 0]["return_pct"].mean()
    avg_loss = trades[trades["return_pct"] <= 0]["return_pct"].mean()

    print(f"\n{'='*60}")
    print(f"回測結果")
    print(f"{'='*60}")
    print(f"初始資金     : {init_capital:>12,.0f}")
    print(f"最終資產     : {final:>12,.0f}")
    print(f"總報酬       : {total_ret:>+11.2f}%")
    print(f"年化報酬（CAGR）: {cagr:>+8.2f}%")
    print(f"最大回撤     : {max_dd:>+11.2f}%")

    print(f"\n--- 交易統計 ---")
    print(f"總交易次數   : {n_trades:,}")
    print(f"勝（獲利）   : {n_win:,}  ({win_rate:.1%})")
    print(f"敗（虧損）   : {n_loss:,}  ({1-win_rate:.1%})")
    print(f"平均獲利     : {avg_win:>+.2f}%")
    print(f"平均虧損     : {avg_loss:>+.2f}%")
    print(f"獲利因子     : {abs(avg_win / avg_loss):.2f}x")
    if "stop_pct" in trades.columns:
        print(f"平均初始停損 : -{trades['stop_pct'].mean():.2f}%（ATR動態）")
        print(f"平均目標幅度 : +{trades['target_pct'].mean():.2f}%（ATR動態）")

    # 出場原因分佈
    print(f"\n--- 出場原因 ---")
    for reason, cnt in trades["exit_reason"].value_counts().items():
        pct = cnt / n_trades
        bar = "█" * int(pct * 30)
        print(f"  {reason:<12} {bar:<30}  {cnt:,} ({pct:.1%})")

    # 各年績效
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    yearly = trades.groupby("year").agg(
        n=("pnl", "count"), total_pnl=("pnl", "sum"), win_rate=("return_pct", lambda x: (x > 0).mean())
    )
    print(f"\n--- 各年交易統計 ---")
    for yr, row in yearly.iterrows():
        bar = "█" * int(row["win_rate"] * 20)
        sign = "+" if row["total_pnl"] >= 0 else ""
        print(
            f"  {yr}  {bar:<20}  勝率 {row['win_rate']:.1%}  "
            f"PnL {sign}{row['total_pnl']:,.0f}  ({int(row['n'])} 筆)"
        )

    print(f"{'='*60}")
