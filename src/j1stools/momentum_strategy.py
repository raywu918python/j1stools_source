"""動能策略：創 60 日新高＋量 2 倍＋120 日報酬前 10%。

每天 13:10 用當天價格跑一次，訊號和出場都在「當天收盤」成交（避開台股開盤的大漲大跌）。
- 最多 MAX_POS 檔，同族群最多 2 檔，漲停（漲幅 > 9.5%）不買
- 0050 在 MKT_MA 日均線上才開新倉（已持有的照規則出場）
- 部位依波動調整：每檔基本 1/MAX_POS，日波動超過 VOL_TARGET 的按比例買少
- 出場（動態停利）：持有滿 HOLD 天時報酬未達 RUN_THR 就出場；有達到就繼續抱，
  直到收盤跌破 EXIT_MA 日均線，最長 MAX_HOLD 天
參數用 2015–2023 挑選（lab/close_grid.py、lab/momentum_exit_variant.py）。
"""

import glob

import numpy as np
import pandas as pd

MAX_POS = 5
HOLD = 10
RUN_THR = 0.10
EXIT_MA = 10
MAX_HOLD = 120
TRAIL = None
VOL_TARGET = 0.04
GROUP_LIMIT = 2
MKT_MA = 120
FEE = 0.003
MIN_VOLUME = 500_000  # 20 日均量（股）
MIN_PRICE = 10
INIT_CASH = 1_000_000


def _wide(df, col):
    return df.pivot(index="date", columns="stock_id", values=col).sort_index()


def load_data(price_dir="db/price", info_path="db/info/info.parquet", st="2014-01-01"):
    px = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{price_dir}/*.parquet")])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"] >= st].drop_duplicates(["date", "stock_id"], keep="last")
    etf = px[px["stock_id"] == "0050"].set_index("date")["close"].sort_index()
    px = px[~px["stock_id"].str.startswith("00")]
    d = {c: _wide(px, c) for c in ["high", "close", "volume"]}
    d["close"] = d["close"].ffill(limit=5)
    d["etf"] = etf.reindex(d["close"].index).ffill()
    info = pd.read_parquet(info_path).set_index("stock_id")
    d["group"] = info["group"].reindex(d["close"].columns).fillna("未知").values
    return d


def signals(d, mkt_ma=MKT_MA):
    C, H, V = d["close"], d["high"], d["volume"]
    vol20 = V.rolling(20).mean()
    ret = C.pct_change(fill_method=None)
    r120 = C.pct_change(120, fill_method=None)
    univ = (vol20 >= MIN_VOLUME) & (C >= MIN_PRICE)
    sig = (C >= H.rolling(60).max().shift(1)) & (V > 2 * vol20.shift(1)) & (r120.rank(axis=1, pct=True) >= 0.9)
    sig &= univ & ~(ret > 0.095)
    entry_ok = d["etf"] > d["etf"].rolling(mkt_ma).mean() if mkt_ma else pd.Series(True, index=C.index)
    return sig, r120, ret.rolling(20).std(), entry_ok


def backtest(d, st="2024-01-01", max_pos=MAX_POS, hold=HOLD, trail=TRAIL, vol_target=VOL_TARGET, mkt_ma=MKT_MA,
             run_thr=RUN_THR, exit_ma=EXIT_MA, max_hold=MAX_HOLD):
    """回傳 (equity_df, trades_df, open_positions dict, close_df)，欄位與 RFC 回測相同，網站可直接讀。"""
    sig, r120, std20, entry_ok = signals(d, mkt_ma)
    C = d["close"]
    idx = C.index[C.index >= st]
    c = C.loc[idx].values
    ma = C.rolling(exit_ma).mean().loc[idx].values if exit_ma else None
    s = sig.loc[idx].fillna(False).values
    score, sd = r120.loc[idx].values, std20.loc[idx].values
    ok = entry_ok.loc[idx].fillna(False).values
    cols, group = C.columns, d["group"]

    cash, pos, trades, records = float(INIT_CASH), {}, [], []
    prev_eq = cash
    for i, dt in enumerate(idx):
        # 1. 出場：今天收盤判斷、今天收盤賣
        for j in list(pos):
            p, px = pos[j], c[i, j]
            if not np.isfinite(px):
                continue
            p["highest"] = max(p["highest"], px)
            days, gain = i - p["i"], px / p["entry_price"] - 1
            running = run_thr is not None and days >= hold and gain >= run_thr  # 賺到門檻就不受 hold 限制
            if trail and px <= p["highest"] * (1 - trail):
                why = "sl_trail"
            elif running and exit_ma and px < ma[i, j]:
                why = "ma_exit"
            elif days >= max_hold or (days >= hold and not running):
                why = "hold_days"
            else:
                why = None
            if not why:
                continue
            value = p["shares"] * px * (1 - FEE)
            cash += value
            del pos[j]
            trades.append(
                {
                    "stock_id": cols[j],
                    "entry_date": idx[p["i"]],
                    "entry_price": p["entry_price"],
                    "stop": round(p["highest"] * (1 - trail), 4) if trail else None,
                    "target": None,
                    "exit_date": dt,
                    "exit_price": px,
                    "exit_reason": why,
                    "highest": p["highest"],
                    "cost": p["cost"],
                    "pnl": value - p["cost"],
                    "pnl_pct": (px / p["entry_price"] - 1) * 100,
                    "return_pct": (value - p["cost"]) / p["cost"] * 100,
                    "hold_days": i - p["i"],
                }
            )

        # 2. 進場：今天的訊號、今天收盤買
        if ok[i]:
            cand = np.where(s[i])[0]
            cand = cand[np.argsort(-np.nan_to_num(score[i, cand], nan=-9))]
            for j in cand:
                if len(pos) >= max_pos:
                    break
                if j in pos or not np.isfinite(c[i, j]) or c[i, j] <= 0:
                    continue
                if sum(group[k] == group[j] for k in pos) >= GROUP_LIMIT:
                    continue
                size = prev_eq / max_pos
                if vol_target and np.isfinite(sd[i, j]) and sd[i, j] > 0:
                    size *= min(1.0, vol_target / sd[i, j])
                cost = min(size, cash)
                if cost < size * 0.5:
                    break
                cash -= cost
                pos[j] = {"shares": cost * (1 - FEE) / c[i, j], "entry_price": c[i, j], "i": i, "highest": c[i, j], "cost": cost}

        mv = sum(p["shares"] * (c[i, j] if np.isfinite(c[i, j]) else p["entry_price"]) for j, p in pos.items())
        records.append({"date": dt, "cash": round(cash, 2), "market_value": round(mv, 2), "total": round(cash + mv, 2)})
        prev_eq = cash + mv

    equity = pd.DataFrame(records)
    trades_df = pd.DataFrame(trades)
    open_positions = {
        cols[j]: {
            "entry_price": float(p["entry_price"]),
            "entry_date": idx[p["i"]],
            "highest": float(p["highest"]),
            "cost": float(p["cost"]),
            "stop": round(float(p["highest"]) * (1 - trail), 4) if trail else None,
            "target": None,
        }
        for j, p in pos.items()
    }
    return equity, trades_df, open_positions, C.loc[idx]
