"""動能策略：創 60 日新高＋量 2 倍＋120 日報酬前 10%。

- t 日收盤出現訊號 → t+1 開盤買；出場在收盤判斷 → 隔天開盤賣
- 最多 5 檔，同族群最多 2 檔，開盤漲停不買
- 0050 在 200 日均線上才開新倉（已持有的照規則出場）
- 部位依波動調整：每檔基本 1/5，日波動超過 4% 的按比例買少
- 出場：從持有期間最高收盤回落 20%，或持有滿 20 個交易日
"""

import glob

import numpy as np
import pandas as pd

MAX_POS = 5
HOLD = 20
TRAIL = 0.20
VOL_TARGET = 0.04
GROUP_LIMIT = 2
MKT_MA = 200
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
    d = {c: _wide(px, c) for c in ["open", "high", "close", "volume"]}
    d["close"] = d["close"].ffill(limit=5)
    d["etf"] = etf.reindex(d["close"].index).ffill()
    info = pd.read_parquet(info_path).set_index("stock_id")
    d["group"] = info["group"].reindex(d["close"].columns).fillna("未知").values
    return d


def signals(d):
    C, H, V = d["close"], d["high"], d["volume"]
    vol20 = V.rolling(20).mean()
    ret = C.pct_change(fill_method=None)
    r120 = C.pct_change(120, fill_method=None)
    univ = (vol20 >= MIN_VOLUME) & (C >= MIN_PRICE)
    sig = (C >= H.rolling(60).max().shift(1)) & (V > 2 * vol20.shift(1)) & (r120.rank(axis=1, pct=True) >= 0.9)
    sig &= univ & ~(ret > 0.095)
    entry_ok = d["etf"] > d["etf"].rolling(MKT_MA).mean()
    return sig, r120, ret.rolling(20).std(), entry_ok


def backtest(d, st="2024-01-01"):
    """回傳 (equity_df, trades_df, open_positions dict, close_df)，欄位與 RFC 回測相同，網站可直接讀。"""
    sig, r120, std20, entry_ok = signals(d)
    C, O = d["close"], d["open"]
    idx = C.index[C.index >= st]
    c, o = C.loc[idx].values, O.loc[idx].values
    s = sig.loc[idx].fillna(False).values
    score, sd = r120.loc[idx].values, std20.loc[idx].values
    ok = entry_ok.loc[idx].fillna(False).values
    cols, group = C.columns, d["group"]

    cash, pos, trades, records = float(INIT_CASH), {}, [], []
    to_sell, prev_eq = {}, cash
    for i, dt in enumerate(idx):
        for j, why in list(to_sell.items()):
            p = pos.pop(j)
            px = o[i, j] if np.isfinite(o[i, j]) and o[i, j] > 0 else c[i - 1, j]
            value = p["shares"] * px * (1 - FEE)
            cash += value
            trades.append(
                {
                    "stock_id": cols[j],
                    "entry_date": idx[p["i"]],
                    "entry_price": p["entry_price"],
                    "stop": round(p["highest"] * (1 - TRAIL), 4),
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
        to_sell.clear()

        if i > 0 and ok[i - 1]:
            cand = np.where(s[i - 1])[0]
            cand = cand[np.argsort(-np.nan_to_num(score[i - 1, cand], nan=-9))]
            for j in cand:
                if len(pos) >= MAX_POS:
                    break
                if j in pos or not np.isfinite(o[i, j]) or o[i, j] <= 0 or o[i, j] >= c[i - 1, j] * 1.095:
                    continue
                if sum(group[k] == group[j] for k in pos) >= GROUP_LIMIT:
                    continue
                size = prev_eq / MAX_POS
                if np.isfinite(sd[i - 1, j]) and sd[i - 1, j] > 0:
                    size *= min(1.0, VOL_TARGET / sd[i - 1, j])
                cost = min(size, cash)
                if cost < size * 0.5:
                    break
                cash -= cost
                pos[j] = {"shares": cost * (1 - FEE) / o[i, j], "entry_price": o[i, j], "i": i, "highest": o[i, j], "cost": cost}

        for j, p in pos.items():
            px = c[i, j]
            if not np.isfinite(px):
                continue
            p["highest"] = max(p["highest"], px)
            if px <= p["highest"] * (1 - TRAIL):
                to_sell[j] = "sl_trail"
            elif i - p["i"] >= HOLD:
                to_sell[j] = "hold_days"

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
            "stop": round(float(p["highest"]) * (1 - TRAIL), 4),
            "target": None,
            "exit_tomorrow": j in to_sell,
        }
        for j, p in pos.items()
    }
    return equity, trades_df, open_positions, C.loc[idx]
