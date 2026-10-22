"""動能策略每日回測：產生 db/backtest/momentum_v1/ 的 equity / trades / open / market，推回 HF 給網站讀。

規則見 src/j1stools/momentum_strategy.py。這支只產生回測結果，不下單。
"""

import os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from j1stools import hf_sync, momentum_strategy

MODEL_NAME = "momentum_v1"
START = "2024-01-01"
_TW = timezone(timedelta(hours=8))


def run(push=True):
    if push:
        hf_sync.pull(["db/price", "db/info"])
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    out_dir = f"db/backtest/{MODEL_NAME}"

    d = momentum_strategy.load_data()
    equity, trades_df, open_pos, close_df = momentum_strategy.backtest(d, st=START)

    # 大盤對比（0050），從同一天開始、同樣起始金額
    etf = d["etf"].reindex(pd.to_datetime(equity["date"])).ffill()
    market_df = pd.DataFrame({"date": equity["date"].values, "total": (etf / etf.iloc[0] * equity["total"].iloc[0]).round(2).values})

    pv = equity.set_index("date")["total"]
    if not trades_df.empty:
        trades_df["weight"] = (trades_df["cost"] / trades_df["entry_date"].map(pv)).round(4)

    os.makedirs(out_dir, exist_ok=True)
    equity.to_parquet(f"{out_dir}/equity_{today}.parquet", index=False)
    trades_df.to_parquet(f"{out_dir}/trades_{today}.parquet", index=False)
    market_df.to_parquet(f"{out_dir}/market_{today}.parquet", index=False)

    if open_pos:
        last_price = close_df.iloc[-1]
        total_value = float(equity["total"].iloc[-1])
        open_df = pd.DataFrame(
            [
                {
                    "stock_id": sid,
                    **{k: v for k, v in pos.items() if k != "exit_tomorrow"},
                    "last_date": close_df.index[-1],
                    "pnl_pct": round((last_price.get(sid, pos["entry_price"]) / pos["entry_price"] - 1) * 100, 2),
                    "weight": round(pos["cost"] / total_value, 4),
                }
                for sid, pos in open_pos.items()
            ]
        )
        open_df.to_parquet(f"{out_dir}/open_{today}.parquet", index=False)
        print("=== 目前持倉 ===")
        print(open_df.to_string(index=False))
    else:
        print("=== 沒有持倉 ===")

    print(f"淨值 {equity['total'].iloc[-1]:,.0f}，交易 {len(trades_df)} 筆")
    print(f"saved backtest → {out_dir}/ ({today})")
    if push:
        hf_sync.push([out_dir])


if __name__ == "__main__":
    run(push=os.environ.get("NO_PUSH") != "1")
