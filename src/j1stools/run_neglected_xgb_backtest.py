import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta, timezone

import pandas as pd

from j1stools import backtest_platform, parquet_db
from run_neglected_xgb_predict import MODEL_NAME, predict

_TW = timezone(timedelta(hours=8))
today = datetime.now(_TW).strftime("%Y-%m-%d")
out_dir = f"db/backtest/{MODEL_NAME}"


def backtest(st: str = "2024-01-01", end: str = "2099-01-01"):
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    signal = predict(stocks, st, end)

    portfolio_value, trades_df, positions, close_df = backtest_platform.prepare_data_backtest(
        signal,
        top_n=5,
        threshold=0.35,
        max_positions=5,
        use_sl_trail=False,
        use_fixed_sl=True,
        sl_stop=0.10,
        use_fixed_tp=True,
        tp_stop=0.10,
        use_hold_days=True,
        hold_days=10,
        group_limit=2,
        min_volume=200,
    )

    # 大盤對比（0050）
    eq_dates = pd.to_datetime(portfolio_value["date"]).dt.normalize()
    mkt = parquet_db.query_price(["0050"], st, end)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
    mkt = mkt[mkt["date"].isin(eq_dates)].reset_index(drop=True)
    if mkt.empty:
        mkt = parquet_db.query_price(["0050"], st, end)
        mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
        mkt = mkt[mkt["date"] >= eq_dates.iloc[0]].reset_index(drop=True)
    start_val = portfolio_value["total"].iloc[0]
    mkt["total"] = (mkt["close"] / mkt["close"].iloc[0] * start_val).round(2)
    market_df = mkt[["date", "total"]]

    return portfolio_value, trades_df, positions, close_df, market_df


portfolio_value, trades_df, open_df, close_df, market_df = backtest(st="2024-01-01")

if not trades_df.empty:
    pv = portfolio_value.copy()
    pv["date"] = pd.to_datetime(pv["date"]).dt.normalize()
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.normalize()
    trades_df = trades_df.merge(
        pv[["date", "total"]].rename(columns={"date": "entry_date", "total": "_pv"}),
        on="entry_date",
        how="left",
    )
    trades_df["weight"] = (trades_df["cost"] / trades_df["_pv"]).round(4)
    trades_df = trades_df.drop(columns=["_pv"])

os.makedirs(out_dir, exist_ok=True)
portfolio_value.to_parquet(f"{out_dir}/equity_{today}.parquet", index=False)
trades_df.to_parquet(f"{out_dir}/trades_{today}.parquet", index=False)
market_df.to_parquet(f"{out_dir}/market_{today}.parquet", index=False)

if open_df:
    last_price = close_df.iloc[-1]
    total_value = float(portfolio_value["total"].iloc[-1])
    open_positions = pd.DataFrame(
        [
            {
                "stock_id": sid,
                **{k: v for k, v in pos.items() if k != "entry_bar"},
                "last_date": close_df.index[-1],
                "pnl_pct": round((last_price.get(sid, pos["entry_price"]) / pos["entry_price"] - 1) * 100, 2),
                "weight": round(pos["cost"] / total_value, 4),
            }
            for sid, pos in open_df.items()
        ]
    )
    open_positions.to_parquet(f"{out_dir}/open_{today}.parquet", index=False)
    print("=== 目前持倉 ===")
    print(open_positions.to_string(index=False))

print(f"saved backtest → {out_dir}/ ({today})")


"""
完成。兩個 script：

run_neglected_xgb_predict.py — 載入 neglected_xgb.pkl + GMM，產生信號格式 date, stock_id, 0, 1
run_neglected_xgb_backtest.py — 呼叫 predict → backtest_platform.prepare_data_backtest，輸出 parquet 到 db/backtest/neglected_xgb/
流程：

先跑 MODE = "release" 產生 db/models/neglected_xgb.pkl
再跑 run_neglected_xgb_backtest.py
threshold=0.6、sl_stop=0.10、tp_stop=0.10、hold_days=10 和 RFC backtest 一樣，可以直接對比。
"""
