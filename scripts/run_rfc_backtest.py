import os, sys
from datetime import datetime, timezone, timedelta

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync

hf_sync.pull(["db/price", "db/active_stocks", "db/feature_cols", "db/stock_group", "models"])

from release.rfc_macd import backtest_platform

MODEL_NAME = "rfc_macd"
today = datetime.now(_TW).strftime("%Y-%m-%d")
out_dir = f"db/backtest/{MODEL_NAME}"

portfolio_value, trades_df, open_df, close_df = backtest_platform.main(st="2024-01-01")

os.makedirs(out_dir, exist_ok=True)
portfolio_value.to_parquet(f"{out_dir}/equity_{today}.parquet", index=False)
trades_df.to_parquet(f"{out_dir}/trades_{today}.parquet", index=False)
if len(open_df):
    open_df.to_parquet(f"{out_dir}/open_{today}.parquet", index=False)

print(f"saved backtest → {out_dir}/ ({today})")
hf_sync.push([f"db/backtest/{MODEL_NAME}"])
