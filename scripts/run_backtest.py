import os, sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# pull 必須在 import ma60_model_run 之前
from j1stools import hf_sync
hf_sync.pull(["db/price", "db/ib", "db/margin", "db/active_stocks", "db/feature_cols", "models"])

from j1stools import ma60_model_run

today = datetime.now().strftime("%Y-%m-%d")

equity_df, trades_df, open_df = ma60_model_run.main(st="2024-01-01")

os.makedirs("db/backtest", exist_ok=True)
equity_df.to_parquet(f"db/backtest/equity_{today}.parquet", index=False)
trades_df.to_parquet(f"db/backtest/trades_{today}.parquet", index=False)
if len(open_df):
    open_df.to_parquet(f"db/backtest/open_{today}.parquet", index=False)

print(f"saved backtest → db/backtest/ ({today})")
hf_sync.push(["db/backtest"])
