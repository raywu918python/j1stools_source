import os, sys
from datetime import datetime, timezone, timedelta

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# pull 必須在 import ma60_model_run 之前
from j1stools import hf_sync
hf_sync.pull(["db/price", "db/ib", "db/margin", "db/active_stocks", "db/feature_cols", "models"])

from j1stools import ma60_model_run

MODEL_NAME = "lgbm_timeseries_ensemble"
today = datetime.now(_TW).strftime("%Y-%m-%d")
out_dir = f"db/backtest/{MODEL_NAME}"

equity_df, trades_df, open_df, market_df = ma60_model_run.main(st="2024-01-01")

os.makedirs(out_dir, exist_ok=True)
equity_df.to_parquet(f"{out_dir}/equity_{today}.parquet", index=False)
trades_df.to_parquet(f"{out_dir}/trades_{today}.parquet", index=False)
market_df.to_parquet(f"{out_dir}/market_{today}.parquet", index=False)
if len(open_df):
    open_df.to_parquet(f"{out_dir}/open_{today}.parquet", index=False)

print(f"saved backtest → {out_dir}/ ({today})")
hf_sync.push([f"db/backtest/{MODEL_NAME}"])
