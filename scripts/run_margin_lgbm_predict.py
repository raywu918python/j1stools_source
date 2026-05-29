import os, sys
from datetime import datetime, timedelta, timezone

from release_model.ma60_lgbm import margin_lgbm_main

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# pull 必須在 import margin_lgbm_main 之前，因為它的 default 參數會讀 db/
from j1stools import hf_sync

hf_sync.pull(["db/price", "db/ib", "db/margin", "db/active_stocks", "db/feature_cols", "models"])

from j1stools import parquet_db

stocks = parquet_db.activate_stocks()
stocks = list(set(stocks) - {"0050", "0052", "0056"})

today = datetime.now(_TW).strftime("%Y-%m-%d")
st = (datetime.now(_TW) - timedelta(days=10)).strftime("%Y-%m-%d")

df = margin_lgbm_main.predict(stocks, st, "2099-01-01")

MODEL_NAME = "lgbm_timeseries_ensemble"

if df is not None:
    out_dir = f"db/predictions/{MODEL_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/pred_{today}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df)} rows → {out_path}")
    hf_sync.push([f"db/predictions/{MODEL_NAME}"])
else:
    print("市場狀態不佳，今日不選股")
