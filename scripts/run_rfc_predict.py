import os, sys
from datetime import datetime, timedelta, timezone

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# pull 必須在 import rfc_main 之前，因為它的 default 參數會讀 db/
from j1stools import hf_sync

hf_sync.pull(["db/price", "db/active_stocks", "db/feature_cols", "models"])

from release.rfc_macd import rfc_main
from j1stools import parquet_db

stocks = parquet_db.activate_stocks()
stocks = list(set(stocks) - {"0050", "0052", "0056"})

today = datetime.now(_TW).strftime("%Y-%m-%d")
st = (datetime.now(_TW) - timedelta(days=10)).strftime("%Y-%m-%d")

df = rfc_main.predict(stocks, st, "2099-01-01")

MODEL_NAME = "rfc_macd_6xx"

if df is not None and not df.empty:
    out_dir = f"db/predictions/{MODEL_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/pred_{today}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df)} rows → {out_path}")
    hf_sync.push([f"db/predictions/{MODEL_NAME}"])
else:
    print("今日無預測結果")
