import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync, margin_lgbm_main, parquet_db

hf_sync.pull(["db/price", "db/ib", "db/margin", "db/active_stocks", "db/feature_cols", "models"])

stocks = parquet_db.activate_stocks()
stocks = list(set(stocks) - {"0050", "0052", "0056"})

today = datetime.now().strftime("%Y-%m-%d")
st = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

df = margin_lgbm_main.predict(stocks, st, "2099-01-01")

if df is not None:
    os.makedirs("db/predictions", exist_ok=True)
    out_path = f"db/predictions/pred_{today}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df)} rows → {out_path}")
    hf_sync.push(["db/predictions"])
else:
    print("市場狀態不佳，今日不選股")
