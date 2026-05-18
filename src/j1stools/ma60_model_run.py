import random

from j1stools import parquet_db
from j1stools.ma60_model import run

# stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
stocks = parquet_db.query_stocks_ids_list()
price_df = parquet_db.query_price(stocks, "2015-01-01", "2024-01-01")
market_df = parquet_db.query_price(["0050"], "2015-01-01", "2024-01-01")
results, data = run(price_df, market_df)
