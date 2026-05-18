import random

import pandas as pd

from j1stools import parquet_db
from j1stools.find_falling_google import find_wedge
from j1stools.ma60_model import run
from j1stools.triangle_google import find_refined_triangle

# stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
stocks = parquet_db.query_stocks_ids_list()
price_df = parquet_db.query_price(stocks, "2015-01-01", "2024-01-01")
market_df = parquet_db.query_price(["0050"], "2015-01-01", "2024-01-01")
triangle_df: pd.DataFrame = find_refined_triangle(price_df)
wedge_df = find_wedge(price_df)

print(triangle_df.shape)

# print(triangle_df.head())
run(price_df, market_df, triangle_df, wedge_df)
