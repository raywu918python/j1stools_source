import random

import pandas as pd

from j1stools import ma_cross_xgb, parquet_db


def main():
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
    stocks = parquet_db.query_stocks_ids_list()
    price_df = parquet_db.query_price(stocks, "2015-01-01", "2024-01-01")
    market_df = parquet_db.query_price(["0050"], "2015-01-01", "2024-01-01")
    # triangle_df: pd.DataFrame = find_refined_triangle(price_df)
    # wedge_df = find_wedge(price_df)

    # print(triangle_df.shape)

    # print(triangle_df.head())
    ma_cross_xgb.run(price_df, market_df)


main()
