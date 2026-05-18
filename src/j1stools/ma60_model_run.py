import random

import pandas as pd

from j1stools import ma_cross_xgb, momentum_stats, parquet_db, triangle_stats
from j1stools.train_flow import print_target_counts
import ma_backtest


def main():
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
    stocks = parquet_db.query_stocks_ids_list()
    price_df = parquet_db.query_price(stocks, "2015-01-01", "2026-01-01")
    market_df = parquet_db.query_price(["0050"], "2015-01-01", "2026-01-01")
    # triangle_df: pd.DataFrame = find_refined_triangle(price_df)
    # wedge_df = find_wedge(price_df)

    # print(triangle_df.shape)

    # print(triangle_df.head())
    # triangle_stats.run(triangle_df, price_df)

    momentum_stats.run(price_df, market_df)
    # ma_backtest.run_backtest(signals, price_df, init_capital=1_000_000)
    # ma_backtest.run_backtest(signals, price_df, init_capital=1_000_000)


main()
