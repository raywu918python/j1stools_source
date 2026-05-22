from os import times
import os
import random
import time

import pandas as pd
from pandas import Timestamp

from j1stools import ma_cross_xgb, margin_lgbm_main, momentum_backtest, momentum_stats, parquet_db, triangle_stats
from j1stools.train_flow import print_target_counts
import ma_backtest


def sort_signals(signals: pd.DataFrame, pred) -> pd.DataFrame:
    import pandas as pd

    # 確保格式一致
    signals["date"] = pd.to_datetime(signals["date"])
    signals["stock_id"] = signals["stock_id"].astype(str).str.strip()

    pred["date"] = pd.to_datetime(pred["date"])
    pred["stock_id"] = pred["stock_id"].astype(str).str.strip()

    # merge：只保留兩者都有的 (date, stock_id)
    selected = signals.merge(pred[["date", "stock_id", "pred_score"]], on=["date", "stock_id"], how="inner")

    # 每天按 pred_score 排序，取前 N 名
    selected = selected.sort_values(["date", "pred_score"], ascending=[True, False]).reset_index(drop=True)

    print(f"原始訊號：{len(signals):,} 筆")
    print(f"有排序的：{len(selected):,} 筆")
    return selected


def get_sord_margin(stocks, st, end):
    from datetime import date

    today = date.today()
    find_name = "margin" + today.strftime("%Y%m%d") + st + end + ".csv"

    if os.path.exists(find_name):
        df_margin = pd.read_csv(find_name)
    else:
        df_margin = margin_lgbm_main.predict(stocks, st, end)
        df_margin.to_csv(find_name, index=False)

    # df_margin = df_margin[df_margin["pred_score"] > 0.2]
    return df_margin


def good_search(signals, df_price, df_market):
    """
    momentum_stats.USE_COND5 = False
    momentum_stats.USE_COND3 = False
    --- 大盤比較 ---
                                  策略          大盤          超額
    總報酬                     +156.03%     +71.34%     +84.69%
    年化報酬(CAGR)               +76.83%     +38.61%     +38.22%
    最大回撤                     -14.55%     -27.48%     +12.94%
    """
    momentum_backtest.run_backtest(
        signals,
        df_price,
        df_market=df_market,
        stop_pct=0.07,
        target_pct=0.15,
        # exit_mode="trailing",
        trail_pct=0.10,
        max_positions=3,
        position_pct=0.33,
        max_hold=30,
        sort_by=[
            # momentum_backtest.SortBy.PRED_SCORE.desc(),  # 分數大→小
            momentum_backtest.SortBy.MA60_TURN_DAYS.asc(),  # 天數小→大
            momentum_backtest.SortBy.DEV_MA60.asc(),  # 距離小→大
        ],
    )


def main():
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
    st = "2024-01-01"
    end = "2026-07-01"
    # stocks = parquet_db.query_stocks_ids_list()
    stocks = parquet_db.query_stocks_no_etf()
    df_price = parquet_db.query_price(stocks, st, end)
    df_market = parquet_db.query_price(["0050"], st, end)

    #############################################################
    # momentum_stats.USE_DEDUP = False
    momentum_stats.USE_COND5 = False
    momentum_stats.USE_COND3 = False
    # 調整門檻（例如 30%）
    momentum_stats.MAX_DEV_MA60 = 0.2
    momentum_stats.USE_MAX_DEV_MA60 = False
    signals = momentum_stats.run(df_price, df_market)
    signals = sort_signals(signals, get_sord_margin(stocks, st, end))

    #############################################################
    if "pred_score" in signals.columns:
        momentum_backtest.run_backtest(
            signals,
            df_price,
            df_market=df_market,
            stop_pct=0.07,
            target_pct=0.15,
            # exit_mode="trailing",
            trail_pct=0.10,
            max_positions=3,
            position_pct=0.33,
            max_hold=30,
            sort_by=[
                # momentum_backtest.SortBy.PRED_SCORE.desc(),  # 分數大→小
                momentum_backtest.SortBy.MA60_TURN_DAYS.asc(),  # 天數小→大
                momentum_backtest.SortBy.DEV_MA60.asc(),  # 距離小→大
            ],
        )
    else:
        momentum_backtest.run_backtest(
            signals,
            df_price,
            df_market=df_market,
        )


if __name__ == "__main__":
    main()
