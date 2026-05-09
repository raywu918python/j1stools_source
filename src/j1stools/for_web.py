from j1stools import j1s_chart, lgbm_main, parquet_db
from j1stools.backtest_engine import backtest_engine
from j1stools.backtest_platform import PrepareDate, prepare_data_backtest


def web_backtest(stocks):
    """for web"""
    signal = lgbm_main.predict(stocks, "2025-01-01", "2099-01-01")
    portfolio_value, trades_df, positions, close = prepare_data_backtest(
        signal,
        threshold=0.6,
        max_positions=10,
        group_limit=99,
        tp_stop=0.15,
        sl_stop=0.15,
        use_sl_trail=False,
    )

    return j1s_chart.plot_performance(
        portfolio_value=portfolio_value,
        trades_df=trades_df,
        is_web=True,
    )


def web_query_last(
    st="2024-01",
    end="2099-01",
):
    """for web"""
    signal = lgbm_main.predict(parquet_db.query_stocks_no_etf(), st, end)
    p = PrepareDate(signal=signal, top_n=3, threshold=0.8)
    portfolio_value, trades_df, positions = backtest_engine(
        use_sl_trail=False,
        sl_trail=0.15,
        use_fixed_sl=True,
        sl_stop=0.20,
        use_fixed_tp=True,
        tp_stop=0.15,
        use_hold_days=True,
        hold_days=30,
        #############################################################
        use_fixed_sl_tp=False,
        use_proba_sizing=False,
        #############################################################
        max_positions=5,
        group_limit=1,
        stock_group=p.stock_group,
        #############################################################
        init_cash=1_000_000,
        fee=0.001,
        close=p.close,
        entries=p.entries,
        exits=p.exits,
        df_proba=p.proba,
    )

    # j1s_chart.plot_performance(
    #     portfolio_value=portfolio_value,
    #     trades_df=trades_df,
    #     is_web=True,
    # )
    return portfolio_value, trades_df, positions
