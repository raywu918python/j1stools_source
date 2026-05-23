# import joblib

# from j1stools import j1s_chart, margin_lgbm_main, parquet_db, rfc_main
# from j1stools.backtest_engine import backtest_engine
# from j1stools.backtest_platform import PrepareDate, prepare_data_backtest, win6XX
# from j1stools.django_orm import *
# from j1stools.lgbm_orgin_main import add_day


# def web_backtest(stocks):
#     """for web"""
#     signal = margin_lgbm_main.predict(stocks, "2025-01-01", "2099-01-01")
#     portfolio_value, trades_df, positions, close = prepare_data_backtest(
#         signal,
#         threshold=0.6,
#         max_positions=10,
#         group_limit=99,
#         tp_stop=0.15,
#         sl_stop=0.15,
#         use_sl_trail=False,
#     )

#     return j1s_chart.plot_performance(
#         portfolio_value=portfolio_value,
#         trades_df=trades_df,
#         is_web=True,
#     )


# def web_query_last(
#     st="2024-01",
#     end="2099-01",
# ):
#     """for web"""
#     signal = rfc_main.predict(parquet_db.query_stocks_no_etf(), st, end)
#     portfolio_value, trades_df, positions = win6XX(signal)

#     # j1s_chart.plot_performance(
#     #     portfolio_value=portfolio_value,
#     #     trades_df=trades_df,
#     #     is_web=True,
#     # )
#     return portfolio_value, trades_df, positions


# def predict_margin_model(model_path="models/lgbm_timeseries_ensemble.joblib"):
#     stocks = list(ActiveStocks.objects.values_list("stock_id", flat=True))
#     stocks = list(set(stocks) - set(["0050", "0052", "0056"]))
#     st = "2024-01-01"
#     end = "2099-01-01"
#     model = joblib.load(model_path)
#     df = margin_lgbm_main.predict(stocks, st, end, model=model)
#     return df


# if __name__ == "__main__":
#     print("============== web query last ==============")
#     df = predict_margin_model()
#     print(df.head())
#     # web_query_last()
#     pass
