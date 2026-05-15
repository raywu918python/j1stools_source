from calendar import c
from operator import index
import sqlite3

import requests
import pandas as pd

# TaiwanStockDayTrading
# def margin():
#     """融資"""
#     """
#     日期	# date
#     股票代碼	# stock_id
#     融資買進	# MarginPurchaseBuy
#     融資現金償還	# MarginPurchaseCashRepayment
#     融資限額	# MarginPurchaseLimit
#     融資賣出	# MarginPurchaseSell
#     融資今日餘額# MarginPurchaseTodayBalance
#     融資昨日餘額# MarginPurchaseYesterdayBalance
#     註記	# Note
#     資券互抵	# OffsetLoanAndShort
#     融券買進	# ShortSaleBuy
#     融券償還	# ShortSaleCashRepayment
#     融券限額	# ShortSaleLimit
#     融券賣出	# ShortSaleSell
#     融券今日餘額	# ShortSaleTodayBalance
#     融券昨日餘額# ShortSaleYesterdayBalance
#     """

#     url = "https://api.finmindtrade.com/api/v4/data"
#     params = {
#         "dataset": "TaiwanStockMarginPurchaseShortSale",
#         "data_id": "2330",
#         "start_date": "2026-05-06",
#         "end_date": "2099-01-01",
#         # "token": "你的token",  # 免費版不填也行，有限制
#     }

#     r = requests.get(url, params=params)
#     data = r.json()

#     df = pd.DataFrame(data["data"])
#     print(df.T)


# margin()


def download_stock_info():
    url = "https://api.finmindtrade.com/api/v4/data"
    r = requests.get(url, params={"dataset": "TaiwanStockInfo"})
    df_info = pd.DataFrame(r.json()["data"])
    print(df_info.head())
    df_info.to_csv("stock_info.csv")
    # conn = sqlite3.connect("../just1stock_web/db.sqlite3 ")


# download_stock_info()
