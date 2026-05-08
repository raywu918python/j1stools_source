import sqlite3

import requests
import pandas as pd


def f1():
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": "2330",
        "start_date": "2026-05-06",
        "end_date": "2099-01-01",
        # "token": "你的token",  # 免費版不填也行，有限制
    }

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data["data"])
    print(df.T)


# date
# stock_id
# MarginPurchaseBuy
# MarginPurchaseCashRepayment
# MarginPurchaseLimit
# MarginPurchaseSell
# MarginPurchaseTodayBalance
# MarginPurchaseYesterdayBalance
# Note
# OffsetLoanAndShort
# ShortSaleBuy
# ShortSaleCashRepayment
# ShortSaleLimit
# ShortSaleSell
# ShortSaleTodayBalance
# ShortSaleYesterdayBalance


def update_stock_info():
    url = "https://api.finmindtrade.com/api/v4/data"
    r = requests.get(url, params={"dataset": "TaiwanStockInfo"})
    df_info = pd.DataFrame(r.json()["data"])
    print(df_info.head())

    conn = sqlite3.connect("../just1stock_web/db.sqlite3 ")
