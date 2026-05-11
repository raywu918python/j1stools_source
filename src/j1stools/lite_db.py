import pandas as pd

from db_models.peewee_models import MyappActivestocks, MyappStocksmargin
from j1stools import parquet_db


def query():
    d = MyappActivestocks.select()
    print("**********")
    for row in d:
        print(row.stock_id)


def margin(stocks, st, end):
    data = (
        MyappStocksmargin.select()
        .where(
            MyappStocksmargin.stock_id.in_(stocks),
            MyappStocksmargin.date >= st,
            MyappStocksmargin.date < end,
        )
        .dicts()
    )
    margin = pd.DataFrame(data)
    margin["date"] = pd.to_datetime(margin["date"])
    margin = margin.drop(columns=["id", "note"])
    margin.sort_values(by=["date", "stock_id"], inplace=True)

    price = parquet_db.query_price(margin["stock_id"].unique(), st, end)
    df = pd.merge(price, margin, on=["date", "stock_id"], how="left")
    return df
    # for row in data:
    # print(row)


# margin(["2330"], "2025-05-01", "2099-01-01")
