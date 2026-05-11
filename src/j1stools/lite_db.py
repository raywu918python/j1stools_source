import pandas as pd

from db_models.peewee_models import MyappActivestocks, MyappStocksinfo, MyappStocksmargin
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

    price = parquet_db.query_price(stocks, st, end)
    df = pd.merge(price, margin, on=["date", "stock_id"], how="left")
    return df
    # for row in data:
    # print(row)


def group(stocks, st, end):
    group = MyappStocksinfo().select().where(MyappStocksinfo.stock_id.in_(stocks)).dicts()
    group = pd.DataFrame(group)
    group = group[["stock_id", "group"]]
    price = parquet_db.query_price(stocks, st, end)
    df = pd.merge(price, group, on=["stock_id"], how="left")
    return df


# query(["2330", "2308", "3105"], "2024-01-01", "2099-01-01")

# margin(["2330"], "2025-05-01", "2099-01-01")
