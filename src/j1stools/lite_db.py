import pandas as pd

from db_models.peewee_models import MyappActivestocks, MyappStocksibbuysell, MyappStocksinfo, MyappStocksmargin
from j1stools import parquet_db


def query():
    d = MyappActivestocks.select()
    print("**********")
    for row in d:
        print(row.stock_id)


def margin(stocks, st, end):
    """
    日期	# date
    股票代碼	# stock_id
    融資買進	# margin_purchase_buy
    融資現金償還	# margin_purchase_cash_repayment
    融資限額	# margin_purchase_limit
    融資賣出	# margin_purchase_sell
    融資今日餘額	# margin_purchase_today_balance
    融資昨日餘額	# margin_purchase_yesterday_balance
    資券互抵	# offset_loan_and_short
    融券買進	# short_sale_buy
    融券償還	# short_sale_cash_repayment
    融券限額	# short_sale_limit
    融券賣出	# short_sale_sell
    融券今日餘額	# short_sale_today_balance
    融券昨日餘額	# short_sale_yesterday_balance
    """

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
    print(df.head().T)
    return df


def margin_group(stocks, st, end):
    df_margin = margin(stocks, st, end)
    df_group = MyappStocksinfo().select().where(MyappStocksinfo.stock_id.in_(stocks)).dicts()
    df_group = pd.DataFrame(df_group)
    df = pd.merge(df_margin, df_group, on=["stock_id"], how="left")
    print(df.head().T)
    return df


def ibbuysell(stocks, st, end):
    # date: str, # 日期
    # stock_id: str, # 股票代碼
    # buy: int64, # 買進
    # name: str, # 類別
    # sell: int64 # 賣出
    data = (
        MyappStocksibbuysell.select()
        .where(
            MyappStocksibbuysell.stock_id.in_(stocks),
            MyappStocksibbuysell.date >= st,
            MyappStocksibbuysell.date < end,
        )
        .dicts()
    )
    df = pd.DataFrame(data)
    df = df.drop(columns=["id"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# margin_group(["2330", "2308", "3105"], "2024-01-01", "2099-01-01")

# margin(["2330"], "2025-05-01", "2099-01-01")
