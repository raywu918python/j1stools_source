from datetime import datetime
import os
import threading

from numpy import add
import pandas as pd
from regex import F, P
import requests

from myapp.models import ActiveStocks, StocksIbBuySell, StocksMargin, StocksUpdateFlag

from myapp.models import StocksInfo

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoic3VwZXJ3bXIiLCJlbWFpbCI6InN1cGVyd21yQGdtYWlsLmNvbSIsInRva2VuX3ZlcnNpb24iOjB9.tjvjaid1Xp8PBiGFhmTgRRVv6obyYYCO412RmtZNmAo"


def init_stocksinfo():
    df = pd.read_csv("stock_info.csv", dtype={"stock_id": str}, parse_dates=["date"])  # .reset_index(drop=True)
    df = df[["industry_category", "stock_id", "stock_name", "type", "date"]]
    df.rename(columns={"industry_category": "group", "type": "market_type"}, inplace=True)
    df = df[df["stock_id"].astype(str).str.len() == 4]
    df.sort_values(by=["date"], inplace=True)
    StocksInfo.objects.all().delete()
    StocksInfo(stock_id=df["stock_id"], name=df["stock_name"], market_type=df["market_type"], group=df["group"])
    for _, row in df.iterrows():
        StocksInfo.objects.update_or_create(
            stock_id=row["stock_id"],  # 用於查找的條件欄位
            defaults={  # 需要建立或更新的資料
                "name": row["stock_name"],
                "market_type": row["market_type"],
                "group": row["group"],
            },
        )


def init_activestocks():
    """取csv檔名前4碼"""
    target_dir = r"data/"
    unique_prefixes = set()

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            # 取出檔案名稱（不含路徑）的前 4 碼
            prefix = file[:4]
            if file and file[0].isdigit():
                prefix = file[:4]
                unique_prefixes.add(prefix)

    # 印出所有不重複的前 4 碼
    for p in sorted(unique_prefixes):
        print(p)

    ActiveStocks.objects.all().delete()
    ActiveStocks.objects.bulk_create([ActiveStocks(stock_id=p) for p in sorted(unique_prefixes)])


def update_stocks_margin(start_date="2015-01-01"):
    """融資"""
    """
    日期	# date
    股票代碼	# stock_id
    融資買進	# MarginPurchaseBuy
    融資現金償還	# MarginPurchaseCashRepayment
    融資限額	# MarginPurchaseLimit
    融資賣出	# MarginPurchaseSell
    融資今日餘額# MarginPurchaseTodayBalance
    融資昨日餘額# MarginPurchaseYesterdayBalance
    註記	# Note
    資券互抵	# OffsetLoanAndShort
    融券買進	# ShortSaleBuy
    融券償還	# ShortSaleCashRepayment
    融券限額	# ShortSaleLimit
    融券賣出	# ShortSaleSell
    融券今日餘額	# ShortSaleTodayBalance
    融券昨日餘額# ShortSaleYesterdayBalance
    """
    flag = "margin"
    date_str = datetime.now().strftime("%Y-%m-%d")
    print("date_str", date_str)

    def add_flag(stock_id):
        StocksUpdateFlag.objects.update_or_create(
            stock_id=stock_id,
            date=date_str,
            note=flag,
            defaults={
                "flag": 1,
            },
        )

    def get_wait_update_stocks():
        stocks = ActiveStocks.objects.all()
        updated_stocks = StocksUpdateFlag.objects.filter(date=date_str, flag=1, note=flag)

        stocks = [s.stock_id for s in stocks]
        updated_stocks = [s.stock_id for s in updated_stocks]
        stocks = list(set(stocks) - set(updated_stocks))
        return stocks

    def download(stock_id):
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockMarginPurchaseShortSale",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": "2099-01-01",
            "token": token,  # 免費版不填也行，有限制
        }

        r = requests.get(url, params=params)
        data = r.json()
        df = pd.DataFrame(data["data"])
        return df

    def add_margin(df):
        # StocksMargin.objects.all().delete()
        StocksMargin.objects.bulk_create(
            [
                StocksMargin(
                    stock_id=row["stock_id"],
                    date=row["date"],
                    margin_purchase_buy=row["MarginPurchaseBuy"],
                    margin_purchase_cash_repayment=row["MarginPurchaseCashRepayment"],
                    margin_purchase_limit=row["MarginPurchaseLimit"],
                    margin_purchase_sell=row["MarginPurchaseSell"],
                    margin_purchase_today_balance=row["MarginPurchaseTodayBalance"],
                    margin_purchase_yesterday_balance=row["MarginPurchaseYesterdayBalance"],
                    note=row["Note"],
                    offset_loan_and_short=row["OffsetLoanAndShort"],
                    short_sale_buy=row["ShortSaleBuy"],
                    short_sale_cash_repayment=row["ShortSaleCashRepayment"],
                    short_sale_limit=row["ShortSaleLimit"],
                    short_sale_sell=row["ShortSaleSell"],
                    short_sale_today_balance=row["ShortSaleTodayBalance"],
                    short_sale_yesterday_balance=row["ShortSaleYesterdayBalance"],
                )
                for _, row in df.iterrows()
            ],
            update_conflicts=True,
            unique_fields=["stock_id", "date"],
            update_fields=[
                "margin_purchase_buy",
                "margin_purchase_cash_repayment",
                "margin_purchase_limit",
                "margin_purchase_sell",
                "margin_purchase_today_balance",
                "margin_purchase_yesterday_balance",
                "note",
                "offset_loan_and_short",
                "short_sale_buy",
                "short_sale_cash_repayment",
                "short_sale_limit",
                "short_sale_sell",
                "short_sale_today_balance",
                "short_sale_yesterday_balance",
            ],
        )

    try:
        stocks = get_wait_update_stocks()
        print("還有", len(stocks), "個股票未更新")
        for stock_id in stocks:
            df = download(stock_id)
            add_margin(df)
            add_flag(stock_id)
            print(stock_id, "更新完成")
    except Exception as e:
        print(e)


def update_stocks_ib_buy_sell(start_date="2015-01-01"):
    """法人"""
    flag = "ib_buy_sell"
    date_str = datetime.now().strftime("%Y-%m-%d")

    def add_flag(stock_id):
        StocksUpdateFlag.objects.update_or_create(
            stock_id=stock_id,
            date=date_str,
            note=flag,
            defaults={
                "flag": 1,
            },
        )

    def get_wait_update_stocks():
        stocks = ActiveStocks.objects.all()
        updated_stocks = StocksUpdateFlag.objects.filter(date=date_str, flag=1, note=flag)

        stocks = [s.stock_id for s in stocks]
        updated_stocks = [s.stock_id for s in updated_stocks]
        wait_update_stocks = list(set(stocks) - set(updated_stocks))
        return wait_update_stocks

    def download(stock_id):
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": "2099-01-01",
            "token": token,  # 免費版不填也行，有限制
        }

        r = requests.get(url, params=params)
        data = r.json()
        df = pd.DataFrame(data["data"])
        return df

    def insert_db(df):
        StocksIbBuySell.objects.bulk_create(
            [
                StocksIbBuySell(
                    stock_id=row["stock_id"],
                    date=row["date"],
                    buy=row["buy"],
                    sell=row["sell"],
                    name=row["name"],
                )
                for _, row in df.iterrows()
            ],
            update_conflicts=True,
            unique_fields=["stock_id", "date", "name"],
            update_fields=[
                "buy",
                "sell",
            ],
        )

    try:
        stocks = get_wait_update_stocks()
        print("還有", len(stocks), "個股票未更新")
        for stock_id in stocks:
            df = download(stock_id)
            insert_db(df)
            add_flag(stock_id)
            print(stock_id, "更新完成")
    except Exception as e:
        print(e)


# StocksUpdateFlag.objects.all().delete()
# StocksIbBuySell.objects.all().delete()
#
# init_stocks_ib_buy_sell(start_date="2026-05-01")
# init_stocks_margin(start_date="2026-05-01")
