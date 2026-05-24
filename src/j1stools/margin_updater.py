from datetime import datetime
import os
import threading

from numpy import add
import pandas as pd
from regex import F, P
import requests


from j1stools.parquet_db import activate_stocks

token = os.environ.get("FINMIND_TOKEN", "")


_COL_MAP = {
    "MarginPurchaseBuy": "margin_purchase_buy",
    "MarginPurchaseCashRepayment": "margin_purchase_cash_repayment",
    "MarginPurchaseLimit": "margin_purchase_limit",
    "MarginPurchaseSell": "margin_purchase_sell",
    "MarginPurchaseTodayBalance": "margin_purchase_today_balance",
    "MarginPurchaseYesterdayBalance": "margin_purchase_yesterday_balance",
    "Note": "note",
    "OffsetLoanAndShort": "offset_loan_and_short",
    "ShortSaleBuy": "short_sale_buy",
    "ShortSaleCashRepayment": "short_sale_cash_repayment",
    "ShortSaleLimit": "short_sale_limit",
    "ShortSaleSell": "short_sale_sell",
    "ShortSaleTodayBalance": "short_sale_today_balance",
    "ShortSaleYesterdayBalance": "short_sale_yesterday_balance",
}


def _margin_file_path():
    now = datetime.now()
    return f"db/margin/{now.year}_{now.month}.parquet"


def _download_margin(stock_id, start_date):
    r = requests.get(
        "https://api.finmindtrade.com/api/v4/data",
        params={
            "dataset": "TaiwanStockMarginPurchaseShortSale",
            "data_id": stock_id,
            "start_date": str(start_date),
            "end_date": "2099-01-01",
            "token": token,
        },
    )
    data = r.json()
    if "data" not in data or not data["data"]:
        return pd.DataFrame()
    return pd.DataFrame(data["data"])


def _save_margin(new_df, file_path):
    new_df = new_df.rename(columns=_COL_MAP)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    if os.path.exists(file_path):
        old_df = pd.read_parquet(file_path)
        old_df["date"] = pd.to_datetime(old_df["date"]).dt.strftime("%Y-%m-%d")
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    new_df.sort_values(["date", "stock_id"], inplace=True)
    new_df.to_parquet(file_path, index=False)


_FLAG_PATH = "db/margin_flags/margin_flag.parquet"


def _update_flag(stock_id, date_str):
    new_row = pd.DataFrame([{"stock_id": stock_id, "date": date_str}])
    if os.path.exists(_FLAG_PATH):
        df = pd.read_parquet(_FLAG_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
        df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    else:
        df = new_row
    os.makedirs(os.path.dirname(_FLAG_PATH), exist_ok=True)
    df.to_parquet(_FLAG_PATH, index=False)


def _get_wait_update_stocks(date_str):
    all_stocks = set(activate_stocks())
    if os.path.exists(_FLAG_PATH):
        df = pd.read_parquet(_FLAG_PATH)
        done = set(df[df["date"] == date_str]["stock_id"].tolist())
    else:
        done = set()
    return list(all_stocks - done)


def update_margin(start_date: str = None):
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

    now = datetime.now()
    file_path = _margin_file_path()
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs("db/margin", exist_ok=True)

    if start_date is None:
        if os.path.exists(file_path):
            start_date = pd.read_parquet(file_path)["date"].max()
        else:
            start_date = f"{now.year}-{now.month:02d}-01"

    stocks = _get_wait_update_stocks(date_str)
    print("還有", len(stocks), "個股票未更新")
    for stock_id in stocks:
        try:
            df = _download_margin(stock_id, start_date)
            if not df.empty:
                _save_margin(df, file_path)
            _update_flag(stock_id, date_str)
            print(stock_id, "下載完成")
        except Exception as e:
            print(f"{stock_id} 失敗: {e}")

    print(f"全部完成: {file_path}，共 {len(pd.read_parquet(file_path))} 筆")


if __name__ == "__main__":
    # qs = ActiveStocks.objects.all()
    # df = pd.DataFrame(list(qs.values()))
    # path = "db/active_stocks/"
    # os.makedirs(path, exist_ok=True)
    # df.to_parquet(f"{path}stocks.parquet")
    # print(df.head())
    # print(activate_stocks())

    update_margin()
