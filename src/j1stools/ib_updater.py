from datetime import datetime, timezone, timedelta

_TW = timezone(timedelta(hours=8))
import os

import pandas as pd
import requests

from j1stools.parquet_db import activate_stocks

token = os.environ.get("FINMIND_TOKEN", "")

_FLAG_PATH = "db/ib_flags/ib_flag.parquet"


def _ib_file_path():
    now = datetime.now(_TW)
    return f"db/ib/{now.year}_{now.month}.parquet"


def _download_ib(stock_id, start_date):
    r = requests.get(
        "https://api.finmindtrade.com/api/v4/data",
        params={
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
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


def _save_ib(new_df, file_path):
    new_df = new_df[["stock_id", "date", "name", "buy", "sell"]].copy()
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    if os.path.exists(file_path):
        old_df = pd.read_parquet(file_path)
        old_df["date"] = pd.to_datetime(old_df["date"]).dt.strftime("%Y-%m-%d")
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.drop_duplicates(subset=["stock_id", "date", "name"], keep="last", inplace=True)
    new_df.sort_values(["date", "stock_id", "name"], inplace=True)
    new_df.to_parquet(file_path, index=False)


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


def update_ib(start_date: str = None):
    """法人買賣超"""
    now = datetime.now(_TW)
    file_path = _ib_file_path()
    date_str = now.strftime("%Y-%m-%d")
    os.makedirs("db/ib", exist_ok=True)

    if start_date is None:
        if os.path.exists(file_path):
            start_date = pd.read_parquet(file_path)["date"].max()
        else:
            start_date = f"{now.year}-{now.month:02d}-01"

    stocks = _get_wait_update_stocks(date_str)
    print("還有", len(stocks), "個股票未更新")
    for stock_id in stocks:
        try:
            df = _download_ib(stock_id, start_date)
            if not df.empty:
                _save_ib(df, file_path)
                _update_flag(stock_id, date_str)
                print(stock_id, "下載完成")
        except Exception as e:
            print(f"{stock_id} 失敗: {e}")

    print(f"全部完成: {file_path}，共 {len(pd.read_parquet(file_path))} 筆")


if __name__ == "__main__":
    update_ib()
