"""
上櫃當沖查詢
"""

import argparse
import requests
import pandas as pd
from datetime import date

API_URL = "https://www.tpex.org.tw/www/zh-tw/intraday/stat"


def get_all_day_trading(query_date: str) -> pd.DataFrame:
    """取出指定日期所有上櫃股票的當沖明細，query_date 格式: 'YYYY/MM/DD'"""
    payload = {"type": "Daily", "date": query_date, "id": "", "response": "json"}
    resp = requests.post(API_URL, data=payload, verify=False)
    resp.raise_for_status()
    data = resp.json()

    for table in data.get("tables", []):
        fields = table.get("fields", [])
        rows = table.get("data", [])
        if "證券代號" in fields and rows:
            return pd.DataFrame(rows, columns=fields)
    return pd.DataFrame()


def get_day_trading(stock_id: str, query_date: str) -> pd.DataFrame:
    """查詢單一上櫃個股當沖明細，query_date 格式: 'YYYY/MM/DD'"""
    df = get_all_day_trading(query_date)
    if df.empty:
        return df
    return df[df["證券代號"] == stock_id].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查詢上櫃當日沖銷明細")
    parser.add_argument("date", nargs="?", default=date.today().strftime("%Y/%m/%d"), help="查詢日期，格式 YYYY/MM/DD")
    parser.add_argument("stock_id", nargs="?", default=None, help="股票代號，例如 1240")
    args = parser.parse_args()

    # if args.stock_id:
    #     print(get_day_trading(args.stock_id, args.date))
    # else:
    #     print(get_all_day_trading(args.date))

    df = get_all_day_trading("2026/06/03")
    df.to_csv("2026_06_03.csv", index=False)
