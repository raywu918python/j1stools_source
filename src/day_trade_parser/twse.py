"""
上市當沖查詢
"""

import requests
import json
import pandas as pd


def get_all_day_trading(date: str) -> pd.DataFrame:
    """取出指定日期所有上市股票的當沖明細，date 格式: 'YYYYMMDD'

    回傳 DataFrame，欄位: 證券代號, 證券名稱, 暫停現股賣出後現款買進當沖註記,
                          當日沖銷交易成交股數, 當日沖銷交易買進成交金額, 當日沖銷交易賣出成交金額
    """
    url = "https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U"
    params = {"date": date, "response": "json"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    for table in data.get("tables", []):
        fields = table.get("fields", [])
        rows = table.get("data", [])
        if "證券代號" in fields and rows:
            return pd.DataFrame(rows, columns=fields)
    return pd.DataFrame()


def fetch_twse_daytrading_data(date, stock_no):
    """
    爬取證交所當日沖銷交易資料

    參數:
        date: 日期，格式為 YYYYMMDD (例如: 20260529)
        stock_no: 股票代號 (例如: 1102)

    回傳:
        JSON 資料 (dict)
    """
    url = f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U2"
    params = {
        "date": date,
        "stockNo": stock_no,
        "response": "json",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"請求失敗，狀態碼: {response.status_code}")
        return None


def for_loop():

    for d in range(1, 13):
        date = f"20260{d}01"
        stock_no = "1102"

        # 獲取資料
        data = fetch_twse_daytrading_data(date, stock_no)

        if data:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            json.dump(data, open(f"{date}_{stock_no}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        else:
            print(f"無法獲取 {date} 的資料")


if __name__ == "__main__":
    # 設定參數

    for_loop()
