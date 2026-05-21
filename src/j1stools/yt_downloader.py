from colorsys import TWO_THIRD
from enum import Enum
from os import path
import time

import numpy as np
from patsy.mgcv_cubic_splines import te
from pytest import mark
from regex import P
import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from db_models.peewee_models import MyappActivestocks, MyappStocksinfo
from j1stools import lite_db
from j1stools.utils import get_base_url, stock


class INTERVAL(Enum):
    m1 = "1m"
    m5 = "5m"
    day = "1d"


def download(
    interval: INTERVAL,
    select_tickers: list = None,
    period="5d",
):

    def tickers_to_download(select_tickers: list = None):
        if select_tickers is None:
            stocks = MyappActivestocks.select(MyappActivestocks.stock_id)
            stocks = list([x.stock_id for x in stocks])
        else:
            stocks = select_tickers

        info = (
            MyappStocksinfo.select(MyappStocksinfo.stock_id, MyappStocksinfo.market_type)
            .where(MyappStocksinfo.stock_id.in_(stocks))
            .dicts()
        )

        all_tickers = []
        for i in info:
            stock_id = i.get("stock_id")
            market_type = "TW" if i.get("market_type") == "twse" else "TWO"
            ticker = f"{stock_id}.{market_type}"
            all_tickers.append(ticker)

        return all_tickers

    interval_value = interval.value

    all_tickers = tickers_to_download(select_tickers)
    chunk_size = 50  # 每次下載 50 檔

    for i in range(0, len(all_tickers), chunk_size):
        batch = all_tickers[i : i + chunk_size]

        df = yf.download(
            batch,
            period=period,
            auto_adjust=True,
            repair=True,
            interval=interval_value,
            group_by="ticker",
            threads=True,
        )
        if interval == INTERVAL.m1 or interval == INTERVAL.m5:
            df.index = df.index.tz_convert("Asia/Taipei")

        save(batch, df, interval)

        print(f"已完成第 {i+chunk_size} 檔...")

        time.sleep(2)


def save(all_path, dfall, interval: INTERVAL):
    interval_value = interval.value

    def save_and_merge(ticker, new_data):
        new_data = new_data[["Open", "High", "Low", "Close", "Volume"]]
        # 1. 定義檔案路徑
        file_path = f"{get_base_url() + ticker}_{interval_value}.csv"

        # 2. 檢查舊檔是否存在
        if os.path.exists(file_path):
            # 讀取舊檔，並將 Date 設為索引以利合併
            old_data = pd.read_csv(
                file_path,
                index_col=0,
                parse_dates=True,
            )
            if interval == INTERVAL.m1 or interval == INTERVAL.m5:
                old_data.index = old_data.index.tz_convert("Asia/Taipei")

            # 更好的做法是 combine_first：它會以新資料為主，補足舊資料沒有的部分
            combined = new_data.combine_first(old_data)

            # 4. 確保日期排序且刪除重複項 (避免重複下載同一天)
            combined = combined.sort_index().drop_duplicates()
        else:
            # 如果沒舊檔，就直接用新抓到的資料
            combined = new_data

        # 5. 存檔
        combined.to_csv(file_path)
        print(f"{ticker} 資料已更新並合併。")

    for path in all_path:
        df = dfall[path]
        stock = path.split(".")[0]
        save_and_merge(stock, df)


# download(INTERVAL.day)
# download(INTERVAL.m1)
# download(INTERVAL.m5)

# download(INTERVAL.day, ["6546"], period="3000d")
# download(INTERVAL.m1, ["1609"], period="60d")
# download(INTERVAL.m5, ["1609"], period="60d")
