from colorsys import TWO_THIRD
from enum import Enum
from os import path
import time

import numpy as np
from patsy.mgcv_cubic_splines import te
from pytest import mark
import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from j1stools.utils import get_base_url


class MARKET(Enum):
    tw = "TW"
    two = "TWO"


class TIME(Enum):
    m1 = "1m"
    m5 = "5m"
    day = "d"
    day2015_2099 = "allday"


def download_from_yf(stock, time: TIME, market: MARKET):
    template = None

    if time == TIME.day2015_2099:
        load_dotenv()
        s = os.getenv("START")
        e = os.getenv("END")

        template = yf.download(f"{stock}.{market.value}", start=s, end=e, repair=True)
    elif time == TIME.day:
        print(f"下載 {stock} 日線")
        tickers = "2330.TW,0050.TW,2303.TW,0052.TW,0056.TW"
        # template = yf.download("2330.TW,0050.TW", period="5d", repair=True)
        template = yf.download(
            tickers,
            period="50d",
            auto_adjust=True,
            repair=True,
            interval="1d",
            group_by="ticker",
            threads=True,
        )
        # template = yf.download(tickers, period="5d", repair=True)
        # template = yf.download(tickers, period="5d")
    else:
        template = yf.download(
            tickers=f"{stock}.{[market]}",
            period="5d",
            interval=time.value,
            repair=True,
        )

        template.index = template.index.tz_convert("Asia/Taipei")

    # 如果你希望移除時區符號 (+08:00)，只保留時間數值
    template.index = template.index.tz_localize(None)

    # 1. 先只抓取特定的欄位 (通常 yf 下載回來會有 'Adj Close')
    # template = template[["Open", "High", "Low", "Close", "Volume"]]

    # 2. 再進行重新命名 (如果你需要自定義名稱或大小寫)
    # template.columns = ["Open", "High", "Low", "Close", "Volume"]

    return template


#
# 產生檔案
#
def format_stock_file(file_name):
    df = pd.read_csv(f"{get_base_url() + file_name}")
    # print(df.head(5)["代號"].re('=',''))

    # code = df.head(5).to_string.strip('="')
    df["代號"] = df["代號"].str.replace("=", "", regex=False).str.replace('"', "", regex=False)
    # df['代號'] = df['代號'].str.replace('=', '', regex=False)
    df = df[df["代號"].str.len() == 4]

    new_df = df[["代號", "名稱", "市場"]]
    # code = df['代號']
    # print(new_df.head)
    return new_df


def save_download_stock_file():
    # https://goodinfo.tw/tw/StockList.asp?MARKET_CAT=%E7%86%B1%E9%96%80%E6%8E%92%E8%A1%8C&INDUSTRY_CAT=%E6%88%90%E4%BA%A4%E5%BC%B5%E6%95%B8+%28%E9%AB%98%E2%86%92%E4%BD%8E%29%40%40%E6%88%90%E4%BA%A4%E5%BC%B5%E6%95%B8%40%40%E7%94%B1%E9%AB%98%E2%86%92%E4%BD%8E&REINIT=46113.9056944444
    df1 = format_stock_file("StockList1.csv")
    df2 = format_stock_file("StockList2.csv")
    df3 = format_stock_file("StockList3.csv")
    df4 = format_stock_file("StockList4.csv")
    df5 = format_stock_file("StockList5.csv")
    df6 = format_stock_file("StockList6.csv")
    df7 = format_stock_file("StockList7.csv")
    df7 = format_stock_file("StockList8.csv")
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7]).drop_duplicates(subset=["代號"], ignore_index=True)
    # )  # .reset_index(drop=True)
    # df = df.drop_duplicates(subset=["代號"], keep="first").reset_index()

    # df.to_csv(get_base_url() + "stock_list.csv", index=False, header=False)
    df.to_csv(get_base_url() + "stock_list.csv")


def save_price_file(is_min, filter_stock=None):
    df_from = pd.read_csv(get_base_url() + "stock_list.csv", dtype={"代號": str})

    test = False
    if test:
        # 5871
        df_from = df_from[df_from["代號"] == "0050"]

        for stock, market in zip(df_from["代號"], df_from["市場"]):
            pass
            # download_m(stock, 5, market)
            # download_from_yf(stock, 0, market)

            # df_now = download_from_yf(stock, 0, market)
            # print(f'{df_now.head}')
            # df_old = pd.read_csv(f'{stock}_m1.csv', dtype={'代號': str}
            #                     ,parse_dates=['Datetime'] # 這裡請填入你 CSV 裡的日期欄位名
            #                     ,index_col='Datetime')

            # utils.save_csv(df_now, f'{stock}_m1')

    elif is_min:
        for stock, market in zip(df_from["代號"], df_from["市場"]):
            print(f"代號: {stock}, 市場: {market}")
            download_m(stock, 1, market)
            download_m(stock, 5, market)

    else:
        for stock, market in zip(df_from["代號"], df_from["市場"]):

            if filter_stock != None:
                if stock != filter_stock:
                    continue

            # for stock, market in zip(["0050"], ["市"]):
            df_new = download_from_yf(stock, 0, market)
            if not df_new.empty:
                filename = f"{get_base_url()+stock}_d1.csv"
                try:
                    df_old = pd.read_csv(
                        filename,
                        dtype={"代號": str},
                        parse_dates=["Date"],  # 這裡請填入你 CSV 裡的日期欄位名
                        index_col="Date",
                    )
                except FileNotFoundError:
                    print(f"找不到檔案 {filename}，將建立新的 DataFrame")
                    df_old = pd.DataFrame()
                df_combined = (
                    pd.concat([df_old, df_new])
                    .reset_index()
                    .drop_duplicates(subset=["Date"], keep="last")
                    .set_index("Date")
                    .sort_index()
                )
                df_combined.to_csv(f"{get_base_url()+stock}_d1.csv")


def download_m(stock, min, market):
    df_now = download_from_yf(stock, min, market)
    if not df_now.empty:
        filename = f"{get_base_url()+stock}_m{min}.csv"
        try:
            df_old = pd.read_csv(
                filename,
                dtype={"代號": str},
                parse_dates=["Datetime"],
                index_col="Datetime",
            )
        except FileNotFoundError:
            print(f"找不到檔案 {filename}，將建立新的 DataFrame")
            df_old = pd.DataFrame()

        df_combined = (
            pd.concat([df_old, df_now])
            .reset_index()
            .drop_duplicates(subset=["Datetime"], keep="last")
            .set_index("Datetime")
            .sort_index()
        )
        df_combined.to_csv(f"{get_base_url()+stock}_m{min}.csv")


def load_m30():
    # 1. 讀取資料 (確保 Datetime 是 index 並轉換為時間格式)
    path = get_base_url() + "0050_m5.csv"

    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)

    # 2. 過濾出每天 09:00 到 09:30 的資料
    # between_time 會檢查索引中的「時間」部分，忽略日期
    df_first_30m = df.between_time("09:00", "09:30")

    # 3. 查看結果
    print(df_first_30m.head(35))  # 可以看到跨日期的前 30 分鐘


def clear_format(file_name):
    # ,Date,Unnamed: 0.1,Unnamed: 0,Open,High,Low,Close,Volume
    df = pd.read_csv(f"{get_base_url() + file_name}_d1.csv")
    d = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    d.set_index("Date").to_csv(get_base_url() + file_name + "_d1.csv")
    # print(d.head())


def save_and_merge(ticker, new_data):
    new_data = new_data[["Open", "High", "Low", "Close", "Volume"]]
    # 1. 定義檔案路徑
    file_path = f"{get_base_url() + ticker}_d1.csv"

    # 2. 檢查舊檔是否存在
    if os.path.exists(file_path):
        # 讀取舊檔，並將 Date 設為索引以利合併
        old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 3. 合併資料
        # concat 會將新舊資料接在一起
        # combined = pd.concat([old_data, new_data])

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


def download_day():
    dfcsv = pd.read_csv(get_base_url() + "stock_list.csv", dtype={"代號": str})
    dfcsv["download_path"] = dfcsv["代號"] + "." + np.where(dfcsv["市場"] == "市", "TW", "TWO")
    all_tickers = dfcsv["download_path"].tolist()
    chunk_size = 50  # 每次下載 50 檔

    for i in range(0, len(all_tickers), chunk_size):
        batch = all_tickers[i : i + chunk_size]

        df = yf.download(
            batch,
            period="10d",
            auto_adjust=True,
            repair=True,
            interval="1d",
            group_by="ticker",
            threads=True,
        )

        save(batch, df)

        print(f"已完成第 {i+chunk_size} 檔...")

        time.sleep(2)


def save(stocks, dfall):
    for path in stocks:
        df = dfall[path]
        stock = path.split(".")[0]
        save_and_merge(stock, df)


# download_day()
