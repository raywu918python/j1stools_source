from colorsys import TWO_THIRD
from enum import Enum
from math import e
from os import path
import time
from types import SimpleNamespace

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


def save(dfall):
    long_df = dfall.stack(level=0).reset_index()
    long_df.rename(
        columns={
            "Ticker": "stock_id",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    long_df["stock_id"] = long_df["stock_id"].str.split(".").str[0]
    columns = ["date", "stock_id", "open", "high", "low", "close", "volume"]
    new: pd.DataFrame = long_df[columns]

    # 5. 分月存檔
    for month, group in new.groupby(new["date"].dt.to_period("M")):
        path = f"{month.strftime('%Y_%m')}.parquet"

        if os.path.exists(path):
            old = pd.read_parquet(path=path, engine="pyarrow")
        else:
            old = pd.DataFrame(columns=columns)

        write = (
            pd.concat([old, group], axis=0)
            .sort_values(by=["date", "stock_id"], ascending=[True, True])
            .drop_duplicates(subset=["date", "stock_id"], keep="last")
        )
        write.to_parquet(path, engine="pyarrow")


def find_tickers_to_download(select_tickers: list = None):
    """
    回傳 dict
    dict.stock_id
    dict.market_type
    """

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
        ticker = SimpleNamespace({"stock_id": stock_id, "market_type": market_type})
        all_tickers.append(ticker)

    return all_tickers


def download(
    interval: INTERVAL,
    select_tickers: list = None,
    period="5d",
):

    interval_value = interval.value

    all_tickers = find_tickers_to_download(select_tickers)
    chunk_size = 50  # 每次下載 50 檔

    for i in range(0, len(all_tickers), chunk_size):
        batch = [f"{x.stock_id}.{x.market_type}" for x in all_tickers[i : i + chunk_size]]

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

        save(df)

        print(f"已完成第 {i+chunk_size} 檔...")

        time.sleep(2)


def update_price(start_date: str, end_date: str = None):
    """
    通用型股票資料整合函式 (支援單月更新與歷史區間合併)

    用法 1: build_stock_parquet("2015-01", "2026-05") -> 產出 201501_202605.parquet
    用法 2: build_stock_parquet("2026-05")            -> 產出 2026_05.parquet
    """
    # 1. 解析日期邊界 (使用 Pandas Timestamp 處理跨月/跨年邏輯最無腦且精準)
    st_dt = pd.to_datetime(f"{start_date}-01")

    if end_date:
        # 用法 1: 取到結束月份的下個月 1 號 (左閉右開)
        end_dt = pd.to_datetime(f"{end_date}-01") + pd.DateOffset(months=1)
        file_name = (
            f"{start_date}.parquet"
            if start_date == end_date
            else f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        )
    else:
        # 用法 2: 沒傳 end_date，代表只做單月。結束時間就是下個月 1 號
        end_dt = st_dt + pd.DateOffset(months=1)
        file_name = f"{st_dt.year}_{st_dt.month:02d}.parquet"

    file_path = f"db/price/{file_name}"
    print(f"開始處理區間: {st_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')} -> 目標檔案: {file_path}")

    # 2. 抓取今日/新撈到的資料
    stocks = utils.get_file_list()
    new_tables = []

    for stock_id in stocks:
        try:
            df = utils.stock(stock_id)
            if df.empty:
                continue

            # 轉換為 datetime 以利精準比較
            df["date"] = pd.to_datetime(df["date"])

            # 過濾時間區間 [start, end)
            df = df[(df["date"] >= st_dt) & (df["date"] < end_dt)]

            if not df.empty:
                # 轉 PyArrow 前先去除該股票內部的重複值
                df = df.drop_duplicates(subset=["date"])

                # 確保欄位順序與 Schema 一致，並剔除可能多出來的 index 欄位
                df = df[MY_SCHEMA.names]

                new_tables.append(pa.Table.from_pandas(df, schema=MY_SCHEMA))
        except Exception as e:
            print(f"stock_id {stock_id} 處理失敗: {e}")

    if not new_tables:
        print("❌ 該區間內找不到任何新資料")
        return

    # 合併新撈出來的所有股票資料
    combined_table = pa.concat_tables(new_tables)

    # 3. 如果是「每日更新單月」的模式，需要讀取舊檔進行增量合併
    if end_date is None and os.path.exists(file_path):
        try:
            existing_table = pq.read_table(file_path).cast(MY_SCHEMA)
            combined_table = pa.concat_tables([existing_table, combined_table])
            print("讀取既有月檔成功，進行增量合併...")
        except Exception as e:
            print(f"讀取舊檔失敗 (可能 Schema 衝突)，將直接覆蓋。錯誤: {e}")

    # 4. 全局去重 (轉回 Pandas 做最保險，避免跨股票或增量更新造成的重複)
    df_all = combined_table.to_pandas()
    df_all.drop_duplicates(subset=["date", "stock_id"], keep="last", inplace=True)

    # 5. 重新包回 Table 並強制套用 Schema 與排序
    final_table = pa.Table.from_pandas(df_all, schema=MY_SCHEMA)
    indices = pc.sort_indices(final_table, sort_keys=[("date", "ascending"), ("stock_id", "ascending")])
    final_table = final_table.take(indices)

    # 6. 寫入 Parquet
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pq.write_table(final_table, file_path, compression="snappy")
    print(f"🎉 處理完成！總筆數: {len(final_table)}\n")


download(INTERVAL.day, ["6546", "2330", "0050"], period="35d")
# download(INTERVAL.m1)
# download(INTERVAL.m5)

# download(INTERVAL.day, ["6546"], period="3000d")
# download(INTERVAL.m1, ["1609"], period="60d")
# download(INTERVAL.m5, ["1609"], period="60d")
