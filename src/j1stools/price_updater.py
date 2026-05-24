from colorsys import TWO_THIRD
from enum import Enum
from math import e
from os import path
import time
from types import SimpleNamespace
import pyarrow as pa
from patsy.mgcv_cubic_splines import te
from pytest import mark
from regex import P
import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from j1stools import parquet_db
from j1stools.utils import get_base_url, stock

MY_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("stock_id", pa.string()),
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("volume", pa.int64()),
    ]
)


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
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")
    columns = ["date", "stock_id", "open", "high", "low", "close", "volume"]
    new: pd.DataFrame = long_df[columns]

    # 轉型
    for col in ["open", "high", "low", "close"]:
        new[col] = new[col].astype("float32")

    # 5. 分月存檔
    os.makedirs("db/price", exist_ok=True)
    for month, group in new.groupby(new["date"].str[:7].str.replace("-", "_")):
        path = f"db/price/{month}.parquet"

        if os.path.exists(path):
            old = pd.read_parquet(path=path, engine="pyarrow")
        else:
            old = pd.DataFrame(columns=columns)

        write = (
            pd.concat([old, group], axis=0)
            .sort_values(by=["date", "stock_id"], ascending=[True, True])
            .drop_duplicates(subset=["date", "stock_id"], keep="last")
        )
        write[["open", "high", "low", "close"]] = write[["open", "high", "low", "close"]].astype("float32")
        write.to_parquet(path, engine="pyarrow", schema=MY_SCHEMA, compression="zstd")


def find_tickers_to_download(select_tickers: list = None):
    """
    回傳 dict
    dict.stock_id
    dict.market_type
    """

    if select_tickers is None:
        stocks = parquet_db.activate_stocks()
    else:
        stocks = select_tickers

    info_df = parquet_db.query_stock_info()
    info_df = info_df[info_df["stock_id"].isin(stocks)][["stock_id", "market_type"]]

    all_tickers = []
    for _, row in info_df.iterrows():
        stock_id = row["stock_id"]
        market_type = "TW" if row["market_type"] == "twse" else "TWO"
        ticker = SimpleNamespace(stock_id=stock_id, market_type=market_type)
        all_tickers.append(ticker)

    return all_tickers


def update(
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


if __name__ == "__main__":
    # price_updater(INTERVAL.day)
    update(INTERVAL.day, period="10d")
