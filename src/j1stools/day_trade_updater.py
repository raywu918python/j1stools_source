"""
當沖資料更新

歷史資料 (2015-01-01 ~ 2026-05-31) → db/day_trade/201501_202605.parquet
月份資料 (2026-06+)               → db/day_trade/YYYY_M.parquet

每次呼叫都以「日期」為單位，TWSE + TPEX 各抓一次（全部股票），合併後存檔。
"""

from datetime import datetime, timezone, timedelta
import time
import os
import warnings

import pandas as pd
import requests

from j1stools import parquet_db

_TW = timezone(timedelta(hours=8))
_FLAG_PATH = "db/day_trade_flags/day_trade_flag.parquet"
_HISTORICAL_PATH = "db/day_trade/201501_202605.parquet"

_TWSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
_TPEX_API = "https://www.tpex.org.tw/www/zh-tw/intraday/stat"


# ── fetch ─────────────────────────────────────────────────────────────────────


def _fetch_twse(date_yyyymmdd: str) -> pd.DataFrame:
    """TWSE TWTB4U：一次取得當日所有上市股票的當沖資料"""
    url = "https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U"
    params = {"date": date_yyyymmdd, "response": "json"}
    r = requests.get(url, params=params, headers=_TWSE_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    for table in data.get("tables", []):
        fields = table.get("fields", [])
        rows = table.get("data", [])
        if "證券代號" in fields and rows:
            return pd.DataFrame(rows, columns=fields)
    return pd.DataFrame()


def _fetch_tpex(date_slash: str) -> pd.DataFrame:
    """TPEX intraday/stat：一次取得當日所有上櫃股票的當沖資料"""
    payload = {"type": "Daily", "date": date_slash, "id": "", "response": "json"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = requests.post(_TPEX_API, data=payload, verify=False, timeout=20)
    r.raise_for_status()
    data = r.json()
    for table in data.get("tables", []):
        fields = table.get("fields", [])
        rows = table.get("data", [])
        if "證券代號" in fields and rows:
            return pd.DataFrame(rows, columns=fields)
    return pd.DataFrame()


def _normalize(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """把 TWSE / TPEX 的欄位統一成 stock_id, date, volume, buy_amount, sell_amount"""
    rename = {
        "證券代號": "stock_id",
        "當日沖銷交易成交股數": "volume",
        "當日沖銷交易買進成交金額": "buy_amount",
        "當日沖銷交易賣出成交金額": "sell_amount",
    }
    df = df.rename(columns=rename)
    cols = [c for c in ["stock_id", "volume", "buy_amount", "sell_amount"] if c in df.columns]
    df = df[cols].copy()
    for col in ["volume", "buy_amount", "sell_amount"]:
        df[col] = df[col].str.replace(",", "", regex=False).astype("int64")
    df["stock_id"] = df["stock_id"].str.strip()
    df["date"] = date_str
    return df


def _fetch_day(date: datetime) -> pd.DataFrame:
    """同時抓 TWSE + TPEX，合併回傳；非交易日回傳空 DataFrame"""
    date_str = date.strftime("%Y-%m-%d")
    parts = []

    try:
        twse_df = _fetch_twse(date.strftime("%Y%m%d"))
        if not twse_df.empty:
            parts.append(_normalize(twse_df, date_str))
    except Exception as e:
        print(f"  TWSE {date_str} 失敗: {e}")

    try:
        tpex_df = _fetch_tpex(date.strftime("%Y/%m/%d"))
        if not tpex_df.empty:
            parts.append(_normalize(tpex_df, date_str))
    except Exception as e:
        print(f"  TPEX {date_str} 失敗: {e}")

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── storage ───────────────────────────────────────────────────────────────────


def _save_day_trade(df: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        old = pd.read_parquet(file_path)
        df = pd.concat([old, df], ignore_index=True)
    df.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df.sort_values(["date", "stock_id"], inplace=True)
    df.to_parquet(file_path, index=False)


# ── flag ──────────────────────────────────────────────────────────────────────


def _get_done_dates() -> set:
    if os.path.exists(_FLAG_PATH):
        return set(pd.read_parquet(_FLAG_PATH)["date"].tolist())
    return set()


def _update_flag(date_str: str):
    os.makedirs(os.path.dirname(_FLAG_PATH), exist_ok=True)
    new_row = pd.DataFrame([{"date": date_str}])
    if os.path.exists(_FLAG_PATH):
        df = pd.read_parquet(_FLAG_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
        df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    else:
        df = new_row
    df.to_parquet(_FLAG_PATH, index=False)


# ── public API ────────────────────────────────────────────────────────────────


def _day_trade_file_path(year: int, month: int) -> str:
    if year < 2026 or (year == 2026 and month < 6):
        return _HISTORICAL_PATH
    return f"db/day_trade/{year}_{month}.parquet"


def init_day_trade(start: str = "2015-01-01", end: str = "2026-05-31"):
    """歷史當沖資料初始化，下載 start～end 並存到 db/day_trade/201501_202605.parquet"""
    os.makedirs("db/day_trade", exist_ok=True)
    done = _get_done_dates()
    date_range = pd.date_range(start=start, end=end, freq="B")  # 工作日近似
    batch: list[pd.DataFrame] = []

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in done:
            continue

        df = _fetch_day(date)
        if not df.empty:
            batch.append(df)

        _update_flag(date_str)

        if len(batch) >= 30:
            _save_day_trade(pd.concat(batch, ignore_index=True), _HISTORICAL_PATH)
            batch = []
            print(f"  進度儲存至 {date_str}")

        time.sleep(0.5)

    if batch:
        _save_day_trade(pd.concat(batch, ignore_index=True), _HISTORICAL_PATH)

    if os.path.exists(_HISTORICAL_PATH):
        print(f"完成: {_HISTORICAL_PATH}，共 {len(pd.read_parquet(_HISTORICAL_PATH))} 筆")
    else:
        print("完成: 無資料寫入（所有日期可能皆為非交易日）")


def update_day_trade():
    """更新當月當沖資料，存到 db/day_trade/YYYY_M.parquet"""
    now = datetime.now(_TW)
    file_path = _day_trade_file_path(now.year, now.month)
    os.makedirs("db/day_trade", exist_ok=True)
    done = _get_done_dates()

    month_start = f"{now.year}-{now.month:02d}-01"
    date_range = pd.date_range(start=month_start, end=now.strftime("%Y-%m-%d"), freq="B")

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in done:
            continue

        df = _fetch_day(date)
        if not df.empty:
            _save_day_trade(df, file_path)
        _update_flag(date_str)
        print(f"{date_str} 完成，共 {len(df)} 筆")
        time.sleep(0.5)

    if os.path.exists(file_path):
        print(f"全部完成: {file_path}，共 {len(pd.read_parquet(file_path))} 筆")
    else:
        print(f"全部完成: {file_path}，今日尚無資料")


if __name__ == "__main__":
    # init_day_trade()
    update_day_trade()
    # df = parquet_db.query_day_trade(["2330"])
    # print(df.tail())
