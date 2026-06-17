from __future__ import annotations

import argparse
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from j1stools import parquet_db

_TW = timezone(timedelta(hours=8))

_API_URL = "https://mops.twse.com.tw/mops/api/t05st02"
_DEFAULT_START = "110-01-01"
_DEFAULT_UPDATE_LOOKBACK_DAYS = 5
_OUTPUT_DIR = "db/news_mops"
_FLAG_PATH = "db/news_mops_flags/news_mops_flag.parquet"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}

_COLUMNS = [
    "date",
    "roc_date",
    "time",
    "stock_id",
    "company_name",
    "title",
    "mops_id",
    "company_id",
    "market_kind",
    "enter_date",
    "serial_number",
    "api_name",
    "source_query_date",
    "source_query_roc_date",
]


def _today_tw() -> date:
    return datetime.now(_TW).date()


def _parse_date(value: str | date | datetime) -> date:
    """Parse ROC dates like 110-01-01 and AD dates like 2021-01-01."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    raw = str(value).strip()
    if not raw:
        raise ValueError("date value is empty")

    normalized = raw.replace("/", "-")
    if "-" in normalized:
        parts = normalized.split("-")
    else:
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if len(digits) not in (7, 8):
            raise ValueError(f"unsupported date format: {value!r}")
        parts = [digits[:-4], digits[-4:-2], digits[-2:]]

    if len(parts) != 3:
        raise ValueError(f"unsupported date format: {value!r}")

    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    if year < 1911:
        year += 1911
    return date(year, month, day)


def _to_roc_slash(day: date) -> str:
    return f"{day.year - 1911:03d}/{day.month:02d}/{day.day:02d}"


def _to_roc_compact(day: date) -> str:
    return f"{day.year - 1911:03d}{day.month:02d}{day.day:02d}"


def _to_roc_payload(day: date) -> dict[str, str]:
    return {
        "year": str(day.year - 1911),
        "month": str(day.month),
        "day": f"{day.day:02d}",
    }


def _roc_to_ad_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("-", "/")
    parts = text.split("/")
    if len(parts) != 3:
        return None
    try:
        year = int(parts[0]) + 1911
        month = int(parts[1])
        day = int(parts[2])
        return date(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


_PUA_CONTROL_RE = re.compile("[\u0000-\u001f\u007f-\u009f\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")


def _clean_title(value: Any) -> str:
    if value is None:
        return ""
    text = "".join(str(value).split())
    return _PUA_CONTROL_RE.sub("", text)


def _request_day(
    session: requests.Session,
    query_date: date,
    retries: int = 3,
    timeout: int = 20,
) -> dict[str, Any]:
    payload = _to_roc_payload(query_date)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.post(_API_URL, json=payload, headers=_HEADERS, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 200:
                if data.get("message") == "查無相符資料":
                    return {"code": data.get("code"), "message": data.get("message"), "result": {"data": []}}
                raise RuntimeError(f"MOPS API error: {data.get('message', data)}")
            return data
        except Exception as exc:  # requests/json/runtime errors should retry the same way.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"{_to_roc_slash(query_date)} request failed: {last_error}") from last_error


def _normalize_rows(rows: list[list[Any]], query_date: date) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    query_date_str = query_date.strftime("%Y-%m-%d")
    query_roc_date = _to_roc_slash(query_date)

    for row in rows:
        if len(row) < 6:
            continue

        roc_date, announce_time, stock_id, company_name, title, detail = row[:6]
        params = detail.get("parameters", {}) if isinstance(detail, dict) else {}
        api_name = detail.get("apiName", "") if isinstance(detail, dict) else ""

        company_id = str(params.get("companyId") or stock_id or "").strip()
        market_kind = str(params.get("marketKind") or "").strip()
        enter_date = str(params.get("enterDate") or "").strip()
        serial_number = params.get("serialNumber")
        mops_id = f"{company_id}_{market_kind}_{enter_date}_{serial_number or ''}"

        records.append(
            {
                "date": _roc_to_ad_string(roc_date),
                "roc_date": _clean_text(roc_date),
                "time": _clean_text(announce_time),
                "stock_id": str(stock_id or "").strip(),
                "company_name": _clean_text(company_name),
                "title": _clean_title(title),
                "mops_id": mops_id,
                "company_id": company_id,
                "market_kind": market_kind,
                "enter_date": enter_date,
                "serial_number": serial_number,
                "api_name": api_name,
                "source_query_date": query_date_str,
                "source_query_roc_date": query_roc_date,
            }
        )

    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    if df.empty:
        return df

    df["serial_number"] = pd.to_numeric(df["serial_number"], errors="coerce").astype("Int64")
    return df


def fetch_mops_news_day(query_date: str | date | datetime, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch one MOPS query date and return normalized announcement rows."""
    day = _parse_date(query_date)
    close_session = session is None
    session = session or requests.Session()
    try:
        data = _request_day(session, day)
        rows = data.get("result", {}).get("data", []) or []
        return _normalize_rows(rows, day)
    finally:
        if close_session:
            session.close()


def _default_output_path(start: date, end: date) -> str:
    return os.path.join(_OUTPUT_DIR, f"{_to_roc_compact(start)}_{_to_roc_compact(end)}.parquet")


def _latest_output_path() -> str | None:
    if not os.path.isdir(_OUTPUT_DIR):
        return None

    paths = [
        os.path.join(_OUTPUT_DIR, entry.name)
        for entry in os.scandir(_OUTPUT_DIR)
        if entry.is_file() and entry.name.endswith(".parquet")
    ]
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _filter_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df.empty:
        return df
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    return df[df["date"].fillna("").between(start_str, end_str)].copy()


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)

    df = df.copy()
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[_COLUMNS]
    df["serial_number"] = pd.to_numeric(df["serial_number"], errors="coerce").astype("Int64")
    df.drop_duplicates(subset=["mops_id"], keep="last", inplace=True)
    df.sort_values(["date", "time", "stock_id", "serial_number"], inplace=True, na_position="last")
    df.reset_index(drop=True, inplace=True)
    return df


def _save_news(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        old = pd.read_parquet(output_path)
        df = pd.concat([old, df], ignore_index=True)

    write = _finalize(df)
    write.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")


def _get_done_query_dates(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    if not os.path.exists(_FLAG_PATH):
        return set()
    df = pd.read_parquet(_FLAG_PATH)
    if "query_date" not in df.columns or "output_path" not in df.columns:
        return set()
    df = df[df["output_path"].astype(str) == output_path]
    return set(df["query_date"].astype(str).tolist())


def _update_flag(query_date: date, rows: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(_FLAG_PATH), exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "query_date": query_date.strftime("%Y-%m-%d"),
                "query_roc_date": _to_roc_slash(query_date),
                "rows": rows,
                "output_path": output_path,
                "updated_at": datetime.now(_TW).isoformat(timespec="seconds"),
            }
        ]
    )

    if os.path.exists(_FLAG_PATH):
        df = pd.read_parquet(_FLAG_PATH)
        df = pd.concat([df, row], ignore_index=True)
        df.drop_duplicates(subset=["output_path", "query_date"], keep="last", inplace=True)
    else:
        df = row

    df.sort_values(["output_path", "query_date"], inplace=True)
    df.to_parquet(_FLAG_PATH, index=False, engine="pyarrow", compression="zstd")


def download_mops_news(
    start: str | date | datetime = _DEFAULT_START,
    end: str | date | datetime | None = None,
    output_path: str | None = None,
    batch_days: int = 30,
    sleep_seconds: float = 0.35,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download MOPS announcements and save them as parquet.

    Dates may be ROC (110-01-01) or AD (2021-01-01). The default range is
    110-01-01 through Taiwan-time today.
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end or _today_tw())
    if start_date > end_date:
        raise ValueError("start must be earlier than or equal to end")

    output_path = os.path.normpath(output_path or _default_output_path(start_date, end_date))

    today = _today_tw()
    query_end = end_date + timedelta(days=1)
    if query_end > today:
        query_end = today

    query_dates = [ts.date() for ts in pd.date_range(start=start_date, end=query_end, freq="D")]
    last_query_date = query_dates[-1] if query_dates else None
    done_dates = set() if force else _get_done_query_dates(output_path)
    failures: list[str] = []
    batch: list[pd.DataFrame] = []

    with requests.Session() as session:
        for idx, query_date in enumerate(query_dates, start=1):
            query_date_str = query_date.strftime("%Y-%m-%d")
            if query_date_str in done_dates and query_date != last_query_date:
                continue

            try:
                df = fetch_mops_news_day(query_date, session=session)
            except Exception as exc:
                failures.append(query_date_str)
                print(f"{query_date_str} 下載失敗: {exc}")
                continue

            rows = len(df)
            df = _filter_date_range(df, start_date, end_date)
            if not df.empty:
                batch.append(df)

            _update_flag(query_date, rows, output_path)
            print(f"{idx}/{len(query_dates)} {query_date_str} 完成，API {rows} 筆，保留 {len(df)} 筆")

            if len(batch) >= batch_days:
                _save_news(pd.concat(batch, ignore_index=True), output_path)
                batch = []
                print(f"  進度儲存: {output_path}")

            time.sleep(sleep_seconds)

    if batch:
        _save_news(pd.concat(batch, ignore_index=True), output_path)

    if os.path.exists(output_path):
        result = pd.read_parquet(output_path)
    else:
        result = pd.DataFrame(columns=_COLUMNS)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")

    if failures:
        print("以下 query date 下載失敗，可重新執行補抓: " + ", ".join(failures))

    print(f"全部完成: {output_path}，共 {len(result)} 筆")
    return result


def update_mops_news(
    target_date: str | date | datetime | None = None,
    lookback_days: int = _DEFAULT_UPDATE_LOOKBACK_DAYS,
    output_path: str | None = None,
    sleep_seconds: float = 0.35,
) -> pd.DataFrame:
    """Fetch target_date through the previous lookback_days and append/deduplicate parquet."""
    if lookback_days < 0:
        raise ValueError("lookback_days must be greater than or equal to 0")

    end_date = _parse_date(target_date or _today_tw())
    start_date = end_date - timedelta(days=lookback_days)
    output_path = os.path.normpath(output_path or _latest_output_path() or _default_output_path(start_date, end_date))

    print("更新 MOPS: " f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} " f"存入 {output_path}")

    return download_mops_news(
        start=start_date,
        end=end_date,
        output_path=output_path,
        batch_days=1,
        sleep_seconds=sleep_seconds,
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MOPS announcements to parquet.")
    parser.add_argument("--start", default=_DEFAULT_START, help="ROC or AD start date, default: 110-01-01")
    parser.add_argument("--end", default=None, help="ROC or AD end date, default: Taiwan-time today")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--batch-days", type=int, default=30, help="Save progress after this many fetched dates")
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds to sleep between API requests")
    parser.add_argument("--force", action="store_true", help="Ignore done flags and refetch all query dates")
    parser.add_argument("--update", action="store_true", help="Fetch --date through the previous --lookback-days")
    parser.add_argument("--date", default=None, help="ROC or AD update date, default: Taiwan-time today")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=_DEFAULT_UPDATE_LOOKBACK_DAYS,
        help="Days before --date to include in --update mode, default: 5",
    )
    args = parser.parse_args()

    if args.update:
        update_mops_news(
            target_date=args.date,
            lookback_days=args.lookback_days,
            output_path=args.output,
            sleep_seconds=args.sleep,
        )
        return

    download_mops_news(
        start=args.start,
        end=args.end,
        output_path=args.output,
        batch_days=args.batch_days,
        sleep_seconds=args.sleep,
        force=args.force,
    )


if __name__ == "__main__":
    # main()
    download_mops_news()
