import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

REPO_ID = "raywu918python/j1s-data"
DEFAULT_MODEL = "lgbm_timeseries_ensemble"
_API_KEY = os.environ.get("API_KEY", "")
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_key_header = APIKeyHeader(name="X-API-Key")


def _verify_key(key: str = Security(_key_header)):
    if not _API_KEY or key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _pred_url(date_str: str, model: str) -> str:
    return (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
        f"/db/predictions/{model}/pred_{date_str}.parquet"
    )


def _backtest_url(date_str: str, model: str, kind: str) -> str:
    return (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
        f"/db/backtest/{model}/{kind}_{date_str}.parquet"
    )


def _read_parquet(url: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    import io, httpx
    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return _read_parquet(io.BytesIO(r.content))


def _load_latest_pred(model: str):
    for delta in range(7):
        d = (date.today() - timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            return _read_parquet(_pred_url(d, model)), d
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="No prediction found in last 7 days")


def _load_latest_backtest(model: str):
    for delta in range(7):
        d = (date.today() - timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            equity = _read_parquet(_backtest_url(d, model, "equity"))
            trades = _read_parquet(_backtest_url(d, model, "trades"))
            try:
                open_pos = _read_parquet(_backtest_url(d, model, "open"))
            except Exception:
                open_pos = pd.DataFrame()
            try:
                market = _read_parquet(_backtest_url(d, model, "market"))
            except Exception:
                market = pd.DataFrame()
            return equity, trades, open_pos, market, d
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="No backtest found in last 7 days")


@app.get("/predictions")
def get_predictions(model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    df, d = _load_latest_pred(model)
    return {"date": d, "model": model, "data": df.to_dict(orient="records")}


@app.get("/predictions/{date_str}")
def get_predictions_by_date(date_str: str, model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    try:
        df = _read_parquet(_pred_url(date_str, model))
    except Exception:
        raise HTTPException(status_code=404, detail=f"No prediction for {date_str} model={model}")
    return {"date": date_str, "model": model, "data": df.to_dict(orient="records")}


@app.get("/backtest")
def get_backtest(model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    equity, trades, open_pos, market, d = _load_latest_backtest(model)
    return {
        "date": d,
        "model": model,
        "equity": equity.to_dict(orient="records"),
        "trades": trades.to_dict(orient="records"),
        "open": open_pos.to_dict(orient="records"),
        "market": market.to_dict(orient="records"),
    }


@app.get("/backtest/{date_str}")
def get_backtest_by_date(date_str: str, model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    try:
        equity = _read_parquet(_backtest_url(date_str, model, "equity"))
        trades = _read_parquet(_backtest_url(date_str, model, "trades"))
        try:
            open_pos = _read_parquet(_backtest_url(date_str, model, "open"))
        except Exception:
            open_pos = pd.DataFrame()
        try:
            market = _read_parquet(_backtest_url(date_str, model, "market"))
        except Exception:
            market = pd.DataFrame()
    except Exception:
        raise HTTPException(status_code=404, detail=f"No backtest for {date_str} model={model}")
    return {
        "date": date_str,
        "model": model,
        "equity": equity.to_dict(orient="records"),
        "trades": trades.to_dict(orient="records"),
        "open": open_pos.to_dict(orient="records"),
        "market": market.to_dict(orient="records"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
