from datetime import date, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

REPO_ID = "raywu918python/j1s-data"
MODEL_NAME = "lgbm_timeseries_ensemble"


def _parquet_url(date_str: str) -> str:
    return (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
        f"/db/predictions/{MODEL_NAME}/pred_{date_str}.parquet"
    )


def _load_latest() -> pd.DataFrame:
    for delta in range(7):
        d = (date.today() - timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            return pd.read_parquet(_parquet_url(d)), d
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="No prediction found in last 7 days")


@app.get("/predictions")
def get_predictions():
    df, d = _load_latest()
    return {"date": d, "model": MODEL_NAME, "data": df.to_dict(orient="records")}


@app.get("/predictions/{date_str}")
def get_predictions_by_date(date_str: str):
    try:
        df = pd.read_parquet(_parquet_url(date_str))
    except Exception:
        raise HTTPException(status_code=404, detail=f"No prediction for {date_str}")
    return {"date": date_str, "model": MODEL_NAME, "data": df.to_dict(orient="records")}


@app.get("/health")
def health():
    return {"status": "ok"}
