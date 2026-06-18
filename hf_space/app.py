import json
import os
from datetime import datetime, timedelta, timezone

_TW = timezone(timedelta(hours=8))
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from ai_chat import (
    get_provider,
    list_topics,
    load_topic,
    run_autonomous_roundtable,
    run_qa,
    run_roundtable_for_topic,
    save_topic,
    set_provider,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "raywu918python/j1s-data"
DEFAULT_MODEL = "margin_lgbm"
_API_KEY = os.environ.get("API_KEY", "")
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_GITHUB_REPO = "raywu918python/j1stools_source"
_key_header = APIKeyHeader(name="X-API-Key")


def _verify_key(key: str = Security(_key_header)):
    if not _API_KEY or key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _rag_url(date_str: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/db/rag/daily_picks_{date_str}.json"


def _pred_url(date_str: str, model: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main" f"/db/predictions/{model}/pred_{date_str}.parquet"


def _backtest_url(date_str: str, model: str, kind: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main" f"/db/backtest/{model}/{kind}_{date_str}.parquet"


def _to_records(df: pd.DataFrame) -> list:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _read_parquet(url: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    import io, httpx

    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _load_latest_pred(model: str):
    for delta in range(7):
        d = (datetime.now(_TW).date() - timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            return _read_parquet(_pred_url(d, model)), d
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="No prediction found in last 7 days")


def _load_latest_backtest(model: str):
    for delta in range(7):
        d = (datetime.now(_TW).date() - timedelta(days=delta)).strftime("%Y-%m-%d")
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


def _read_rag(date_str: str) -> dict:
    import json as _json, httpx
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    r = httpx.get(_rag_url(date_str), headers=headers, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return _json.loads(r.content)


@app.get("/rag")
def get_rag_picks(_=Depends(_verify_key)):
    """最新一天的 RAG 選股結果（往前找最近 7 天）。"""
    for delta in range(7):
        d = (datetime.now(_TW).date() - timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            return _read_rag(d)
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="No RAG picks found in last 7 days")


@app.get("/rag/{date_str}")
def get_rag_picks_by_date(date_str: str, _=Depends(_verify_key)):
    """指定日期的 RAG 選股結果。"""
    try:
        return _read_rag(date_str)
    except Exception:
        raise HTTPException(status_code=404, detail=f"No RAG picks for {date_str}")


@app.get("/predictions")
def get_predictions(model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    df, d = _load_latest_pred(model)
    return {"date": d, "model": model, "data": _to_records(df)}


@app.get("/predictions/{date_str}")
def get_predictions_by_date(date_str: str, model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    try:
        df = _read_parquet(_pred_url(date_str, model))
    except Exception:
        raise HTTPException(status_code=404, detail=f"No prediction for {date_str} model={model}")
    return {"date": date_str, "model": model, "data": _to_records(df)}


@app.get("/backtest")
def get_backtest(model: Optional[str] = Query(default=DEFAULT_MODEL), _=Depends(_verify_key)):
    equity, trades, open_pos, market, d = _load_latest_backtest(model)
    return {
        "date": d,
        "model": model,
        "equity": _to_records(equity),
        "trades": _to_records(trades),
        "open": _to_records(open_pos),
        "market": _to_records(market),
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
        "equity": _to_records(equity),
        "trades": _to_records(trades),
        "open": _to_records(open_pos),
        "market": _to_records(market),
    }


def _trigger_workflow(workflow_file: str) -> bool:
    import httpx

    url = f"https://api.github.com/repos/{_GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches"
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        json={"ref": "main"},
        timeout=10,
    )
    return r.status_code == 204


@app.get("/trigger/run-rag")
def trigger_run_rag(token: str = Query(default="")):
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    ok = _trigger_workflow("run_rag.yml")
    return {"ok": ok}


@app.get("/trigger/update-price")
def trigger_update_price(token: str = Query(default="")):
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    ok = _trigger_workflow("update_price.yml")
    return {"ok": ok}


@app.get("/trigger/margin-lgbm")
def trigger_margin_lgbm(token: str = Query(default="")):
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    ok = _trigger_workflow("margin_lgbm.yml")
    return {"ok": ok}


@app.get("/trigger/rfc-macd-6xx")
def trigger_rfc_macd_6xx(token: str = Query(default="")):
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    ok = _trigger_workflow("rfc_macd.yml")
    return {"ok": ok}


@app.get("/trigger/predict-all")
def trigger_predict_all(token: str = Query(default="")):
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    results = {name: _trigger_workflow(yml) for name, yml in _MODELS.items()}
    return {"ok": all(results.values()), "results": results}


_MODELS = {
    "rfc_macd_6xx": "rfc_macd.yml",
    "rfc_macd_ib_6xx": "rfc_ib_macd.yml",
    "margin_lgbm": "margin_lgbm.yml",
}


@app.get("/models")
def list_models():
    return {"models": list(_MODELS.keys())}


# ─── 議題聊天室 REST API ──────────────────────────────────────────────────────


@app.get("/topics")
async def get_topics(date: str = Query(default="")):
    """列出指定日期所有議題摘要（不帶 date 則回傳今日）。

    回傳範例：
    {
      "date": "2026-06-18",
      "topics": [
        {"topic_id": "rfc_macd_ib", "title": "RFC MACD IB 持倉分析", "summary": "...",
         "created_at": "2026-06-18 09:00", "source": "auto", "qa_count": 3},
        ...
      ]
    }
    """
    import asyncio as _aio
    date_str = date or datetime.now(_TW).strftime("%Y-%m-%d")
    topics = await _aio.to_thread(list_topics, date_str)
    return {"date": date_str, "topics": topics}


@app.get("/topics/{topic_id}")
async def get_topic(topic_id: str, date: str = Query(default="")):
    """取得指定議題的完整資料（discussion + qa）。"""
    import asyncio as _aio
    date_str = date or datetime.now(_TW).strftime("%Y-%m-%d")
    topic = await _aio.to_thread(load_topic, topic_id, date_str)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"議題 {topic_id} 不存在")
    return topic


class AskBody(BaseModel):
    nickname: str = "匿名"
    message: str


@app.post("/topics/{topic_id}/ask")
async def ask_topic(topic_id: str, body: AskBody, background_tasks: BackgroundTasks, date: str = Query(default="")):
    """對現有議題追問。LLM 答完立即回傳，HF 存檔在背景執行。

    回傳：{"question": {...}, "answer": {...}}
    """
    text = body.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message 不可為空")
    date_str = date or datetime.now(_TW).strftime("%Y-%m-%d")
    try:
        qa_entry, updated_topic = await run_qa(topic_id, text, date_str, body.nickname)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    background_tasks.add_task(save_topic, updated_topic, date_str)
    return qa_entry


class NewTopicBody(BaseModel):
    title: str
    question: str
    nickname: str = "匿名"


@app.post("/topics")
async def create_topic(body: NewTopicBody, background_tasks: BackgroundTasks):
    """用戶自開新議題，觸發完整圓桌（背景執行）。

    立即回傳 topic_id，前端輪詢 GET /topics/{topic_id} 查看結果。
    """
    title = body.title.strip()
    question = body.question.strip()
    if not title or not question:
        raise HTTPException(status_code=400, detail="title 和 question 不可為空")

    import uuid as _uuid
    topic_id = f"user_{_uuid.uuid4().hex[:8]}"
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")

    background_tasks.add_task(
        run_roundtable_for_topic,
        topic_id=topic_id,
        title=title,
        question=question,
        date_str=date_str,
        source="user",
    )
    return {
        "topic_id": topic_id,
        "title": title,
        "status": "processing",
        "message": "圓桌會議已啟動，請稍後用 GET /topics/{topic_id} 查看結果",
    }


@app.get("/chat/model")
def get_current_model():
    """查看目前使用的 LLM。"""
    return {"provider": get_provider()}


@app.post("/chat/model")
def switch_model(provider: str = Query(..., description="gemini 或 qwen")):
    """切換 LLM（gemini / qwen）。前端直接呼叫，不需 token。重啟後還原預設值。"""
    try:
        set_provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"provider": provider, "message": f"已切換到 {provider}"}


@app.post("/trigger/roundtable")
async def trigger_roundtable(background_tasks: BackgroundTasks, token: str = Query(default="")):
    """觸發每日自動圓桌會議。接 GitHub Action 每日收盤後呼叫：
      curl -X POST "https://your-space.hf.space/trigger/roundtable?token=YOUR_API_KEY"
    """
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    background_tasks.add_task(run_autonomous_roundtable)
    return {"ok": True, "message": "AI 圓桌會議已在背景啟動"}


@app.get("/health")
def health():
    return {"status": "ok"}
