"""AI 聊天室後端 — 本機開發版

HF Space 版本在 hf_space/ai_chat.py，使用 HTTP 存取 HF Dataset。
本機版直接 import rag.py 的函式，適合開發測試。

API（由 hf_space/app.py 定義）：
  POST /chat/message  → 存訊息 + AI 回應 → 存 HF
  GET  /chat/history  → 讀 HF 當日聊天記錄

結論：Gemini 2.5 Flash，夠用就好，之後資源不足再換 HF 模型。

需要部署的時候，記得在 HF Space 的 Secrets 設定這幾個環境變數：

GEMINI_API_KEY
HF_TOKEN
API_KEY
QDRANT_TOKEN
QDRANT_PATH
VOYAGE_TOKEN
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from google import genai
from google.genai import types as T

_TW = timezone(timedelta(hours=8))

_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
REPO_ID = "raywu918python/j1s-data"
_CHAT_MODEL = "gemini-2.5-flash"

_gemini = genai.Client(api_key=_GEMINI_API_KEY)


# ─── Chat History Storage ─────────────────────────────────────────────────────


def _chat_path(date_str: str) -> str:
    return f"db/chat/chat_{date_str}.json"


def _chat_url(date_str: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{_chat_path(date_str)}"


def read_chat_history(date_str: str = "") -> list[dict]:
    if not date_str:
        date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    try:
        r = httpx.get(_chat_url(date_str), headers=headers, follow_redirects=True, timeout=15)
        r.raise_for_status()
        return json.loads(r.content)
    except Exception:
        return []


def append_messages(msgs: list[dict], date_str: str) -> None:
    if not _HF_TOKEN:
        return
    from huggingface_hub import HfApi

    existing = read_chat_history(date_str)
    content = json.dumps(existing + msgs, ensure_ascii=False, indent=2).encode()
    try:
        HfApi(token=_HF_TOKEN).upload_file(
            path_or_fileobj=content,
            path_in_repo=_chat_path(date_str),
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"chat {date_str}",
        )
    except Exception:
        pass


# ─── Agent Tools ──────────────────────────────────────────────────────────────


def get_daily_picks(date_str: str = "") -> str:
    """查詢 AI 模型推薦股票。本機版直接呼叫 rag.query_sort_stock。"""
    try:
        from j1stools.rag import query_sort_stock

        data = query_sort_stock(st=date_str or None)
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception as e:
        return f"查詢失敗: {e}"


def search_mops(query: str, stock_id: str = "") -> str:
    """語義搜尋 MOPS 公告。本機版直接用 rag.query_any_string。"""
    try:
        from j1stools.rag import query_any_string

        hits = query_any_string(query=query, top_k=5, stock_id=stock_id or None)
        return json.dumps(hits, ensure_ascii=False, default=str)
    except Exception as e:
        return f"MOPS 搜尋失敗: {e}"


_TOOL_FN_MAP = {
    "get_daily_picks": get_daily_picks,
    "search_mops": search_mops,
}

_TOOLS = [
    T.Tool(
        function_declarations=[
            T.FunctionDeclaration(
                name="get_daily_picks",
                description="取得 AI 模型推薦台股清單（ABCD+外資+融券多因子）。未指定日期查最新。",
                parameters=T.Schema(
                    type=T.Type.OBJECT,
                    properties={
                        "date_str": T.Schema(
                            type=T.Type.STRING,
                            description="查詢日期 YYYY-MM-DD。留空查最新。",
                        )
                    },
                ),
            ),
            T.FunctionDeclaration(
                name="search_mops",
                description="語義搜尋公開資訊觀測站(MOPS)公告。",
                parameters=T.Schema(
                    type=T.Type.OBJECT,
                    properties={
                        "query": T.Schema(type=T.Type.STRING, description="搜尋關鍵字"),
                        "stock_id": T.Schema(
                            type=T.Type.STRING,
                            description="限定股票代號，例如 2330",
                        ),
                    },
                    required=["query"],
                ),
            ),
        ]
    )
]

_SYSTEM_PROMPT = """你是 J1S 股票分析助手，協助用戶了解 AI 模型選股結果與台股市場資訊。

工具：
- get_daily_picks：查詢 ABCD評分+外資+融券多因子模型的推薦股票
- search_mops：搜尋公開資訊觀測站最新公告

回答風格：繁體中文、簡潔具體、引用數據時說明來源。
免責聲明：資料僅供參考，不構成投資建議。"""


def _build_history(history: list[dict]) -> list[T.Content]:
    result = []
    for msg in history[-10:]:
        role = "user" if msg.get("type") == "user" else "model"
        result.append(T.Content(role=role, parts=[T.Part(text=msg["message"])]))
    return result


async def run_agent(user_message: str, date_str: str) -> str:
    """ReAct agent：讀今日聊天記錄作背景素材 → Gemini tool call 迴圈。"""
    today_msgs = await asyncio.to_thread(read_chat_history, date_str)
    system = _SYSTEM_PROMPT
    if today_msgs:
        recent = today_msgs[-20:]
        lines = [f"[{m.get('nickname','?')}] {m.get('message','')}" for m in recent]
        system += "\n\n今日聊天室最近討論：\n" + "\n".join(lines)

    chat = await asyncio.to_thread(
        _gemini.chats.create,
        model=_CHAT_MODEL,
        config=T.GenerateContentConfig(system_instruction=system, tools=_TOOLS),
    )

    response = await asyncio.to_thread(chat.send_message, user_message)

    for _ in range(5):
        fn_calls = [p.function_call for p in response.candidates[0].content.parts if p.function_call]
        if not fn_calls:
            break
        parts = []
        for fc in fn_calls:
            fn = _TOOL_FN_MAP.get(fc.name)
            result = await asyncio.to_thread(fn, **dict(fc.args)) if fn else f"未知工具: {fc.name}"
            parts.append(T.Part(function_response=T.FunctionResponse(name=fc.name, response={"result": result})))
        response = await asyncio.to_thread(chat.send_message, parts)

    return response.text
