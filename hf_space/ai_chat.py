"""AI 圓桌會議後端 — LangGraph Multi-Agent Supervisor 教學實作

架構圖
------
（自動 or 用戶觸發）
  │
  ▼
supervisor（主持人）← ← ← ← ←┐
  │                            │
  ├─── chip_analyst（籌碼師）──┤
  │                            │
  ├─── news_analyst（新聞師）──┘
  │
  └─── synthesizer（整合）──→ END

議題驅動流程
-----------
1. 每日自動 → run_autonomous_roundtable()
   → 各議題存 db/chat/{date}/topic_{id}.json
   → 更新 db/chat/{date}/index.json

2. 用戶追問 → run_qa(topic_id, message)
   → 輕量 LLM 回答（帶議題背景，不觸發完整圓桌）
   → qa 欄位追加回 topic 檔

3. 用戶開新議題 → run_roundtable_for_topic()
   → 觸發完整圓桌，存新 topic 檔

API
---
GET  /topics              → list_topics()
GET  /topics/{id}         → load_topic()
POST /topics/{id}/ask     → run_qa()
POST /topics              → run_roundtable_for_topic()（背景執行）
POST /trigger/roundtable  → run_autonomous_roundtable()（背景執行）
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, TypedDict
import operator

import httpx
from langchain_core.callbacks import BaseCallbackHandler          # fallback 失敗記錄用
from langchain_core.messages import HumanMessage, SystemMessage  # 輕量，留在頂層
from langchain_core.tools import tool                            # @tool 裝飾器需要
from pydantic import BaseModel

# ⚠️ 重量級套件全部 lazy import（第一次呼叫時才載入，不影響啟動速度）：
#   langchain_groq / langchain_openai → 在 _llm() 內
#   langgraph              → 在 _get_roundtable() 內
#   create_react_agent     → 在 _make_agent_node() 內

_TW = timezone(timedelta(hours=8))
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_JUST1STOCK", "")  # 第二個 Groq 帳號，額度獨立
_CEREBRAS_TOKEN = os.environ.get("CEREBRES_TOKEN", "")
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_QDRANT_TOKEN = os.environ.get("QDRANT_TOKEN", "")
_QDRANT_PATH = os.environ.get("QDRANT_PATH", "")
_VOYAGE_TOKEN = os.environ.get("VOYAGE_TOKEN", "")
REPO_ID = "raywu918python/j1s-data"
_EMBED_MODEL_VOYAGE = "voyage-3"
_EMBED_DIM = 1024


# ─── LLM 工廠 ─────────────────────────────────────────────────────────────────
# LLM_PROVIDER 環境變數控制使用哪個模型：
#   groq（預設）   → 多個 Groq 模型自動 fallback，每個模型額度獨立，主模型額度用完自動換下一個
#   openrouter     → 免費模型（OpenAI 相容 API），工具呼叫不穩定（測試多個模型都有問題），實驗用
#   qwen           → Qwen3-8B on HF Serverless，實驗用、不建議上線：
#                      - 工具呼叫(bind_tools)實際走計費的 Inference Providers 額度，不是真正免費
#                      - 思考模式(thinking)關掉會跟 bind_tools 衝突，開著又可能把 token 用在思考、生成空白回應
#
# Gemini 不放在這裡 — 額度留給 RAG 用，避免互搶。
#
# HF Space Secrets 設定：LLM_PROVIDER=groq（預設值，通常不需要特別設定）

# Groq 上各自獨立速率限制的模型，依序當主力/備援（全部測試過 structured output + 工具呼叫）
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# Cerebras 上的模型 — 走 OpenAI 相容端點（langchain-cerebras 套件相依 langchain-core 太舊，
# 會把 langgraph/langchain-groq 需要的 1.x 版本擠掉，所以不用那個套件）。
_CEREBRAS_MODELS = ["gpt-oss-120b", "zai-glm-4.7"]

_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
_current_provider: str = _LLM_PROVIDER  # 可透過 API 動態切換，重啟後還原預設值


def get_provider() -> str:
    return _current_provider


def set_provider(provider: str) -> None:
    global _current_provider
    if provider not in ("groq", "openrouter", "qwen"):
        raise ValueError(f"不支援的 provider: {provider}，可選 groq / openrouter / qwen")
    _current_provider = provider


def _llm():
    if _current_provider == "qwen":
        # HF Serverless Inference（task="text-generation"）— 真正免費無額度上限。
        # 注意：這條路徑不支援 with_structured_output()（會 raise NotImplementedError），
        # 所以 supervisor_node 用這個 provider 會壞掉 — 僅供內部測試，不對外開放。
        # bind_tools()（chip_analyst/news_analyst 用）測試正常，可以用。
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-8B",
            huggingfacehub_api_token=_HF_TOKEN,
            task="text-generation",
            max_new_tokens=2048,
        )
        return ChatHuggingFace(llm=endpoint)

    if _current_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="nvidia/nemotron-nano-9b-v2:free",
            api_key=_OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    # 預設：Groq + Cerebras，多模型 + 多帳號 fallback — 每個模型、每個帳號的速率限制額度
    # 都獨立，Groq 10 組（5模型×2帳號）用完才輪到 Cerebras，前端完全感覺不到切換。
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    keys = [k for k in (_GROQ_API_KEY, _GROQ_API_KEY_2) if k]
    combos = [(m, k) for m in _GROQ_MODELS for k in keys]
    llms = [ChatGroq(model=m, api_key=k, callbacks=[_FallbackLogger(m)]) for m, k in combos]

    if _CEREBRAS_TOKEN:
        llms += [
            ChatOpenAI(
                model=m,
                api_key=_CEREBRAS_TOKEN,
                base_url="https://api.cerebras.ai/v1",
                callbacks=[_FallbackLogger(f"cerebras:{m}")],
            )
            for m in _CEREBRAS_MODELS
        ]

    primary, *fallbacks = llms
    return primary.with_fallbacks(fallbacks)


class _FallbackLogger(BaseCallbackHandler):
    """模型呼叫失敗時印出來，方便在 HF Space Logs 看到 fallback 有沒有觸發。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        print(f"[ai_chat] Groq 模型 {self.model_name} 失敗，切換下一個: {type(error).__name__}: {error}")


# ─── Topic Storage (HF Dataset) ──────────────────────────────────────────────
# 每個議題一個檔案，加上每日 index 供前端快速列出

def _topic_path(topic_id: str, date_str: str) -> str:
    return f"db/chat/{date_str}/topic_{topic_id}.json"


def _index_path(date_str: str) -> str:
    return f"db/chat/{date_str}/index.json"


def _hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path}"


def _save_to_hf(path: str, data, commit_msg: str) -> None:
    if not _HF_TOKEN:
        print(f"[ai_chat] 跳過存檔 {path}：未設定 HF_TOKEN")
        return
    content = json.dumps(data, ensure_ascii=False, indent=2).encode()
    try:
        from huggingface_hub import HfApi
        HfApi(token=_HF_TOKEN).upload_file(
            path_or_fileobj=content,
            path_in_repo=path,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        print(f"[ai_chat] 已存檔 {path}")
    except Exception as e:
        print(f"[ai_chat] 存檔失敗 {path}: {e}")


def load_topic(topic_id: str, date_str: str = "") -> dict | None:
    """從 HF 讀取指定議題的完整資料（含 discussion + qa）。"""
    if not date_str:
        date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    try:
        r = httpx.get(_hf_url(_topic_path(topic_id, date_str)), headers=headers, follow_redirects=True, timeout=15)
        r.raise_for_status()
        return json.loads(r.content)
    except Exception:
        return None


def list_topics(date_str: str = "") -> list[dict]:
    """從 HF 讀取當日議題摘要列表（index.json）。"""
    if not date_str:
        date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    try:
        r = httpx.get(_hf_url(_index_path(date_str)), headers=headers, follow_redirects=True, timeout=15)
        r.raise_for_status()
        return json.loads(r.content)
    except Exception:
        return []


def save_topic(topic: dict, date_str: str) -> None:
    """把 topic dict 存到 HF，並更新當日 index。"""
    _save_to_hf(
        _topic_path(topic["topic_id"], date_str),
        topic,
        f"topic {topic['topic_id']} {date_str}",
    )
    # 更新 index（summary 用，不含完整 discussion）
    index = list_topics(date_str)
    entry = {
        "topic_id": topic["topic_id"],
        "title": topic["title"],
        "summary": topic.get("summary", ""),
        "created_at": topic.get("created_at", ""),
        "source": topic.get("source", "auto"),
        "qa_count": len(topic.get("qa", [])),
    }
    index = [e for e in index if e["topic_id"] != topic["topic_id"]]
    index.append(entry)
    _save_to_hf(_index_path(date_str), index, f"index {date_str}")


# ─── LangChain Tools（用 @tool 裝飾器）─────────────────────────────────────────
#
# LangGraph 概念：@tool 讓函式變成 LangChain 工具物件
#   - .name        : 工具名稱（LLM 呼叫時使用）
#   - .description : LLM 用來決定要不要呼叫這個工具
#   - .invoke()    : 實際執行函式
#
# bind_tools([tool1, tool2]) 讓 LLM 知道有哪些工具可用


@tool
def get_daily_picks(date_str: str = "") -> str:
    """查詢 AI 模型推薦股票（ABCD評分+外資+融券多因子）。未指定日期查最新一天。"""
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    dates = (
        [date_str]
        if date_str
        else [(datetime.now(_TW).date() - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(7)]
    )
    for d in dates:
        try:
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/db/rag/daily_picks_{d}.json"
            r = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
            r.raise_for_status()
            return json.dumps(json.loads(r.content), ensure_ascii=False)
        except Exception:
            continue
    return "找不到選股結果"


@tool
def search_mops(query: str, stock_id: str = "") -> str:
    """語義搜尋公開資訊觀測站(MOPS)公告，找出相關股票新聞。"""
    if not all([_VOYAGE_TOKEN, _QDRANT_PATH, _QDRANT_TOKEN]):
        return "MOPS 搜尋服務未設定（缺 VOYAGE_TOKEN / QDRANT_TOKEN / QDRANT_PATH）"
    try:
        r = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {_VOYAGE_TOKEN}"},
            json={"input": [query], "model": _EMBED_MODEL_VOYAGE, "output_dimension": _EMBED_DIM},
            timeout=15,
        )
        r.raise_for_status()
        vector = r.json()["data"][0]["embedding"]
    except Exception as e:
        return f"Voyage embedding 失敗: {e}"

    payload: dict = {"vector": vector, "limit": 5, "with_payload": True}
    if stock_id:
        payload["filter"] = {"must": [{"key": "stock_id", "match": {"value": stock_id}}]}
    try:
        r = httpx.post(
            f"https://{_QDRANT_PATH}/collections/mops_news/points/search",
            headers={"api-key": _QDRANT_TOKEN},
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        hits = [
            {
                "stock_id": h["payload"].get("stock_id"),
                "title": h["payload"].get("title"),
                "date": h["payload"].get("date"),
                "score": round(h["score"], 3),
            }
            for h in r.json().get("result", [])
        ]
        return json.dumps(hits, ensure_ascii=False)
    except Exception as e:
        return f"Qdrant 搜尋失敗: {e}"


# ─── 主題資料來源 ─────────────────────────────────────────────────────────────
# 圓桌開場前自動抓取兩個主題，供 context_loader_node 填入 State


def _fetch_rfc_open() -> dict:
    """取得 rfc_macd_ib_6xx 最新持倉股票。讀 db/backtest/rfc_macd_ib_6xx/open_*.parquet"""
    import io
    import pandas as pd

    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    for delta in range(7):
        d = (datetime.now(_TW).date() - timedelta(days=delta)).strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main" f"/db/backtest/rfc_macd_ib_6xx/open_{d}.parquet"
        try:
            r = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
            r.raise_for_status()
            df = pd.read_parquet(io.BytesIO(r.content))
            stocks = df["stock_id"].astype(str).tolist() if "stock_id" in df.columns else []
            return {"date": d, "model": "rfc_macd_ib_6xx", "stocks": stocks}
        except Exception:
            continue
    return {"date": "", "model": "rfc_macd_ib_6xx", "stocks": []}


def _fetch_radar_top3() -> dict:
    """取得 AI 雷達最新選股前 3 支。讀 db/rag/daily_picks_*.json 的 top5[:3]"""
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
    for delta in range(7):
        d = (datetime.now(_TW).date() - timedelta(days=delta)).strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main" f"/db/rag/daily_picks_{d}.json"
        try:
            r = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
            r.raise_for_status()
            data = json.loads(r.content)
            top3 = data.get("top5", [])[:3]
            return {"date": d, "model": "radar", "stocks": top3}
        except Exception:
            continue
    return {"date": "", "model": "radar", "stocks": []}


def get_today_topics() -> dict:
    """前端顯示用：回傳今日兩個討論主題的摘要。"""
    rfc = _fetch_rfc_open()
    radar = _fetch_radar_top3()
    return {
        "rfc_macd_ib": {
            "label": "RFC MACD IB 持倉",
            "date": rfc["date"],
            "stocks": rfc["stocks"],
        },
        "radar_top3": {
            "label": "AI 雷達 Top3",
            "date": radar["date"],
            "stocks": radar["stocks"],
        },
    }


# ─── LangGraph State ──────────────────────────────────────────────────────────
#
# LangGraph 概念：State 是整個圖共享的狀態
#   - 每個節點接收 state，回傳「要更新哪些欄位」的 dict
#   - Annotated[list, operator.add] 代表這個欄位是「累加」，不是覆蓋
#     → chip_analyst 和 news_analyst 各自加一條訊息，supervisor 整合時看得到全部


class RoundtableState(TypedDict):
    question: str  # 用戶原始問題
    load_daily_context: bool  # 是否載入今日 RFC/雷達廣播（只有每日自動圓桌需要）
    rfc_context: str  # 主題 1：RFC 持倉（context_loader 填入）
    radar_context: str  # 主題 2：AI 雷達 Top3（context_loader 填入）
    chip_result: str  # 籌碼AI 揭露的客觀數據
    news_result: str  # 新聞AI 摘要的公告重點
    discussion: Annotated[list, operator.add]  # 所有 agent 的發言（前端顯示用）
    next: str  # supervisor 決定的下一步


def _entry(nickname: str, message: str) -> dict:
    now = datetime.now(_TW)
    return {
        "type": "ai",
        "nickname": nickname,
        "message": message,
        "timestamp": now.strftime("%H:%M"),
        "date": now.strftime("%Y-%m-%d"),
    }


# ─── Context Loader Node ─────────────────────────────────────────────────────
# 圖的第一個節點：圓桌開場前預載入今日主題，讓 supervisor 和各分析師都知道要聚焦哪些股票


def context_loader_node(state: RoundtableState) -> dict:
    """預載入兩個討論主題，寫入 State 供後續所有節點使用。

    只有每日自動圓桌（load_daily_context=True）才廣播今日主題，
    使用者自訂議題只聚焦在自己的問題，不會被無關的 RFC/雷達內容干擾。
    """
    if not state.get("load_daily_context"):
        return {}

    rfc = _fetch_rfc_open()
    radar = _fetch_radar_top3()

    rfc_summary = f"RFC MACD IB 持倉（{rfc['date']}）：{', '.join(rfc['stocks']) or '無資料'}"
    radar_summary = (
        f"AI 雷達 Top3（{radar['date']}）："
        + "、".join(f"{s.get('stock_id','')} {s.get('company_name','')}" for s in radar["stocks"])
        or "無資料"
    )

    # 以「系統」身份在聊天室公告今日主題（前端可選擇顯示）
    context_msg = _entry(
        "系統",
        f"📋 今日圓桌主題\n1️⃣ {rfc_summary}\n2️⃣ {radar_summary}",
    )
    context_msg["type"] = "system"

    return {
        "rfc_context": rfc_summary,
        "radar_context": radar_summary,
        "discussion": [context_msg],
    }


# ─── Agent 角色定義（在這裡加角色） ───────────────────────────────────────────
#
# 每個角色的設定：
#   nickname : 顯示在聊天室的名稱
#   tools    : 這個角色能使用的工具（LangChain @tool 物件）
#   persona  : 角色的 system prompt — 決定它的分析風格和專業範疇
#   result_key: 把分析結果寫入 RoundtableState 的哪個欄位
#
# 新增角色只需要在這裡加一個 dict，不用改其他地方。

_AGENTS: dict[str, dict] = {
    "chip_analyst": {
        "nickname": "籌碼AI",
        "tools": [get_daily_picks],
        "persona": (
            "你是籌碼數據助手，唯一任務是把外資買賣超、融券增減、ABCD量化評分等"
            "「已經發生的客觀數據」條列出來。\n"
            "規則（絕對遵守）：\n"
            "1. 只陳述數據本身，例如：今日外資買超 X 張、投信賣超 Y 張、近 5 日三大法人累計買超 Z 張。\n"
            "2. 禁止對數據做任何解讀或推論，例如禁止說「代表主力看好」「建議跟單」「後續會上漲」。\n"
            "3. 禁止使用「看好」「看壞」「建議」「進場」「出場」等字眼。\n"
            "4. 用繁體中文條列式回答。"
        ),
        "result_key": "chip_result",
    },
    "news_analyst": {
        "nickname": "新聞AI",
        "tools": [search_mops],
        "persona": (
            "你是新聞摘要助手，唯一任務是把公開資訊觀測站公告的「字面重點」整理出來。\n"
            "規則（絕對遵守）：\n"
            "1. 只做新聞/公告內容的摘要，例如：依據 XX 公告，該公司表示...\n"
            "2. 禁止對新聞做主觀行情解讀，例如禁止說「這絕對是大利多」「股價要噴了」「建議加碼」。\n"
            "3. 可以引用公告原文揭露的風險字眼（如停業、重大訴訟），但不要自己加上「地雷」「利多」之類的價值判斷標籤。\n"
            "4. 用繁體中文條列式回答，引用時註明資料來源。"
        ),
        "result_key": "news_result",
    },
    # 未來可以加：
    # "tech_analyst": {
    #     "nickname":   "技術AI",
    #     "tools":      [get_price_chart],
    #     "persona":    "你是技術數據助手，只陳述K線、均線、量價等客觀數據，不做解讀...",
    #     "result_key": "tech_result",
    # },
}


# 前端顯示用：聊天室裡所有 AI 角色清單（GET /chat/roles）
_ROLES_INFO = [
    {"nickname": "籌碼AI", "description": "彙整外資、融券、ABCD評分等客觀籌碼數據，不做主觀解讀"},
    {"nickname": "新聞AI", "description": "摘要公開資訊觀測站公告重點，不做行情解讀"},
    {"nickname": "主持人AI", "description": "彙整會議揭露的客觀資訊、回答議題追問、把關不當問題，不給投資建議"},
    {"nickname": "散戶AI", "description": "陪用戶聊股票以外的話題，關心用戶生活"},
]


def list_chat_roles() -> list[dict]:
    """前端顯示用：聊天室裡有哪些 AI 角色。"""
    return _ROLES_INFO


# ─── Specialist Agent Nodes ───────────────────────────────────────────────────
#
# LangGraph 概念：create_react_agent 建立一個完整的 ReAct 子圖
#   prompt = agent 的 persona（system prompt）
#   內部循環：LLM 思考 → 決定呼叫工具 → 執行工具 → LLM 讀結果 → 繼續或結束
#
# _make_agent_node 是工廠函式：讀 _AGENTS 設定，產生對應的 LangGraph 節點函式。
# 這樣新增角色不需要手寫新的 node function。


def _make_agent_node(agent_key: str):
    cfg = _AGENTS[agent_key]

    def node(state: RoundtableState) -> dict:
        from langgraph.prebuilt import create_react_agent
        subgraph = create_react_agent(
            _llm(),
            tools=cfg["tools"],
            prompt=cfg["persona"],  # ← 角色在這裡注入
        )
        result = subgraph.invoke({"messages": [HumanMessage(state["question"])]})
        analysis = result["messages"][-1].content
        return {
            cfg["result_key"]: analysis,
            "discussion": [_entry(cfg["nickname"], analysis)],
        }

    node.__name__ = f"{agent_key}_node"
    return node


chip_analyst_node = _make_agent_node("chip_analyst")
news_analyst_node = _make_agent_node("news_analyst")


# ─── Supervisor Node ──────────────────────────────────────────────────────────
#
# LangGraph 概念：supervisor 使用 with_structured_output 做結構化決策
#   → LLM 不回傳自由文字，而是回傳符合 Pydantic schema 的物件
#   → 確保輸出一定是合法的路由選項，不會出現解析錯誤


class _RouterDecision(BaseModel):
    next: Literal["chip_analyst", "news_analyst", "FINISH", "BLOCKED", "casual_chat"]
    reason: str  # 讓 LLM 說明決定理由（提高思考品質）


def supervisor_node(state: RoundtableState) -> dict:
    """主持人：把關問題合規性 + 根據已完成的分析決定下一步路由。

    用 with_structured_output() 強制回傳合法路由值，比純文字判斷可靠
    （純文字判斷實測會有模型誤判、重複路由同一個分析師的問題）。
    注意：Qwen（HF Serverless）不支援這個方法，但它已不對外開放，無影響。

    BLOCKED / casual_chat 只在「尚未開始任何分析」時才會判斷，
    避免分析做到一半又被切去這兩個分支。
    """
    done = []
    if state.get("chip_result"):
        done.append(f"籌碼分析完成：{state['chip_result'][:150]}…")
    if state.get("news_result"):
        done.append(f"新聞分析完成：{state['news_result'][:150]}…")

    done_str = "\n".join(done) if done else "尚未開始任何分析"

    daily_context = ""
    if state.get("rfc_context") or state.get("radar_context"):
        daily_context = f"""
今日圓桌討論主題：
  主題 1｜{state.get('rfc_context', '')}
  主題 2｜{state.get('radar_context', '')}
"""

    system = f"""你是投資研究圓桌會議主持人，協調各專家依序發言，並把關不當問題。
{daily_context}
用戶問題：{state['question']}

目前進度：
{done_str}

決策規則（依序判斷，BLOCKED 跟 casual_chat 只在「尚未開始任何分析」時可選）：
- 尚未開始任何分析，且用戶要求具體買賣建議、預測股價漲跌、內幕消息、或其他違反法規的內容 → BLOCKED
- 尚未開始任何分析，且用戶問題明顯跟股票分析無關（閒聊、問候、生活話題、心情）→ casual_chat
- 尚未進行籌碼分析 → chip_analyst
- 籌碼完成但尚未進行新聞分析 → news_analyst
- 兩者都完成 → FINISH（進入整合結論）"""

    decision = _llm().with_structured_output(_RouterDecision).invoke(
        [SystemMessage(system), HumanMessage("請決定下一步。")]
    )
    return {"next": decision.next}


# ─── Synthesizer Node ─────────────────────────────────────────────────────────


def synthesizer_node(state: RoundtableState) -> dict:
    """主持人AI：總結會議揭露的客觀資訊，絕對不做綜合研判或給出方向性結論。"""
    system = """你是會議流程總結助手，唯一任務是把籌碼AI和新聞AI揭露的「客觀事實」彙整呈現。

規則（絕對遵守）：
1. 只做客觀事實的彙整陳述，不做「綜合研判」「給出結論」「給出方向」的動作。
2. 禁止使用「看多」「看空」「建議買進/賣出/加碼/減碼」「值得介入」等任何投資建議用語。
3. 禁止把籌碼面和新聞面的資訊「串連推論」成一個市場方向（例如禁止說「籌碼集中加上利多消息，後續看漲」）。
4. 結尾固定加上：「以上僅為會議揭露之公開資訊彙整，不構成投資建議，請投資人自行審慎評估風險。」
5. 用繁體中文回答，分點陳述。

範例（安全寫法）：
「感謝籌碼AI與新聞AI的數據整理。今日會議揭露的公開資訊如下：三大法人買賣超情況為...；
媒體關注焦點為...。以上僅為會議揭露之公開資訊彙整，不構成投資建議，請投資人自行審慎評估風險。」"""

    daily_context = ""
    if state.get("rfc_context") or state.get("radar_context"):
        daily_context = f"""今日圓桌主題：
{state.get('rfc_context', '')}
{state.get('radar_context', '')}

"""

    user = f"""{daily_context}用戶問題：{state['question']}

【籌碼AI】
{state.get('chip_result') or '無資料'}

【新聞AI】
{state.get('news_result') or '無資料'}

請彙整以上揭露的客觀資訊。"""

    response = _llm().invoke([SystemMessage(system), HumanMessage(user)])
    return {"discussion": [_entry("主持人AI", response.content)]}


# ─── Host Refuse Node ─────────────────────────────────────────────────────────
# supervisor 判斷問題違規（要求買賣建議/股價預測/內幕消息等）時，直接拒絕，
# 不進行籌碼/新聞分析，避免讓 chip_analyst/news_analyst 處理到不當問題。


def host_refuse_node(_state: RoundtableState) -> dict:
    """主持人AI：偵測到不合規問題，直接拒絕。"""
    msg = (
        "您的問題涉及具體買賣建議、股價預測，或其他本平台無法回應的內容。"
        "本平台僅提供公開資訊彙整，不提供投資建議，請自行審慎評估風險。"
    )
    return {"discussion": [_entry("主持人AI", msg)]}


# ─── Retail Investor Node ─────────────────────────────────────────────────────
# supervisor 判斷問題跟股票分析無關（閒聊、問候）時，由散戶AI陪聊，
# 同樣禁止給出投資建議。


def retail_investor_node(state: RoundtableState) -> dict:
    """散戶AI：陪用戶聊股票分析以外的話題，展現關心，但同樣不能給投資建議或違法內容。"""
    system = """你是散戶AI，站在發問用戶這邊，像朋友一樣聊天、關心用戶的生活與心情。

規則（絕對遵守）：
1. 只聊跟股票分析無關的閒聊話題，展現關心、陪伴的語氣。
2. 即使話題聊開了，也絕對不能給出任何投資建議、預測股價、或違反法規的內容。
3. 若用戶想聊股票分析，引導用戶換個方式提出問題（例如直接問某支股票的籌碼/新聞）。
4. 用繁體中文回答，語氣輕鬆自然。"""

    response = _llm().invoke([SystemMessage(system), HumanMessage(state["question"])])
    return {"discussion": [_entry("散戶AI", response.content)]}


# ─── 建立 LangGraph ────────────────────────────────────────────────────────────
#
# StateGraph(State) → 建立以 State 為基礎的有向圖
# add_node(name, fn) → fn 接收 state，回傳部分更新
# add_edge(A, B) → A 完成後固定跳到 B
# add_conditional_edges(src, router_fn, mapping) → 根據 router_fn 的回傳值決定下一個節點
# set_entry_point(name) → 圖的起點
# compile() → 產生可執行的 Runnable（支援 invoke / stream / astream）


def _route(state: RoundtableState) -> str:
    """Conditional edge router：讀取 supervisor 設定的 next 欄位。"""
    return state.get("next", "FINISH")


_roundtable = None


def _get_roundtable():
    global _roundtable
    if _roundtable is None:
        from langgraph.graph import END, StateGraph
        g = StateGraph(RoundtableState)
        g.add_node("context_loader", context_loader_node)
        g.add_node("supervisor", supervisor_node)
        g.add_node("chip_analyst", chip_analyst_node)
        g.add_node("news_analyst", news_analyst_node)
        g.add_node("synthesizer", synthesizer_node)
        g.add_node("host_refuse", host_refuse_node)
        g.add_node("retail_chat", retail_investor_node)
        g.set_entry_point("context_loader")
        g.add_edge("context_loader", "supervisor")
        g.add_conditional_edges(
            "supervisor",
            _route,
            {
                "chip_analyst": "chip_analyst",
                "news_analyst": "news_analyst",
                "FINISH": "synthesizer",
                "BLOCKED": "host_refuse",
                "casual_chat": "retail_chat",
            },
        )
        g.add_edge("chip_analyst", "supervisor")
        g.add_edge("news_analyst", "supervisor")
        g.add_edge("synthesizer", END)
        g.add_edge("host_refuse", END)
        g.add_edge("retail_chat", END)
        _roundtable = g.compile()
    return _roundtable
#
# 圖視覺化（本機開發可用）：
#   from IPython.display import Image
#   Image(roundtable.get_graph().draw_mermaid_png())


# ─── 對外介面 ─────────────────────────────────────────────────────────────────


async def run_rfc_macd_ib_roundtable() -> dict:
    """每日自動啟動 RFC MACD IB 持倉議題（收盤後資料才齊，較晚觸發）。

    由 GET/POST /trigger/roundtable/rfc-macd-ib 觸發。
    """
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    print(f"[ai_chat] RFC MACD IB 議題開始 {date_str}")
    rfc = _fetch_rfc_open()
    rfc_stocks = "、".join(rfc["stocks"][:5]) or "無持倉資料"

    topic = await run_roundtable_for_topic(
        topic_id="rfc_macd_ib",
        title=f"RFC MACD IB 持倉分析（{rfc['date']}）",
        question=(
            f"請針對 RFC MACD IB 模型持倉股票進行分析：\n"
            f"持倉清單：{rfc_stocks}\n"
            f"請從籌碼面與新聞面分別提出看法。"
        ),
        date_str=date_str,
        source="auto",
    )
    print(f"[ai_chat] RFC MACD IB 議題完成 {date_str}")
    return topic


async def run_radar_top3_roundtable() -> dict:
    """每日自動啟動 AI 雷達 Top3 議題（盤前資料即齊，較早觸發）。

    由 GET/POST /trigger/roundtable/radar-top3 觸發。
    """
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    print(f"[ai_chat] AI 雷達 Top3 議題開始 {date_str}")
    radar = _fetch_radar_top3()
    radar_stocks = (
        "、".join(f"{s.get('stock_id','')} {s.get('company_name','')}" for s in radar["stocks"])
        or "無雷達資料"
    )

    topic = await run_roundtable_for_topic(
        topic_id="radar_top3",
        title=f"AI 雷達 Top3 分析（{radar['date']}）",
        question=(
            f"請針對 AI 雷達選出的前 3 支股票進行分析：\n"
            f"股票清單：{radar_stocks}\n"
            f"請從籌碼面與新聞面分別提出看法。"
        ),
        date_str=date_str,
        source="auto",
    )
    print(f"[ai_chat] AI 雷達 Top3 議題完成 {date_str}")
    return topic


async def run_roundtable(user_message: str, date_str: str, load_daily_context: bool = True) -> list[dict]:
    """執行 AI 圓桌會議，回傳所有 agent 的發言。

    load_daily_context=True（每日自動圓桌）才會廣播 RFC/雷達今日主題；
    使用者自訂議題應該傳 False，避免顯示跟自己問題無關的內容。
    """
    init_state: RoundtableState = {
        "question": user_message,
        "load_daily_context": load_daily_context,
        "rfc_context": "",
        "radar_context": "",
        "chip_result": "",
        "news_result": "",
        "discussion": [],
        "next": "",
    }
    result = await asyncio.to_thread(_get_roundtable().invoke, init_state)
    return result["discussion"]


async def run_roundtable_for_topic(
    topic_id: str,
    title: str,
    question: str,
    date_str: str = "",
    source: str = "auto",
) -> dict:
    """觸發完整圓桌，結果打包成 topic dict 存入 HF。

    回傳 topic dict（含 discussion + 空 qa 列表）。
    """
    if not date_str:
        date_str = datetime.now(_TW).strftime("%Y-%m-%d")

    print(f"[ai_chat] 議題開始 {topic_id} ({title})")
    discussion = await run_roundtable(question, date_str, load_daily_context=(source == "auto"))
    print(f"[ai_chat] 議題討論完成 {topic_id}，共 {len(discussion)} 則發言")

    # synthesizer 的輸出就是摘要
    summary = next(
        (m["message"] for m in reversed(discussion) if m.get("nickname") == "主持人AI"),
        "",
    )

    topic = {
        "topic_id": topic_id,
        "title": title,
        "summary": summary,
        "discussion": discussion,
        "qa": [],
        "created_at": datetime.now(_TW).strftime("%Y-%m-%d %H:%M"),
        "source": source,
    }
    await asyncio.to_thread(save_topic, topic, date_str)
    return topic


async def run_qa(
    topic_id: str,
    user_message: str,
    date_str: str = "",
    nickname: str = "匿名",
) -> tuple[dict, dict]:
    """對已完成的議題追問 — 輕量 LLM 回答，不重跑完整圓桌。

    回傳 (qa_entry, topic)。
    呼叫方負責把 save_topic 丟背景執行，不在此阻塞。
    """
    if not date_str:
        date_str = datetime.now(_TW).strftime("%Y-%m-%d")

    topic = await asyncio.to_thread(load_topic, topic_id, date_str)
    if topic is None:
        raise ValueError(f"議題 {topic_id} 不存在")

    ctx_parts = []
    if topic.get("summary"):
        ctx_parts.append(f"圓桌會議結論：\n{topic['summary']}")
    recent_qa = topic.get("qa", [])[-5:]
    if recent_qa:
        lines = []
        for qa in recent_qa:
            lines.append(f"問：{qa['question']['message']}")
            lines.append(f"答：{qa['answer']['message']}")
        ctx_parts.append("近期問答：\n" + "\n".join(lines))

    system = (
        f"你是主持人AI，協助用戶查閱圓桌會議揭露的客觀資訊。\n"
        f"議題：{topic.get('title', '')}\n\n"
        + "\n\n".join(ctx_parts)
        + "\n\n規則（絕對遵守）：\n"
        "1. 只能根據以上會議揭露的客觀資訊回答，不做超出資料範圍的預測或推論。\n"
        "2. 禁止給出任何買進/賣出/加碼/減碼等投資建議或方向性判斷。\n"
        "3. 若用戶詢問是否該買賣，請回覆無法提供投資建議，並引導用戶自行判斷風險。\n"
        "4. 用繁體中文簡潔回答。"
    )

    now = datetime.now(_TW)
    ts = now.strftime("%H:%M")
    date_out = now.strftime("%Y-%m-%d")

    response = await asyncio.to_thread(_llm().invoke, [SystemMessage(system), HumanMessage(user_message)])

    qa_entry = {
        "question": {"type": "user", "nickname": nickname, "message": user_message, "timestamp": ts, "date": date_out},
        "answer": {"type": "ai", "nickname": "主持人AI", "message": response.content, "timestamp": ts, "date": date_out},
    }

    topic["qa"] = topic.get("qa", []) + [qa_entry]
    return qa_entry, topic
