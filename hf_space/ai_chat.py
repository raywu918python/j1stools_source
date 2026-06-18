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
from langchain_core.messages import HumanMessage, SystemMessage  # 輕量，留在頂層
from langchain_core.tools import tool                            # @tool 裝飾器需要
from pydantic import BaseModel

# ⚠️ 重量級套件全部 lazy import（第一次呼叫時才載入，不影響啟動速度）：
#   langchain_google_genai → 在 _llm() 內
#   langchain_huggingface  → 在 _llm() 內
#   langgraph              → 在 _get_roundtable() 內
#   create_react_agent     → 在 _make_agent_node() 內

_TW = timezone(timedelta(hours=8))
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_QDRANT_TOKEN = os.environ.get("QDRANT_TOKEN", "")
_QDRANT_PATH = os.environ.get("QDRANT_PATH", "")
_VOYAGE_TOKEN = os.environ.get("VOYAGE_TOKEN", "")
REPO_ID = "raywu918python/j1s-data"
_CHAT_MODEL = "gemini-2.5-flash"
_EMBED_MODEL_VOYAGE = "voyage-3"
_EMBED_DIM = 1024


# ─── LLM 工廠 ─────────────────────────────────────────────────────────────────
# LLM_PROVIDER 環境變數控制使用哪個模型：
#   gemini（預設）→ Gemini 2.5 Flash，免費 1500 RPD，tool calling 最穩
#   qwen          → Qwen3-8B on HF Serverless，用 HF_TOKEN，免費無 RPD 硬限
#
# HF Space Secrets 設定：LLM_PROVIDER=qwen  即可切換

_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "qwen")
_current_provider: str = _LLM_PROVIDER  # 可透過 API 動態切換，重啟後還原預設值


def get_provider() -> str:
    return _current_provider


def set_provider(provider: str) -> None:
    global _current_provider
    if provider not in ("gemini", "qwen"):
        raise ValueError(f"不支援的 provider: {provider}，可選 gemini / qwen")
    _current_provider = provider


def _llm():
    if _current_provider == "qwen":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-8B",
            huggingfacehub_api_token=_HF_TOKEN,
            task="text-generation",
            max_new_tokens=2048,
            # Qwen3 有 thinking 模式，關掉讓 tool calling 輸出更乾淨
            model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return ChatHuggingFace(llm=endpoint)

    # 預設：Gemini 2.5 Flash
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=_CHAT_MODEL,
        google_api_key=_GEMINI_API_KEY,
    )


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
    except Exception:
        pass


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
    rfc_context: str  # 主題 1：RFC 持倉（context_loader 填入）
    radar_context: str  # 主題 2：AI 雷達 Top3（context_loader 填入）
    chip_result: str  # 籌碼分析師的結論
    news_result: str  # 新聞分析師的結論
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


def context_loader_node(_state: RoundtableState) -> dict:
    """預載入兩個討論主題，寫入 State 供後續所有節點使用。"""
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
        "nickname": "籌碼分析師",
        "tools": [get_daily_picks],
        "persona": (
            "你是籌碼分析師，專注外資買賣超、融券增減、ABCD量化評分分析。"
            "用繁體中文條列式回答，重點放在數字變化與趨勢。"
        ),
        "result_key": "chip_result",
    },
    "news_analyst": {
        "nickname": "新聞分析師",
        "tools": [search_mops],
        "persona": (
            "你是新聞分析師，專注公開資訊觀測站公告解讀，"
            "找出地雷訊號（掏空、財報異常、停業）或利多消息（合約、增資）。"
            "用繁體中文條列式回答，標出風險等級。"
        ),
        "result_key": "news_result",
    },
    # 未來可以加：
    # "tech_analyst": {
    #     "nickname":   "技術分析師",
    #     "tools":      [get_price_chart],
    #     "persona":    "你是技術分析師，專注 K 線、均線、量價關係...",
    #     "result_key": "tech_result",
    # },
}


# ─── Specialist Agent Nodes ───────────────────────────────────────────────────
#
# LangGraph 概念：create_react_agent 建立一個完整的 ReAct 子圖
#   state_modifier = agent 的 persona（system prompt）
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
            state_modifier=cfg["persona"],  # ← 角色在這裡注入
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
    next: Literal["chip_analyst", "news_analyst", "FINISH"]
    reason: str  # 讓 LLM 說明決定理由（提高思考品質）


def supervisor_node(state: RoundtableState) -> dict:
    """主持人：根據已完成的分析，決定下一步路由。"""
    done = []
    if state.get("chip_result"):
        done.append(f"籌碼分析完成：{state['chip_result'][:150]}…")
    if state.get("news_result"):
        done.append(f"新聞分析完成：{state['news_result'][:150]}…")

    done_str = "\n".join(done) if done else "尚未開始任何分析"

    system = f"""你是投資研究圓桌會議主持人，協調各專家依序發言。

今日圓桌討論主題：
  主題 1｜{state.get('rfc_context', '未載入')}
  主題 2｜{state.get('radar_context', '未載入')}

用戶問題：{state['question']}

目前進度：
{done_str}

決策規則：
- 尚未進行籌碼分析 → chip_analyst
- 籌碼完成但尚未進行新聞分析 → news_analyst
- 兩者都完成 → FINISH（進入整合結論）"""

    decision = _llm().with_structured_output(_RouterDecision).invoke([SystemMessage(system)])
    return {"next": decision.next}


# ─── Synthesizer Node ─────────────────────────────────────────────────────────


def synthesizer_node(state: RoundtableState) -> dict:
    """主持人整合：匯總所有專家意見，給出最終結論。"""
    system = """你是投資研究圓桌會議主持人。
根據各專家意見整合最終結論，分點說明，結尾加上「資料僅供參考，不構成投資建議」。
用繁體中文回答。"""

    user = f"""今日圓桌主題：
{state.get('rfc_context', '')}
{state.get('radar_context', '')}

用戶問題：{state['question']}

【籌碼分析師】
{state.get('chip_result') or '無資料'}

【新聞分析師】
{state.get('news_result') or '無資料'}

請整合以上分析，給出圓桌會議結論。"""

    response = _llm().invoke([SystemMessage(system), HumanMessage(user)])
    return {"discussion": [_entry("主持人（整合結論）", response.content)]}


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
        g.set_entry_point("context_loader")
        g.add_edge("context_loader", "supervisor")
        g.add_conditional_edges(
            "supervisor",
            _route,
            {
                "chip_analyst": "chip_analyst",
                "news_analyst": "news_analyst",
                "FINISH": "synthesizer",
            },
        )
        g.add_edge("chip_analyst", "supervisor")
        g.add_edge("news_analyst", "supervisor")
        g.add_edge("synthesizer", END)
        _roundtable = g.compile()
    return _roundtable
#
# 圖視覺化（本機開發可用）：
#   from IPython.display import Image
#   Image(roundtable.get_graph().draw_mermaid_png())


# ─── 對外介面 ─────────────────────────────────────────────────────────────────


async def run_autonomous_roundtable() -> list[dict]:
    """每日自動啟動圓桌會議 — 各議題分開存檔。

    由 POST /trigger/roundtable 觸發，接 GitHub Action 每日排程。
    """
    date_str = datetime.now(_TW).strftime("%Y-%m-%d")
    rfc = _fetch_rfc_open()
    radar = _fetch_radar_top3()

    rfc_stocks = "、".join(rfc["stocks"][:5]) or "無持倉資料"
    radar_stocks = (
        "、".join(f"{s.get('stock_id','')} {s.get('company_name','')}" for s in radar["stocks"])
        or "無雷達資料"
    )

    t1 = await run_roundtable_for_topic(
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

    t2 = await run_roundtable_for_topic(
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

    return [t1, t2]


async def run_roundtable(user_message: str, date_str: str) -> list[dict]:
    """執行 AI 圓桌會議，回傳所有 agent 的發言。"""
    init_state: RoundtableState = {
        "question": user_message,
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

    discussion = await run_roundtable(question, date_str)

    # synthesizer 的輸出就是摘要
    summary = next(
        (m["message"] for m in reversed(discussion) if m.get("nickname") == "主持人（整合結論）"),
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
        f"你是股票分析助手，協助用戶深入了解投資研究圓桌會議分析。\n"
        f"議題：{topic.get('title', '')}\n\n"
        + "\n\n".join(ctx_parts)
        + "\n\n請根據以上背景資料用繁體中文簡潔回答。"
    )

    now = datetime.now(_TW)
    ts = now.strftime("%H:%M")
    date_out = now.strftime("%Y-%m-%d")

    response = await asyncio.to_thread(_llm().invoke, [SystemMessage(system), HumanMessage(user_message)])

    qa_entry = {
        "question": {"type": "user", "nickname": nickname, "message": user_message, "timestamp": ts, "date": date_out},
        "answer": {"type": "ai", "nickname": "AI助手", "message": response.content, "timestamp": ts, "date": date_out},
    }

    topic["qa"] = topic.get("qa", []) + [qa_entry]
    return qa_entry, topic
