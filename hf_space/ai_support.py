"""Just 1 Stock 客服 AI — RAG 問答

知識庫由 src/j1stools/ai_support.py 的 build_support_index() 一次性 embed 後存進
Qdrant 的 support_faq collection；這裡只做 runtime 查詢：

  使用者問題 → Voyage embedding → Qdrant 搜尋 support_faq → top-k 片段
            → 組 prompt（身份/合規規則 + 片段）→ Groq（沿用 ai_chat._llm）→ 回答

身份/合規規則寫死在 _ALWAYS_ON_RULES，每次都會送進 system prompt，不靠語意搜尋
命中與否決定 —— 否則使用者問法剛好沒搜到 guardrail 內容時，AI 可能講出投資建議。

Qdrant/Voyage 任何一步失敗都直接跳過 retrieval（context 留空），讓 Groq 不帶
FAQ 內容直接回答，靠 _ALWAYS_ON_RULES 裡「沒有資料就老實說不知道」的規則處理，
不另外寫忙碌訊息邏輯。
"""

from __future__ import annotations

import os

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_chat import _llm

_HF_TOKEN = os.environ.get("HF_TOKEN", "")

_VOYAGE_TOKEN = os.environ.get("VOYAGE_TOKEN", "")
_QDRANT_TOKEN = os.environ.get("QDRANT_TOKEN", "")
# QDRANT_PATH 在不同環境格式不一致（有的含 https:// 前綴，有的只是 host），
# 統一去掉 scheme 後自己補 https://，避免組出 https://https://... 連不到。
_QDRANT_PATH = os.environ.get("QDRANT_PATH", "").removeprefix("https://").removeprefix("http://")
_EMBED_MODEL_VOYAGE = "voyage-3"
_EMBED_DIM = 1024
_COLLECTION = "support_faq"
_TOP_K = 3

_ALWAYS_ON_RULES = """你是 Just 1 Stock 的 AI 客服顧問。任務是幫助新使用者快速了解網站、引導使用者找到適合功能、回答平台功能問題、回答訂閱方案問題、解釋量化策略與 AI 功能。

規則（絕對遵守）：
1. 你不提供投資建議、不保證獲利、不推薦買賣任何股票，也不回答應該買哪一檔股票。
2. Just 1 Stock 不是投顧，提供的是研究工具與數據分析服務；所有資料僅供研究參考，投資一定有風險。
3. 只能根據下面提供的知識庫內容回答問題；知識庫沒有提到的問題，要老實說不知道，並建議使用者聯繫人工客服。
4. 當使用者的問題很模糊、不知道要問什麼時（例如「我不知道要問什麼」「能幹嘛」），改成詢問：
   「請問您比較偏向哪一種需求呢？A 我是第一次來看看、B 我想找值得研究的股票、C 我想驗證策略績效、
   D 我想了解 AI 如何分析股票、E 我沒有時間研究，希望直接看重點、F 我想知道會員值不值得加入。」
   根據使用者的回答，再導向對應的功能說明。
5. 用繁體中文回答，語氣親切簡潔。"""


def _embed_query(text: str) -> list[float] | None:
    if not _VOYAGE_TOKEN:
        return None
    try:
        r = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {_VOYAGE_TOKEN}"},
            json={"input": [text], "model": _EMBED_MODEL_VOYAGE, "output_dimension": _EMBED_DIM},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[ai_support] Voyage embedding 失敗: {e}")
        return None


def _search_faq(vector: list[float]) -> list[dict]:
    if not all([_QDRANT_TOKEN, _QDRANT_PATH]):
        return []
    try:
        r = httpx.post(
            f"https://{_QDRANT_PATH}/collections/{_COLLECTION}/points/search",
            headers={"api-key": _QDRANT_TOKEN},
            json={"vector": vector, "limit": _TOP_K, "with_payload": True},
            timeout=15,
        )
        r.raise_for_status()
        return [hit["payload"] for hit in r.json().get("result", [])]
    except Exception as e:
        print(f"[ai_support] Qdrant 搜尋失敗: {e}")
        return []


def _llm_hf_primary():
    """客服優先用 HF Serverless（Qwen3-8B，免費無額度上限），失敗才 fallback 到 Groq+Cerebras。

    只在這支檔案這樣排序，不動 ai_chat.py 的共用 _llm() —— qwen provider 在 bind_tools()/
    with_structured_output() 上有已知問題（見 ai_chat.py 註解），但 ask_support() 只是
    單純的 system prompt + invoke()，沒有用到工具呼叫，不會碰到那些問題。

    這樣排序的好處：客服平常的流量不會去吃 Groq 的額度，把 Groq 額度留給圓桌會議
    （那邊沒辦法用 qwen，因為要用 bind_tools）。
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    qwen = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-8B",
            huggingfacehub_api_token=_HF_TOKEN,
            task="text-generation",
            max_new_tokens=2048,
        )
    )
    return qwen.with_fallbacks([_llm()])


def ask_support(message: str, history: list[dict] | None = None) -> str:
    """客服問答主入口。

    message : 使用者本次問題
    history : 前端帶上來的對話紀錄（可選），格式 [{"role": "user"|"assistant", "content": "..."}]
              採無狀態設計，後端不存對話紀錄，多輪對話靠前端把歷史一起傳上來。

    回傳：AI 回答的文字字串。
    """
    vector = _embed_query(message)
    hits = _search_faq(vector) if vector else []
    context = "\n\n".join(f"【{h['title']}】{h['content']}" for h in hits) if hits else "（沒有找到相關資料）"

    system = f"{_ALWAYS_ON_RULES}\n\n以下是跟使用者問題最相關的知識庫內容：\n{context}"

    messages = [SystemMessage(system)]
    for h in (history or [])[-6:]:
        cls = HumanMessage if h.get("role") == "user" else AIMessage
        messages.append(cls(h.get("content", "")))
    messages.append(HumanMessage(message))

    response = _llm_hf_primary().invoke(messages)
    return response.content
