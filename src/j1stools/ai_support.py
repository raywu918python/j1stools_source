"""
Just 1 Stock 客服知識庫 — ingestion
====================================

把 FAQ_CHUNKS 逐筆 embed（Voyage）後存進 Qdrant Cloud 的 support_faq collection，
供 hf_space/ai_support.py 在客服問答時做語意搜尋（RAG）。

跟 mops_news（股票新聞）共用同一個 Qdrant Cloud 帳號，但是不同 collection，
資料不會互相干擾。

身份/合規規則（不提供投資建議等）不放在這裡 — 那些規則要「每次都在」，
不能靠語意搜尋命不命中決定，所以寫死在 hf_space/ai_support.py 的 system prompt 裡。
這裡只放「使用者問了才需要查」的知識性內容。

執行方式
--------
    from j1stools.ai_support import build_support_index
    build_support_index()

環境變數
--------
    VOYAGE_TOKEN    Voyage AI embedding API token
    QDRANT_TOKEN    Qdrant Cloud API token
    QDRANT_PATH     Qdrant Cloud endpoint URL
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

QDRANT_COLLECTION = "support_faq"
_EMBED_MODEL_VOYAGE = "voyage-3"
_EMBED_DIM = 1024

FAQ_CHUNKS: list[dict] = [
    {
        "id": "002",
        "title": "第一次進站導覽",
        "content": "第一次使用 Just 1 Stock，建議按照以下順序體驗：策略研究紀錄、策略發想過程、Just Alpha 績效、RAG 雷達、AI 圓桌會議、AI 主播、LINE 通知。這樣可以最快了解平台如何運作。",
        "tags": ["新手導覽", "第一次使用", "onboarding"],
    },
    {
        "id": "003",
        "title": "平台定位",
        "content": "Just 1 Stock 是結合量化交易、RAG 語意資料庫、多 Agent AI 與自動化研究流程的投資研究平台。目標是用 AI 幫助投資人更有效率地研究市場、發現機會與驗證策略。平台不只是選股網站，更重視策略如何產生、如何驗證、如何公開追蹤。",
        "tags": ["平台介紹", "Just 1 Stock", "定位"],
    },
    {
        "id": "004",
        "title": "Just Alpha 介紹",
        "content": "Just Alpha 是平台核心量化模型，透過 AI 與量化分析找出高機率標的，並公開追蹤模型績效與研究紀錄。績效每日 13:10 更新。首頁報酬率是模型研究紀錄累積出的績效結果，僅供研究參考，不代表未來報酬。",
        "tags": ["Just Alpha", "量化模型", "績效"],
    },
    {
        "id": "005",
        "title": "Just Alpha 指標說明",
        "content": "最大回撤（MDD）代表策略歷史上從高點回落的最大幅度，數字越小代表風險控制越好。夏普比率（Sharpe Ratio）是風險報酬指標，數值越高代表承擔同樣風險下獲得更好的報酬。",
        "tags": ["MDD", "Sharpe Ratio", "夏普比率", "風險指標"],
    },
    {
        "id": "006",
        "title": "Just Alpha 策略說明與持倉",
        "content": "首頁將滑鼠移到「策略說明」可查看選股邏輯、持股數限制、資金配置、停利停損規則與風險控制機制。會員可查看完整持倉資訊，包含股票代號、納入日期、配置比例與目前績效。",
        "tags": ["策略說明", "持倉", "會員功能"],
    },
    {
        "id": "007",
        "title": "Just Alpha 歷史紀錄",
        "content": "會員可查看完整歷史案例，包含進場時間、出場時間與報酬結果。歷史紀錄可用來了解模型過去的研究成果，但不代表未來績效。",
        "tags": ["歷史紀錄", "回測", "會員功能"],
    },
    {
        "id": "008",
        "title": "RAG 雷達介紹",
        "content": "RAG 雷達利用 MOPS 公告、RAG 語意資料庫與 AI 分析模型，每天篩選值得關注的股票。RAG 是 Retrieval Augmented Generation，結合知識庫與大型語言模型，讓 AI 根據真實資料進行分析。",
        "tags": ["RAG 雷達", "RAG", "語意資料庫"],
    },
    {
        "id": "009",
        "title": "RAG 雷達選股流程",
        "content": "RAG 雷達流程包含：收集公開資訊、建立語意索引、模型篩選股票、Gemini 分析內容、產生綜合評分。分數代表 AI 對該股票研究價值與品質的綜合評估，分數越高代表越值得關注，但不代表一定會上漲。",
        "tags": ["RAG 選股", "Gemini", "AI 評分"],
    },
    {
        "id": "010",
        "title": "市場焦點",
        "content": "市場焦點會自動整理 MOPS 的重要公告，讓使用者快速掌握市場重要事件。包含股利公告、除權息、現金減資、增資、公司治理、重大投資、董事會決議等資訊。",
        "tags": ["市場焦點", "MOPS", "公告"],
    },
    {
        "id": "011",
        "title": "AI 圓桌會議",
        "content": "AI 圓桌會議是由多個 AI Agent 組成的投資討論室。每天固定討論 Just Alpha 持倉與 RAG 雷達股票。目前包含籌碼AI、新聞AI、主持人AI、散戶AI 等角色，不同角色提供不同觀點。",
        "tags": ["AI 圓桌會議", "Multi-Agent", "討論室"],
    },
    {
        "id": "012",
        "title": "AI 圓桌會議互動",
        "content": "使用者可以直接參與 AI 圓桌會議討論，也可以自己開主題，例如某檔股票、某個產業、大盤看法、新聞事件等。",
        "tags": ["AI 圓桌會議", "使用者互動", "自訂主題"],
    },
    {
        "id": "013",
        "title": "AI 主播",
        "content": "AI 主播會整理 AI 圓桌會議內容，自動生成講稿與影音摘要。內容主要來自 AI 圓桌會議、RAG 雷達分析與 Just Alpha 持倉研究。AI 主播每日更新，讓使用者不用閱讀大量內容，也能透過影片快速掌握重點。",
        "tags": ["AI 主播", "影音摘要", "每日更新"],
    },
    {
        "id": "014",
        "title": "策略發想過程",
        "content": "策略製作共有五個步驟：想出賺錢方法、蒐集資料、AI 找訊號、驗證績效、公開追蹤。公開研究流程是為了讓使用者知道策略如何建立，而不只是看到績效。",
        "tags": ["策略發想", "研究流程", "公開追蹤"],
    },
    {
        "id": "015",
        "title": "策略資料來源與 AI 找訊號",
        "content": "策略資料來源包含證交所、櫃買中心、FinMind、Yahoo Finance、公開市場資訊。AI 找訊號是指 AI 從數百種市場特徵中，尋找與未來績效有統計關聯性的訊號。",
        "tags": ["資料來源", "AI 訊號", "特徵工程"],
    },
    {
        "id": "016",
        "title": "策略研究紀錄",
        "content": "策略研究紀錄完全公開、不需付費。免費公開是為了讓使用者先了解模型，建立信任後再決定是否訂閱。看到的績效是研究紀錄，每日更新並完整保留歷史資料。",
        "tags": ["策略研究紀錄", "免費", "公開績效"],
    },
    {
        "id": "017",
        "title": "實際價格差異",
        "content": "實際價格可能與研究紀錄不同，因為市場成交價會受到流動性、波動、時間差影響。",
        "tags": ["成交價", "流動性", "價格差異"],
    },
    {
        "id": "018",
        "title": "LINE 通知",
        "content": "加入官方 LINE 並完成綁定即可使用 LINE 通知。通知時間為每日 13:10，內容包含新研究結果、持股更新與模型訊號更新。",
        "tags": ["LINE 通知", "綁定", "13:10"],
    },
    {
        "id": "019",
        "title": "訂閱方案",
        "content": "目前提供免費體驗 30 天、月方案 300 元、年方案 3000 元。免費體驗不需要信用卡，且可以完整體驗會員功能。",
        "tags": ["訂閱方案", "價格", "免費體驗"],
    },
    {
        "id": "020",
        "title": "會員功能",
        "content": "會員可以查看完整持倉、歷史研究紀錄、模型分析、回測資料與 AI 功能。適合想每天收到研究結果、查看完整持倉、查看完整歷史案例、使用 AI 分析功能或節省研究時間的使用者。",
        "tags": ["會員功能", "付費方案", "訂閱價值"],
    },
    {
        "id": "022",
        "title": "情境導覽：新手投資人",
        "content": "當使用者說「我第一次來」「我看不懂」「我要從哪裡開始」「有沒有推薦看的功能」，建議依序推薦策略研究紀錄、策略發想過程、Just Alpha、RAG 雷達、AI 圓桌會議、AI 主播。",
        "tags": ["情境導覽", "新手投資人"],
    },
    {
        "id": "023",
        "title": "情境導覽：忙碌上班族",
        "content": "當使用者說「我沒時間研究股票」「我工作很忙」「想快速知道重點」，推薦 AI 主播、LINE 通知、RAG 雷達。這三個功能最適合時間有限的投資人。",
        "tags": ["情境導覽", "忙碌上班族", "快速重點"],
    },
    {
        "id": "024",
        "title": "情境導覽：喜歡選股的人",
        "content": "當使用者說「想找股票」「今天有什麼股票可以研究」「推薦我看什麼」，建議查看 RAG 雷達、AI 圓桌會議、Just Alpha 持倉。",
        "tags": ["情境導覽", "選股", "股票研究"],
    },
    {
        "id": "025",
        "title": "情境導覽：想驗證績效的人",
        "content": "當使用者質疑「你們績效真的假的」「怎麼驗證」「有沒有回測」，推薦查看策略研究紀錄、策略發想過程與 Just Alpha。平台重視公開驗證，而不是單純展示結果。",
        "tags": ["情境導覽", "績效驗證", "回測"],
    },
    {
        "id": "026",
        "title": "情境導覽：技術派投資人",
        "content": "當使用者說「我看技術分析」「我看指標」「我看量價」，推薦策略發想過程、Just Alpha、AI 圓桌會議，協助了解模型如何從大量技術指標找出有效訊號。",
        "tags": ["情境導覽", "技術分析", "量價"],
    },
    {
        "id": "027",
        "title": "情境導覽：基本面投資人",
        "content": "當使用者說「我看財報」「我看公司基本面」「我想研究公司」，推薦 RAG 雷達、市場焦點、AI 圓桌會議，協助查看 MOPS 公告、重大事件與 AI 解讀。",
        "tags": ["情境導覽", "基本面", "財報"],
    },
    {
        "id": "028",
        "title": "情境導覽：AI 愛好者",
        "content": "當使用者問「AI 怎麼分析股票」「用什麼模型」「有什麼 AI 功能」，介紹平台包含 RAG 語意資料庫、Gemini 分析模型、Multi-Agent 系統、AI 主播、AI 圓桌會議。建議從 RAG 雷達與 AI 圓桌會議開始體驗。",
        "tags": ["情境導覽", "AI 功能", "Gemini", "Multi-Agent"],
    },
    {
        "id": "029",
        "title": "情境導覽：短線交易者",
        "content": "當使用者說「我做短線」「我想找強勢股」「我要找飆股」，推薦 RAG 雷達、市場焦點、AI 圓桌會議，用來追蹤 AI 評分高的股票、最新重大公告與市場討論方向。不得保證上漲或推薦買賣。",
        "tags": ["情境導覽", "短線", "強勢股"],
    },
    {
        "id": "030",
        "title": "情境導覽：存股族",
        "content": "當使用者說「我是存股族」「我長期投資」「我不做短線」，推薦 Just Alpha、策略研究紀錄、策略發想過程，用來觀察模型長期績效、長期勝率、風險控制與策略邏輯基礎。",
        "tags": ["情境導覽", "存股族", "長期投資"],
    },
    {
        "id": "031",
        "title": "情境導覽：工程師或量化研究者",
        "content": "當使用者說「我想知道模型怎麼做」「我是工程師」「我做量化」，推薦策略發想過程、AI 找出關鍵訊號、歷史驗證績效、公開追蹤成果，協助理解策略研究流程、特徵工程邏輯、驗證期間與方法。",
        "tags": ["情境導覽", "工程師", "量化研究"],
    },
    {
        "id": "032",
        "title": "情境導覽：是否適合加入會員",
        "content": "當使用者問「值得訂閱嗎」「我適合會員嗎」「要不要付費」，如果使用者想每天收到研究結果、查看完整持倉、查看完整歷史案例、使用 AI 分析功能或節省研究時間，會員功能會很適合。若不確定，建議先使用免費體驗方案，再決定是否訂閱。",
        "tags": ["情境導覽", "會員判斷", "訂閱轉換"],
    },
]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    import voyageai

    vo = voyageai.Client(api_key=os.environ["VOYAGE_TOKEN"])
    result = vo.embed(texts, model=_EMBED_MODEL_VOYAGE, output_dimension=_EMBED_DIM)
    return result.embeddings


def _qdrant() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_PATH"], api_key=os.environ["QDRANT_TOKEN"], timeout=60)


def build_support_index() -> None:
    """把 FAQ_CHUNKS 全部 embed 後覆寫存進 Qdrant support_faq collection。

    資料量小（30 筆），每次直接全量重建，不用像 mops_news 那樣做增量更新判斷。
    """
    client = _qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"[ai_support] 建立 collection '{QDRANT_COLLECTION}'")

    texts = [f"{c['title']}\n{c['content']}" for c in FAQ_CHUNKS]
    vectors = _embed_batch(texts)

    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={"chunk_id": c["id"], "title": c["title"], "content": c["content"], "tags": c["tags"]},
        )
        for i, (c, vec) in enumerate(zip(FAQ_CHUNKS, vectors))
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"[ai_support] 已寫入 {len(points)} 筆到 {QDRANT_COLLECTION}")


if __name__ == "__main__":
    build_support_index()
