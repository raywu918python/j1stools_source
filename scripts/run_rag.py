import os, sys
from datetime import datetime, timedelta, timezone

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync

if __name__ == "__main__":
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    print(f"[run_rag] 台灣今日：{today}（UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}）")

    # 1. 拉取需要的資料與模型
    hf_sync.pull([
        "db/price",
        "db/active_stocks",
        "db/feature_cols",
        "db/info",
        "db/margin",
        "db/ib",
        "db/news_mops",
        "models",
        "db/models",
    ])

    # 2. 執行 RAG pipeline（內部會 push rag/ 到 HF Space）
    from j1stools.rag import update_mops_index, query_sort_stock
    import news_mops

    # 更新今日 MOPS 公告（明確傳台灣今日）
    news_mops.update_mops_news(target_date=today, lookback_days=1)

    # 更新今日公告向量到 Qdrant
    update_mops_index(lookback_days=1)

    # 跑全流程，結果自動推到 HF Space
    result = query_sort_stock(use_qdrant=True, push_hf=True)

    print(f"[{today}] RAG 完成：candidates={len(result['candidates'])}，top5={len(result['top5'])}")
