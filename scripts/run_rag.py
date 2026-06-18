import os, sys
from datetime import datetime, timedelta, timezone

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync

if __name__ == "__main__":
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    print(f"[run_rag] 台灣今日：{today}（UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}）")

    # 1. 拉取需要的資料與模型
    hf_sync.pull(
        [
            "db/price",
            "db/active_stocks",
            "db/feature_cols",
            "db/info",
            "db/margin",
            "db/ib",
            "db/day_trade",
            "db/news_mops",
            "models",
        ]
    )

    # HF 上 models/ 與 db/ 同層，本地程式碼預期 db/models/，補一個 symlink
    db_models = "db/models"
    if not os.path.exists(db_models):
        os.makedirs("db", exist_ok=True)
        os.symlink(os.path.abspath("models"), os.path.abspath(db_models))

    from j1stools.rag import update_mops_index, query_sort_stock, query_any_string, upload_to_hf
    import news_mops

    # 更新今日 MOPS 公告（明確傳台灣今日）
    news_mops.update_mops_news(target_date=today, lookback_days=1)

    # 更新今日公告向量到 Qdrant
    update_mops_index(lookback_days=1)

    # AI雷達
    radar = query_sort_stock(use_qdrant=True)

    # 市場焦點
    focus = query_any_string(
        query="除息 除權 股利分配 配股配息 重大事件",
        # date_from="2020-01-01",
        # date_to="2020-06-17",
        # lookback_days=180,
        # query="重大事件",
        top_k=30,
    )

    # 組合並上傳
    output = {
        "date": today,
        "updated_at": datetime.now(_TW).isoformat(timespec="seconds"),
        "AI雷達": radar,
        "市場焦點": focus,
    }
    # print(output)
    upload_to_hf(output)

    print(
        f"[{today}] RAG 完成：candidates={len(radar['candidates'])}，top5={len(radar['top5'])}，市場焦點={len(focus)}"
    )
