"""
MOPS 公告地雷過濾 pipeline — 完整 RAG 架構
============================================

RAG 三步驟
----------
  Corpus     = 所有歷史 MOPS 公告向量，存於 Qdrant Cloud
  Retrieval  = 用風險 query 或股票 ID 從 Qdrant 取回相關公告
  Generation = Gemini 讀取取回的公告作為 context 後評分

Pipeline 流程
-------------
  Stage 1  margin_lgbm 全市場打分，取最新交易日前 CANDIDATE_N 名
             → 量化篩選，過濾市場整體狀態差的日子
  Stage 2  GMM 籌碼形態過濾，排除 EXCLUDE_CLUSTERS
             → 排除籌碼形態差的叢集
  Stage 3a 地雷排除（二選一）
             MODE A（預設）：關鍵字黑名單，從 parquet 掃描，無 API 成本
             MODE B        ：Qdrant 語意搜尋，embed 一次 risk query 掃全市場
  Stage 3b Augmented Generation
             Retrieve：從 MOPS parquet 取回各股近期真實公告
             Generate：把公告當 context 送 Gemini LLM 評分，取前 TOP_N 名

執行方式
--------
  一次性準備：build_mops_index()   — 全歷史 802,135 筆 → Qdrant（用 Ollama）
  每日排程  ：update_mops_index()  — 當天新公告 upsert → Qdrant（用 Gemini）
              run()                — 執行完整 pipeline，回傳最終選股 DataFrame

工具與依賴
----------
  qdrant-client      Qdrant Cloud 向量資料庫，存放 MOPS 公告向量
                     collection: mops_news，1024 維，int8 scalar quantization
  google-genai       Gemini API
                       embedding : gemini-embedding-2（output_dimensionality=1024）
                       generation: gemini-2.5-flash（LLM 評分）
  Ollama             本機推理伺服器（初始建索引用）
                       model: mxbai-embed-large（1024 維，多語言）
  margin_lgbm_main   Stage 1 量化打分（release.ma60_lgbm）
  gmm_classify       Stage 2 GMM 籌碼叢集模型（j1stools）
  parquet_db         取得全市場啟用股票清單（j1stools）
  news_mops parquet  MOPS 公告原始資料來源（db/news_mops/*.parquet）
                     由 news_mops.download_mops_news() 下載產生

環境變數
--------
  GEMINI_API_KEY  Gemini API key（embedding + generation）
  QDRANT_TOKEN    Qdrant Cloud API token
  QDRANT_PATH     Qdrant Cloud endpoint URL
  EMBED_BACKEND   ollama（初始建立）或 gemini（雲端每日更新，預設）
  OLLAMA_URL      Ollama 伺服器位址（預設 http://localhost:11434）
"""

from __future__ import annotations

import glob
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ServerError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from j1stools import gmm_classify, parquet_db
from release.ma60_lgbm import margin_lgbm_main

load_dotenv()

_TW = timezone(timedelta(hours=8))

CANDIDATE_N = 40
NEWS_TOP_N = 15
TOP_N = 5
MOPS_LOOKBACK_DAYS = 10
MOPS_PARQUET_DIR = "db/news_mops"
QDRANT_COLLECTION = "mops_news"
# Embedding backend：
#   EMBED_BACKEND=ollama  → 本機 mxbai-embed-large（初始建立用）
#   EMBED_BACKEND=gemini  → Gemini Cloud（雲端每日更新用，預設）
_EMBED_BACKEND = os.getenv("EMBED_BACKEND", "gemini")
_EMBED_MODEL_GEMINI = "gemini-embedding-2"
_EMBED_MODEL_OLLAMA = "mxbai-embed-large"
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_EMBED_DIM = 1024  # mxbai 固定 1024；Gemini 截至 1024 → 兩邊一致
_UPSERT_BATCH = 200

_RISK_KEYWORDS: set[str] = {
    "重大訊息",
    "財報重編",
    "掏空",
    "背信",
    "偽造",
    "停業",
    "解散",
    "下市",
    "違約",
    "票據退票",
    "強制執行",
    "假扣押",
    "假處分",
    "財務困難",
    "延遲申報",
    "更換會計師",
    "內控缺失",
    "重大異常",
    "減資",
    "清算",
    "破產",
    "重整",
    "私募",
}

_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _qdrant() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_PATH"], api_key=os.environ["QDRANT_TOKEN"])


def _log(stage: str, msg: str) -> None:
    ts = datetime.now(_TW).strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}")


def _mops_id_to_uuid(mops_id: str) -> str:
    """mops_id → 固定 UUID，確保同一筆公告重複 upsert 不會重複存入。"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, mops_id))


def _embed_one_gemini(text: str) -> list[float]:
    for attempt in range(5):
        try:
            result = _gemini.models.embed_content(
                model=_EMBED_MODEL_GEMINI,
                contents=text,
                config=genai_types.EmbedContentConfig(output_dimensionality=_EMBED_DIM),
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt == 4:
                raise
            wait = 30 * (2**attempt)
            _log("EMBED", f"Gemini 429，等 {wait}s（retry {attempt+1}/5）")
            time.sleep(wait)


def _embed_one_ollama(text: str) -> list[float]:
    import requests as _req

    r = _req.post(
        f"{_OLLAMA_URL}/api/embeddings",
        json={"model": _EMBED_MODEL_OLLAMA, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _embed_batch(texts: list[str], sleep_sec: float = 0.05) -> list[list[float]]:
    """
    EMBED_BACKEND=ollama → 本機 mxbai-embed-large（初始建立，快）
    EMBED_BACKEND=gemini → Gemini Cloud（雲端每日更新，預設）
    兩者都輸出 1024 維，可共用同一個 Qdrant collection。
    """
    embed_fn = _embed_one_ollama if _EMBED_BACKEND == "ollama" else _embed_one_gemini
    vectors: list[list[float]] = []
    for i, text in enumerate(texts):
        vectors.append(embed_fn(text))
        if (i + 1) % 100 == 0:
            _log("EMBED", f"  進度 {i+1}/{len(texts)}")
        time.sleep(sleep_sec)
    return vectors


# ── 0. 一次性：把歷史 MOPS 全部 embed 存進 Qdrant ────────────────────────────


def _fetch_existing_ids(client: QdrantClient, ids: list[str]) -> set[str]:
    """查詢 Qdrant 哪些 UUID 已存在（分批查，每批 1000）。"""
    existing: set[str] = set()
    for i in range(0, len(ids), 1000):
        chunk = ids[i : i + 1000]
        results = client.retrieve(collection_name=QDRANT_COLLECTION, ids=chunk, with_payload=False, with_vectors=False)
        existing.update(str(r.id) for r in results)
    return existing


def build_mops_index(days: int | None = None) -> None:
    """
    把 db/news_mops/*.parquet 的公告 embed 後存進 Qdrant。
    - days=None → 全部歷史（802,135 筆，用 Ollama 跑約 4-5 小時）
    - days=30   → 近 30 天（~9,000 筆）
    支援中斷續跑：已在 Qdrant 的 UUID 自動跳過，不重複 embed。
    每 {_UPSERT_BATCH} 筆 embed 完就立即 upsert，中斷最多損失一批進度。
    """
    _log("BUILD", f"=== build_mops_index 開始（近 {days if days else '全部'} 天）===")
    _log("BUILD", f"Embedding backend: {_EMBED_BACKEND}")

    files = glob.glob(os.path.join(MOPS_PARQUET_DIR, "*.parquet"))
    if not files:
        _log("BUILD", f"找不到 {MOPS_PARQUET_DIR}/*.parquet，請先執行 news_mops.download_mops_news()")
        return

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if days is not None:
        cutoff = (datetime.now(_TW) - timedelta(days=days)).strftime("%Y-%m-%d")
        df = df[df["date"] >= cutoff]
    df = df.dropna(subset=["title", "mops_id"]).drop_duplicates(subset=["mops_id"])
    _log("BUILD", f"讀取 {len(files)} 個 parquet，篩選後共 {len(df)} 筆公告")

    client = _qdrant()
    _ensure_collection(client)

    # 查哪些已存在，跳過（支援中斷續跑）
    all_uuids = [_mops_id_to_uuid(mid) for mid in df["mops_id"]]
    _log("BUILD", "查詢 Qdrant 已存在的向量（可能需要幾秒）...")
    existing = _fetch_existing_ids(client, all_uuids)
    _log("BUILD", f"已存在 {len(existing)} 筆，跳過；待 embed {len(df) - len(existing)} 筆")

    rows = df.to_dict("records")
    batch_rows: list[dict] = []
    done = 0
    skipped = 0

    for row, uid in zip(rows, all_uuids):
        if uid in existing:
            skipped += 1
            continue
        batch_rows.append((row, uid))

        if len(batch_rows) >= _UPSERT_BATCH:
            titles = [r["title"] for r, _ in batch_rows]
            vectors = _embed_batch(titles, sleep_sec=0.05)
            points = [
                PointStruct(
                    id=u,
                    vector=vec,
                    payload={
                        "stock_id": str(r["stock_id"]),
                        "date": str(r["date"]),
                        "title": str(r["title"]),
                        "company_name": str(r.get("company_name", "")),
                    },
                )
                for (r, u), vec in zip(batch_rows, vectors)
            ]
            done += len(points)
            total_target = len(df) - len(existing)
            _log("BUILD", f"進度 {done}/{total_target}（跳過 {skipped}）批次 upsert 完成")
            batch_rows = []

    # 最後一批
    if batch_rows:
        titles = [r["title"] for r, _ in batch_rows]
        vectors = _embed_batch(titles, sleep_sec=0.05)
        points = [
            PointStruct(
                id=u,
                vector=vec,
                payload={
                    "stock_id": str(r["stock_id"]),
                    "date": str(r["date"]),
                    "title": str(r["title"]),
                    "company_name": str(r.get("company_name", "")),
                },
            )
            for (r, u), vec in zip(batch_rows, vectors)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        done += len(points)
        _log("BUILD", f"進度 {done}/{len(df) - len(existing)}（最後一批）upsert 完成")

    _log("BUILD", f"=== build_mops_index 完成：新增 {done} 筆，跳過 {skipped} 筆 ===")


# ── 每日更新：把新公告 upsert 進 Qdrant ──────────────────────────────────────


def update_mops_index(lookback_days: int = MOPS_LOOKBACK_DAYS) -> None:
    """
    把最近 lookback_days 天的新公告 embed 後 upsert 進 Qdrant。
    應在 news_mops.update_mops_news() 之後執行。
    """
    _log("UPDATE", f"=== update_mops_index 開始（近 {lookback_days} 天）===")
    files = glob.glob(os.path.join(MOPS_PARQUET_DIR, "*.parquet"))
    if not files:
        _log("UPDATE", "找不到 MOPS parquet，跳過")
        return

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cutoff = (datetime.now(_TW) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = df[df["date"] >= cutoff].dropna(subset=["title", "mops_id"]).drop_duplicates(subset=["mops_id"])
    _log("UPDATE", f"近 {lookback_days} 天公告：{len(df)} 筆")

    if df.empty:
        _log("UPDATE", "無新公告，結束")
        return

    client = _qdrant()
    _ensure_collection(client)

    vectors = _embed_batch(df["title"].tolist())
    points = [
        PointStruct(
            id=_mops_id_to_uuid(row["mops_id"]),
            vector=vec,
            payload={
                "stock_id": str(row["stock_id"]),
                "date": str(row["date"]),
                "title": str(row["title"]),
                "company_name": str(row.get("company_name", "")),
            },
        )
        for (_, row), vec in zip(df.iterrows(), vectors)
    ]

    for i in range(0, len(points), _UPSERT_BATCH):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i : i + _UPSERT_BATCH])

    _log("UPDATE", f"upsert 完成，{len(points)} 筆進入 Qdrant")


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        sample = _embed_batch(["test"])
        dim = len(sample[0])
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,  # 量化向量常駐記憶體，加速查詢
                )
            ),
        )
        _log("BUILD", f"建立 collection '{QDRANT_COLLECTION}'（dim={dim}，int8 量化，2.46GB→0.62GB）")


# ── Stage 1 ───────────────────────────────────────────────────────────────────


def _stage1(stocks: list[str], st: str, end: str) -> pd.DataFrame:
    """
    量化候選池篩選。

    工具：margin_lgbm_main.predict()（release.ma60_lgbm）
    輸入：全市場股票清單、日期範圍
    輸出：最新交易日前 CANDIDATE_N 名，按 pred_score 降冪排列

    use_filter=True 代表在市場整體狀態差時（如大盤崩跌）直接回傳空，
    避免在系統性風險期間強選股。
    """
    _log("STAGE1", f"margin_lgbm 全市場打分，股票池 {len(stocks)} 檔，日期 {st} ~ {end}")
    df = margin_lgbm_main.predict(stocks, st, end, top_n=CANDIDATE_N, use_filter=True)
    if df is None or df.empty:
        _log("STAGE1", "市場狀態不佳（use_filter 擋掉），今日無候選，結束")
        return pd.DataFrame(columns=["date", "stock_id", "pred_score"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest = df["date"].max()
    out = df[df["date"] == latest].sort_values("pred_score", ascending=False).reset_index(drop=True)
    _log("STAGE1", f"最新交易日 {latest.date()}，取前 {CANDIDATE_N} 名 → 候選池 {len(out)} 檔")
    return out


# ── Stage 2 ───────────────────────────────────────────────────────────────────


def _stage2(df: pd.DataFrame, st: str, end: str) -> pd.DataFrame:
    """
    GMM 籌碼形態過濾。

    工具：gmm_classify.IBMarginGMM（j1stools），模型從 BREAKOUT_GMM_MODEL_PATH 載入
    輸入：Stage 1 候選股 DataFrame
    輸出：排除 EXCLUDE_CLUSTERS 後的候選股

    叢集中歷史勝率差（爆量失敗、外資連買後回吐等）的 cluster 在訓練時已標記為
    EXCLUDE_CLUSTERS，此處直接排除，不需再計算勝率。
    若特徵不完整則跳過此 Stage，維持 Stage 1 結果。
    """
    _log("STAGE2", f"GMM 籌碼形態過濾，排除叢集 {gmm_classify.EXCLUDE_CLUSTERS}")
    if df.empty:
        return df
    clf = gmm_classify.IBMarginGMM.load(gmm_classify.BREAKOUT_GMM_MODEL_PATH)
    _log("STAGE2", f"載入 GMM 模型，特徵數 {len(clf._fitted_features)}")
    df_feat = gmm_classify.load_data(df["stock_id"].tolist(), st, end, min_atr_pct=None, require_complete=False)
    missing = [c for c in clf._fitted_features if c not in df_feat.columns]
    if missing:
        _log("STAGE2", f"缺少特徵 {missing}，跳過 GMM 過濾")
        return df
    sub = df_feat[["date", "stock_id"] + clf._fitted_features].copy()
    sub["gmm_cluster"] = clf.predict(sub).values
    out = df.merge(sub[["date", "stock_id", "gmm_cluster"]], on=["date", "stock_id"], how="left")
    before = len(out)
    out = out[out["gmm_cluster"].isna() | ~out["gmm_cluster"].isin(gmm_classify.EXCLUDE_CLUSTERS)].reset_index(
        drop=True
    )
    _log("STAGE2", f"過濾完成：{before} → {len(out)} 檔（排除 {before - len(out)} 筆壞叢集）")
    return out


# ── Stage 3a：地雷排除（兩模式可切換）──────────────────────────────────────
#
# MODE A（預設）：關鍵字 — 快速、無 API 成本，但只能精確比對
# MODE B        ：Qdrant  — 語意搜尋，需先執行 update_mops_index() 累積向量
#
# MODE B 的優勢：
#   不需要知道「要查哪支股票」，直接用 risk query 掃全市場近期公告
#   → 找出語意上像「財務困難/破產/掏空」的公告屬於哪些股票
#   → 跟候選股交集 → 排除
#
# 切換方式：_stage3a(df, use_qdrant=True)

_RISK_QUERY = "財務困難 破產 掏空 下市 強制執行 票據退票 重整 內控缺失"
_QDRANT_RISK_THRESHOLD = 0.75
_QDRANT_RISK_TOP_K = 50  # 從全市場近期公告取前 50 筆最像地雷的


def _to_stock_titles(mops: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for sid, grp in mops.groupby("stock_id"):
        titles = grp["title"].dropna().tolist()
        if titles:
            result[str(sid)] = titles
    return result


def _stage3a_keyword(df: pd.DataFrame) -> pd.DataFrame:
    """MODE A：關鍵字黑名單預篩，從 parquet 讀取（無 API 成本）。"""
    _log("STAGE3A", f"[MODE A 關鍵字] 輸入 {len(df)} 檔")
    mops = _load_mops(df["stock_id"].tolist())
    if mops.empty:
        _log("STAGE3A", "無 MOPS 公告，跳過")
        return df

    kw_exclude: set[str] = set()
    for sid, grp in mops.groupby("stock_id"):
        for title in grp["title"].dropna():
            if any(k in title for k in _RISK_KEYWORDS):
                kw_exclude.add(str(sid))
                _log("STAGE3A", f"  → {sid} 命中：{title[:40]}")
                break

    before = len(df)
    out = df[~df["stock_id"].isin(kw_exclude)].reset_index(drop=True)
    _log("STAGE3A", f"完成：{before} → {len(out)} 檔（排除 {before - len(out)} 筆）")
    return out


def _stage3a_qdrant(df: pd.DataFrame) -> pd.DataFrame:
    """
    MODE B：Qdrant 語意搜尋排除地雷股。

    流程：
      1. 把 _RISK_QUERY embed 成向量（1 次 API 呼叫）
      2. 在 Qdrant 全市場近期公告裡找語意最像地雷的前 K 筆
         （不限定 stock_id，掃全市場）
      3. 取出這些公告的 stock_id → 跟候選股交集 → 排除

    需要先執行 update_mops_index() 讓 Qdrant 有資料。
    """
    _log("STAGE3A", f"[MODE B Qdrant] 輸入 {len(df)} 檔")
    candidate_ids = set(df["stock_id"].tolist())
    cutoff = (datetime.now(_TW) - timedelta(days=MOPS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    # Step 1：embed risk query（1 次 API 呼叫）
    _log("STAGE3A", f"embed risk query：「{_RISK_QUERY}」")
    risk_vec = _embed_batch([_RISK_QUERY])[0]

    # Step 2：Qdrant 語意搜尋（掃全市場近期公告，不限 stock_id）
    _log("STAGE3A", f"Qdrant 語意搜尋，取前 {_QDRANT_RISK_TOP_K} 筆，門檻 {_QDRANT_RISK_THRESHOLD}")
    client = _qdrant()
    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=risk_vec,
        query_filter=Filter(must=[FieldCondition(key="date", range=Range(gte=cutoff))]),
        limit=_QDRANT_RISK_TOP_K,
        with_payload=True,
    )

    # Step 3：找出命中的候選股
    qdrant_exclude: set[str] = set()
    for hit in hits:
        if hit.score < _QDRANT_RISK_THRESHOLD:
            continue
        sid = hit.payload.get("stock_id", "")
        title = hit.payload.get("title", "")
        if sid in candidate_ids:
            qdrant_exclude.add(sid)
            _log("STAGE3A", f"  → {sid} 命中（score={hit.score:.3f}）：{title[:40]}")

    before = len(df)
    out = df[~df["stock_id"].isin(qdrant_exclude)].reset_index(drop=True)
    _log("STAGE3A", f"完成：{before} → {len(out)} 檔（排除 {before - len(out)} 筆）")
    return out


def _stage3a(df: pd.DataFrame, use_qdrant: bool = False) -> pd.DataFrame:
    """地雷排除：use_qdrant=False 用關鍵字，use_qdrant=True 用 Qdrant 語意搜尋。"""
    if use_qdrant:
        return _stage3a_qdrant(df)
    return _stage3a_keyword(df)


# ── Stage 3b：parquet 取回公告 → Gemini 評分（Augmented Generation）─────────


def _generate_with_retry(prompt: str, model: str = "gemini-2.5-flash", max_retries: int = 5, base_wait: int = 15):
    for attempt in range(max_retries):
        try:
            return _gemini.models.generate_content(model=model, contents=prompt)
        except ServerError as e:
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (2**attempt)
            _log("GEMINI", f"503（{e}），{wait}s 後重試 ({attempt + 1}/{max_retries})")
            time.sleep(wait)


def _stage3b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augmented Generation：
    1. Retrieve — 從 parquet 取回各股近期真實公告
    2. Generate  — 把公告當 context 送給 Gemini 評分
    """
    shortlist = df.head(NEWS_TOP_N).copy()
    _log("STAGE3B", f"LLM 評分，送入 {len(shortlist)} 檔 → 最終取前 {TOP_N} 名")
    if shortlist.empty:
        return shortlist

    # Retrieval：從 parquet 取回各股公告作為 context
    _log("STAGE3B", "Retrieval：從 MOPS parquet 取回各股近期公告...")
    mops = _load_mops(shortlist["stock_id"].tolist())
    stock_titles = _to_stock_titles(mops) if not mops.empty else {}

    for sid in shortlist["stock_id"]:
        n = len(stock_titles.get(sid, []))
        _log("STAGE3B", f"  {sid}：{n} 筆公告")

    if not stock_titles:
        _log("STAGE3B", "無 MOPS 公告，Gemini 將以空資料評分")

    # Generation：把取回的公告當 context 送給 Gemini
    prompt = f"""你是台股公告分析助理。
以下是各股票近期在 MOPS（公開資訊觀測站）的真實公告標題。
請根據這些公告評估每支股票的投資風險（score 0~100，100=完全無風險，0=極高風險）。

評分規則：
1. 只根據提供的公告判斷，不使用自身記憶
2. 財務困難/法律糾紛/減資/下市相關公告大幅扣分
3. 常規性公告（股東會、法說會、股利）不影響分數
4. 無公告的股票給 70 分（資訊不足，中性）
5. 只輸出 JSON，不加 markdown

輸出格式：
{{
  "results": [
    {{"stock_id": "xxxx", "score": 85, "reason": "一句話說明評分依據"}}
  ]
}}

各股公告資料（來源：MOPS 近 {MOPS_LOOKBACK_DAYS} 天）：
{json.dumps(stock_titles, ensure_ascii=False)}
"""

    _log("STAGE3B", f"Generation：呼叫 gemini-2.5-flash，context = {len(stock_titles)} 股公告...")
    response = _generate_with_retry(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    df_score = pd.DataFrame(json.loads(text)["results"])
    df_score["score"] = pd.to_numeric(df_score["score"], errors="coerce")

    out = shortlist.merge(df_score[["stock_id", "score", "reason"]], on="stock_id", how="left")
    out = out.dropna(subset=["score"]).sort_values("score", ascending=False).head(TOP_N).reset_index(drop=True)

    _log("STAGE3B", f"評分完成，最終輸出 {len(out)} 檔：")
    for _, row in out.iterrows():
        _log("STAGE3B", f"  #{int(row.name)+1} {row['stock_id']}  score={row['score']}  {row.get('reason', '')}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    stocks: list[str] | None = None,
    st: str | None = None,
    end: str = "2099-01-01",
    use_qdrant: bool = False,  # True = Qdrant 語意搜尋；False = 關鍵字（預設）
) -> pd.DataFrame:
    """
    執行完整 MOPS RAG pipeline，回傳最終選股結果。

    Stage 1  → margin_lgbm 量化打分，取前 CANDIDATE_N 名
    Stage 2  → GMM 排除籌碼形態差的叢集
    Stage 3a → 地雷排除（use_qdrant=False: 關鍵字 / True: Qdrant 語意搜尋）
    Stage 3b → Gemini LLM 讀公告評分，取前 TOP_N 名

    參數
    ----
    stocks     : 指定股票清單；None 代表全市場（排除 ETF）
    st         : 開始日期；None 代表今天往前 10 天
    end        : 結束日期
    use_qdrant : Stage 3a 是否使用 Qdrant 語意搜尋
                 True  → 需先執行 build_mops_index() / update_mops_index()
                 False → 直接從 parquet 關鍵字比對，無需 Qdrant 有資料

    回傳
    ----
    DataFrame，欄位：stock_id, pred_score, gmm_cluster, score, reason
    任一 Stage 無結果則回傳空 DataFrame。
    """
    if stocks is None:
        stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    if st is None:
        st = (datetime.now(_TW) - timedelta(days=10)).strftime("%Y-%m-%d")

    mode = "Qdrant 語意搜尋" if use_qdrant else "關鍵字"
    _log("RUN", f"=== MOPS RAG pipeline 開始（Stage3a={mode}）===  股票池 {len(stocks)} 檔")

    candidates = _stage1(stocks, st, end)
    if candidates.empty:
        return candidates

    candidates = _stage2(candidates, st, end)
    if candidates.empty:
        return candidates

    candidates = _stage3a(candidates, use_qdrant=use_qdrant)
    if candidates.empty:
        return candidates

    result = _stage3b(candidates)
    _log("RUN", "=== pipeline 完成 ===")
    return result


def test():
    import glob, pandas as pd

    files = glob.glob("db/news_mops/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files])
    print(f"共 {len(df)} 筆，{df['date'].min()} ～ {df['date'].max()}")


if __name__ == "__main__":
    # test()
    build_mops_index()

    # df = run()
    # if df.empty:
    #     print("無結果")
    # else:
    #     print(df[["stock_id", "score", "reason"]])
    #     df.to_csv("rag_top.csv", index=False, encoding="utf-8-sig")
