"""
News + Margin + IB 三層 funnel：選出股票新聞給使用者看，不是交易訊號。

  Stage 1 量化篩選     : margin_lgbm 全市場打分，取最新交易日前 CANDIDATE_N 名
  Stage 2 籌碼形態過濾 : IBMarginGMM 叢集標籤，排除 EXCLUDE_CLUSTERS（可選）
  Stage 3 新聞摘要     : news_rank 對 Stage1/2 後的候選跑 LLM 評分，
                        只留有評到分的股票，依新聞分數排序取前 TOP_N 名

Stage1/2 只負責「挑出哪些股票的新聞值得看」，Stage3 才是真正的輸出：
依 Gemini 新聞分數排序，不跟量化分混合計分（這支工具的目的是整理新聞給人看，
不是產生交易訊號）。news_rank 逐股呼叫 Gemini + RSS（有延遲與 API 成本），
只能用在小名單，所以放在 funnel 最後一層，而不是用來掃全市場。
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from j1stools import gmm_classify, news_rank, parquet_db
from release.ma60_lgbm import margin_lgbm_main

_TW = timezone(timedelta(hours=8))

CANDIDATE_N = 40  # Stage 1：margin_lgbm 候選池大小
NEWS_TOP_N = 15  # Stage 3：送進 LLM 評分的候選數（API 延遲/成本限制）
TOP_N = 5  # 最終輸出檔數


def get_quant_candidates(
    stocks: list,
    st: str,
    end: str,
    candidate_n: int = CANDIDATE_N,
    use_filter: bool = True,
) -> pd.DataFrame:
    """Stage 1：margin_lgbm 全市場打分，取最新交易日的前 candidate_n 名。"""
    df = margin_lgbm_main.predict(stocks, st, end, top_n=candidate_n, use_filter=use_filter)
    if df is None or df.empty:
        print("margin_lgbm：市場狀態不佳或無資料，今日不選股")
        return pd.DataFrame(columns=["date", "stock_id", "pred_score"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    out = df[df["date"] == latest_date].sort_values("pred_score", ascending=False).reset_index(drop=True)
    print(f"Stage1 量化候選池：{latest_date.date()}　{len(out)} 檔")
    return out


def apply_gmm_filter(
    df_candidates: pd.DataFrame,
    st: str,
    end: str,
    exclude_clusters: list | None = None,
) -> pd.DataFrame:
    """Stage 2：用已訓練的 IBMarginGMM 算候選股的籌碼叢集，排除歷史達標率差的叢集。"""
    if df_candidates.empty:
        return df_candidates

    exclude = exclude_clusters if exclude_clusters is not None else gmm_classify.EXCLUDE_CLUSTERS
    clf = gmm_classify.IBMarginGMM.load(gmm_classify.BREAKOUT_GMM_MODEL_PATH)

    stock_ids = df_candidates["stock_id"].unique().tolist()
    df_feat = gmm_classify.load_data(stock_ids, st, end, min_atr_pct=None, require_complete=False)

    missing = [c for c in clf._fitted_features if c not in df_feat.columns]
    if missing:
        print(f"Stage2 GMM：缺少特徵 {missing}，跳過籌碼形態過濾")
        return df_candidates

    sub = df_feat[["date", "stock_id"] + clf._fitted_features].copy()
    sub["gmm_cluster"] = clf.predict(sub).values
    sub["gmm_max_prob"] = clf.predict_proba(sub).max(axis=1).values

    out = df_candidates.merge(sub[["date", "stock_id", "gmm_cluster", "gmm_max_prob"]], on=["date", "stock_id"], how="left")
    before = len(out)
    out = out[out["gmm_cluster"].isna() | ~out["gmm_cluster"].isin(exclude)].reset_index(drop=True)
    print(f"Stage2 GMM 叢集過濾（排除 {exclude}）：{before} → {len(out)} 檔")
    return out


def apply_news_gate(
    df_candidates: pd.DataFrame,
    news_top_n: int = NEWS_TOP_N,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Stage 3：對候選跑新聞 LLM 評分，只留有評到分的股票，依新聞分數取前 top_n 名。"""
    if df_candidates.empty:
        return df_candidates

    shortlist = df_candidates.head(news_top_n).copy()
    df_news = news_rank.rank_stocks(shortlist["stock_id"].tolist())

    out = shortlist.merge(
        df_news[["stock_id", "score", "summary", "positive_factors", "negative_factors", "news"]],
        on="stock_id",
        how="left",
    )

    before = len(out)
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    print(f"Stage3 新聞評分（僅留有評到分的股票）：{before} → {len(out)} 檔")

    return out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def select_top10(
    stocks: list | None = None,
    st: str | None = None,
    end: str = "2099-01-01",
    candidate_n: int = CANDIDATE_N,
    news_top_n: int = NEWS_TOP_N,
    top_n: int = TOP_N,
    use_filter: bool = True,
    use_gmm: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """三層 funnel：margin_lgbm 全市場排名 → GMM 籌碼形態過濾 → news_rank 新聞把關。

    verbose=True 時，每一層算完就把該層的候選表印出來（方便確認某層篩掉/留下了哪些股票，
    不用等到 Stage3 打完 Gemini 才知道哪裡出問題）。
    """
    if stocks is None:
        stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    if st is None:
        st = (datetime.now(_TW) - timedelta(days=10)).strftime("%Y-%m-%d")

    candidates = get_quant_candidates(stocks, st, end, candidate_n=candidate_n, use_filter=use_filter)
    if candidates.empty:
        return candidates
    if verbose:
        print("── Stage1 候選（margin_lgbm）──")
        print(candidates[["date", "stock_id", "pred_score"]])

    if use_gmm:
        candidates = apply_gmm_filter(candidates, st, end)
        if verbose:
            cols = [c for c in ["stock_id", "pred_score", "gmm_cluster", "gmm_max_prob"] if c in candidates.columns]
            print("── Stage2 候選（GMM 過濾後）──")
            print(candidates[cols])

    if verbose:
        n_news = min(len(candidates), news_top_n)
        print(f"── Stage3 將對前 {n_news} 檔呼叫 news_rank（Gemini）──")
        print(candidates.head(news_top_n)[["stock_id"]].to_string(index=False))

    return apply_news_gate(candidates, news_top_n=news_top_n, top_n=top_n)


if __name__ == "__main__":
    df = select_top10()
    summary_cols = [c for c in ["stock_id", "score"] if c in df.columns]
    print(df[summary_cols])

    for _, row in df.iterrows():
        print(f"\n{'='*60}")
        print(f"{row['stock_id']}　Gemini分數：{row.get('score', float('nan'))}")
        if "summary" in row:
            print(f"摘要：{row['summary']}")
        if "news" in row:
            print(f"新聞：{row['news']}")

    df.to_csv("news_margin_ib_top10.csv", index=False, encoding="utf-8-sig")
