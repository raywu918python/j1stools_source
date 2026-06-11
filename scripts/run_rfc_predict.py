import os, sys
from datetime import datetime, timedelta, timezone

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_utils import drop_na_inf
from j1stools import data_filter, feature_builder, hf_sync, label_builder, parquet_db


def prepare_data(stocks, st, end, use_ib=False):
    df = parquet_db.query_price(stocks, st, end)
    print("filter.before:", df.shape)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    if use_ib:
        from j1stools.rfc_main import _add_ib_features

        df = _add_ib_features(df, stocks, st, end)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    print("filter.after:", df.shape)
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)
    return df


def batter_predict(model, xtest):
    yproba = model.predict_proba(xtest)
    dfyproba = pd.DataFrame(yproba, index=xtest.index)
    dfyproba.reset_index(drop=False, inplace=True)
    dfyproba.columns = dfyproba.columns.astype(str)
    return dfyproba


def predict(stocks, st, end, use_news_filter=False):
    model = joblib.load(f"models/{MODEL_NAME}.joblib")
    warmup_st = (pd.Timestamp(st) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    df = prepare_data(stocks, warmup_st, end, use_ib=USE_IB_FEATURES)
    _, x, _, y = rfc_split_date(df, is_gen_test=True, is_gen_train=False, trainging_idx=0)
    x = x[[col for col in x.columns if col.startswith("f_")]] if x is not None else None
    x, y = drop_na_inf(x, y)
    expected_features = model.feature_names_in_
    x = x[expected_features]
    result = batter_predict(model, x)
    result["date"] = pd.to_datetime(result["date"])
    result = result[result["date"] >= pd.Timestamp(st)].reset_index(drop=True)
    if use_news_filter:
        today = datetime.now(_TW).strftime("%Y-%m-%d")
        result = _apply_news_filter(result, today)
    return result


USE_IB_FEATURES = os.getenv("USE_IB_FEATURES", "false").lower() == "true"
MODEL_NAME = "rfc_macd_ib_6xx" if USE_IB_FEATURES else "rfc_macd_6xx"
USE_NEWS_FILTER = False  # False → 跳過新聞過濾，直接用模型分數
NEWS_SCORE_MIN = 50  # 新聞分數低於此值的股票從今日預測中剔除
RFC_THRESHOLD = 0.6  # 送進 news_rank 的候選門檻（與 backtest 一致）
NEWS_CANDIDATE_LIMIT = 20  # 最多送這幾支進 news_rank（取 prob 最高的前 N）


def _apply_news_filter(df: pd.DataFrame, today: str) -> pd.DataFrame:
    from j1stools.news_rank import rank_stocks as rank_by_news

    today_ts = pd.Timestamp(today)
    today_mask = df["date"] == today_ts
    candidates = df[today_mask & (df["1"] >= RFC_THRESHOLD)].nlargest(NEWS_CANDIDATE_LIMIT, "1")["stock_id"].tolist()

    if not candidates:
        return df

    news_df = rank_by_news(candidates)
    keep = set(news_df[news_df["score"] >= NEWS_SCORE_MIN]["stock_id"])
    dropped = set(candidates) - keep
    print(f"[news filter] {len(candidates)} 候選 → {len(keep)} 通過 (score ≥ {NEWS_SCORE_MIN})，剔除 {sorted(dropped)}")

    drop_mask = today_mask & (~df["stock_id"].isin(keep))
    return df[~drop_mask].reset_index(drop=True)


if __name__ == "__main__":
    pull_dirs = ["db/price", "db/active_stocks", "db/feature_cols", "db/info", "models"]
    if USE_IB_FEATURES:
        pull_dirs.append("db/ib")
    hf_sync.pull(pull_dirs)

    stocks = parquet_db.activate_stocks()
    stocks = list(set(stocks) - {"0050", "0052", "0056"})

    today = datetime.now(_TW).strftime("%Y-%m-%d")
    st = (datetime.now(_TW) - timedelta(days=200)).strftime("%Y-%m-%d")

    df = predict(stocks, st, "2099-01-01", use_news_filter=USE_NEWS_FILTER)

    if df is not None and not df.empty:

        out_dir = f"db/predictions/{MODEL_NAME}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/pred_{today}.parquet"
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_parquet(out_path, index=False)
        print(f"saved {len(df)} rows → {out_path}")
        hf_sync.push([f"db/predictions/{MODEL_NAME}"])
    else:
        print("今日無預測結果")
