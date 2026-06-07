import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from j1stools import parquet_db
from j1stools.neglected_rfc import RFC_FEATURES, EnsembleModel, load_ensemble
from j1stools.neglected_stock_classify import NeglectedGMM, load_neglected_data

MODEL_NAME = "neglected_ensemble"
MIN_ATR_PCT = 0.02
WARMUP_DAYS = 60  # 滾動特徵需要的暖機天數


def predict(stocks: list, st: str, end: str = "2099-01-01") -> pd.DataFrame:
    """
    產生外資忽視股 XGB 信號。

    Returns
    -------
    DataFrame: date, stock_id, 0, 1
        「1」欄位為 10日內達到 +10% 的機率，供 backtest_platform 使用。
    """
    ensemble = load_ensemble()
    clf_gmm = NeglectedGMM.load()

    # 暖機：多載入 WARMUP_DAYS 天讓滾動特徵穩定
    warmup_st = (pd.Timestamp(st) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    df = load_neglected_data(stocks, warmup_st, end, min_atr_pct=MIN_ATR_PCT)
    df["cluster"] = clf_gmm.predict(df)

    # GMM 軟機率（build_dataset 流程一致）
    proba_gmm = clf_gmm.predict_proba(df)
    df = pd.concat([df, proba_gmm], axis=1)

    avail = [c for c in RFC_FEATURES if c in df.columns]
    X = df[avail].fillna(0.5).values
    prob = ensemble.predict_proba(X)  # (N, 3): 0=盤整, 1=停損, 2=達標

    # XGB 三類輸出：0=盤整, 1=停損, 2=達標，直接對應 backtest_platform 預期欄位
    result = df[["date", "stock_id"]].copy()
    result["0"] = prob[:, 0]
    result["1"] = prob[:, 1]
    result["2"] = prob[:, 2]
    result["date"] = pd.to_datetime(result["date"])

    # 過濾掉暖機期
    result = result[result["date"] >= pd.Timestamp(st)].reset_index(drop=True)
    print(f"XGB 信號：{len(result):,} 筆  日期範圍 {result['date'].min().date()} ~ {result['date'].max().date()}")
    return result


if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone

    _TW = timezone(timedelta(hours=8))
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    st = (datetime.now(_TW) - timedelta(days=15)).strftime("%Y-%m-%d")

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    df = predict(stocks, st)

    if df is not None and not df.empty:
        out_dir = f"db/predictions/{MODEL_NAME}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/pred_{today}.parquet"
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_parquet(out_path, index=False)
        print(f"saved {len(df)} rows → {out_path}")
