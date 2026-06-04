import os, sys
from datetime import datetime, timedelta, timezone

_TW = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_utils import drop_na_inf
from j1stools.train_flow import batter_predict
from j1stools import data_filter, feature_builder, hf_sync, label_builder, parquet_db


def prepare_data(stocks, st, end):
    df = parquet_db.query_price(stocks, st, end)
    print("filter.before:", df.shape)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    print("filter.after:", df.shape)
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)
    return df


def predict(stocks, st, end):
    model = joblib.load("models/rfc_macd_6xx.joblib")
    warmup_st = (pd.Timestamp(st) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    df = prepare_data(stocks, warmup_st, end)
    _, x, _, y = rfc_split_date(df, is_gen_test=True, is_gen_train=False, trainging_idx=0)
    x = x[[col for col in x.columns if col.startswith("f_")]] if x is not None else None
    x, y = drop_na_inf(x, y)
    expected_features = model.feature_names_in_
    x = x[expected_features]
    result = batter_predict(model, x, y)
    result["date"] = pd.to_datetime(result["date"])
    return result[result["date"] >= pd.Timestamp(st)].reset_index(drop=True)


if __name__ == "__main__":
    hf_sync.pull(["db/price", "db/active_stocks", "db/feature_cols", "models"])

    stocks = parquet_db.activate_stocks()
    stocks = list(set(stocks) - {"0050", "0052", "0056"})

    today = datetime.now(_TW).strftime("%Y-%m-%d")
    st = (datetime.now(_TW) - timedelta(days=10)).strftime("%Y-%m-%d")

    df = predict(stocks, st, "2099-01-01")

    MODEL_NAME = "rfc_macd_6xx"

    if df is not None and not df.empty:
        out_dir = f"db/predictions/{MODEL_NAME}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/pred_{today}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"saved {len(df)} rows → {out_path}")
        hf_sync.push([f"db/predictions/{MODEL_NAME}"])
    else:
        print("今日無預測結果")
