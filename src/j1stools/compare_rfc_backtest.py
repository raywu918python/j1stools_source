import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
import numpy as np
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_utils import drop_na_inf
from j1stools import backtest_platform, data_filter, feature_builder, hf_sync, label_builder, parquet_db
from j1stools.rfc_main import _add_ib_features
from run_rfc_predict import batter_predict

BACKTEST_ST = "2024-01-01"
BACKTEST_PARAMS = dict(
    top_n=5,
    threshold=0.6,
    max_positions=5,
    use_sl_trail=False,
    use_fixed_sl=True,
    sl_stop=0.10,
    use_fixed_tp=True,
    tp_stop=0.10,
    use_hold_days=True,
    hold_days=10,
    group_limit=2,
)


def _predict(stocks, st, model_name, use_ib):
    model = joblib.load(f"models/{model_name}.joblib")
    warmup_st = (pd.Timestamp(st) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")

    df = parquet_db.query_price(stocks, warmup_st, "2099-01-01")
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd)
    if use_ib:
        df = _add_ib_features(df, stocks, warmup_st, "2099-01-01")
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)

    _, x, _, y = rfc_split_date(df, is_gen_test=True, is_gen_train=False, trainging_idx=0)
    x = x[[col for col in x.columns if col.startswith("f_")]]
    x, y = drop_na_inf(x, y)
    x = x[model.feature_names_in_]

    result = batter_predict(model, x)
    result["date"] = pd.to_datetime(result["date"])
    return result[result["date"] >= pd.Timestamp(st)].reset_index(drop=True)


def _calc_metrics(pv: pd.DataFrame) -> dict:
    pv = pv.copy()
    pv["date"] = pd.to_datetime(pv["date"])
    pv = pv.sort_values("date")
    ret = pv["total"].pct_change().dropna()
    total_return = (pv["total"].iloc[-1] / pv["total"].iloc[0] - 1) * 100
    sharpe = ret.mean() / ret.std() * (252 ** 0.5) if ret.std() > 0 else 0
    peak = pv["total"].cummax()
    max_dd = ((pv["total"] - peak) / peak).min() * 100
    return {
        "total_return_%": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_%": round(max_dd, 2),
    }


if __name__ == "__main__":
    hf_sync.pull(["db/price", "db/active_stocks", "db/feature_cols", "db/ib", "db/info", "models"])

    stocks = parquet_db.query_stocks_no_etf()

    models = [
        ("rfc_macd_6xx",    False),
        ("rfc_macd_ib_6xx", True),
    ]

    results = {}
    for model_name, use_ib in models:
        print(f"\n{'='*55} {model_name}")
        signal = _predict(stocks, BACKTEST_ST, model_name, use_ib)
        pv, *_ = backtest_platform.prepare_data_backtest(signal, **BACKTEST_PARAMS)
        results[model_name] = _calc_metrics(pv)

    print(f"\n{'='*55} 比較結果")
    df = pd.DataFrame(results).T
    print(df.to_string())

    winner = max(results, key=lambda k: results[k]["sharpe"])
    print(f"\n Sharpe 勝出：{winner}")
