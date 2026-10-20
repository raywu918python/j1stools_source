"""比較 f_macd_hist_div 修掉 center=True 前後的模型與回測（唯讀，不推回 HF、不覆蓋線上模型）。

A. 線上模型 + 舊特徵（center=True，偷看未來 5 天）：等於現在回測報告的數字
B. 線上模型 + 新特徵（確認低點後才標記）：線上模型拿到「實盤看得到」的輸入時的表現
C. 用新特徵重新訓練（同 rfc_main.model_release 的設定）+ 新特徵：修正後的真實表現

新模型存在 models/rfc_macd_ib_6xx_nolook.joblib，會上傳成 workflow artifact，不會 push 到 HF。
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from time import time

import joblib
import pandas as pd

from j1stools import backtest_platform, hf_sync, macdh_divergences, parquet_db
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_builder import gen_rfc_model
from j1stools.model_utils import drop_na_inf
from run_rfc_predict import MODEL_NAME, USE_IB_FEATURES, prepare_data

TRAIN_ST, TRAIN_END = "2015-01-01", "2024-01-01"
TEST_ST = "2024-01-01"
NEW_MODEL_PATH = "models/rfc_macd_ib_6xx_nolook.joblib"
BACKTEST_KW = dict(
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
    min_volume=200,
)

_fixed_lows = macdh_divergences._confirmed_price_lows


def _leaky_lows(close_wide, hist_wide, n):
    """舊版：rolling(center=True)，低點當天就用到之後 n 天的價格。"""
    is_low = close_wide == close_wide.rolling(2 * n + 1, center=True).min()

    def find_prev_val(val_df, mask_df):
        return val_df.where(mask_df).shift(1).ffill(limit=30)

    return is_low, close_wide, hist_wide, find_prev_val(close_wide, is_low), find_prev_val(hist_wide, is_low)


def use_leaky_features(leaky: bool):
    macdh_divergences._confirmed_price_lows = _leaky_lows if leaky else _fixed_lows


def load_xy(stocks, st, end):
    df = prepare_data(stocks, st, end, use_ib=USE_IB_FEATURES)
    x, _, y, _ = rfc_split_date(df, is_gen_train=True, trainging_idx=1)
    x = x[[c for c in x.columns if c.startswith("f_")]]
    return drop_na_inf(x, y)


def gen_signal(model, stocks):
    warmup = (pd.Timestamp(TEST_ST) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    x, _ = load_xy(stocks, warmup, "2099-01-01")
    x = x[model.feature_names_in_]
    proba = model.predict_proba(x)
    out = pd.DataFrame(proba, index=x.index, columns=[str(c) for c in model.classes_]).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"] >= TEST_ST].reset_index(drop=True)


def summarize(name, signal):
    print("\n" + "=" * 20, name, "=" * 20)
    pv, trades, positions, _ = backtest_platform.prepare_data_backtest(signal, **BACKTEST_KW)
    daily_hits = signal.groupby("date")["2"].apply(lambda s: int((s >= 0.6).sum()))
    row = {
        "case": name,
        "final_value": round(float(pv["total"].iloc[-1])),
        "total_return_%": round((float(pv["total"].iloc[-1]) / 1_000_000 - 1) * 100, 1),
        "trades": len(trades),
        "win_rate_%": round((trades["pnl"] > 0).mean() * 100, 1) if len(trades) else None,
        "avg_return_%": round(trades["return_pct"].mean(), 2) if len(trades) else None,
        "days_with_p2>=0.6_%": round((daily_hits > 0).mean() * 100, 1),
        "days_with_p2>=0.6_last60d": int((daily_hits[daily_hits.index >= daily_hits.index.max() - pd.Timedelta(days=60)] > 0).sum()),
    }
    if len(trades):
        t = trades.copy()
        t["year"] = pd.to_datetime(t["entry_date"]).dt.year
        by_year = t.groupby("year").agg(trades=("pnl", "size"), win_rate=("pnl", lambda s: round((s > 0).mean() * 100, 1)), avg_ret=("return_pct", "mean"))
        print(by_year.round(2).to_string())
    print("目前持倉：", list(positions.keys()) if positions else "無")
    return row


if __name__ == "__main__":
    hf_sync.pull(["db/price", "db/active_stocks", "db/feature_cols", "db/info", "models", "db/ib"])
    old_model = joblib.load(f"models/{MODEL_NAME}.joblib")
    test_stocks = list(set(parquet_db.query_stocks_no_etf()) - {"0050", "0052", "0056"})

    rows = []
    use_leaky_features(True)
    rows.append(summarize("A 線上模型 + 舊特徵(偷看)", gen_signal(old_model, test_stocks)))

    use_leaky_features(False)
    rows.append(summarize("B 線上模型 + 新特徵", gen_signal(old_model, test_stocks)))

    t0 = time()
    x, y = load_xy(parquet_db.activate_stocks(), TRAIN_ST, TRAIN_END)
    print(f"\ntrain rows: {len(x)}, class counts: {y.value_counts().sort_index().to_dict()}")
    new_model = gen_rfc_model()
    new_model.fit(x, y)
    print(f"train time: {time() - t0:.0f}s, OOB score: {new_model.oob_score_:.4f}")
    imp = pd.Series(new_model.feature_importances_, index=new_model.feature_names_in_).sort_values(ascending=False)
    print("new model feature importance:\n" + imp.round(4).to_string())
    os.makedirs("models", exist_ok=True)
    joblib.dump(new_model, NEW_MODEL_PATH)
    print(f"saved → {NEW_MODEL_PATH}")
    rows.append(summarize("C 重訓模型 + 新特徵", gen_signal(new_model, test_stocks)))

    print("\n" + "=" * 20, "總表", "=" * 20)
    print(pd.DataFrame(rows).to_string(index=False))
