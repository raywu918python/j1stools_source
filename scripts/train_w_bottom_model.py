"""W 底 + 拉回 特徵版 RFC：重訓並和 C（rfc_macd_ib_6xx_nolook）比較門檻 / 持倉數（唯讀，不推回 HF）。

特徵 = 原本 MACD 5 個 0/1（已修掉 center=True）+ f_atr_just + 外資 4 個 + W 底 6 個。
新模型存在 models/rfc_wbottom_ib.joblib，上傳成 workflow artifact。
"""

import itertools
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from time import time

import joblib
import pandas as pd

from j1stools import backtest_platform, data_filter, feature_builder, hf_sync, label_builder, parquet_db
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_builder import gen_rfc_model
from j1stools.model_utils import drop_na_inf
from j1stools.rfc_main import _add_ib_features
from compare_nolook_model import BACKTEST_KW, NEW_MODEL_PATH, TEST_ST, TRAIN_END, TRAIN_ST
from run_rfc_predict import prepare_data as prepare_data_c

W_MODEL_PATH = "models/rfc_wbottom_ib.joblib"
THRESHOLDS = [0.45, 0.5, 0.55, 0.6]
POSITIONS = [5, 10]


def prepare_data_w(stocks, st, end):
    df = parquet_db.query_price(stocks, st, end)
    df = feature_builder.gen_feature(df, FEATURE_TYPE.macd | FEATURE_TYPE.w_bottom)
    df = _add_ib_features(df, stocks, st, end)
    df = label_builder.profit_label(df)
    df = data_filter.filter(df, True, FILTER_TYPE.none_, FILTER_TYPE.add_)
    df.set_index(["date", "stock_id"], inplace=True)
    df.sort_index(level=["date", "stock_id"], inplace=True)
    return df


def load_xy(prepare, stocks, st, end):
    df = prepare(stocks, st, end)
    x, _, y, _ = rfc_split_date(df, is_gen_train=True, trainging_idx=1)
    x = x[[c for c in x.columns if c.startswith("f_")]]
    return drop_na_inf(x, y)


def gen_signal(model, prepare, stocks):
    warmup = (pd.Timestamp(TEST_ST) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    x, _ = load_xy(prepare, stocks, warmup, "2099-01-01")
    x = x[model.feature_names_in_]
    proba = model.predict_proba(x)
    out = pd.DataFrame(proba, index=x.index, columns=[str(c) for c in model.classes_]).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"] >= TEST_ST].reset_index(drop=True)


def sweep(name, signal):
    rows = []
    for th, pos in itertools.product(THRESHOLDS, POSITIONS):
        kw = {**BACKTEST_KW, "threshold": th, "max_positions": pos, "top_n": pos}
        pv, trades, _, _ = backtest_platform.prepare_data_backtest(signal, **kw)
        eq = pv["total"]
        t = trades.copy()
        t["entry_date"] = pd.to_datetime(t["entry_date"])
        daily_hits = signal.groupby("date")["2"].apply(lambda s: int((s >= th).sum()))
        rows.append(
            {
                "model": name,
                "threshold": th,
                "max_pos": pos,
                "total_return_%": round((float(eq.iloc[-1]) / 1_000_000 - 1) * 100, 1),
                "max_dd_%": round((eq / eq.cummax() - 1).min() * 100, 1),
                "trades": len(t),
                "trades_2026": int((t["entry_date"].dt.year == 2026).sum()),
                "win_rate_%": round((t["pnl"] > 0).mean() * 100, 1) if len(t) else None,
                "avg_return_%": round(t["return_pct"].mean(), 2) if len(t) else None,
                "signal_days_%": round((daily_hits > 0).mean() * 100, 1),
            }
        )
    return rows


if __name__ == "__main__":
    hf_sync.pull(["db/price", "db/active_stocks", "db/feature_cols", "db/info", "db/ib"])
    test_stocks = list(set(parquet_db.query_stocks_no_etf()) - {"0050", "0052", "0056"})

    t0 = time()
    x, y = load_xy(prepare_data_w, parquet_db.activate_stocks(), TRAIN_ST, TRAIN_END)
    print(f"train rows: {len(x)}, class counts: {y.value_counts().sort_index().to_dict()}, prep {time() - t0:.0f}s")
    print("W 特徵觸發比例：", (x[[c for c in x.columns if c.startswith("f_w_")]] > 0).mean().round(4).to_dict())
    print("各 W 特徵下 class 2 比例：")
    for c in ["f_w_bottom_30d", "f_w_pullback_hold", "f_w_pullback_hist_div", "f_w_setup"]:
        print(f"  {c}=1: {(y[x[c] > 0] == 2).mean():.3f}  vs 全體 {(y == 2).mean():.3f}  (n={(x[c] > 0).sum()})")

    t0 = time()
    model = gen_rfc_model()
    model.fit(x, y)
    print(f"train time: {time() - t0:.0f}s, OOB score: {model.oob_score_:.4f}")
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False)
    print("feature importance:\n" + imp.round(4).to_string())
    joblib.dump(model, W_MODEL_PATH)
    print(f"saved → {W_MODEL_PATH}")
    del x, y

    rows = sweep("D W底", gen_signal(model, prepare_data_w, test_stocks))
    if os.path.exists(NEW_MODEL_PATH):
        rows += sweep("C", gen_signal(joblib.load(NEW_MODEL_PATH), lambda s, a, b: prepare_data_c(s, a, b, use_ib=True), test_stocks))

    mkt = parquet_db.query_price(["0050"], TEST_ST, "2099-01-01")
    print(f"\n0050 同期報酬：{(mkt['close'].iloc[-1] / mkt['close'].iloc[0] - 1) * 100:.1f}%")
    print("\n" + "=" * 20, "總表", "=" * 20)
    print(pd.DataFrame(rows).to_string(index=False))
