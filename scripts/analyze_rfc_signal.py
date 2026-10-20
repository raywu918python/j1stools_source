"""診斷 rfc_macd_ib_6xx 為什麼最近沒訊號（唯讀，不寫回 HF）。

1. 模型 feature importance
2. 每日 pred 檔的 P(2) 分布（最近 30 個檔）
3. 2024 起每月 P(2) >= 0.6 的天數，看兩週沒訊號是否常見
4. 最近 60 天：依 MACD 0/1 組合分組看 P(2)，以及外資特徵與 P(2) 的關係
"""

import glob
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd

from j1stools import hf_sync, parquet_db
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_utils import drop_na_inf
from run_rfc_predict import MODEL_NAME, USE_IB_FEATURES, prepare_data

THRESHOLD = 0.6
MACD_COLS = ["f_macd_hist_div", "f_macd_trend_up", "f_macd_bull_regime", "f_macd_price_support", "f_atr_active"]
IB_COLS = ["f_ib_net_foreign_pct", "f_ib_net_foreign_5d_z", "f_ib_net_foreign_10d_z", "f_ib_net_foreign_streak"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(title):
    print("\n" + "=" * 20, title, "=" * 20)


pull_dirs = ["db/price", "db/active_stocks", "db/feature_cols", "db/info", "models", "db/ib", f"db/predictions/{MODEL_NAME}"]
hf_sync.pull(pull_dirs)

model = joblib.load(f"models/{MODEL_NAME}.joblib")
print("model:", MODEL_NAME, "USE_IB_FEATURES:", USE_IB_FEATURES, "classes:", list(model.classes_))

section("1. feature importance")
imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False)
print(imp.round(4).to_string())

section("2. 每日 pred 檔 P(2) 分布（最近 30 檔）")
rows = []
for f in sorted(glob.glob(f"db/predictions/{MODEL_NAME}/pred_*.parquet"))[-30:]:
    d = os.path.basename(f)[5:15]
    p = pd.read_parquet(f)["2"]
    rows.append(
        {
            "date": d,
            "n": len(p),
            "max": p.max(),
            "p99": p.quantile(0.99),
            "p90": p.quantile(0.90),
            "median": p.median(),
            f">={THRESHOLD}": int((p >= THRESHOLD).sum()),
            ">=0.55": int((p >= 0.55).sum()),
            ">=0.5": int((p >= 0.5).sum()),
        }
    )
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 重算 2024 起的特徵與機率（特徵不用未來資料，可以重算）
stocks = list(set(parquet_db.query_stocks_no_etf()) - {"0050", "0052", "0056"})
df = prepare_data(stocks, "2023-09-01", "2099-01-01", use_ib=USE_IB_FEATURES)
_, x, _, y = rfc_split_date(df, is_gen_test=True, is_gen_train=False, trainging_idx=0)
x = x[[c for c in x.columns if c.startswith("f_")]]
x, y = drop_na_inf(x, y)
x = x[model.feature_names_in_]
p2 = model.predict_proba(x)[:, list(model.classes_).index(2)]
res = x.reset_index()
res["p2"] = p2
res["date"] = pd.to_datetime(res["date"])
res = res[res["date"] >= "2024-01-01"]

section("3. 每月：有 >=0.6 候選的天數 / 交易日數")
daily = res.groupby("date").agg(max_p2=("p2", "max"), n_hit=("p2", lambda s: int((s >= THRESHOLD).sum())))
daily["has_signal"] = daily["n_hit"] > 0
monthly = daily.groupby(daily.index.to_period("M")).agg(
    trade_days=("has_signal", "size"), signal_days=("has_signal", "sum"), avg_hits=("n_hit", "mean"), max_p2=("max_p2", "max")
)
print(monthly.round(3).to_string())
# 最長連續無訊號天數
streak, longest, longest_end = 0, 0, None
for d, h in daily["has_signal"].items():
    streak = 0 if h else streak + 1
    if streak > longest:
        longest, longest_end = streak, d
print(f"最長連續無訊號：{longest} 個交易日（結束於 {longest_end.date() if longest_end is not None else '-'}）")
print("最近 20 個交易日：")
print(daily.tail(20).round(3).to_string())

recent = res[res["date"] >= res["date"].max() - pd.Timedelta(days=60)].copy()

section("4a. 最近 60 天依 MACD 組合分組")
recent["macd_combo"] = recent[MACD_COLS].astype(int).astype(str).agg("".join, axis=1)
combo = recent.groupby("macd_combo")["p2"].agg(["size", "mean", "max", lambda s: int((s >= THRESHOLD).sum())])
combo.columns = ["n", "mean_p2", "max_p2", f"n>={THRESHOLD}"]
print("組合順序：", MACD_COLS)
print(combo.sort_values("max_p2", ascending=False).round(3).to_string())

section("4b. 外資特徵與 P(2) 的關係（最近 60 天）")
print("Spearman 相關：")
print(recent[IB_COLS + ["p2"]].corr(method="spearman")["p2"].drop("p2").round(3).to_string())
for col in IB_COLS:
    try:
        bins = pd.qcut(recent[col], 5, duplicates="drop")
    except ValueError:
        continue
    g = recent.groupby(bins, observed=True)["p2"].agg(["size", "mean", "max"])
    print(f"\n{col} 分五等份 → P(2)")
    print(g.round(3).to_string())

section("4c. 全期間（2024 起）外資分位 vs P(2)，對照用")
print(res[IB_COLS + ["p2"]].corr(method="spearman")["p2"].drop("p2").round(3).to_string())
hits = res[res["p2"] >= THRESHOLD]
print(f"\n2024 起 P(2)>={THRESHOLD} 的樣本數：{len(hits)}，各特徵平均 vs 全體平均：")
print(pd.DataFrame({"hits_mean": hits[IB_COLS + MACD_COLS].mean(), "all_mean": res[IB_COLS + MACD_COLS].mean()}).round(3).to_string())

section("4d. 今天最接近門檻的 10 檔")
last = res[res["date"] == res["date"].max()].nlargest(10, "p2")
print(last[["date", "stock_id", "p2"] + MACD_COLS + IB_COLS].round(3).to_string(index=False))
