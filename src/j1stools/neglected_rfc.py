"""
外資忽視股 RFC 選股模型

流程：
  1. GMM 篩選叢集 2/3（穩健技術面群）
  2. RFC 在這個宇宙裡學「哪些股票接下來會漲」
  3. OOS 回測驗證效果

不是偷看答案：
  - GMM 只用當下特徵分群（無未來資訊）
  - RFC 的 Y = 未來 N 日漲跌（正確的監督學習目標）
  - 訓練用 2015~2023，測試用 2024~
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from j1stools import parquet_db
from j1stools.label_builder import profit_label
from j1stools.neglected_stock_classify import (
    NeglectedGMM,
    load_neglected_data,
    NEGLECTED_FEATURES,
)

RFC_MODEL_PATH = "db/models/neglected_rfc.pkl"

# RFC 特徵：NEGLECTED_FEATURES（GMM 用的技術面）+ GMM 沒用的維度
# prob_k 是 NEGLECTED_FEATURES 的非線性組合，移除以避免冗餘
RFC_FEATURES = NEGLECTED_FEATURES + [
    # GMM 叢集編號：讓 RFC 自己學要不要用叢集資訊
    "cluster",
    # 價格動能：GMM 只有 slope 代理，這裡加直接報酬
    "f_return_5d_xrank",
    "f_return_10d_xrank",
    "f_return_20d_xrank",
    # 相對強度 vs 大盤：GMM 完全沒用
    "f_relative_strength_5d_xrank",
    "f_relative_strength_20d_xrank",
    # 更長週期融資：GMM 只有 5d
    "f_margin_balance_change_10d_pct_xrank",
    # 融券絕對變化：GMM 只用比率
    "f_short_balance_change_pct_xrank",
    # 個股 vs 大盤波動：GMM 沒用
    "f_volatility_vs_market_xrank",
    # 大盤環境（raw，截面 xrank 無意義）
    "f_market_volatility_20d",
]


def build_dataset(
    clf_gmm: NeglectedGMM,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    clusters: list[int] | None = None,
    hold_days: int = 10,
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """
    載入外資忽視股 → GMM 分群 → 計算未來報酬標籤。

    Parameters
    ----------
    clusters  : 預測時要保留的叢集；None = 不過濾（訓練模式，學全部叢集）
    hold_days : 持有天數
    """
    df = load_neglected_data(stocks, st, end, min_atr_pct=min_atr_pct)
    df["cluster"] = clf_gmm.predict(df)

    # GMM 軟機率（保留欄位但不加入 RFC_FEATURES，供後續分析用）
    proba = clf_gmm.predict_proba(df)
    df = pd.concat([df, proba], axis=1)

    # 叢集過濾：None = 訓練時用全部叢集，非 None = 預測/eval 時過濾
    if clusters is not None:
        df = df[df["cluster"].isin(clusters)].copy()

    # 補回 high/low（profit_label 需要，load_data 已 drop）
    df_price_hl = parquet_db.query_price(df["stock_id"].unique().tolist(), st, end)
    df_price_hl["date"] = pd.to_datetime(df_price_hl["date"])
    df = df.merge(df_price_hl[["date", "stock_id", "high", "low"]], on=["date", "stock_id"], how="left")

    # 未來收盤報酬（eval_rfc 回測用）
    df["future_return"] = df.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)

    # profit_label：持有期間任何一天碰到目標就算成功（2=達標, 1=停損, 0=盤整）
    df = profit_label(df, hold_days=hold_days)
    df["Y"] = (df["target"] == 2).astype(int)

    # dropna 用 future_return 而非 target（profit_label 對末端列不會輸出 NaN）
    df = df.dropna(subset=["future_return"]).drop(columns=["target", "high", "low"], errors="ignore")

    cluster_info = f"叢集 {clusters}" if clusters is not None else "全叢集"
    print(f"資料集：{len(df):,} 筆  正樣本(達標)：{df['Y'].mean():.1%}  {cluster_info}  持有 {hold_days} 日")
    return df


def train_rfc(
    df_train: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 6,
    features: list | None = None,
) -> RandomForestClassifier:
    """在訓練集上訓練 RFC。features=None 時使用全域 RFC_FEATURES。"""
    feat_list = features if features is not None else RFC_FEATURES
    avail = [c for c in feat_list if c in df_train.columns]
    X = df_train[avail].fillna(0.5).values
    y = df_train["Y"].values

    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rfc.fit(X, y)
    rfc._fitted_features = avail

    auc = roc_auc_score(y, rfc.predict_proba(X)[:, 1])
    print(f"訓練集 AUC：{auc:.4f}（in-sample，供參考）")
    return rfc


def eval_rfc(
    rfc: RandomForestClassifier,
    df_test: pd.DataFrame,
    top_n: int = 10,
    hold_days: int = 10,
) -> pd.DataFrame:
    """
    OOS 評估：
      - AUC / 分類報告
      - 每日選 top_n 高機率股票，計算組合報酬
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    avail = rfc._fitted_features
    X_test = df_test[avail].fillna(0.5).values
    y_test = df_test["Y"].values
    prob = rfc.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, prob)
    print(f"\n{'='*55}")
    print(f"OOS AUC：{auc:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_test, (prob >= 0.5).astype(int), target_names=["跌", "漲"]))

    # 特徵重要性
    fi = pd.Series(rfc.feature_importances_, index=avail).sort_values(ascending=False)
    print(f"\n特徵重要性 Top 10：")
    print(fi.head(10).map("{:.4f}".format).to_string())

    # 信心度分層勝率
    df_cal = pd.DataFrame({"prob": prob, "Y": y_test, "future_return": df_test["future_return"].values})
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"]
    df_cal["bucket"] = pd.cut(df_cal["prob"], bins=bins, labels=labels, right=False)
    cal = df_cal.groupby("bucket", observed=True).agg(
        筆數=("Y", "count"),
        平均信心=("prob", "mean"),
        勝率=("Y", "mean"),
        平均報酬=("future_return", "mean"),
    )
    print(f"\n{'='*60}")
    print("信心度分層勝率")
    print(f"{'='*60}")
    print(cal.to_string(float_format=lambda x: f"{x:.2%}"))

    # 每日 top_n 組合回測（非重疊：每 hold_days 天換一次倉）
    df_eval = df_test.copy()
    df_eval["prob"] = prob
    # clip 極端值，避免單一股票暴漲/暴跌炸掉累積計算
    df_eval["future_return_clip"] = df_eval["future_return"].clip(-0.5, 1.0)

    all_dates = sorted(df_eval["date"].unique())
    # 每 hold_days 天取一個換倉日（非重疊窗口）
    rebal_dates = all_dates[::hold_days]

    rfc_returns, base_returns = [], []
    for date in rebal_dates:
        g = df_eval[df_eval["date"] == date]
        if len(g) == 0:
            continue
        # RFC top_n
        top_rfc = g.nlargest(top_n, "prob")
        rfc_returns.append(
            {
                "date": date,
                "mean_return": top_rfc["future_return_clip"].mean(),
                "win_rate": (top_rfc["future_return"] > 0).mean(),
            }
        )
        # Baseline：從同一叢集隨機選 top_n
        top_base = g.sample(min(top_n, len(g)), random_state=42)
        base_returns.append(
            {
                "date": date,
                "mean_return": top_base["future_return_clip"].mean(),
                "win_rate": (top_base["future_return"] > 0).mean(),
            }
        )

    df_perf = pd.DataFrame(rfc_returns).set_index("date")
    df_base = pd.DataFrame(base_returns).set_index("date")
    cum_rfc = (1 + df_perf["mean_return"]).cumprod()
    cum_base = (1 + df_base["mean_return"]).cumprod()

    print(f"\n{'='*60}")
    print(f"非重疊換倉 Top-{top_n} 組合（每 {hold_days} 日換一次）")
    print(f"{'='*60}")
    print(f"{'':15} {'RFC':>10} {'隨機 Baseline':>15}")
    print(f"  平均報酬   {df_perf['mean_return'].mean():>9.2%} {df_base['mean_return'].mean():>14.2%}")
    print(f"  勝率       {df_perf['win_rate'].mean():>9.1%} {df_base['win_rate'].mean():>14.1%}")
    print(f"  累積報酬   {cum_rfc.iloc[-1]-1:>9.2%} {cum_base.iloc[-1]-1:>14.2%}")

    # 累積報酬比較圖
    _, axes = plt.subplots(2, 1, figsize=(12, 8))
    cum_rfc.plot(ax=axes[0], color="steelblue", label=f"RFC Top-{top_n}")
    cum_base.plot(ax=axes[0], color="tomato", linestyle="--", label=f"隨機 Baseline")
    axes[0].axhline(1, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"外資忽視股 RFC vs 隨機 Baseline — Top-{top_n} 每 {hold_days} 日換倉")
    axes[0].set_ylabel("累積報酬倍數")
    axes[0].legend()

    df_perf["win_rate"].rolling(5).mean().plot(ax=axes[1], color="steelblue", label="RFC")
    df_base["win_rate"].rolling(5).mean().plot(ax=axes[1], color="tomato", linestyle="--", label="隨機")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("滾動 5 期勝率")
    axes[1].set_ylabel("勝率")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return df_perf


def save_rfc(rfc: RandomForestClassifier, path: str = RFC_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rfc, f)
    print(f"RFC 模型已儲存：{path}")


def load_rfc(path: str = RFC_MODEL_PATH) -> RandomForestClassifier:
    with open(path, "rb") as f:
        rfc = pickle.load(f)
    print(f"RFC 模型已載入：{path}")
    return rfc


def predict_today_rfc(
    top_n: int = 10,
    clusters: list[int] | None = None,
    st: str = "2025-01-01",
    min_atr_pct: float = 0.02,
) -> pd.DataFrame:
    """今日候選：GMM 叢集 2/3 → RFC 打分 → 回傳 top_n。"""
    if clusters is None:
        clusters = [1, 2, 3]

    clf_gmm = NeglectedGMM.load()
    rfc = load_rfc()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    df = load_neglected_data(stocks, st=st, min_atr_pct=min_atr_pct)
    df["cluster"] = clf_gmm.predict(df)
    proba_gmm = clf_gmm.predict_proba(df)
    df = pd.concat([df, proba_gmm], axis=1)

    latest = df["date"].max()
    today = df[(df["date"] == latest) & df["cluster"].isin(clusters)].copy()

    avail = rfc._fitted_features
    today["rfc_prob"] = rfc.predict_proba(today[avail].fillna(0.5).values)[:, 1]

    result = today.nlargest(top_n, "rfc_prob")[["date", "stock_id", "cluster", "rfc_prob"]]
    print(f"\n分類日期：{latest.date()}  GMM 叢集 {clusters} 共 {len(today)} 支 → RFC Top-{top_n}：")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_ST, TRAIN_END = "2015-01-01", "2023-12-31"
    EVAL_ST = "2024-01-01"
    HOLD_DAYS = 10
    CLUSTERS = [1, 2, 3]
    TOP_N = 10
    MIN_ATR = 0.02

    # ── 切換模式 ─────────────────────────────────────────── #
    MODE = "train+eval"  # "train+eval" | "eval_only" | "predict_today"
    # ────────────────────────────────────────────────────── #

    clf_gmm = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train+eval":
        print("建立訓練集...")
        df_train = build_dataset(
            clf_gmm, stocks, TRAIN_ST, TRAIN_END, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR
        )
        print("\n建立測試集...")
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)

        # Ablation：同一訓練集，比較有無 cluster 特徵
        features_with = RFC_FEATURES
        features_without = [f for f in RFC_FEATURES if f != "cluster"]

        print("\n" + "=" * 60)
        print("【含 cluster 特徵】")
        print("=" * 60)
        rfc_with = train_rfc(df_train, features=features_with)
        eval_rfc(rfc_with, df_test, top_n=TOP_N, hold_days=HOLD_DAYS)

        # print("\n" + "="*60)
        # print("【不含 cluster 特徵】")
        # print("="*60)
        # rfc_without = train_rfc(df_train, features=features_without)
        # eval_rfc(rfc_without, df_test, top_n=TOP_N, hold_days=HOLD_DAYS)

        # 儲存較好的（預設存含 cluster 版本）
        save_rfc(rfc_with)

    elif MODE == "eval_only":
        rfc = load_rfc()
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)
        eval_rfc(rfc, df_test, top_n=TOP_N, hold_days=HOLD_DAYS)

    elif MODE == "predict_today":
        predict_today_rfc(top_n=TOP_N, clusters=CLUSTERS)
