from math import e
from re import L
import signal
from time import time

import random
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from numpy import add
import pandas as pd
import numpy as np
from regex import P
from sklearn.model_selection import TimeSeriesSplit

from j1stools import data_filter, feature_builder, label_builder, lite_db, parquet_db, rfc_main
from j1stools.CONFIG import (
    BaseDataBuilderConfig,
    BaseLabelConfig,
    LgbmTrainConfig,
    MACDDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, TRAIN_TYPE, MODEL_TYPE


from j1stools.j1s_split_date import lgbm_split_date
from j1stools.lgbm_backtestt import run_backtest_engin
from j1stools.market_filter import add_market_filter, apply_filter
from j1stools.model_builder import gen_lgbm_orgin_model, gen_lgbm_r_model
from j1stools.model_utils import drop_na_inf
from j1stools.train_flow import get_full_name, keep_latest_ten_files, start_train
import joblib

from j1stools.train_function import lgbm_r_function_train


def predict(
    stocks,
    st,
    end,
):

    models = joblib.load("models/lgbm_timeseries_ensemble.joblib")

    df_feature, df_market = prepare_data(stocks, st, end, model="predict")
    df_select_stocks = select_stocks(df_today=df_feature, df_market_history=df_market, models=models)
    if df_select_stocks is None:
        return
    df = df_select_stocks.head(30).sort_values(by=["date"], ascending=False)
    print(df.head(30))


def main():
    """
        每天收盤後：
      → select_stocks → 更新候選清單

    每週一：
      → calc_feature_ic → 監控特徵健康度

    每月底：
      → walk_forward_train → 重新訓練模型
      → evaluate_selection → 確認新模型有效
      → 下個月用新模型
    """
    stocks = list(set(parquet_db.query_stocks_ids_list()) - set(["0050", "0052", "0056"]))
    st = "2026-03-01"
    end = "2026-05-07"  # "2026-02-01"
    # models = train(stocks=stocks, st=st, end=end)
    # joblib.dump(models, "models/lgbm_timeseries_ensemble.joblib")

    predict(
        stocks=stocks,
        st=st,
        end=end,
    )


def prepare_data(stocks, st, end, model="train"):
    df_margin = lite_db.margin(stocks, st, end)
    print("check date", df_margin["date"].max())
    df_market = parquet_db.query_price(["0050"], st, end)
    df_ibbuysell = lite_db.ibbuysell(stocks, st, end)
    df_margin = label_builder.add_target(
        df_margin,
        df_market=df_market,
        forward_days=20,
    )
    df_feature = feature_builder.gen_feature(
        None,
        FEATURE_TYPE.margin_ibbuysell,
        dfs=[df_margin, df_ibbuysell, df_market],
        argv=model,
    )
    return df_feature, df_market


def train(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2024-01-01",
    end="2026-02-01",
    pick_import_feature=False,
):
    print(f"=" * 60, "lgbm start")
    print("=" * 60, "train")
    keep_latest_ten_files("./model")

    df_feature, _ = prepare_data(stocks, st, end)

    models, scores = walk_forward_train(
        df_feature,
        n_splits=5,
    )
    result_df, stock_df, bottom_df = evaluate_selection(df_feature, models)
    summary = analyze_frequent(stock_df)

    return models


def start_backtest(
    stocks=parquet_db.query_stocks_no_etf(),
    st="2015-01-01",
    end="2024-01-01",
):
    print("=" * 60, "backtest")
    df_margin = lite_db.margin(stocks, st, end)
    df_market = parquet_db.query_price(["0050"], st, end)
    df_ibbuysell = lite_db.ibbuysell(stocks, st, end)
    df_margin = label_builder.add_target_forward(df_margin, df_market=df_market)
    df_feature = feature_builder.gen_feature(
        None,
        FEATURE_TYPE.margin_ibbuysell,
        dfs=[df_margin, df_ibbuysell, df_market],
    )

    models = joblib.load("models/lgbm_timeseries_ensemble_2024_1.joblib")

    for top_n in [5, 10, 15, 20, 30]:
        print(f"\ntop_n = {top_n}")
        result = run_backtest_engin(
            df_feature,
            models[-2:],
            df_market,
            use_filter=True,
            top_n=top_n,
            start_date="2024-01-01",
        )

    raise Exception("未完成")
    # models = joblib.load("models/lgbm_timeseries_ensemble_2025_1.joblib")

    # result_df = backtest_engin(
    #     df_feature,
    #     [models[-2]],
    #     df_market,
    #     top_n=20,
    #     forward_days=20,
    #     use_filter=True,
    #     start_date="2024-01-01",
    # )
    # print(result_df[result_df["actual_return"] > 0.5][["date", "actual_return", "stocks"]])
    # 看那一期選了哪些股票，以及它們的實際報酬

    # result_df = backtest_engin(
    #     df_feature,
    #     [models[-1]],
    #     df_market,
    #     top_n=20,
    #     forward_days=20,
    #     use_filter=True,
    #     start_date="2024-01-01",
    # )

    # result_df = backtest_engin(
    #     df_feature,
    #     models,
    #     df_market,
    #     top_n=20,
    #     forward_days=20,
    #     use_filter=True,
    #     start_date="2024-01-01",
    # )

    # joblib.dump(model, get_full_name("lgbm_r"))
    # print("=" * 60, "test")

    # expected_features = model.feature_names_in_
    # xtest = xtest[expected_features]
    # model.train(xtest, ytest)
    # dfyproba = batter_predict(model, xtest, ytest)
    # print("=" * 60, "signal")
    # print(dfyproba.head())
    # print_ft_important(model)


PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,
    "learning_rate": 0.03,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1,
}


def walk_forward_train(df, params=PARAMS, n_splits=5):
    """
    擴張窗口 Walk-forward 訓練

    每個 fold 的訓練資料逐步擴張（累積所有歷史），
    驗證資料為該 fold 之後的時間段。
    使用 early stopping 防止過擬合。

    Parameters
    ----------
    df       : 完整特徵 df，需包含 date, stock_id, target, f_* 欄位
    params   : LGBM 參數字典，預設使用 PARAMS
    n_splits : fold 數量，預設5

    Returns
    -------
    models : list，所有 fold 的 LGBMRegressor 模型
    scores : list，每個 fold 的 IC（Information Coefficient）

    Notes
    -----
    實際選股時使用最後2個fold的模型平均（models[-2:]），
    因為這兩個 fold 的訓練資料最新，最貼近當前市場。
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    tss = TimeSeriesSplit(n_splits=n_splits)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    scores, models = [], []

    for fold, (train_idx, val_idx) in enumerate(tss.split(dates)):
        train = df[df["date"].isin(dates[train_idx])]
        val = df[df["date"].isin(dates[val_idx])]
        X_train, y_train = train[feature_cols], train["target"]
        X_val, y_val = val[feature_cols], val["target"]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )

        score = np.corrcoef(model.predict(X_val), y_val)[0, 1]
        print(f"Fold {fold+1} IC: {score:.4f}")
        scores.append(score)
        models.append(model)

    print(f"\n平均 IC: {np.mean(scores):.4f}")
    return models, scores


def evaluate_selection(df, models, start_date=None, forward_days=20, top_n=20):
    """
    驗證 A 模型選股品質

    A 模型的 target 是最大漲幅，不適合用固定出場的累積報酬衡量。
    正確的驗證方式是：
      - 勝率：選出的股票有多少比例在 forward_days 內跑贏大盤
      - 最大漲幅：選出的股票平均能達到多高的超額報酬

    Parameters
    ----------
    df           : 完整特徵 df，需包含 date, stock_id, target, f_* 欄位
    models       : walk_forward_train 回傳的模型列表
    start_date   : 驗證起始日，若無則使用全部資料
    forward_days : 換倉頻率（天），預設20
    top_n        : 每期選幾支股票，預設20

    Returns
    -------
    result_df : 每期統計結果（勝率、漲幅）
    stock_df  : 每期選出的個股明細
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    all_dates = sorted(df["date"].unique())
    if start_date:
        all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
    rebalance_dates = all_dates[::forward_days]

    all_results, all_stocks, all_bottom = [], [], []

    for date in rebalance_dates:
        today = df[df["date"] == date]
        if len(today) == 0:
            continue

        preds = np.mean([m.predict(today[feature_cols]) for m in models[-2:]], axis=0)
        today = today.copy()
        today["pred_score"] = preds

        top = today.nlargest(top_n, "pred_score")
        bottom = today.nsmallest(top_n, "pred_score")

        all_results.append(
            {
                "date": date,
                "win_rate_0": (top["target"] > 0).mean(),
                "win_rate_5": (top["target"] > 0.05).mean(),
                "win_rate_10": (top["target"] > 0.10).mean(),
                "avg_max_return": top["target"].mean(),
                "median_return": top["target"].median(),
                "min_return": top["target"].min(),
                "max_return": top["target"].max(),
                "bot_win_rate_0": (bottom["target"] > 0).mean(),
                "bot_win_rate_5": (bottom["target"] > 0.05).mean(),
                "bot_avg_return": bottom["target"].mean(),
                "bot_max_return": bottom["target"].max(),
                "bot_min_return": bottom["target"].min(),
            }
        )
        for stock_id, target in zip(top["stock_id"], top["target"]):
            all_stocks.append({"date": date, "stock_id": stock_id, "target": target})
        for stock_id, target in zip(bottom["stock_id"], bottom["target"]):
            all_bottom.append({"date": date, "stock_id": stock_id, "target": target})

    result_df = pd.DataFrame(all_results)
    stock_df = pd.DataFrame(all_stocks)
    bottom_df = pd.DataFrame(all_bottom)

    print("=" * 50)
    print(f"驗證期間：{result_df['date'].min().date()} ~ {result_df['date'].max().date()}")
    print(f"總期數：{len(result_df)}")
    print(f"\n【前{top_n}名（模型看好）】")
    print(f"勝率 > 0%  : {result_df['win_rate_0'].mean():.1%}")
    print(f"勝率 > 5%  : {result_df['win_rate_5'].mean():.1%}")
    print(f"勝率 > 10% : {result_df['win_rate_10'].mean():.1%}")
    print(f"平均最大超額：  {result_df['avg_max_return'].mean():.2%}")
    print(f"中位數最大超額：{result_df['median_return'].mean():.2%}")
    print(f"平均最差股票：  {result_df['min_return'].mean():.2%}")
    print(f"平均最強股票：  {result_df['max_return'].mean():.2%}")
    print(f"\n【後{top_n}名（模型看壞）】")
    print(f"勝率 > 0%  : {result_df['bot_win_rate_0'].mean():.1%}")
    print(f"勝率 > 5%  : {result_df['bot_win_rate_5'].mean():.1%}")
    print(f"平均最大超額：  {result_df['bot_avg_return'].mean():.2%}")
    print(f"平均最差股票：  {result_df['bot_min_return'].mean():.2%}")
    print(f"平均最強股票：  {result_df['bot_max_return'].mean():.2%}")
    print(f"\n【前後對比（區別能力）】")
    print(f"勝率差距：  {result_df['win_rate_0'].mean() - result_df['bot_win_rate_0'].mean():.1%}")
    print(f"超額差距：  {result_df['avg_max_return'].mean() - result_df['bot_avg_return'].mean():.2%}")
    print("=" * 50)

    return result_df, stock_df, bottom_df


def analyze_frequent(stock_df, min_count=5):
    """
    分析常客股票的實際表現

    常客股票是模型持續看好的股票，值得深入研究。
    若表現持續優異，可作為 B 模型的優先候選。

    Parameters
    ----------
    stock_df  : evaluate_selection 回傳的 stock_df
    min_count : 最少出現幾次才算常客，預設5次

    Returns
    -------
    summary : 每支常客股票的出現次數、平均超額、勝率、最大/最小漲幅
    """
    stock_counts = stock_df["stock_id"].value_counts()
    frequent = stock_counts[stock_counts >= min_count].index.tolist()
    freq_df = stock_df[stock_df["stock_id"].isin(frequent)]

    summary = (
        freq_df.groupby("stock_id")["target"]
        .agg(
            [
                ("出現次數", "count"),
                ("平均超額", "mean"),
                ("勝率", lambda x: (x > 0).mean()),
                ("最大漲幅", "max"),
                ("最小漲幅", "min"),
            ]
        )
        .sort_values("平均超額", ascending=False)
    )

    print(f"\n=== 常客股票（出現 {min_count} 次以上）===")
    print(f"總共 {len(frequent)} 支")
    print(summary.round(3).to_string())
    return summary


def calc_feature_ic(df, period_start, period_end):
    """
    計算指定期間內每個特徵的 IC（Information Coefficient）

    用於定期監控特徵是否失效。
    建議每季末執行一次，IC < 0.02 的特徵可考慮移除或替換。

    Parameters
    ----------
    df           : 完整特徵 df，需包含 date, f_* 欄位, target
    period_start : 計算期間起始日（字串或 datetime）
    period_end   : 計算期間結束日（字串或 datetime）

    Returns
    -------
    ic_df : 每個特徵的 IC 和 abs_IC，依 abs_IC 降序排列
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= period_start) & (df["date"] <= period_end)
    period_df = df[mask].dropna()
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    ic_results = []
    for col in feature_cols:
        ic = np.corrcoef(period_df[col].fillna(0), period_df["target"])[0, 1]
        ic_results.append({"feature": col, "ic": ic, "abs_ic": abs(ic)})

    return pd.DataFrame(ic_results).sort_values("abs_ic", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# TRADE
# ══════════════════════════════════════════════════════════════


def select_stocks(
    df_today, models, df_market_history, top_n=20, use_filter=True, vol_threshold=0.008, trend_threshold=0
):
    """
    每日選股，回傳候選股票清單交給 B 模型操作

    使用最後2個fold的模型平均預測，
    因為這兩個模型的訓練資料最新，最貼近當前市場規律。

    市場狀態過濾（use_filter=True 時啟用）：
      - 大盤過去20日波動率 > vol_threshold（排除過於平靜的市場）
      - 大盤過去20日報酬 > trend_threshold（排除趨勢向下的市場）
      兩個條件同時滿足才進行選股，否則回傳 None（空手）。

    Parameters
    ----------
    df_today           : 今天的特徵資料（已跑完 add_feature 的單日資料）
    models             : walk_forward_train 回傳的模型列表
    df_market_history  : 大盤歷史資料，至少需要60個交易日
                         欄位需包含 date, close
    top_n              : 選幾支股票，預設20
    use_filter         : 是否啟用市場狀態過濾，預設開啟
    vol_threshold      : 波動率門檻，預設 0.008
    trend_threshold    : 趨勢門檻，預設 0（大盤20日報酬 > 0 才交易）

    Returns
    -------
    DataFrame（stock_id, pred_score）依 pred_score 降序排列
    或 None（市場狀態不佳，建議空手）
    """
    if use_filter:
        mkt = df_market_history.copy().sort_values("date")
        market_vol = mkt["close"].pct_change(1).rolling(20).std().iloc[-1]
        market_trend = mkt["close"].pct_change(20).iloc[-1]

        if not ((market_vol > vol_threshold) and (market_trend > trend_threshold)):
            print(f"市場狀態不佳，今日不交易")
            print(f"波動率：{market_vol:.4f}  門檻：{vol_threshold}")
            print(f"20日趨勢：{market_trend:.2%}  門檻：{trend_threshold:.2%}")
            return None

    feature_cols = [c for c in df_today.columns if c.startswith("f_")]

    results = []

    for date, group in df_today.groupby("date"):
        preds = np.mean([m.predict(group[feature_cols]) for m in models[-2:]], axis=0)
        group = group.copy()
        group["pred_score"] = preds

        top = (
            group[["date", "stock_id", "pred_score"]]
            .sort_values("pred_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        results.append(top)

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


main()
# predict(
#     stocks=random.sample(parquet_db.query_stocks_no_etf(), 500),
#     st="2024-01-01",
#     end="2099-01-01",
# )
