import json
from math import e
from re import L
import select
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

    model = joblib.load("models/lgbm_timeseries_ensemble.joblib")
    feature_cols = json.load(open("models/feature_cols.json"))

    df_feature, df_market = prepare_data(stocks, st, end, model="predict")
    df_select_stocks = select_stocks(
        df_today=df_feature,
        df_market_history=df_market,
        feature_cols=feature_cols,
        model=model,
        top_n=20,
    )
    if df_select_stocks is None:
        return

    feature_cols = json.load(open("models/feature_cols.json"))
    evaluate_selection(df=df_feature, feature_cols=feature_cols, model=model)

    df_select_stocks = df_select_stocks.sort_values(by=["date", "pred_score"], ascending=[False, False])
    df_select_stocks.to_csv("lgbm_signal_today.csv", index=False)
    # print(df_select_stocks.head())
    return df_select_stocks


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
    # st = "2015-01-01"
    # end = "2024-01-01"  # "2026-02-01"
    # models = train(
    #     stocks=stocks,
    #     st=st,
    #     end=end,
    # )

    predict(
        stocks=stocks,
        st="2024-01-01",
        end="2026-01-01",
    )


def prepare_data(stocks, st, end, model="train"):
    df_margin = lite_db.margin(stocks, st, end)
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
    models, scores = walk_forward_train(df_feature)

    final_model, feature_cols = train_final_model(df_feature, models)
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    joblib.dump(final_model, "models/lgbm_timeseries_ensemble.joblib")

    # feature_ic = calc_feature_ic(df_feature, period_start=st, period_end=end)
    # feature_ic.to_csv("feature_ic.csv", index=False)
    # result_df, stock_df, bottom_df = evaluate_selection(df_feature, models)
    # summary = analyze_frequent(stock_df)

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


def walk_forward_train(df, params=PARAMS, test_ratio=0.2):
    """
    驗證模型有效性

    用最後 test_ratio 的資料驗證，確認模型沒有過擬合。
    職責是評估，不是預測。預測請用 train_final_model。

    Parameters
    ----------
    df         : 完整特徵 df，需包含 date, stock_id, target, f_* 欄位
    params     : LGBM 參數字典，預設使用 PARAMS
    test_ratio : 驗證資料比例，預設最後 20%

    Returns
    -------
    model : 驗證用的 LGBMRegressor（不用於實際預測）
    score : IC（Information Coefficient），衡量模型預測能力
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    split_idx = int(len(dates) * (1 - test_ratio))
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    train = df[df["date"].isin(train_dates)]
    val = df[df["date"].isin(val_dates)]

    X_train, y_train = train[feature_cols], train["target"]
    X_val, y_val = val[feature_cols], val["target"]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    score = np.corrcoef(model.predict(X_val), y_val)[0, 1]

    print(f"訓練期：{pd.to_datetime(train_dates[0]).date()} ~ {pd.to_datetime(train_dates[-1]).date()}")
    print(f"驗證期：{pd.to_datetime(val_dates[0]).date()} ~ {pd.to_datetime(val_dates[-1]).date()}")
    print(f"IC：{score:.4f}")

    return model, score


def evaluate_selection(df, model, feature_cols, start_date=None, forward_days=20, top_n=20):
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

    all_dates = sorted(df["date"].unique())
    if start_date:
        all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
    rebalance_dates = all_dates[::forward_days]

    all_results, all_stocks, all_bottom = [], [], []

    for date in rebalance_dates:
        today = df[df["date"] == date]
        if len(today) == 0:
            continue

        today = today.copy()
        today["pred_score"] = model.predict(today[feature_cols])

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


def calc_feature_ic(df, lookback_days=60):
    """
    計算最近 N 個交易日每個特徵的 IC（Information Coefficient）

    用途：監控市場結構是否改變，決定要不要重新訓練模型
    不用於篩選特徵（篩選特徵請用 Feature Importance）

    使用時機：
      每季末執行一次，觀察 IC 是否大幅下降
      IC 全面下降 → 市場結構改變 → 觸發重新訓練
      IC 某特徵歸零 → 該訊號失效 → 考慮移除該特徵

    Parameters
    ----------
    df           : 完整特徵 df，需包含 date, f_* 欄位, target
    lookback_days: 回顧天數，預設60個交易日（約一季）

    Returns
    -------
    ic_df : 每個特徵的 IC 和 abs_IC，依 abs_IC 降序排列
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 自動抓最後 lookback_days 個交易日
    all_dates = sorted(df["date"].unique())
    cutoff_date = all_dates[-lookback_days] if len(all_dates) >= lookback_days else all_dates[0]
    period_df = df[df["date"] >= cutoff_date].dropna()

    print(f"IC 計算期間：{cutoff_date.date()} ~ {all_dates[-1].date()}（{lookback_days} 個交易日）")

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
    df_today, model, feature_cols, df_market_history, top_n=20, use_filter=True, vol_threshold=0.008, trend_threshold=0
):
    """
    每日選股，回傳候選股票清單交給 B 模型操作

    使用 train_final_model 訓練的單一最終模型預測。

    市場狀態過濾（use_filter=True 時啟用）：
      - 大盤過去20日波動率 > vol_threshold（排除過於平靜的市場）
      - 大盤過去20日報酬 > trend_threshold（排除趨勢向下的市場）
      兩個條件同時滿足才進行選股，否則回傳 None（空手）。

    Parameters
    ----------
    df_today           : 特徵資料，可以是單日或區間
                         已跑完 add_feature（mode="predict"）
    model              : train_final_model 回傳的單一最終模型
    feature_cols       : train_final_model 回傳的特徵欄位列表
                         必須跟訓練時一致
    df_market_history  : 大盤歷史資料，至少需要60個交易日
                         欄位需包含 date, close
    top_n              : 每日選幾支股票，預設20
    use_filter         : 是否啟用市場狀態過濾，預設開啟
    vol_threshold      : 波動率門檻，預設 0.008
    trend_threshold    : 趨勢門檻，預設 0（大盤20日報酬 > 0 才交易）

    Returns
    -------
    DataFrame（date, stock_id, pred_score）依日期和 pred_score 排列
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

    results = []

    for date, group in df_today.groupby("date"):
        group = group.copy()
        group["pred_score"] = model.predict(group[feature_cols])

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


def train_final_model(df, val_model, params=PARAMS, use_importance_filter=True, importance_threshold=0):
    """
    用全部歷史資料訓練最終模型，專門用於實際預測

    與 walk_forward_train 不同：
      - 沒有 validation set，不計算 IC
      - 使用全部資料，充分利用最新市場規律
      - 用 walk_forward_train 回傳的驗證模型的 Feature Importance 篩選特徵
        （Feature Importance 反映模型實際用了哪些特徵，
          考慮非線性和特徵交互，比 IC 更準確）

    Parameters
    ----------
    df                   : 完整特徵 df，需包含 date, target, f_* 欄位
                           mode="train" 的資料（已過濾 target NaN）
    val_model            : walk_forward_train 回傳的單一驗證模型
                           用來取得 Feature Importance
    params               : LGBM 參數，預設使用 PARAMS
    use_importance_filter: 是否用 Feature Importance 篩選特徵，預設開啟
    importance_threshold : 重要性門檻，低於此值的特徵會被移除，預設 0
                           （預設移除完全沒用到的特徵）

    Returns
    -------
    model        : 訓練好的 LGBMRegressor
    feature_cols : 實際使用的特徵欄位列表（預測時需傳入相同特徵）

    Notes
    -----
    回傳 feature_cols 很重要，預測時必須用同樣的特徵欄位。
    特徵篩選用 Feature Importance，不用 IC。
    IC 只用來監控市場結構是否改變，決定要不要重新訓練。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [c for c in df.columns if c.startswith("f_")]

    if use_importance_filter:
        # 用驗證模型的 gain 篩選特徵
        # gain = 每次分裂帶來的誤差改善，比 split（次數）更能反映特徵真正的貢獻
        importance = pd.DataFrame(
            {"feature": feature_cols, "importance": val_model.booster_.feature_importance(importance_type="gain")}
        ).sort_values("importance", ascending=False)

        feature_cols = importance[importance["importance"] > importance_threshold]["feature"].tolist()

        print(f"Feature Importance（gain）篩選：保留 {len(feature_cols)} 個特徵")

    X = df[feature_cols]
    y = df["target"]

    # 移除特徵有 NaN 的資料
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    print(f"train_final_model 完成：{len(X)} 筆資料，{len(feature_cols)} 個特徵")
    return model, feature_cols


def work_flow():
    # 1. 評估模型
    models, scores = walk_forward_train(df_feat, params)

    # 2. 訓練最終模型（用 Feature Importance 篩選）
    final_model, feature_cols = train_final_model(df_feat, models)

    # 3. 每季監控（只是警報器，不篩選特徵）
    ic_df = calc_feature_ic(df_feat)
    # IC 全面下降 → 重新跑步驟 1 和 2

    # 4. 預測
    df_today = add_feature(df, df_ibbuysell, df_market, mode="predict")
    candidates = select_stocks(df_today, [final_model], df_market_history)


# main()
# predict(
#     stocks=random.sample(parquet_db.query_stocks_no_etf(), 500),
#     st="2024-01-01",
#     end="2099-01-01",
# )
