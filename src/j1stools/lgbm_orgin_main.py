from datetime import datetime, timedelta
from re import L
from time import time

import random

from numpy import add
import pandas as pd
from regex import T
from websockets import Data

from j1stools import parquet_db
from j1stools.CONFIG import (
    BaseDataBuilderConfig,
    BaseLabelConfig,
    LgbmTrainConfig,
    MACDDataBuilterConfig,
)
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, TRAIN_TYPE, MODEL_TYPE


from j1stools.data_builder import DataBuilder
from j1stools.model_builder import gen_lgbm_c_model, gen_lgbm_orgin_model
from j1stools.train_flow import start_train
import joblib


def add_rfc_feature(df, data: BaseDataBuilderConfig):
    from j1stools import rfc_main
    signal = rfc_main.test(
        stocks=data.stocks,
        st=data.st,
        end=data.end,
        pick_import_feature=data.pick_import_feature,
    )
    signal.rename(columns={"y_proba": "f_rfc"}, inplace=True)
    signal = signal[["date", "stock_id", "f_rfc"]]
    #
    return pd.merge(
        df,
        signal[["date", "stock_id", "f_rfc"]],
        on=["date", "stock_id"],
        how="left",  # 只取索引部分  # 以全時段為準
    ).fillna(
        0
    )  # 沒預測到的（ATR太小的）補 0


def query_no_rfc(stocks, st, end):

    model = joblib.load("models/lgbm_20231231_no_rfc.joblib")
    return exec(
        stocks,
        st,
        end,
        0,
        model,
    )


def predict(stocks, st, end):
    model = joblib.load("models/lgbm.joblib")
    return exec(
        stocks=stocks,
        st=st,
        end=end,
        model=model,
        is_using_rfc=True,
        pick_import_feature=False,
        run_type=TRAIN_TYPE.predict,
    )


def main():
    end = "2025-12-29"
    train = LgbmTrainConfig()
    train.model_type = MODEL_TYPE.lgbm_orgin
    cfg = MACDDataBuilterConfig()
    cfg.feature_type = FEATURE_TYPE.test_lgbm_feature
    cfg.train_config = train
    cfg.st = add_day(end, days=-1000)
    cfg.end = end
    cfg.atrcfg = FILTER_TYPE.del_
    data = DataBuilder(cfg).build()

    data.xtest = (
        data.xtest[[col for col in data.xtest.columns if col.startswith("f_")]] if data.xtest is not None else None
    )
    data.xtrain = (
        data.xtrain[[col for col in data.xtrain.columns if col.startswith("f_")]] if data.xtrain is not None else None
    )
    data.xval = data.xval[[col for col in data.xval.columns if col.startswith("f_")]] if data.xval is not None else None

    st = time()
    # train
    import lightgbm as lgb

    X = data.xtrain
    y = data.ytrain
    train_data = lgb.Dataset(X, label=y, categorical_feature=["f_group"], free_raw_data=False)
    params = {
        # 任務類型
        "objective": "binary",  # 二元分類，產出機率
        "metric": "auc",  # 評估信心排序的準確度
        "boosting_type": "gbdt",
        # 模型結構 (針對股市噪音多，不宜太深)
        "num_leaves": 20,  # 限制葉子數量，防止過擬合
        "max_depth": 5,  # 限制深度
        "min_data_in_leaf": 50,  # 每個葉子最少要有50筆資料
        # 隨機性 (增加泛化能力)
        "learning_rate": 0.01,  # 學習率
        "feature_fraction": 0.7,  # 每次選取 70% 的 f_ 特徵
        "bagging_fraction": 0.8,  # 每次選取 80% 的樣本
        "bagging_freq": 5,
        # 類別特徵優化 (針對 f_group)
        "cat_smooth": 10,  # 減少小產業類別的波動
        # 處理不平衡數據 (好股票通常較少)
        "is_unbalance": True,  # 將數據分成好股票與壞股票
        "verbose": -1,  # 隱藏多餘訊息
        "seed": 42,  # 固定隨機種子
    }
    # 訓練模型
    # num_boost_round 決定疊加幾棵樹
    model = lgb.train(params, train_data, num_boost_round=100)
    #############################################################
    # test
    # expected_features = train_cfg.model.feature_names_in_
    # data.xtest = data.xtest[expected_features]

    # 取得信心分數 (機率值)
    X = data.xtest
    X["confidence_score"] = model.predict(X)
    print("【feature_importance】")
    importance = model.feature_importance(importance_type="gain")
    print(importance)
    #############################################################
    print("【信心分數最高的前 10 名標的】")
    # X = X[["confidence_score"]]
    X.reset_index(inplace=True)
    # print(X.head())
    test_df = X
    # 假設你的測試集/最新資料叫 test_df
    # 1. 取得最新一天的資料 (通常我們只看最近期的評分)
    latest_date = test_df["date"].max()
    top_picks = test_df[test_df["date"] == latest_date].sort_values(by="confidence_score", ascending=False)

    print(f"--- 基準日期: {latest_date} ---")
    print(f"【信心分數最高的前 10 名標的】{end}")
    # 印出 股票代碼, 產業類別, 信心分數, 以及核心特徵 (如券資比)
    cols_to_show = ["stock_id", "f_group", "confidence_score", "f_short_margin_ratio", "f_alpha_5d"]
    # print(top_picks[cols_to_show].head(10))

    #############################################################

    # 只顯示最後一天（最新）的預測結果，並帶上日期
    latest_date = test_df["date"].max()
    result_with_date = test_df[test_df["date"] == latest_date][["date", "stock_id", "f_group", "confidence_score"]]
    result_with_date.sort_values(by="confidence_score", ascending=False).to_csv("signal_today.csv", index=False)
    top10 = result_with_date.sort_values(by="confidence_score", ascending=False).head(10)
    # top10: pd.DataFrame = top10[top10["confidence_score"] > 0.9]
    if top10.shape[0] == 0:
        print("沒有符合信心門檻的股票。")
        return

    # print(result_with_date.sort_values(by="confidence_score", ascending=False).head(10))

    # 搭配你的 RFC 訊號 (假設訊號欄位叫 rfc_signal)
    # 只有當 RFC 有訊號且 LGBM 信心分 > 0.7 時才動作
    # final_buy_list = df[(df["rfc_signal"] == 1) & (df["confidence_score"] > 0.7)]

    top10stocks = top10["stock_id"].to_list()
    st = cfg.end
    end = add_day(st, 10)
    df10day = parquet_db.query_price(top10stocks, st, end)
    # print(df10day.head())
    # print(df10day)
    print(top10.head(10))
    back_test(df10day)
    #############################################################

    # 1. 取得預測分數 (建議用這組洗乾淨的特徵)
    # 2. 依照分數分成五組 (由低到高)
    # test_df["rank"] = pd.qcut(test_df["confidence_score"], 5, labels=["級低", "低", "中", "高", "極高"])

    # # 3. 計算每一組的「平均真實漲幅」
    # # 假設你有存 10 日後的報酬率欄位 f_ret_10d
    # performance = test_df.groupby("rank")["f_ret_10d"].mean()

    # print("--- 信心等級 vs 實際報酬率 ---")
    # print(performance)


def back_test(df):
    import pandas as pd
    import numpy as np

    # ==========================================
    # 🛠️ 自訂參數設定區
    # ==========================================
    UP_THRESHOLD = 0.10  # 上漲門檻 (10%)
    DOWN_THRESHOLD = 0.10  # 下跌門檻 (10%)
    # ==========================================

    # 【核心修正】強制清洗 stock_id 欄位，去除前後空格並轉為字串
    df["stock_id"] = df["stock_id"].astype(str).str.strip()

    # 1. 確保時間排序正確
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 2. 找出基準價與第一天
    df["base_close"] = df.groupby("stock_id")["close"].transform("first")
    df["is_first_day"] = df["date"] == df.groupby("stock_id")["date"].transform("first")

    # 3. 排除第一天後，計算每日是否觸發門檻
    df["hit_up"] = (~df["is_first_day"]) & (
        (df["high"] >= df["base_close"] * (1 + UP_THRESHOLD)) | (df["close"] >= df["base_close"] * (1 + UP_THRESHOLD))
    )
    df["hit_down"] = (~df["is_first_day"]) & (
        (df["low"] <= df["base_close"] * (1 - DOWN_THRESHOLD))
        | (df["close"] <= df["base_close"] * (1 - DOWN_THRESHOLD))
    )

    # 4. 定義逐股時間優先權判定函式
    def 判定第一時間順序(group):
        up_indices = group.index[group["hit_up"]]
        down_indices = group.index[group["hit_down"]]

        first_up = up_indices[0] if len(up_indices) > 0 else float("inf")
        first_down = down_indices[0] if len(down_indices) > 0 else float("inf")

        if first_up == float("inf") and first_down == float("inf"):
            return 0
        if first_up < first_down:
            return 2
        elif first_down < first_up:
            return 1
        else:
            當天資料 = group.loc[first_up]
            return 2 if 當天資料["close"] >= 當天資料["base_close"] else 1

    # 5. 計算每檔股票的最終唯一標籤
    每檔股票標籤 = df.groupby("stock_id", group_keys=False).apply(判定第一時間順序)

    # 6. 將結果對應回原始長表格
    df["target_label"] = df["stock_id"].map(每檔股票標籤.to_dict())

    # 7. 移除暫存欄位
    df = df.drop(columns=["base_close", "is_first_day", "hit_up", "hit_down"])

    # -------------------------------------------------------------
    # 📊 精確數量統計報告
    # -------------------------------------------------------------
    總股票數 = len(每檔股票標籤)
    次數分配 = 每檔股票標籤.value_counts()

    print("====== 📝 精確時間序機率分析 ======")
    print(f"📊 測試設定：上漲 +{UP_THRESHOLD*100:.1f}% | 下跌 -{DOWN_THRESHOLD*100:.1f}%")
    print(f"實際辨識出的獨立股票代碼：{list(每檔股票標籤.index)}")
    print(f"總股票數量檢查：{總股票數} 檔\n")

    for 狀態 in [2, 1, 0]:
        個數 = 次數分配.get(狀態, 0)
        機率 = (個數 / 總股票數) * 100 if 總股票數 > 0 else 0
        涵義 = {2: "先上漲突破", 1: "先下跌跌破", 0: "兩者皆未觸發"}[狀態]
        print(f"🔹 狀態 {狀態} ({涵義}) -> 數量: {個數} 檔 | 機率: {機率:.2f}%")
    print("====================================")


def add_day(date_str, days=10):
    # 1. 將字串轉為 datetime 物件
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # 2. 加上 10 天
    new_date_obj = date_obj + timedelta(days=days)

    # 3. 轉回字串格式
    return new_date_obj.strftime("%Y-%m-%d")


# def main1():
#     train = LgbmTrainConfig()
#     train.is_use_rfc = False
#     train.model = gen_lgbm_orgin_model()
#     # train.rfc_proba = add_rfc_feature(df, data)
#     # train.model = (joblib.load("models/rfc_macd.joblib"),)
#     ##############################################################
#     # l = BaseLabelConfig()
#     # l.hold_days = 20
#     # l.profit_target = 0.1
#     # l.stop_loss = -0.1
#     #############################################################
#     data = MACDDataBuilterConfig()
#     # data.is_continuous = True
#     # data.label_cfg = l
#     data.feature_type = FEATURE_TYPE.test_lgbm_feature
#     # data.feature_type = FEATURE_TYPE.today
#     data.atrcfg = FILTER_TYPE.none_
#     data.train_config = train
#     data.st = "2024-01-01"
#     data.end = "2099-01-01"
#     signal = start_train_flow(cfg=data)
#     #############################################################
#     signal = signal[signal["y_proba"] > 0.5]
#     signal.sort_values(by=["date", "y_proba"], inplace=True)
#     signal.to_csv("signal_today.csv", index=False)


import pandas as pd


def backtest_validation(df, model, features, threshold=0.05, confidence_cutoff=0.8):
    """
    df: 包含特徵與原始量價的長表格 (需有 f_ 開頭特徵與 high, close)
    model: 訓練好的 LGBM 原生模型
    features: 訓練時使用的 f_ 開頭特徵清單
    threshold: 漲幅目標 (0.05 代表 5%)
    confidence_cutoff: 你認為多少分算「有信心」
    """
    # 1. 準備驗證用的資料 (排除掉最後 10 天，因為那些還沒有答案)
    # 這裡假設你的 df 是按日期排序的
    dates = sorted(df["date"].unique())
    if len(dates) < 11:
        print("資料長度不足以進行 10 日驗證")
        return None

    # 選取一個「有答案」的最晚日期作為測試基準
    # 例如：倒數第 11 天，這樣我們剛好可以看到它後面 10 天的表現
    test_date = dates[-11]
    test_set = df[df["date"] == test_date].copy()

    # 2. 執行信心預測 (對答案前先打分)
    test_set["confidence_score"] = model.predict(test_set[features])

    # 3. 計算未來 10 天的真實表現 (對答案)
    # 從原始全量 df 中抓取 test_date 之後 10 天的最高價
    results = []
    for idx, row in test_set.iterrows():
        sid = row["stock_id"]
        curr_close = row["close"]

        # 抓取該股票在該日期之後 10 筆的最高價
        future_data = df[(df["stock_id"] == sid) & (df["date"] > test_date)].head(10)

        if not future_data.empty:
            max_future_high = future_data["high"].max()
            actual_return = (max_future_high - curr_close) / curr_close
            hit = 1 if actual_return >= threshold else 0
        else:
            actual_return = np.nan
            hit = 0

        results.append({"actual_return": actual_return, "is_success": hit})

    result_df = pd.concat([test_set.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    # 4. 篩選出：模型有信心 (Score > cutoff) 且 真的成功 (is_success == 1) 的股票
    final_verification = result_df[result_df["confidence_score"] >= confidence_cutoff].sort_values(
        "confidence_score", ascending=False
    )

    print(f"--- 驗證基準日: {test_date.date()} ---")
    print(f"模型推薦 (信心 > {confidence_cutoff}) 的股票數量: {len(final_verification)}")

    if len(final_verification) > 0:
        success_rate = final_verification["is_success"].mean()
        print(f"這些推薦股票在 10 日內達成漲幅 {threshold*100}% 的成功率: {success_rate:.2%}")
        print("\n【驗證成功的推薦名單】")
        print(
            final_verification[final_verification["is_success"] == 1][
                ["stock_id", "f_group", "confidence_score", "actual_return"]
            ].head(10)
        )
    else:
        print("沒有符合信心門檻的股票。")

    return final_verification


# 執行驗證

# main()
# predict(
#     stocks=random.sample(parquet_db.query_stocks_no_etf(), 500),
#     st="2024-01-01",
#     end="2099-01-01",
# )
