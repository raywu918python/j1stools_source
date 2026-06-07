"""
簡單模型檢查：三角收斂在交易上有沒有用

目標：
    1. 用 triangle human_only / keep_best 當訊號樣本
    2. 建立未來 N 天最高報酬是否達標的 label
    3. 用簡單 RandomForest 做時間切分驗證
    4. 比較「全部三角」和「模型高分三角」的勝率/平均報酬

快速使用：
    PYTHONPATH=src python src/j1stools/ma60_triangle.py

Notebook 使用：
    from j1stools.ma60_triangle import run_from_cache
    result, signals = run_from_cache()
"""

import os
import warnings

import numpy as np
import pandas as pd

from j1stools import parquet_db
from j1stools.pattern_triangle import (
    add_triangle_compare_columns,
    add_triangle_overlap_info,
    filter_triangle_by_overlap,
    find_triangle,
    load_or_create_triangle_cache,
)

warnings.filterwarnings("ignore")


FORWARD_DAYS = 20
TARGET_RET = 0.10
TEST_RATIO = 0.30
MA60_BREAKOUT_WINDOW = 10
LONG_CACHE_PATH = "triangle_human_cache_2021_2025.pkl"

PRICE_FEATURES = [
    "f_ret_3d",  # 近 3 日報酬，觀察三角前短線動能
    "f_ret_5d",  # 近 5 日報酬，觀察一週動能
    "f_ret_10d",  # 近 10 日報酬，觀察半個月強弱
    "f_ret_20d",  # 近 20 日報酬，觀察一個月趨勢
    "f_dev_ma20",  # 收盤價偏離 MA20 的比例，判斷是否漲多/貼近均線
    "f_dev_ma60",  # 收盤價偏離 MA60 的比例，判斷中期位置
    "f_ma20_slope",  # MA20 近 5 日斜率，判斷短中期趨勢方向
    "f_ma60_slope",  # MA60 近 10 日斜率，判斷中期趨勢方向
    "f_vol_ratio",  # 5 日均量 / 20 日均量，判斷量能是否放大
    "f_range_pct",  # 當日高低振幅 / 收盤價，判斷波動大小
    "f_body_pct",  # K 棒實體大小 / 收盤價，判斷當天實體強弱
    "f_upper_shadow_pct",  # 上影線比例，判斷上方壓力
    "f_lower_shadow_pct",  # 下影線比例，判斷下方支撐
    "f_rel_mkt_5d",  # 個股 5 日報酬 - 大盤 5 日報酬，判斷短線相對強弱
    "f_rel_mkt_20d",  # 個股 20 日報酬 - 大盤 20 日報酬，判斷月線相對強弱
]

TRIANGLE_FEATURES = [
    "triangle_score",  # fuzzy 三角品質分數，越高代表越像乾淨收斂
    "width_reduction",  # 三角寬度縮小比例，越高代表收斂越明顯
    "overlap_ratio",  # 高點線和低點線的時間重疊比例
    "upper_break_ratio",  # 高價刺破上緣壓力線的比例
    "lower_break_ratio",  # 低價跌破下緣支撐線的比例
    "close_inside_ratio",  # 收盤價落在三角通道內的比例
    "touch_high_count",  # 壓力線使用的高點數，2 或 3
    "touch_low_count",  # 支撐線使用的低點數，2 或 3
    "triangle_span_bars",  # 三角從第一個 pivot 到最後一個 pivot 橫跨幾根 K 棒
    "overlap_pair_count",  # 與同股票其他三角高度重疊的 pair 數
    "overlap_group_size",  # 同一個重疊群內共有幾個三角
    "overlap_rank",  # 在重疊群內的排序，分數高且日期新者排名較前
    "max_overlap_ratio_short",  # 與其他三角最大重疊比例，以較短三角為分母
    "max_overlap_ratio_union",  # 與其他三角最大重疊比例，以聯集長度為分母
    "bars_after_triangle_end_to_ma60_breakout",  # MA60 突破日距離三角最後 pivot 幾根 K
]


def _clean_price(df_price):
    d = df_price.copy()
    d.columns = d.columns.str.strip().str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d["stock_id"] = d["stock_id"].astype(str).str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[d["close"] > 0].dropna(subset=["close"])
    return d.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _clean_market(df_market):
    if df_market is None or len(df_market) == 0:
        return None

    m = df_market.copy()
    m.columns = m.columns.str.strip().str.lower()
    m["date"] = pd.to_datetime(m["date"])
    m["close"] = pd.to_numeric(m["close"], errors="coerce")
    return (
        m[["date", "close"]]
        .rename(columns={"close": "market_close"})
        .dropna(subset=["market_close"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _add_price_features_and_labels(df_price, df_market=None, forward_days=FORWARD_DAYS, target_ret=TARGET_RET):
    price = _clean_price(df_price)
    market = _clean_market(df_market)
    if market is not None:
        price = price.merge(market, on="date", how="left")
    else:
        price["market_close"] = np.nan

    rows = []
    for _, grp in price.groupby("stock_id"):
        d = grp.sort_values("date").reset_index(drop=True).copy()
        d["bar_index"] = np.arange(len(d))
        c = d["close"]

        d["f_ret_3d"] = c.pct_change(3)
        d["f_ret_5d"] = c.pct_change(5)
        d["f_ret_10d"] = c.pct_change(10)
        d["f_ret_20d"] = c.pct_change(20)

        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        d["ma60"] = ma60
        d["ma60_breakout_up"] = (c > ma60) & (c.shift(1) <= ma60.shift(1))
        d["f_dev_ma20"] = (c - ma20) / ma20.clip(lower=1e-9)
        d["f_dev_ma60"] = (c - ma60) / ma60.clip(lower=1e-9)
        d["f_ma20_slope"] = (ma20 - ma20.shift(5)) / ma20.shift(5).clip(lower=1e-9)
        d["f_ma60_slope"] = (ma60 - ma60.shift(10)) / ma60.shift(10).clip(lower=1e-9)

        vol = d["volume"]
        d["f_vol_ratio"] = vol.rolling(5).mean() / vol.rolling(20).mean().clip(lower=1e-9)

        high = d["high"].fillna(c)
        low = d["low"].fillna(c)
        open_ = d["open"].fillna(c)
        day_range = (high - low).replace(0, np.nan)
        d["f_range_pct"] = (high - low) / c
        d["f_body_pct"] = (c - open_).abs() / c
        d["f_upper_shadow_pct"] = (high - np.maximum(open_, c)) / day_range
        d["f_lower_shadow_pct"] = (np.minimum(open_, c) - low) / day_range

        m = d["market_close"]
        d["f_rel_mkt_5d"] = c.pct_change(5) - m.pct_change(5)
        d["f_rel_mkt_20d"] = c.pct_change(20) - m.pct_change(20)

        future_high = pd.concat([high.shift(-i) for i in range(1, forward_days + 1)], axis=1).max(axis=1)
        future_low = pd.concat([low.shift(-i) for i in range(1, forward_days + 1)], axis=1).min(axis=1)
        future_close = c.shift(-forward_days)

        d["future_max_ret"] = future_high / c - 1
        d["future_min_ret"] = future_low / c - 1
        d["future_close_ret"] = future_close / c - 1
        d["label"] = np.where(d["future_max_ret"].notna(), (d["future_max_ret"] >= target_ret).astype(int), np.nan)

        rows.append(d)

    return pd.concat(rows, ignore_index=True)


def _ensure_triangle_end_date(tri):
    if "triangle_end_date" in tri.columns:
        tri["triangle_end_date"] = pd.to_datetime(tri["triangle_end_date"], errors="coerce")
        return tri

    pivot_date_cols = [
        col
        for col in ["h1_date", "h2_date", "h3_date", "l1_date", "l2_date", "l3_date"]
        if col in tri.columns
    ]
    if not pivot_date_cols:
        tri["triangle_end_date"] = pd.to_datetime(tri["date"], errors="coerce")
        return tri

    for col in pivot_date_cols:
        tri[col] = pd.to_datetime(tri[col], errors="coerce")
    tri["triangle_end_date"] = tri[pivot_date_cols].max(axis=1)
    return tri


def _apply_ma60_breakout_window(tri, price_feat, window_bars=MA60_BREAKOUT_WINDOW):
    '''
    過濾三角訊號：
        MA60 上穿日必須落在三角最後 pivot 後 window_bars 根 K 以內。

    通過後會把樣本 date 改成 ma60_breakout_date，
    讓模型特徵與 label 都以真正可能進場的 MA60 突破日計算。
    '''
    tri = _ensure_triangle_end_date(tri.copy())
    tri["triangle_signal_date"] = pd.to_datetime(tri["date"], errors="coerce")

    price_lookup = price_feat[
        ["stock_id", "date", "bar_index", "close", "ma60", "ma60_breakout_up"]
    ].copy()
    price_lookup["date"] = pd.to_datetime(price_lookup["date"])
    price_lookup["stock_id"] = price_lookup["stock_id"].astype(str).str.strip()

    rows = []
    for _, row in tri.iterrows():
        stock_price = price_lookup[price_lookup["stock_id"] == row["stock_id"]]
        if stock_price.empty or pd.isna(row["triangle_end_date"]):
            continue

        end_match = stock_price[stock_price["date"] >= row["triangle_end_date"]].head(1)
        if end_match.empty:
            continue

        end_bar = int(end_match.iloc[0]["bar_index"])
        window = stock_price[
            (stock_price["bar_index"] >= end_bar)
            & (stock_price["bar_index"] <= end_bar + window_bars)
            & (stock_price["ma60_breakout_up"])
        ].head(1)
        if window.empty:
            continue

        breakout = window.iloc[0]
        out = row.copy()
        out["date"] = breakout["date"]
        out["ma60_breakout_date"] = breakout["date"]
        out["triangle_end_bar"] = end_bar
        out["ma60_breakout_bar"] = int(breakout["bar_index"])
        out["bars_after_triangle_end_to_ma60_breakout"] = int(breakout["bar_index"] - end_bar)
        out["ma60_breakout_close"] = breakout["close"]
        out["ma60_breakout_ma60"] = breakout["ma60"]
        rows.append(out)

    if not rows:
        return tri.iloc[0:0].copy()

    result = pd.DataFrame(rows).reset_index(drop=True)
    result["ma60_breakout_within_triangle_window"] = True
    return result


def _prepare_triangle_signals(
    df_price,
    triangle_df=None,
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
):
    if triangle_df is not None:
        tri_source = triangle_df.copy()
    else:
        tri_source = find_triangle(df_price)

    if "human_only" in tri_source.columns:
        if overlap_filter == "all":
            tri = tri_source[tri_source["human_only"].fillna(False)].copy()
        else:
            _, tri, _ = filter_triangle_by_overlap(
                tri_source,
                signal_col="human_only",
                overlap_filter=overlap_filter,
                min_overlap_ratio=min_overlap_ratio,
            )
    else:
        tri = tri_source.copy()

    tri = tri.copy()
    tri.columns = tri.columns.str.strip()
    tri["date"] = pd.to_datetime(tri["date"])
    tri["stock_id"] = tri["stock_id"].astype(str).str.strip()
    return tri


def build_dataset(
    df_price,
    df_market=None,
    triangle_df=None,
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    forward_days=FORWARD_DAYS,
    target_ret=TARGET_RET,
    require_ma60_breakout_window=True,
    ma60_breakout_window=MA60_BREAKOUT_WINDOW,
):
    """
    建立模型資料集。

    回傳每一筆三角訊號，包含：
        - 三角本身的幾何特徵
        - 訊號日的價格/均線/量能特徵
        - 未來 forward_days 的報酬 label
    """
    price_feat = _add_price_features_and_labels(
        df_price,
        df_market=df_market,
        forward_days=forward_days,
        target_ret=target_ret,
    )
    tri = _prepare_triangle_signals(
        df_price,
        triangle_df=triangle_df,
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
    )

    before_ma60_filter = len(tri)
    if require_ma60_breakout_window:
        tri = _apply_ma60_breakout_window(
            tri,
            price_feat,
            window_bars=ma60_breakout_window,
        )
        print(f"MA60 突破需在三角最後日期後 {ma60_breakout_window} 根 K 內：{before_ma60_filter:,} -> {len(tri):,}")

    merge_cols = ["date", "stock_id"]
    keep_price_cols = merge_cols + PRICE_FEATURES + [
        "close",
        "ma60",
        "ma60_breakout_up",
        "future_max_ret",
        "future_min_ret",
        "future_close_ret",
        "label",
    ]
    data = tri.merge(price_feat[keep_price_cols], on=merge_cols, how="left", suffixes=("", "_price"))
    data = data.dropna(subset=["label", "future_max_ret", "future_close_ret"])
    data["label"] = data["label"].astype(int)
    return data.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _make_model_matrix(data):
    feature_cols = [c for c in PRICE_FEATURES + TRIANGLE_FEATURES if c in data.columns]
    x = data[feature_cols].copy()

    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        if x[col].notna().sum() == 0:
            x[col] = 0.0
        else:
            x[col] = x[col].fillna(x[col].median())

    if "triangle_match_type" in data.columns:
        # 將 fuzzy_2h2l / fuzzy_3h2l / fuzzy_2h3l / fuzzy_3h3l 轉成模型可吃的 0/1 欄位。
        cat = pd.get_dummies(data["triangle_match_type"].fillna("unknown"), prefix="type")
        x = pd.concat([x, cat], axis=1)

    return x, list(x.columns)


def _time_split(data, test_ratio=TEST_RATIO):
    dates = np.array(sorted(data["date"].dropna().unique()))
    if len(dates) < 4:
        raise ValueError("日期太少，無法做時間切分")

    cut_idx = int(len(dates) * (1 - test_ratio))
    cut_idx = min(max(cut_idx, 1), len(dates) - 1)
    split_date = pd.Timestamp(dates[cut_idx])

    train_idx = data["date"] < split_date
    test_idx = data["date"] >= split_date
    return train_idx, test_idx, split_date


def _selection_stats(df, name, score_col=None):
    if len(df) == 0:
        return {
            "group": name,
            "count": 0,
            "win_rate": np.nan,
            "avg_max_ret": np.nan,
            "avg_close_ret": np.nan,
            "avg_min_ret": np.nan,
            "median_close_ret": np.nan,
            "avg_score": np.nan,
        }

    return {
        "group": name,
        "count": len(df),
        "win_rate": df["label"].mean(),
        "avg_max_ret": df["future_max_ret"].mean(),
        "avg_close_ret": df["future_close_ret"].mean(),
        "avg_min_ret": df["future_min_ret"].mean(),
        "median_close_ret": df["future_close_ret"].median(),
        "avg_score": df[score_col].mean() if score_col and score_col in df.columns else np.nan,
    }


def _print_report(report):
    show = report.copy()
    for col in ["win_rate", "avg_max_ret", "avg_close_ret", "avg_min_ret", "median_close_ret", "avg_score"]:
        if col in show.columns:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    print(show.to_string(index=False))


def train_simple_model(dataset, test_ratio=TEST_RATIO, random_state=42):
    """
    用時間切分訓練簡單 RandomForest，回傳：
        result_df      : 各組交易統計
        signals_with_pred : 原始樣本 + pred_score
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, roc_auc_score

    data = dataset.copy().sort_values(["date", "stock_id"]).reset_index(drop=True)
    x, feature_cols = _make_model_matrix(data)
    train_idx, test_idx, split_date = _time_split(data, test_ratio=test_ratio)

    train = data.loc[train_idx].copy()
    test = data.loc[test_idx].copy()
    x_train = x.loc[train_idx]
    x_test = x.loc[test_idx]
    y_train = train["label"]
    y_test = test["label"]

    if len(train) < 100 or len(test) < 30:
        raise ValueError(f"樣本太少：train={len(train)}, test={len(test)}")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("train/test label 只有單一類別，無法訓練分類模型")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred_score = model.predict_proba(x_test)[:, 1]
    pred_label = (pred_score >= 0.5).astype(int)

    test = test.copy()
    test["pred_score"] = pred_score
    test["pred_label"] = pred_label

    rows = [
        _selection_stats(test, "test_all"),
        _selection_stats(test[test["pred_label"] == 1], "pred_score>=0.50", score_col="pred_score"),
    ]
    for pct in [0.30, 0.20, 0.10]:
        n = max(1, int(len(test) * pct))
        rows.append(_selection_stats(test.nlargest(n, "pred_score"), f"top_{int(pct * 100)}%", score_col="pred_score"))

    result = pd.DataFrame(rows)
    auc = roc_auc_score(y_test, pred_score)
    precision = precision_score(y_test, pred_label, zero_division=0)

    print("=" * 72)
    print("三角收斂簡單模型")
    print("=" * 72)
    print(f"樣本數 train/test : {len(train):,} / {len(test):,}")
    print(f"時間切分日期       : {split_date.date()}")
    print(f"特徵數             : {len(feature_cols)}")
    print(f"Train 勝率         : {y_train.mean():.2%}")
    print(f"Test  勝率         : {y_test.mean():.2%}")
    print(f"AUC                : {auc:.3f}")
    print(f"Precision@0.50     : {precision:.3f}")
    print("-" * 72)
    _print_report(result)

    feature_importance = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("-" * 72)
    print(feature_importance.head(20).to_string(index=False))

    data_with_pred = data.copy()
    data_with_pred["pred_score"] = np.nan
    data_with_pred["pred_label"] = np.nan
    data_with_pred.loc[test.index, "pred_score"] = test["pred_score"]
    data_with_pred.loc[test.index, "pred_label"] = test["pred_label"]

    return result, data_with_pred, feature_importance


def run(
    df_price,
    df_market=None,
    triangle_df=None,
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    forward_days=FORWARD_DAYS,
    target_ret=TARGET_RET,
    test_ratio=TEST_RATIO,
    require_ma60_breakout_window=True,
    ma60_breakout_window=MA60_BREAKOUT_WINDOW,
):
    dataset = build_dataset(
        df_price,
        df_market=df_market,
        triangle_df=triangle_df,
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
        forward_days=forward_days,
        target_ret=target_ret,
        require_ma60_breakout_window=require_ma60_breakout_window,
        ma60_breakout_window=ma60_breakout_window,
    )
    return train_simple_model(dataset, test_ratio=test_ratio)


def run_from_cache(
    cache_path="triangle_human_cache.pkl",
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    forward_days=FORWARD_DAYS,
    target_ret=TARGET_RET,
    require_ma60_breakout_window=True,
    ma60_breakout_window=MA60_BREAKOUT_WINDOW,
):
    """
    使用 pattern_triangle.py 產生的快取快速跑模型。

    注意：這個快取通常是某一段時間/某批股票的實驗資料，
    速度快，適合先看三角訊號有沒有初步可用性。
    """
    if not os.path.exists(cache_path):
        load_or_create_triangle_cache(cache_path=cache_path)

    df_source = pd.read_pickle(cache_path)
    df_tagged, triangle_df, _ = filter_triangle_by_overlap(
        df_source,
        signal_col="human_only",
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
    )
    return run(
        df_tagged,
        df_market=None,
        triangle_df=triangle_df,
        overlap_filter="all",
        min_overlap_ratio=min_overlap_ratio,
        forward_days=forward_days,
        target_ret=target_ret,
        require_ma60_breakout_window=require_ma60_breakout_window,
        ma60_breakout_window=ma60_breakout_window,
    )


def build_long_triangle_cache(
    start_date="2021-01-01",
    end_date="2025-12-31",
    cache_path=LONG_CACHE_PATH,
    sample_size=None,
    refresh=False,
    warmup_days=220,
    future_buffer_days=60,
):
    '''
    建立長期間三角快取。

    重點：
        - 價格資料會往前抓 warmup_days，讓 MA60 / pivot 有足夠暖身資料
        - 價格資料會往後抓 future_buffer_days，讓 label 可以看未來報酬
        - 只有 start_date ~ end_date 內的三角會被保留成訊號
        - end_date 後的資料只保留價格，不當成三角訊號，避免偷看未來
    '''
    if os.path.exists(cache_path) and not refresh:
        print(f"讀取長期間快取：{cache_path}")
        return pd.read_pickle(cache_path)

    stocks = parquet_db.query_stocks_no_etf()
    if sample_size:
        rng = np.random.default_rng(42)
        stocks = list(rng.choice(stocks, size=min(sample_size, len(stocks)), replace=False))

    query_start = (pd.Timestamp(start_date) - pd.DateOffset(days=warmup_days)).strftime("%Y-%m-%d")
    query_end = (pd.Timestamp(end_date) + pd.DateOffset(days=future_buffer_days)).strftime("%Y-%m-%d")

    print(f"查價格：{len(stocks):,} 檔，{query_start} ~ {query_end}")
    df_price = parquet_db.query_price(stocks, query_start, query_end)

    print("偵測 human triangle...")
    df_source = find_triangle(df_price)
    df_source = add_triangle_compare_columns(df_source)
    df_source["date"] = pd.to_datetime(df_source["date"])

    signal_mask = (df_source["date"] >= pd.Timestamp(start_date)) & (df_source["date"] <= pd.Timestamp(end_date))
    signal_cols = [
        "is_triangle",
        "is_refined_triangle",
        "is_fuzzy_triangle",
        "is_human_triangle",
        "fuzzy_only",
        "human_only",
    ]
    for col in signal_cols:
        if col in df_source.columns:
            df_source[col] = df_source[col].fillna(False).astype(bool) & signal_mask

    df_source.to_pickle(cache_path)
    print(f"已存長期間快取：{cache_path}")
    print(
        "三角數量 total / strict / refined / fuzzy / human / human_only：",
        len(df_source),
        int(df_source["is_triangle"].sum()),
        int(df_source["is_refined_triangle"].sum()),
        int(df_source["is_fuzzy_triangle"].sum()),
        int(df_source["is_human_triangle"].sum()),
        int(df_source["human_only"].sum()),
    )
    return df_source


def run_long_history(
    start_date="2021-01-01",
    end_date="2025-12-31",
    cache_path=LONG_CACHE_PATH,
    sample_size=None,
    refresh=False,
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    require_ma60_breakout_window=True,
    ma60_breakout_window=MA60_BREAKOUT_WINDOW,
):
    '''
    長期間版本的一鍵入口。

    第一次會比較慢，因為會找多年三角；之後會直接讀 cache_path。
    '''
    build_long_triangle_cache(
        start_date=start_date,
        end_date=end_date,
        cache_path=cache_path,
        sample_size=sample_size,
        refresh=refresh,
    )
    return run_from_cache(
        cache_path=cache_path,
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
        require_ma60_breakout_window=require_ma60_breakout_window,
        ma60_breakout_window=ma60_breakout_window,
    )


def build_ma60_comparison_dataset(
    df_price,
    df_market=None,
    triangle_df=None,
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    forward_days=FORWARD_DAYS,
    target_ret=TARGET_RET,
    triangle_lookback_bars=20,
):
    """
    為每一筆 MA60 上穿事件打 tag：前 triangle_lookback_bars 根 K 內是否有三角結束。

    回傳兩組樣本：
        has_triangle=True  : 三角收斂後的 MA60 突破
        has_triangle=False : 純 MA60 突破（無前置三角）
    """
    price_feat = _add_price_features_and_labels(
        df_price,
        df_market=df_market,
        forward_days=forward_days,
        target_ret=target_ret,
    )

    tri = _prepare_triangle_signals(
        df_price,
        triangle_df=triangle_df,
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
    )
    tri = _ensure_triangle_end_date(tri)

    price_lookup = price_feat[["stock_id", "date", "bar_index"]].copy()
    price_lookup["date"] = pd.to_datetime(price_lookup["date"])

    # 把每個三角的 triangle_end_date 對應到 bar_index
    tri_indexed_rows = []
    for stock_id, stock_tri in tri.groupby("stock_id"):
        stock_price = price_lookup[price_lookup["stock_id"] == stock_id].sort_values("date")
        if stock_price.empty:
            continue
        price_dates = stock_price["date"].values
        price_bars = stock_price["bar_index"].values

        for _, row in stock_tri.iterrows():
            end_date = row["triangle_end_date"]
            if pd.isna(end_date):
                continue
            idx = np.searchsorted(price_dates, np.datetime64(pd.Timestamp(end_date)), side="left")
            if idx >= len(price_bars):
                continue
            tri_indexed_rows.append({
                "stock_id": stock_id,
                "tri_end_bar": int(price_bars[idx]),
                "triangle_score": row.get("triangle_score", np.nan),
                "triangle_match_type": row.get("triangle_match_type", None),
                "width_reduction": row.get("width_reduction", np.nan),
            })

    tri_indexed = pd.DataFrame(tri_indexed_rows)

    ma60_crossovers = price_feat[price_feat["ma60_breakout_up"]].dropna(subset=["label"]).copy()

    results = []
    for stock_id, stock_cross in ma60_crossovers.groupby("stock_id"):
        stock_tri = tri_indexed[tri_indexed["stock_id"] == stock_id] if not tri_indexed.empty else pd.DataFrame()
        tri_end_bars = np.sort(stock_tri["tri_end_bar"].values) if not stock_tri.empty else np.array([])

        cross_bars = stock_cross["bar_index"].values.astype(int)
        has_triangle_arr = np.zeros(len(cross_bars), dtype=bool)
        tri_score_arr = np.full(len(cross_bars), np.nan)
        tri_type_arr = np.full(len(cross_bars), None, dtype=object)
        width_reduction_arr = np.full(len(cross_bars), np.nan)

        if len(tri_end_bars) > 0:
            for i, cross_bar in enumerate(cross_bars):
                lo = np.searchsorted(tri_end_bars, cross_bar - triangle_lookback_bars, side="left")
                hi = np.searchsorted(tri_end_bars, cross_bar, side="right")
                if hi > lo:
                    has_triangle_arr[i] = True
                    window = stock_tri[
                        (stock_tri["tri_end_bar"] >= cross_bar - triangle_lookback_bars)
                        & (stock_tri["tri_end_bar"] <= cross_bar)
                    ]
                    if not window.empty:
                        best = window.loc[window["triangle_score"].fillna(-1).idxmax()]
                        tri_score_arr[i] = best["triangle_score"]
                        tri_type_arr[i] = best["triangle_match_type"]
                        width_reduction_arr[i] = best["width_reduction"]

        out = stock_cross.copy()
        out["has_triangle"] = has_triangle_arr
        out["tri_score_at_crossover"] = tri_score_arr
        out["tri_type_at_crossover"] = tri_type_arr
        out["tri_width_reduction"] = width_reduction_arr
        results.append(out)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True).sort_values(["date", "stock_id"]).reset_index(drop=True)


def compare_ma60_with_triangle(
    df_price,
    df_market=None,
    triangle_df=None,
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    forward_days=FORWARD_DAYS,
    target_ret=TARGET_RET,
    triangle_lookback_bars=20,
):
    """
    比較「三角 + MA60 突破」vs「純 MA60 突破（無三角）」。
    """
    data = build_ma60_comparison_dataset(
        df_price=df_price,
        df_market=df_market,
        triangle_df=triangle_df,
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
        forward_days=forward_days,
        target_ret=target_ret,
        triangle_lookback_bars=triangle_lookback_bars,
    )

    with_tri = data[data["has_triangle"]]
    without_tri = data[~data["has_triangle"]]

    rows = [
        _selection_stats(data, "全部 MA60 突破"),
        _selection_stats(with_tri, f"三角+MA60（lookback={triangle_lookback_bars}根）", score_col="tri_score_at_crossover"),
        _selection_stats(without_tri, "純 MA60（無三角）"),
    ]
    result = pd.DataFrame(rows)

    print("=" * 72)
    print(f"MA60 突破比較：有三角 vs 無三角（往前看 {triangle_lookback_bars} 根 K）")
    print(f"Forward days={forward_days}  Target ret={target_ret:.0%}  Overlap={overlap_filter}")
    print("=" * 72)
    _print_report(result)
    print(f"\n三角樣本分佈（match type）：")
    if "tri_type_at_crossover" in with_tri.columns:
        print(with_tri["tri_type_at_crossover"].value_counts().head(10).to_string())

    return data, result


def compare_from_cache(
    cache_path="triangle_human_cache_2020_2025.pkl",
    overlap_filter="keep_best",
    min_overlap_ratio=0.8,
    forward_days=FORWARD_DAYS,
    target_ret=TARGET_RET,
    triangle_lookback_bars=20,
):
    """
    用長期間快取跑「三角 vs 純 MA60 突破」比較。

    快速使用：
        from j1stools.ma60_triangle import compare_from_cache
        data, result = compare_from_cache()
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"快取不存在：{cache_path}，請先跑 build_long_triangle_cache()")

    df_source = pd.read_pickle(cache_path)
    df_tagged, triangle_df, _ = filter_triangle_by_overlap(
        df_source,
        signal_col="human_only",
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
    )
    return compare_ma60_with_triangle(
        df_price=df_tagged,
        df_market=None,
        triangle_df=triangle_df,
        overlap_filter="all",
        min_overlap_ratio=min_overlap_ratio,
        forward_days=forward_days,
        target_ret=target_ret,
        triangle_lookback_bars=triangle_lookback_bars,
    )


def main(st="2021-01-01", end="2025-12-31", sample_size=None):
    """
    查資料、偵測三角、訓練模型的一鍵入口。

    預設改用長期間 cache，先把 2021~2025 的三角找出來。
    """
    return run_long_history(
        start_date=st,
        end_date=end,
        sample_size=sample_size,
        cache_path=LONG_CACHE_PATH,
    )


if __name__ == "__main__":
    main()
