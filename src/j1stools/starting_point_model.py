"""
用實際 K 線學一個最小版 1D CNN：判斷是不是「起漲點」

這支檔案故意寫得簡單，重點不是追求最高績效，而是看懂 CNN 怎麼吃股票序列。

資料來源：
    只使用 parquet_db.query_price() 讀正式報價，不使用 triangle_human_cache.pkl。

問題定義：
    給模型看某檔股票最近 WINDOW_SIZE 根 K 的特徵，
    預測今天之後 FORWARD_DAYS 根 K 內，最高價是否曾經漲到 TARGET_RET。

label:
    1 = 未來 FORWARD_DAYS 根 K 內最高價 >= 今天收盤價 * (1 + TARGET_RET)
    0 = 沒有達標

快速執行：
    PYTHONPATH=src MPLCONFIGDIR=/private/tmp python src/j1stools/starting_point_model.py

Notebook 使用：
    from j1stools.starting_point_model import run_demo
    model, pred, report = run_demo()
"""

import warnings

import numpy as np
import pandas as pd

from j1stools import parquet_db

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # 讓沒有 torch 的環境仍可 import 這支檔案，看程式不會炸掉。
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


WINDOW_SIZE = 60  # CNN 每次看最近 60 根 K
FORWARD_DAYS = 20  # label 看未來 20 根 K
TARGET_RET = 0.10  # 20 根 K 內最高價漲 10% 就算起漲成功
TEST_RATIO = 0.30  # 後 30% 日期當測試集，避免隨機切分偷看未來

FEATURE_COLS = [
    "ret_1d",  # 今天收盤相對昨天收盤的漲跌幅，給 CNN 看短線動能。
    "range_pct",  # 今日高低價差除以收盤價，給 CNN 看波動放大或收縮。
    "body_pct",  # 今日收盤減開盤再除以收盤價，給 CNN 看 K 棒實體方向與力道。
    "upper_shadow_pct",  # 上影線占整根 K 棒高低區間的比例，給 CNN 看上方賣壓。
    "lower_shadow_pct",  # 下影線占整根 K 棒高低區間的比例，給 CNN 看下方承接。
    "vol_ratio",  # 5 日均量除以 20 日均量，給 CNN 看短線量能是放大還是萎縮。
    "dev_ma20",  # 收盤價偏離 MA20 的比例，給 CNN 看股價相對短中均線的位置。
    "dev_ma60",  # 收盤價偏離 MA60 的比例，給 CNN 看股價是否接近中期突破。
    "ma20_slope",  # MA20 相對 5 根 K 前的變化率，給 CNN 看短中期趨勢方向。
    "ma60_slope",  # MA60 相對 10 根 K 前的變化率，給 CNN 看中期趨勢方向。
]


def _require_torch():
    if torch is None:
        raise ImportError("這個範例需要 PyTorch。請先在目前 venv 安裝 torch，或切到有 torch 的環境。")


def clean_price(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    整理價格資料成標準長表格。

    需要欄位：
        date, stock_id, open, high, low, close, volume
    """
    df = df_price.copy()
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str).str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["close"] > 0].dropna(subset=["close"])
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def add_features_and_label(
    df_price: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    target_ret: float = TARGET_RET,
) -> pd.DataFrame:
    """
    對每一檔股票計算 CNN 要看的每日特徵，以及今天的 label。

    注意：
        特徵只用今天以前可知道的資料。
        label 用未來資料，只有訓練時能看，實際預測時不能看。
    """
    price = clean_price(df_price)
    rows = []

    for _, grp in price.groupby("stock_id"):
        d = grp.sort_values("date").reset_index(drop=True).copy()
        close = d["close"]
        high = d["high"].fillna(close)
        low = d["low"].fillna(close)
        open_ = d["open"].fillna(close)
        volume = d["volume"].fillna(0)

        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        day_range = (high - low).replace(0, np.nan)

        d["ret_1d"] = close.pct_change(1)  # 單日報酬率；>0 代表今天收盤比昨天高。
        d["range_pct"] = (high - low) / close  # 高低振幅比例；數值越小通常代表越收斂。
        d["body_pct"] = (close - open_) / close  # K 棒實體比例；>0 偏多，<0 偏空。
        d["upper_shadow_pct"] = (high - np.maximum(open_, close)) / day_range  # 上影線比例；越大代表上方壓力越明顯。
        d["lower_shadow_pct"] = (np.minimum(open_, close) - low) / day_range  # 下影線比例；越大代表下方承接越明顯。
        d["vol_ratio"] = volume.rolling(5).mean() / volume.rolling(20).mean().clip(
            lower=1
        )  # 短均量 / 長均量；>1 代表近期放量。
        d["dev_ma20"] = (close - ma20) / ma20.clip(lower=1e-9)  # 價格離 MA20 多遠；>0 代表站上 MA20。
        d["dev_ma60"] = (close - ma60) / ma60.clip(lower=1e-9)  # 價格離 MA60 多遠；>0 代表站上 MA60。
        d["ma20_slope"] = (ma20 - ma20.shift(5)) / ma20.shift(5).clip(
            lower=1e-9
        )  # MA20 近 5 根 K 的斜率；>0 趨勢轉上。
        d["ma60_slope"] = (ma60 - ma60.shift(10)) / ma60.shift(10).clip(
            lower=1e-9
        )  # MA60 近 10 根 K 的斜率；>0 中期趨勢轉上。

        future_high = pd.concat([high.shift(-i) for i in range(1, forward_days + 1)], axis=1).max(axis=1)
        future_close = close.shift(-forward_days)
        d["future_max_ret"] = future_high / close - 1  # 未來 forward_days 內最高曾經漲多少；這是結果，不是特徵。
        d["future_close_ret"] = future_close / close - 1  # 未來第 forward_days 根 K 收盤報酬；這是結果，不是特徵。
        d["label"] = np.where(
            d["future_max_ret"].notna(), (d["future_max_ret"] >= target_ret).astype(int), np.nan
        )  # 是否達到起漲門檻；訓練答案，不給模型當特徵。

        rows.append(d)

    return pd.concat(rows, ignore_index=True)


def make_cnn_windows(
    df_feature: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols=None,
):
    """
    把長表格轉成 CNN 的張量格式。

    輸出：
        X shape = (樣本數, 特徵數, 時間長度)
        y shape = (樣本數,)
        meta   = 每個樣本對應的 date / stock_id / label / future return

    為什麼要 transpose？
        PyTorch Conv1d 吃的是 (batch, channels, length)
        在股票裡 channels = 特徵，length = K 棒時間。
    """
    _require_torch()

    feature_cols = feature_cols or FEATURE_COLS
    x_list = []
    y_list = []
    meta_rows = []

    for stock_id, grp in df_feature.groupby("stock_id"):
        d = grp.sort_values("date").reset_index(drop=True).copy()
        values = d[feature_cols].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median()).fillna(0.0).values.astype("float32")

        labels = d["label"].values
        for end_pos in range(window_size - 1, len(d)):
            label = labels[end_pos]
            row = d.iloc[end_pos]
            if pd.isna(label) or pd.isna(row["future_max_ret"]) or pd.isna(row["future_close_ret"]):
                continue

            window = values[end_pos - window_size + 1 : end_pos + 1].copy()

            # 每個樣本自己標準化，讓 CNN 主要看形狀，不被股價絕對大小影響。
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True)
            window = (window - mean) / (std + 1e-6)

            x_list.append(window.T)  # (features, time)
            y_list.append(label)
            meta_rows.append(
                {
                    "date": row["date"],
                    "stock_id": stock_id,
                    "label": int(label),
                    "close": row["close"],
                    "future_max_ret": row["future_max_ret"],
                    "future_close_ret": row["future_close_ret"],
                }
            )

    X = torch.tensor(np.array(x_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


_BaseModule = nn.Module if nn is not None else object


class SimpleStartingPointCNN(_BaseModule):
    """
    最小版 1D CNN。

    conv1 看短形狀，例如連續小 K、量縮、短線轉強。
    conv2 把短形狀組合成比較大的型態，例如壓縮後突破。
    classifier 輸出一個 logit，再用 sigmoid 變成 pred_score。
    """

    def __init__(self, num_features: int):
        _require_torch()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.conv(x)).squeeze(1)


def split_by_time(meta: pd.DataFrame, test_ratio: float = TEST_RATIO):
    """
    用日期切 train/test，不隨機切。
    這樣比較接近真實交易：用過去訓練，測未來。
    """
    dates = np.array(sorted(meta["date"].dropna().unique()))
    cut_idx = int(len(dates) * (1 - test_ratio))
    cut_idx = min(max(cut_idx, 1), len(dates) - 1)
    split_date = pd.Timestamp(dates[cut_idx])
    train_mask = meta["date"] < split_date
    test_mask = meta["date"] >= split_date
    return train_mask.values, test_mask.values, split_date


def train_cnn(
    X,
    y,
    meta: pd.DataFrame,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 0.001,
):
    """
    訓練 CNN 並回傳：
        model
        pred  : 測試集每筆樣本的 pred_score
        report: 各分數區間的簡單勝率表
    """
    _require_torch()
    from sklearn.metrics import roc_auc_score

    train_mask, test_mask, split_date = split_by_time(meta)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_test = meta.loc[test_mask].copy().reset_index(drop=True)

    if len(X_train) < 100 or len(X_test) < 30:
        raise ValueError(f"樣本太少：train={len(X_train)}, test={len(X_test)}")

    model = SimpleStartingPointCNN(num_features=X.shape[1])
    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("=" * 72)
    print("Simple 1D CNN 起漲點模型")
    print("=" * 72)
    print(f"train/test     : {len(X_train):,} / {len(X_test):,}")
    print(f"split_date     : {split_date.date()}")
    print(f"train win rate : {y_train.mean().item():.2%}")
    print(f"test  win rate : {y_test.mean().item():.2%}")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(f"epoch {epoch:02d}/{epochs} loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        pred_score = torch.sigmoid(test_logits).numpy()

    pred = meta_test.copy()
    pred["pred_score"] = pred_score

    try:
        auc = roc_auc_score(pred["label"], pred["pred_score"])
    except ValueError:
        auc = np.nan

    report = score_report(pred)
    print("-" * 72)
    print(f"AUC            : {auc:.3f}" if not pd.isna(auc) else "AUC            : N/A")
    print(report.to_string(index=False))
    return model, pred, report


def score_report(pred: pd.DataFrame) -> pd.DataFrame:
    """
    看 CNN 分數是否真的有排序能力。
    如果 top_10% 勝率比 test_all 高，代表 pred_score 有一些資訊量。
    """
    rows = [_group_stats(pred, "test_all")]
    for pct in [0.30, 0.20, 0.10]:
        n = max(1, int(len(pred) * pct))
        rows.append(_group_stats(pred.nlargest(n, "pred_score"), f"top_{int(pct * 100)}%"))
    return pd.DataFrame(rows)


def _group_stats(df: pd.DataFrame, name: str):
    return {
        "group": name,
        "count": len(df),
        "win_rate": df["label"].mean(),
        "avg_max_ret": df["future_max_ret"].mean(),
        "avg_close_ret": df["future_close_ret"].mean(),
        "avg_score": df["pred_score"].mean() if "pred_score" in df.columns else np.nan,
    }


def run_demo(
    st: str = "2023-01-01",
    end: str = "2026-01-01",
    sample_size: int = 120,
    window_size: int = WINDOW_SIZE,
):
    """
    用真實股價資料跑一個最小 CNN 範例。

    sample_size 預設 120 檔，讓你先快速學流程。
    想要更認真驗證，可以把 sample_size=None 跑全市場。
    """
    stocks = parquet_db.query_stocks_no_etf()
    if sample_size:
        rng = np.random.default_rng(42)
        stocks = list(rng.choice(stocks, size=min(sample_size, len(stocks)), replace=False))

    warmup_st = (pd.Timestamp(st) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    query_end = (pd.Timestamp(end) + pd.DateOffset(days=FORWARD_DAYS + 5)).strftime("%Y-%m-%d")
    print(f"查價格：{len(stocks):,} 檔，{warmup_st} ~ {query_end}")
    df_price = parquet_db.query_price(stocks, warmup_st, query_end)

    df_feature = add_features_and_label(df_price)
    df_feature = df_feature[(df_feature["date"] >= pd.Timestamp(st)) & (df_feature["date"] <= pd.Timestamp(end))]
    X, y, meta = make_cnn_windows(df_feature, window_size=window_size)
    print(f"CNN dataset: X={tuple(X.shape)}, y={tuple(y.shape)}")
    return train_cnn(X, y, meta)


if __name__ == "__main__":
    run_demo()
