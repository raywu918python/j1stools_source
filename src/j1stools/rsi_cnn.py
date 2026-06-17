"""
RSI-ATR CNN 分類器：RSI 碰上緣後的走向預測

信號觸發 : RSI 由下往上穿越 upper_band（預設 70），即「剛進超買區」
特徵窗口 : 信號前 lookback 天的 8 個 RSI/ATR/K 線特徵
分類標籤 :
  Y=0  反轉（hold_days 後報酬 < -return_threshold）
  Y=1  盤整（介於中間）
  Y=2  續漲（hold_days 後報酬 >  return_threshold）

RSI_FEATURES (8):
  rsi_val       RSI 值（/100 正規化）
  rsi_slope     RSI 5日變化率
  rsi_div       RSI 背離（close 5日漲跌 × RSI 5日漲跌，正=同向，負=背離）
  atr_pct       ATR14 / close（波動率）
  daily_return  日報酬率
  hl_range      (high - low) / close
  close_pos     (close - low) / (high - low)（收盤位置）
  volume_ratio  成交量 / 20日均量
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

RSI_FEATURES = [
    "rsi_val",
    "rsi_slope",
    "rsi_div",
    "atr_pct",
    "daily_return",
    "hl_range",
    "close_pos",
    "volume_ratio",
]

UPPER_BAND = 70.0
LOWER_BAND = 30.0
RSI_PERIOD = 14
HOLD_DAYS = 5
RETURN_THRESHOLD = 0.03  # 3%，續漲/反轉邊界
LOOKBACK = 20
MODEL_PATH = "db/models/rsi_cnn.pt"


# ── 特徵計算 ────────────────────────────────────────────────────────────────── #


def _rsi(close_series, period=RSI_PERIOD):
    """Wilder 平滑 RSI，回傳 pd.Series（0-100）。"""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def build_rsi_daily_features(stocks: list, st: str, end: str = "2099-01-01"):
    """
    計算 RSI_FEATURES，回傳 DataFrame(date, stock_id, *RSI_FEATURES, atr14_pct)。
    """
    import pandas as pd
    import time as _t
    from j1stools import parquet_db

    t0 = _t.time()
    print(f"  [RSI-CNN] 載入 price（{len(stocks)} 檔，{st}~{end}）...")

    df = parquet_db.query_price(stocks, st, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    print(f"  [RSI-CNN] 計算特徵...")
    parts = []
    for _, grp in df.groupby("stock_id", sort=False):
        grp = grp.copy()
        close = grp["close"].replace(0, float("nan"))
        high = grp["high"]
        low = grp["low"]
        vol = grp["volume"]

        rsi = _rsi(close, RSI_PERIOD)
        grp["rsi_val"] = rsi / 100.0
        grp["rsi_slope"] = rsi.diff(5) / 100.0

        price_slope = close.pct_change(5, fill_method=None)
        rsi_slope5 = rsi.diff(5)
        grp["rsi_div"] = (price_slope.apply(np.sign) * rsi_slope5.apply(np.sign)).fillna(0)

        hl = (high - low).replace(0, float("nan"))
        tr = pd.concat([hl, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        grp["atr_pct"] = tr.rolling(14, min_periods=5).mean() / close
        grp["daily_return"] = close.pct_change(fill_method=None)
        grp["hl_range"] = hl / close
        grp["close_pos"] = (grp["close"] - low) / hl
        grp["volume_ratio"] = vol / vol.rolling(20, min_periods=5).mean()

        parts.append(grp)
    df = pd.concat(parts).reset_index(drop=True)

    # clip 極端值
    df["daily_return"] = df["daily_return"].clip(-0.3, 0.3)
    df["volume_ratio"] = df["volume_ratio"].clip(0, 10)
    df["hl_range"] = df["hl_range"].clip(0, 0.3)
    df["rsi_slope"] = df["rsi_slope"].clip(-0.5, 0.5)
    df["rsi_div"] = df["rsi_div"].clip(-1, 1)

    print(f"  [RSI-CNN] 特徵完成（{_t.time()-t0:.1f}s），{len(df):,} 行")
    return df[["date", "stock_id"] + RSI_FEATURES].copy()


# ── 信號偵測：RSI 上穿 upper_band ─────────────────────────────────────────── #


def detect_rsi_cross(df_feat: "pd.DataFrame", upper_band: float = UPPER_BAND) -> "pd.DataFrame":
    """
    找出 RSI 由下往上穿越 upper_band 的日期（即 yesterday_rsi < upper_band 且 today_rsi >= upper_band）。
    回傳子集 DataFrame（保留原欄位），作為信號日。
    """
    import pandas as pd

    df = df_feat.copy().sort_values(["stock_id", "date"])
    df["_rsi100"] = df["rsi_val"] * 100
    df["_rsi_prev"] = df.groupby("stock_id")["_rsi100"].shift(1)

    mask = (df["_rsi_prev"] < upper_band) & (df["_rsi100"] >= upper_band)
    df_sig = df[mask].drop(columns=["_rsi100", "_rsi_prev"]).reset_index(drop=True)
    print(f"  RSI 上穿 {upper_band} 信號：{len(df_sig):,} 筆")
    return df_sig


# ── 資料集建構 ───────────────────────────────────────────────────────────────── #


def build_rsi_dataset(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = LOOKBACK,
    hold_days: int = HOLD_DAYS,
    return_threshold: float = RETURN_THRESHOLD,
    upper_band: float = UPPER_BAND,
    binary: bool = False,
) -> tuple:
    """
    建立 RSI-CNN 分類資料集。

    binary=True（預設）：二分類，過濾掉盤整樣本
        Y=0 反轉（fwd_return <= -return_threshold）
        Y=1 續漲（fwd_return >=  return_threshold）
    binary=False：三分類
        Y=0 反轉, Y=1 盤整, Y=2 續漲

    回傳 (X, y, df_meta)：
        X       : (n, lookback, n_features) float32
        y       : (n,) int64
        df_meta : 含 date, stock_id, Y, fwd_return 的 DataFrame
    """
    import pandas as pd
    from j1stools import parquet_db

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    end_buf = (pd.Timestamp(end) + pd.DateOffset(days=hold_days * 3)).strftime("%Y-%m-%d")

    print(f"── 載入 RSI 特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_rsi_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # hold_days 後報酬（用 close price）
    df_price = parquet_db.query_price(stocks, st_buf, end_buf)
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_all = df_all.merge(df_price[["date", "stock_id", "close"]], on=["date", "stock_id"], how="left")
    df_all["fwd_return"] = df_all.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    df_all = df_all.drop(columns=["close"])

    # 信號偵測（在有效信號區間內）
    df_in_range = df_all[(df_all["date"] >= pd.Timestamp(st)) & (df_all["date"] < pd.Timestamp(end))]
    df_sig = detect_rsi_cross(df_in_range, upper_band=upper_band)
    df_sig = df_sig[df_sig["fwd_return"].notna()].copy()

    # 標籤
    if binary:
        # 只保留明確反轉或續漲，過濾掉盤整區間
        df_sig = df_sig[df_sig["fwd_return"].abs() >= return_threshold].copy()
        df_sig["Y"] = (df_sig["fwd_return"] >= return_threshold).astype(int)  # 0=反轉, 1=續漲
        label_dist = df_sig["Y"].value_counts(normalize=True).sort_index()
        print(
            f"標籤分布（二分類）：反轉={label_dist.get(0,0):.1%}  續漲={label_dist.get(1,0):.1%}  （共 {len(df_sig):,} 筆）"
        )
    else:
        df_sig["Y"] = 1  # 盤整
        df_sig.loc[df_sig["fwd_return"] >= return_threshold, "Y"] = 2  # 續漲
        df_sig.loc[df_sig["fwd_return"] <= -return_threshold, "Y"] = 0  # 反轉
        label_dist = df_sig["Y"].value_counts(normalize=True).sort_index()
        print(
            f"標籤分布：反轉={label_dist.get(0,0):.1%}  盤整={label_dist.get(1,0):.1%}  續漲={label_dist.get(2,0):.1%}"
        )

    # 建立時序窗口
    df_sig = df_sig.reset_index(drop=True)
    df_sig["_date_str"] = df_sig["date"].astype(str)
    signal_lookup: dict = {(r["stock_id"], r["_date_str"]): i for i, r in df_sig[["stock_id", "_date_str"]].iterrows()}

    sig_stocks = {k[0] for k in signal_lookup}
    all_X, all_meta_idx = [], []

    for sid, grp in df_all.groupby("stock_id"):
        if sid not in sig_stocks:
            continue
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[RSI_FEATURES].fillna(0).values.astype(np.float32)
        dates_str = grp["date"].astype(str).values
        T = len(grp)
        if T < lookback:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(vals, (lookback, vals.shape[1]))
        windows = windows[:, 0, :, :].copy()

        for i, d in enumerate(dates_str[lookback - 1 :]):
            key = (sid, d)
            if key not in signal_lookup:
                continue
            all_X.append(windows[i])
            all_meta_idx.append(signal_lookup[key])

    print(f"時序窗口（lookback={lookback}）：{len(all_X):,} 筆")

    df_meta = df_sig.iloc[all_meta_idx].drop(columns=["_date_str"]).reset_index(drop=True)
    X = np.array(all_X, dtype=np.float32)
    y = df_meta["Y"].values.astype(np.int64)
    return X, y, df_meta


# ── CNN 架構 ──────────────────────────────────────────────────────────────── #


class _RsiCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(4)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64 * 4, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ── 訓練器 ────────────────────────────────────────────────────────────────── #


class RsiCNNClassifier:
    """RSI 碰上緣後的走向分類器（binary=2類：反轉/續漲；或3類：反轉/盤整/續漲）。"""

    def __init__(
        self,
        lookback: int = LOOKBACK,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 128,
        n_classes: int = 3,
    ):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_classes = n_classes
        self._model: _RsiCNN | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weight: bool = True,
        patience: int = 10,
    ) -> "RsiCNNClassifier":
        self._feat_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._feat_std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        X_n = self._normalize(X.astype(np.float32))
        X_val_n = self._normalize(X_val.astype(np.float32)) if X_val is not None else None

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  訓練裝置：{device}")

        n_features = X_n.shape[2]
        self._model = _RsiCNN(n_features, n_classes=self.n_classes).to(device)

        # 類別權重（處理不平衡）
        if class_weight:
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            w = counts.sum() / (self.n_classes * counts + 1e-8)
            weight_t = torch.tensor(w / w.sum() * self.n_classes, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_t)
            print(f"  類別權重：{w/w.sum()*self.n_classes}")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        # (n, lookback, features) → (n, features, lookback) for Conv1d
        X_t = torch.tensor(X_n.transpose(0, 2, 1), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_val_acc = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                self._model.eval()
                msg = f"  epoch {epoch+1:3d}/{self.epochs}  loss={total_loss/len(loader):.4f}"
                if X_val_n is not None:
                    with torch.no_grad():
                        val_t = torch.tensor(X_val_n.transpose(0, 2, 1), dtype=torch.float32).to(device)
                        logits = self._model(val_t).cpu()
                    preds = logits.argmax(dim=1).numpy()
                    acc = (preds == y_val).mean()
                    # 只看反轉(0) vs 續漲(2) 的方向準確
                    mask_02 = (y_val == 0) | (y_val == 2)
                    dir_acc = (preds[mask_02] == y_val[mask_02]).mean() if mask_02.sum() > 0 else 0.0
                    msg += f"  val_acc={acc:.3f}  dir_acc={dir_acc:.3f}"
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                        no_improve = 0
                        msg += " ✓"
                    else:
                        no_improve += 5
                print(msg)
                if X_val_n is not None and no_improve >= patience:
                    print(f"  Early stopping（{patience} epoch 未改善）")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"  最佳 val_acc：{best_val_acc:.3f}")
        self._model.eval().to("cpu")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """回傳 softmax 機率。binary: shape (n,2) 列序 反轉/續漲；三分類: (n,3) 反轉/盤整/續漲"""
        assert self._model is not None
        X_n = self._normalize(X.astype(np.float32))
        X_t = torch.tensor(X_n.transpose(0, 2, 1), dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(X_t)
        return F.softmax(logits, dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳類別預測（0/1/2）"""
        return self.predict_proba(X).argmax(axis=1)

    def predict_continuation_prob(self, X: np.ndarray) -> np.ndarray:
        """回傳續漲機率（最後一個類別的機率），作為排名信號"""
        return self.predict_proba(X)[:, -1]


# ── 儲存 / 載入 ───────────────────────────────────────────────────────────── #


def save_rsi_cnn(clf: RsiCNNClassifier, path: str = MODEL_PATH) -> None:
    import joblib, os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "lookback": clf.lookback,
            "lr": clf.lr,
            "batch_size": clf.batch_size,
            "feat_mean": clf._feat_mean,
            "feat_std": clf._feat_std,
            "n_features": clf._model.conv1.in_channels,
            "n_classes": clf._model.fc2.out_features,
            "weights": {k: v.cpu().numpy() for k, v in clf._model.state_dict().items()},
        },
        path,
    )
    print(f"RSI-CNN 已存：{path}")


def load_rsi_cnn(path: str = MODEL_PATH) -> RsiCNNClassifier:
    import joblib

    state = joblib.load(path)
    clf = RsiCNNClassifier(lookback=state["lookback"], lr=state["lr"], batch_size=state["batch_size"])
    clf._feat_mean = state["feat_mean"]
    clf._feat_std = state["feat_std"]
    clf._model = _RsiCNN(state["n_features"], n_classes=state.get("n_classes", 3))
    clf._model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state["weights"].items()})
    clf._model.eval()
    print(f"RSI-CNN 載入：{path}")
    return clf


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_rsi_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lookback: int = LOOKBACK,
    epochs: int = 60,
    n_runs: int = 3,
    n_classes: int = 3,
) -> RsiCNNClassifier:
    """多次訓練取最佳 val_acc 模型。"""
    best_acc = -1.0
    best_clf = None
    for run in range(n_runs):
        print(f"\n── Run {run+1}/{n_runs} ──")
        clf = RsiCNNClassifier(lookback=lookback, epochs=epochs, lr=1e-3, batch_size=128, n_classes=n_classes)
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        if X_val is not None:
            acc = (clf.predict(X_val) == y_val).mean()
        else:
            acc = 0.0
        print(f"  Run {run+1} val_acc：{acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_clf = clf

    if best_clf is None:
        best_clf = clf
    print(f"\n最佳 val_acc：{best_acc:.3f}")
    return best_clf


# ── 回測信號產生 ─────────────────────────────────────────────────────────── #


def make_rsi_signal(
    clf: RsiCNNClassifier,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = LOOKBACK,
    upper_band: float = UPPER_BAND,
    min_continuation_prob: float = 0.40,
) -> "pd.DataFrame":
    """
    掃描 RSI 上穿信號，回傳續漲機率 >= min_continuation_prob 的信號 DataFrame。
    格式：date, stock_id, cont_prob（供 backtest_platform 使用）。
    """
    import pandas as pd

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    df_all = build_rsi_daily_features(stocks, st_buf, end)
    df_all["date"] = pd.to_datetime(df_all["date"])

    df_in_range = df_all[(df_all["date"] >= pd.Timestamp(st)) & (df_all["date"] < pd.Timestamp(end))]
    df_sig = detect_rsi_cross(df_in_range, upper_band=upper_band)

    stock_feat = {sid: grp.sort_values("date").reset_index(drop=True) for sid, grp in df_all.groupby("stock_id")}

    X_list, valid_rows = [], []
    for _, row in df_sig.iterrows():
        sid = row["stock_id"]
        date = pd.Timestamp(row["date"])
        grp = stock_feat.get(sid)
        if grp is None:
            continue
        pos_arr = grp.index[grp["date"] <= date]
        if len(pos_arr) < lookback:
            continue
        pos = pos_arr[-1]
        window = grp.loc[pos - lookback + 1 : pos, RSI_FEATURES].fillna(0).values
        if window.shape[0] != lookback:
            continue
        X_list.append(window.astype(np.float32))
        valid_rows.append({"date": date, "stock_id": sid})

    if not X_list:
        print("無有效信號")
        return pd.DataFrame(columns=["date", "stock_id", "cont_prob"])

    X = np.array(X_list, dtype=np.float32)
    probs = clf.predict_continuation_prob(X)

    signal = pd.DataFrame(valid_rows)
    signal["cont_prob"] = probs
    signal = signal[signal["cont_prob"] >= min_continuation_prob].reset_index(drop=True)
    signal["date"] = pd.to_datetime(signal["date"])

    print(
        f"RSI 信號：{len(signal):,} 筆（過濾後，cont_prob >= {min_continuation_prob}）"
        f"  {signal['date'].min().date()} ~ {signal['date'].max().date()}"
    )
    return signal


# ── __main__：訓練 / 評估 ─────────────────────────────────────────────────── #

if __name__ == "__main__":
    import os, sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    import pandas as pd
    import matplotlib

    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["STHeiti", "Hiragino Sans GB", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    from sklearn.metrics import classification_report, confusion_matrix

    from j1stools import parquet_db

    # ── 參數 ─────────────────────────────────────────────────────────────────── #
    TRAIN_ST = "2015-01-01"
    EVAL_ST = "2024-01-01"
    # ── 切換模式 ──────────────────────────────────────────────────────────────── #
    #
    #  train    : 【重新訓練】
    #             1. 從 TRAIN_ST ~ EVAL_ST 建立 RSI 上穿信號 + 時序窗口（訓練集）
    #             2. 從 EVAL_ST 之後建立 OOS 測試集
    #             3. 訓練 RsiCNNClassifier（3 runs 取最佳 val_acc）
    #             4. 存模型 → MODEL_PATH
    #             5. 印 classification_report + Confusion Matrix + 分位報酬圖
    #
    #  eval     : 【僅評估，不重訓】
    #             載入現有模型 → 對 OOS 期間重跑 build_rsi_dataset → 印 classification_report
    #             適合調整 EVAL_ST / return_threshold 後快速看 OOS 結果
    #
    #  backtest : 【回測】
    #             載入現有模型 → make_rsi_signal 掃 RSI 上穿信號 → 過濾續漲機率 → backtest_platform
    #             ⚠️  需先跑過 train，MODEL_PATH 必須存在
    #
    # ──────────────────────────────────────────────────────────────────────────── #
    MODE = "backtest"  # "train" | "eval" | "backtest"

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train":
        print("\n══ 訓練集 ══")
        X_train, y_train, meta_train = build_rsi_dataset(stocks, TRAIN_ST, EVAL_ST)

        print("\n══ 測試集（OOS）══")
        X_test, y_test, meta_test = build_rsi_dataset(stocks, EVAL_ST)

        n_classes = len(np.unique(y_train))
        class_names = ["反轉", "續漲"] if n_classes == 2 else ["反轉", "盤整", "續漲"]
        print(f"\n訓練：{X_train.shape}  測試：{X_test.shape}  分類數：{n_classes}")
        print(f"  訓練標籤：{np.bincount(y_train)}")
        print(f"  測試標籤 ：{np.bincount(y_test)}")

        print("\n══ 訓練 RSI-CNN ══")
        clf = train_rsi_cnn(X_train, y_train, X_val=X_test, y_val=y_test, n_classes=n_classes)
        save_rsi_cnn(clf, MODEL_PATH)

        print("\n══ OOS 評估 ══")
        probs = clf.predict_proba(X_test)
        preds = probs.argmax(axis=1)
        print(classification_report(y_test, preds, target_names=class_names))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cm = confusion_matrix(y_test, preds, normalize="true")
        axes[0].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center")
        ticks = list(range(n_classes))
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(class_names)
        axes[0].set_yticks(ticks)
        axes[0].set_yticklabels(class_names)
        axes[0].set_title("Confusion Matrix（OOS）")

        cont_prob = probs[:, -1]
        meta_test["cont_prob"] = cont_prob
        meta_test["fwd_return_pct"] = meta_test["fwd_return"] * 100
        meta_test["prob_decile"] = pd.qcut(cont_prob, q=5, labels=False, duplicates="drop")
        dec = meta_test.groupby("prob_decile")["fwd_return_pct"].mean()
        axes[1].bar(dec.index, dec.values, color="steelblue", alpha=0.8)
        axes[1].axhline(meta_test["fwd_return_pct"].mean(), color="gray", linestyle="--", linewidth=0.8, label="Mean")
        axes[1].set_xlabel("續漲機率分位（0=最低）")
        axes[1].set_ylabel("平均報酬 %")
        axes[1].set_title(f"報酬 by 分位（OOS，{HOLD_DAYS}日持有）")
        axes[1].legend()
        plt.suptitle("RSI-CNN | RSI 碰上緣後走向分類")
        plt.tight_layout()
        plt.show()

    elif MODE == "eval":
        clf = load_rsi_cnn(MODEL_PATH)
        n_classes = clf._model.fc2.out_features
        class_names = ["反轉", "續漲"] if n_classes == 2 else ["反轉", "盤整", "續漲"]
        binary = n_classes == 2
        X_test, y_test, meta_test = build_rsi_dataset(stocks, EVAL_ST, binary=binary)
        preds = clf.predict(X_test)
        print(classification_report(y_test, preds, target_names=class_names))

    elif MODE == "backtest":
        from j1stools import backtest_platform, j1s_chart

        clf = load_rsi_cnn(MODEL_PATH)
        sig = make_rsi_signal(clf, stocks, st=EVAL_ST, min_continuation_prob=0.40)
        sig["2"] = sig["cont_prob"]  # backtest_platform 用欄位名 "2" 作分數

        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=5,
            max_positions=5,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=0.07,
            use_hold_days=True,
            hold_days=HOLD_DAYS,
            min_volume=200,
        )
        j1s_chart.plot_performance(pv, td)
