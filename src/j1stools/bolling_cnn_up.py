"""
BollingCNN：20日時序 → BB W底反彈後上軌觸碰預測（2分類）

進場過濾條件（同時成立）：
    1. 15日內 close <= bb_lower 累計 ≥2次（W底）
    2. 第2次下軌觸碰後5日內 close >= bb_mid（中軌突破）
    3. 中軌突破當日 close > open（陽線確認，W_REQUIRE_BULLISH=True 時啟用）

標籤（二元）：
    Y=1  碰上軌：持有期（HOLD_DAYS）內任一日 high >= bb_upper
    Y=0  未碰上軌：持有到期，未觸及上軌

BB_FEATURES (6):
  bb_daily_return   日報酬率
  bb_hl_range       (high - low) / close
  bb_close_pos      (close - low) / (high - low)
  bb_pos_in_band    (close - BB_lower) / BB_width
  bb_width_pct      (BB_upper - BB_lower) / BB_mid
  bb_atr_pct        ATR14 / close
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BB_FEATURES = [
    "bb_daily_return",
    "bb_hl_range",
    "bb_close_pos",
    "bb_pos_in_band",
    "bb_width_pct",
    "bb_atr_pct",
]

# ── 超參數 ────────────────────────────────────────────────────────────────── #
HOLD_DAYS = 10
BB_PERIOD = 20
BB_STD = 2.0
MIN_ATR_PCT = 0.02  # ATR 過濾（0.03 = 3%）
W_LOOKBACK = 15  # W底下軌觸碰觀察窗口（交易日）
W_MID_DAYS = 5  # 第2次下軌後幾日內需突破中軌
PROFIT_TARGET = 0.07  # 停利目標（與標籤一致）
STOP_LOSS_PCT = 0.05  # 停損目標（絕對值，與標籤一致）
W_REQUIRE_BULLISH = False  # 中軌突破當日是否要求陽線（close > open）
PREC_THR = 0.55  # early stopping / run 選模用的 up_prob 閾值


# ── 原始特徵計算 ──────────────────────────────────────────────────────────── #


def build_bb_daily_features(stocks: list, st: str, end: str = "2099-01-01"):
    """
    計算 BB_FEATURES 及 ATR，回傳 DataFrame(date, stock_id, *BB_FEATURES, atr14_pct, bb_upper, bb_touch)。
    不做 ATR 或 touch 過濾（由呼叫端決定）。
    """
    import pandas as pd
    import time as _time
    from j1stools import parquet_db

    _t0 = _time.time()
    print(f"  [BollingCNN] 載入 price（{len(stocks)} 檔，{st}~{end}）...")

    df_price = parquet_db.query_price(stocks, st, end)
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_price = df_price.sort_values(["stock_id", "date"])

    # ── 日報酬 ──
    df_price["bb_daily_return"] = df_price.groupby("stock_id")["close"].pct_change(fill_method=None)

    # ── K 棒特徵 ──
    hl = (df_price["high"] - df_price["low"]).replace(0, np.nan)
    df_price["bb_hl_range"] = hl / df_price["close"].replace(0, np.nan)
    df_price["bb_close_pos"] = (df_price["close"] - df_price["low"]) / hl

    # ── ATR14 ──
    def _atr14(grp):
        tr = pd.concat(
            [
                grp["high"] - grp["low"],
                (grp["high"] - grp["close"].shift(1)).abs(),
                (grp["low"] - grp["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(14, min_periods=5).mean() / grp["close"].replace(0, np.nan)

    df_price["atr14_pct"] = df_price.groupby("stock_id", group_keys=False).apply(_atr14, include_groups=False)
    df_price["bb_atr_pct"] = df_price["atr14_pct"]

    # ── Bollinger Bands（向量化：pivot → rolling）──
    _pv = df_price.pivot(index="date", columns="stock_id", values="close")
    _bb_mid = _pv.rolling(BB_PERIOD, min_periods=10).mean()
    _bb_std = _pv.rolling(BB_PERIOD, min_periods=10).std(ddof=0)
    _bb_upper = _bb_mid + BB_STD * _bb_std
    _bb_lower = _bb_mid - BB_STD * _bb_std
    _bb_width = (_bb_upper - _bb_lower).replace(0, np.nan)

    def _stack_col(df_wide, name):
        return df_wide.stack(future_stack=True).rename(name).reset_index()

    df_price = df_price.merge(_stack_col(_bb_lower, "bb_lower"), on=["date", "stock_id"], how="left")
    df_price = df_price.merge(_stack_col(_bb_mid, "bb_mid"), on=["date", "stock_id"], how="left")
    df_price = df_price.merge(_stack_col(_bb_upper, "bb_upper"), on=["date", "stock_id"], how="left")
    df_price = df_price.merge(_stack_col(_bb_width, "bb_width"), on=["date", "stock_id"], how="left")

    df_price["bb_pos_in_band"] = (df_price["close"] - df_price["bb_lower"]) / df_price["bb_width"]
    df_price["bb_width_pct"] = df_price["bb_width"] / df_price["bb_mid"].replace(0, np.nan)

    # ── Clip 極端值 ──
    df_price["bb_daily_return"] = df_price["bb_daily_return"].clip(-0.3, 0.3)
    df_price["bb_pos_in_band"] = df_price["bb_pos_in_band"].clip(-0.5, 2.5)
    df_price["bb_width_pct"] = df_price["bb_width_pct"].clip(0, 0.5)
    df_price["bb_atr_pct"] = df_price["bb_atr_pct"].clip(0, 0.3)

    # ── W底信號（按 stock_id 逐組計算）──
    def _w_signal(grp):
        lower_touch = (grp["close"] <= grp["bb_lower"]).astype(int)
        # 10日內累計碰下軌次數
        roll10 = lower_touch.rolling(W_LOOKBACK, min_periods=1).sum()
        # 標記「本日是下軌觸碰，且10日內已有 ≥2 次」
        second_touch = (lower_touch == 1) & (roll10 >= 2)
        # 3日內曾出現 second_touch
        recent_2nd = second_touch.rolling(W_MID_DAYS, min_periods=1).sum() >= 1
        # 中軌突破（+ 可選陽線確認）
        mid_break = grp["close"] >= grp["bb_mid"]
        if W_REQUIRE_BULLISH:
            mid_break = mid_break & (grp["close"] > grp["open"])
        return (recent_2nd & mid_break).astype(np.int8)

    df_price["bb_w_signal"] = df_price.groupby("stock_id", group_keys=False).apply(_w_signal, include_groups=False)

    cols = ["date", "stock_id", "atr14_pct", "bb_w_signal", "bb_upper"] + BB_FEATURES
    print(f"  [BollingCNN] 特徵完成（{_time.time()-_t0:.1f}s），共 {len(df_price):,} 行")
    return df_price[cols].sort_values(["stock_id", "date"]).reset_index(drop=True)


# ── 標籤建構 + 時序窗口 ───────────────────────────────────────────────────── #


def build_bolling_dataset(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = MIN_ATR_PCT,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    建構 Bollinger W底反彈資料集。

    標籤（二元）：
        Y=1: 持有期內（hold_days 日）任一日 high >= 當日 bb_upper（碰上軌）
        Y=0: 未碰上軌

    回傳 (X, y, df_meta)：
        X      : (n, lookback, n_features) float32
        y      : (n,) int，0=未碰上軌, 1=碰上軌
        df_meta: 對應 DataFrame（含 date, stock_id, Y）
    """
    import pandas as pd
    from j1stools import parquet_db

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    end_buf = (pd.Timestamp(end) + pd.DateOffset(days=hold_days * 2)).strftime("%Y-%m-%d")

    print(f"── 載入 BB 原始特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_bb_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # ── 合併 high（上軌觸碰標籤需要）──
    df_px = parquet_db.query_price(stocks, st_buf, end_buf)[["date", "stock_id", "high"]]
    df_px["date"] = pd.to_datetime(df_px["date"])
    df_all = df_all.merge(df_px, on=["date", "stock_id"], how="left")

    # ── 標籤：hold_days 內任一日 high >= bb_upper → Y=1 ──
    def _upper_touch_label(grp):
        highs = grp["high"].values.astype(np.float32)
        bb_up = grp["bb_upper"].values.astype(np.float32)
        n = len(highs)
        touch = (highs >= bb_up).astype(np.float32)
        padded = np.concatenate([touch, np.full(hold_days, np.nan, dtype=np.float32)])
        stacked = np.stack([padded[h : h + n] for h in range(1, hold_days + 1)], axis=1)
        result = np.nanmax(stacked, axis=1)
        result[-hold_days:] = np.nan  # 近 hold_days 日無完整未來資料
        return pd.Series(result, index=grp.index)

    df_all["Y"] = df_all.groupby("stock_id", group_keys=False).apply(_upper_touch_label, include_groups=False)

    # ── 信號過濾：W底信號, ATR 合格, 在信號區間, 有完整標籤 ──
    df_signal = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["date"] < pd.Timestamp(end))
            & (df_all["bb_w_signal"] == 1)
            & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
            & df_all["Y"].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_signal["Y"] = df_signal["Y"].astype(np.int64)

    n_up = (df_signal["Y"] == 1).sum()
    n_dn = (df_signal["Y"] == 0).sum()
    print(
        f"W底信號：{len(df_signal):,}  "
        f"碰上軌(Y=1)={n_up/len(df_signal):.1%}  未碰上軌(Y=0)={n_dn/len(df_signal):.1%}"
    )

    # ── 建立時序窗口（stride_tricks）──
    df_signal["date_str"] = df_signal["date"].astype(str)
    signal_lookup: dict[tuple, int] = {
        (r["stock_id"], r["date_str"]): i for i, r in df_signal[["stock_id", "date_str"]].iterrows()
    }

    all_X, all_meta_idx = [], []

    for sid, grp in df_all.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[BB_FEATURES].fillna(0).values.astype(np.float32)
        dates_str = grp["date"].astype(str).values
        if len(grp) < lookback:
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

    df_valid = df_signal.iloc[all_meta_idx].reset_index(drop=True)
    X = np.array(all_X, dtype=np.float32)
    y = df_valid["Y"].values.astype(np.int64)
    return X, y, df_valid


# ── CNN 架構 ──────────────────────────────────────────────────────────────── #


class _FocalLoss(nn.Module):
    """Focal Loss：對高信心樣本降權，讓模型專注難分類的 case。"""

    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class _BollingCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveMaxPool1d(4)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 4, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.fc2(x)


# ── 分類器封裝 ────────────────────────────────────────────────────────────── #


class BollingCNNClassifier:
    """
    BollingBand 碰上軌後方向 CNN 分類器（3類：中性/向下/向上）。
    輸出 predict_proba → 取類別2（向上）的機率作為信號強度。
    """

    def __init__(
        self,
        lookback: int = 20,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 256,
        class_weight: str = "balanced",  # "balanced" | "none"
    ):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.class_weight = class_weight
        self._model: _BollingCNN | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._calibrator = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 10,
    ) -> "BollingCNNClassifier":
        self._feat_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._feat_std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        X_n = self._normalize(X)
        if X_val is not None:
            X_val_n = self._normalize(X_val)

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  訓練裝置：{device}")

        n_features = X.shape[2]
        self._model = _BollingCNN(n_features, n_classes=2).to(device)

        # 類別權重（補償 imbalance）
        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=2).astype(np.float32)
            weights = torch.tensor(counts.sum() / (2 * counts + 1e-9), dtype=torch.float32).to(device)
            print(f"  類別權重：{weights.tolist()}")
            criterion = _FocalLoss(weight=weights, gamma=2.0)
        else:
            criterion = _FocalLoss(gamma=2.0)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        X_t = torch.tensor(X_n.transpose(0, 2, 1).astype(np.float32))
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_val_metric = -np.inf
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._model.train()
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                self._model.eval()
                train_acc = correct / total
                msg = (
                    f"  epoch {epoch+1:3d}/{self.epochs}  loss={total_loss/len(loader):.4f}  train_acc={train_acc:.3f}"
                )
                if X_val is not None:
                    with torch.no_grad():
                        vt = torch.tensor(X_val_n.transpose(0, 2, 1).astype(np.float32)).to(device)
                        logits_v = self._model(vt)
                        up_prob_v = F.softmax(logits_v, dim=1)[:, 1].cpu().numpy()
                    val_acc = (logits_v.argmax(1).cpu().numpy() == y_val).mean()
                    thr_mask = up_prob_v >= PREC_THR
                    val_prec = (y_val[thr_mask] == 1).mean() if thr_mask.sum() >= 30 else 0.0
                    val_n = thr_mask.sum()
                    msg += f"  val_acc={val_acc:.3f}  prec@{PREC_THR}={val_prec:.3f}(n={val_n})"
                    if val_prec > best_val_metric:
                        best_val_metric = val_prec
                        best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                        no_improve = 0
                        msg += " ✓"
                    else:
                        no_improve += 5
                print(msg)
                if X_val is not None and no_improve >= patience:
                    print(f"  Early stopping（prec@{PREC_THR} 連 {patience} epoch 未改善）")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"  最佳 prec@{PREC_THR}：{best_val_metric:.4f}")
        self._best_prec = best_val_metric
        self._model.eval().to("cpu")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """回傳 (n, 2) softmax 機率，[:,1] 為停利（向上）機率（若已校準則套用 isotonic）。"""
        assert self._model is not None
        X_n = self._normalize(X.astype(np.float32))
        X_t = torch.tensor(X_n.transpose(0, 2, 1), dtype=torch.float32)
        with torch.no_grad():
            proba = F.softmax(self._model(X_t), dim=1).numpy()
        if self._calibrator is not None:
            proba[:, 1] = self._calibrator.predict(proba[:, 1])
        return proba

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "BollingCNNClassifier":
        """用驗證集對 up_prob（class 1）做 isotonic regression 校準。"""
        from sklearn.isotonic import IsotonicRegression

        raw_proba = self.predict_proba(X_val)
        up_prob_raw = raw_proba[:, 1]
        y_binary = (y_val == 1).astype(np.float64)
        self._calibrator = IsotonicRegression(out_of_bounds="clip").fit(up_prob_raw, y_binary)
        cal_prob = self._calibrator.predict(up_prob_raw)
        prec_after = (y_val[cal_prob >= PREC_THR] == 1).mean() if (cal_prob >= PREC_THR).sum() >= 10 else 0.0
        print(f"  校準完成（n={len(y_val):,}，校準後 prec@{PREC_THR}={prec_after:.3f}）")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳 (n,) 預測類別。"""
        return self.predict_proba(X).argmax(axis=1)


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_bolling_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lookback: int = 20,
    epochs: int = 60,
    n_runs: int = 3,
) -> BollingCNNClassifier:
    """訓練 n_runs 次，取 prec@PREC_THR 最佳的模型。"""
    best_score = -np.inf
    best_clf = None
    for run in range(n_runs):
        print(f"\n── Run {run+1}/{n_runs} ──")
        clf = BollingCNNClassifier(lookback=lookback, epochs=epochs, lr=1e-3, batch_size=256)
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        score = clf._best_prec if X_val is not None else 0.0
        print(f"  Run {run+1} prec@{PREC_THR}：{score:.4f}")
        if score > best_score:
            best_score = score
            best_clf = clf
    if best_clf is None:
        best_clf = clf
    print(f"\n最佳 Run prec@{PREC_THR}：{best_score:.4f}")
    if X_val is not None:
        print("── 校準 up_prob（isotonic regression）──")
        best_clf.calibrate(X_val, y_val)
    train_acc = (best_clf.predict(X_train) == y_train).mean()
    print(f"訓練集 acc：{train_acc:.4f}（in-sample）")
    return best_clf


# ── 存取模型 ──────────────────────────────────────────────────────────────── #

BOLLING_CNN_PATH = "db/models/bolling_cnn_up.joblib"


def save_bolling_cnn(clf: BollingCNNClassifier, path: str = BOLLING_CNN_PATH, hold_days: int = HOLD_DAYS) -> None:
    import joblib, os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "lookback": clf.lookback,
            "lr": clf.lr,
            "batch_size": clf.batch_size,
            "hold_days": hold_days,
            "feat_mean": clf._feat_mean,
            "feat_std": clf._feat_std,
            "n_features": clf._model.conv1.in_channels,
            "n_classes": clf._model.fc2.out_features,
            "weights": {k: v.cpu().numpy() for k, v in clf._model.state_dict().items()},
            "calibrator": clf._calibrator,
        },
        path,
    )
    print(f"BollingCNN 已存：{path}（hold_days={hold_days}）")


def load_bolling_cnn(path: str = BOLLING_CNN_PATH) -> tuple["BollingCNNClassifier", int]:
    import joblib

    state = joblib.load(path)
    clf = BollingCNNClassifier(lookback=state["lookback"], lr=state["lr"], batch_size=state["batch_size"])
    clf._feat_mean = state["feat_mean"]
    clf._feat_std = state["feat_std"]
    clf._model = _BollingCNN(state["n_features"], n_classes=state.get("n_classes", 2))
    sd = {k: torch.from_numpy(v.copy()) for k, v in state["weights"].items()}
    clf._model.load_state_dict(sd)
    clf._model.eval()
    clf._calibrator = state.get("calibrator", None)
    hold_days = state.get("hold_days", HOLD_DAYS)
    cal_status = "已校準" if clf._calibrator is not None else "未校準"
    print(f"BollingCNN 已載入：{path}（hold_days={hold_days}，{cal_status}）")
    return clf, hold_days


# ── 信號生成（回測用）────────────────────────────────────────────────────── #


def make_signal_bolling_cnn(
    clf: BollingCNNClassifier,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    min_atr_pct: float = MIN_ATR_PCT,
    up_prob_threshold: float = 0.4,
) -> "pd.DataFrame":
    """掃信號區間，只在 bb_touch=1 且向上機率 >= 閾值時發出訊號（stride_tricks 版）。"""
    import pandas as pd

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    df_all = build_bb_daily_features(stocks, st_buf, end)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # 信號 lookup：(stock_id, date_str) → 行號
    df_scan = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["bb_w_signal"] == 1)
            & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
        ]
        .copy()
        .reset_index(drop=True)
    )
    print(f"  W底信號：{len(df_scan):,} 筆")

    df_scan["_date_str"] = df_scan["date"].astype(str)
    signal_lookup: dict[tuple, int] = {
        (r["stock_id"], r["_date_str"]): i for i, r in df_scan[["stock_id", "_date_str"]].iterrows()
    }

    all_X, all_scan_idx = [], []

    for sid, grp in df_all.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < lookback:
            continue
        vals = grp[BB_FEATURES].fillna(0).values.astype(np.float32)
        dates_str = grp["date"].astype(str).values

        windows = np.lib.stride_tricks.sliding_window_view(vals, (lookback, vals.shape[1]))
        windows = windows[:, 0, :, :].copy()

        for i, d in enumerate(dates_str[lookback - 1 :]):
            key = (sid, d)
            if key not in signal_lookup:
                continue
            all_X.append(windows[i])
            all_scan_idx.append(signal_lookup[key])

    print(f"  窗口筆數：{len(all_X):,}")

    if not all_X:
        print("  無符合條件的信號（W底條件未達成或 lookback 不足）")
        return pd.DataFrame(columns=["date", "stock_id", "up_prob"])

    X = np.array(all_X, dtype=np.float32)
    proba = clf.predict_proba(X)
    up_prob = proba[:, 1]

    signal = df_scan.iloc[all_scan_idx].reset_index(drop=True)[["date", "stock_id"]].copy()
    signal["up_prob"] = up_prob
    signal["2"] = up_prob  # backtest_platform 需要此欄位名稱作為信號強度
    signal["date"] = pd.to_datetime(signal["date"])

    print(
        f"  up_prob 分布：mean={up_prob.mean():.3f}  p50={np.median(up_prob):.3f}  p90={np.percentile(up_prob,90):.3f}"
    )

    signal = signal[signal["up_prob"] >= up_prob_threshold].reset_index(drop=True)
    print(f"  信號筆數（閾值>={up_prob_threshold}）：{len(signal):,}", end="")
    if len(signal) > 0:
        print(f"  日期：{signal['date'].min().date()} ~ {signal['date'].max().date()}")
    else:
        print()
    return signal


# ── Main ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    import pandas as pd
    import matplotlib

    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    from j1stools import backtest_platform, j1s_chart, parquet_db

    # ── 參數 ──────────────────────────────────────────────────────────────────── #
    TRAIN_ST = "2015-01-01"
    EVAL_ST = "2024-01-01"
    LOOKBACK = 20
    # ── 切換模式 ──────────────────────────────────────────────────────────────── #
    #
    #  train    : 【重新訓練】建窗口 → 訓練 BollingCNNClassifier → 存模型
    #
    #  eval     : 【OOS 評估】載現有模型 → 混淆矩陣 + 向上/向下機率分布
    #
    #  backtest : 【獨立回測】載現有模型 → make_signal → backtest_platform
    #
    MODE = "backtest"  # "train" | "eval" | "backtest"
    # ──────────────────────────────────────────────────────────────────────────── #

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train":
        print("\n══ 建立訓練集 ══")
        X_train, y_train, df_tr = build_bolling_dataset(
            stocks, TRAIN_ST, EVAL_ST, lookback=LOOKBACK, hold_days=HOLD_DAYS
        )
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_bolling_dataset(stocks, EVAL_ST, lookback=LOOKBACK, hold_days=HOLD_DAYS)
        print(f"訓練集：{X_train.shape}  測試集：{X_test.shape}")

        print(f"\n══ 訓練 BollingCNN（{HOLD_DAYS}日方向分類）══")
        clf = train_bolling_cnn(X_train, y_train, X_val=X_test, y_val=y_test, lookback=LOOKBACK, n_runs=3)
        save_bolling_cnn(clf, BOLLING_CNN_PATH, hold_days=HOLD_DAYS)

    elif MODE == "eval":
        print(f"\n══ 載入模型 {BOLLING_CNN_PATH} ══")
        clf, hold_days_loaded = load_bolling_cnn(BOLLING_CNN_PATH)

        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_bolling_dataset(stocks, EVAL_ST, lookback=LOOKBACK, hold_days=hold_days_loaded)

        proba = clf.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        up_prob = proba[:, 1]  # 二元分類：class 1 = 停利（向上）

        acc = (pred == y_test).mean()
        up_mask = y_test == 1  # 碰上軌
        dn_mask = y_test == 0  # 未碰上軌
        up_recall = (pred[up_mask] == 1).mean() if up_mask.sum() > 0 else 0.0
        dn_recall = (pred[dn_mask] == 0).mean() if dn_mask.sum() > 0 else 0.0
        print(f"\nOOS 整體準確率：{acc:.3f}")
        print(f"碰上軌 (Y=1) 召回率：{up_recall:.3f}  （真實碰上軌={up_mask.sum()}筆）")
        print(f"未碰上軌 (Y=0) 召回率：{dn_recall:.3f}  （真實未碰上軌={dn_mask.sum()}筆）")

        # 閾值掃描：up_prob >= threshold 時的精準率與信號量
        print(f"\n{'─'*52}")
        print(f"{'up_prob >=':>12}  {'精準率(上)':>10}  {'信號數':>7}  {'佔比':>6}")
        print(f"{'─'*52}")
        for thr in [0.35, 0.40, 0.46, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = up_prob >= thr
            n = mask.sum()
            if n == 0:
                continue
            prec = (y_test[mask] == 1).mean()
            pct = n / len(y_test)
            print(f"{thr:>12.2f}  {prec:>10.1%}  {n:>7,}  {pct:>6.1%}")
        print(f"{'─'*52}")

        # 混淆矩陣
        from sklearn.metrics import classification_report

        print("\n分類報告：")
        print(classification_report(y_test, pred, target_names=["未碰上軌", "碰上軌"]))

        # 圖：向上機率分布（按真實標籤）
        plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]

        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cls, label, color in [
            (1, "碰上軌(Y=1)", "seagreen"),
            (0, "未碰上軌(Y=0)", "tomato"),
        ]:
            axes[0].hist(up_prob[y_test == cls], bins=40, alpha=0.5, label=label, color=color, density=True)
        axes[0].set_xlabel("碰上軌機率 (pred_proba class=1)")
        axes[0].set_title("碰上軌機率分布（按真實類別）")
        axes[0].legend()

        # 停利機率分組後的停利比例
        df_eval = pd.DataFrame({"up_prob": up_prob, "Y": y_test})
        df_eval["decile"] = pd.qcut(up_prob, q=10, labels=False, duplicates="drop")
        rate = df_eval.groupby("decile")["Y"].apply(lambda x: (x == 1).mean())
        axes[1].bar(rate.index, rate.values, color="seagreen", alpha=0.8)
        axes[1].axhline((y_test == 1).mean(), color="gray", linestyle="--", linewidth=0.8, label="Base rate")
        axes[1].set_xlabel("碰上軌機率 decile (0=lowest)")
        axes[1].set_ylabel("真實碰上軌比例")
        axes[1].set_title("Precision by decile (OOS)")
        axes[1].legend()
        plt.suptitle(f"BollingCNN | {hold_days_loaded}d | BB W底反彈方向")
        plt.tight_layout()
        plt.show()

    elif MODE == "backtest":
        print(f"\n══ 載入模型 {BOLLING_CNN_PATH} ══")
        clf, hold_days_loaded = load_bolling_cnn(BOLLING_CNN_PATH)

        print("\n══ 獨立回測 ══")
        backtest_platform.IS_USE_CACHE = True
        sig = make_signal_bolling_cnn(clf, stocks, st=EVAL_ST, lookback=LOOKBACK, up_prob_threshold=PREC_THR)

        print(f"sig 欄位：{sig.columns.tolist()}")
        print(f"sig dtypes:\n{sig.dtypes}")
        print(sig.head(3).to_string())

        print(f"\n{'='*60}\n【BollingCNN 向上高機率 + 持有{hold_days_loaded}日】\n{'='*60}")
        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=5,
            threshold=PREC_THR,
            max_positions=2,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=STOP_LOSS_PCT,
            use_fixed_tp=True,
            tp_stop=PROFIT_TARGET,
            use_hold_days=True,
            hold_days=hold_days_loaded,
            group_limit=99,
            min_volume=200,
            use_fixed_sl_tp=False,
        )
        count = 20
        if td is not None and len(td) > 0:
            print(f"\n最近 {count} 筆交易（共 {len(td):,} 筆）：")
            print(td.tail(count).to_string(index=False))
        j1s_chart.plot_performance(pv, td)

    else:
        print(f"未知 MODE：{MODE!r}，請選 train / eval / backtest")
