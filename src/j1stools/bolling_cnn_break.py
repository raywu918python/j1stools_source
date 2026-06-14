"""
BollingCNNBreak：20日時序 → BB上軌盤整後突破預測（2分類）

進場過濾條件（同時成立）：
    1. 過去 CONSOLIDATION_DAYS 根K棒內，high >= bb_upper（碰上軌）
    2. 過去 CONSOLIDATION_DAYS 根K棒的 (max_high - min_low) / close <= CONSOLIDATION_RANGE（盤整不漲不跌）

標籤（二元）：
    Y=1  突破：持有期（HOLD_DAYS）內最大漲幅 >= PROFIT_TARGET
    Y=0  未突破

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
MIN_ATR_PCT = 0.02
ZONE_DAYS = 3  # 條件1&3 的觀察天數
ZONE_RANGE_PCT = 0.06  # 條件3：3根K max_high - min_low 上限（佔收盤比例）
PROFIT_TARGET = 0.08  # 8% 視為突破成功
PREC_THR = 0.3
THRESHOLD = 0.3

# ── 原始特徵計算 ──────────────────────────────────────────────────────────── #


def build_bb_daily_features(stocks: list, st: str, end: str = "2099-01-01"):
    """
    計算 BB_FEATURES 及 ATR，並標記 bb_consolidation_signal。
    回傳含 date, stock_id, high, low, close, *BB_FEATURES,
    atr14_pct, bb_upper, bb_consolidation_signal 的 DataFrame。
    """
    import pandas as pd
    import time as _time
    from j1stools import parquet_db

    _t0 = _time.time()
    print(f"  [BollingBreak] 載入 price（{len(stocks)} 檔，{st}~{end}）...")

    df = parquet_db.query_price(stocks, st, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"])

    df["bb_daily_return"] = df.groupby("stock_id")["close"].pct_change(fill_method=None)

    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["bb_hl_range"] = hl / df["close"].replace(0, np.nan)
    df["bb_close_pos"] = (df["close"] - df["low"]) / hl

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

    df["atr14_pct"] = df.groupby("stock_id", group_keys=False).apply(_atr14, include_groups=False)
    df["bb_atr_pct"] = df["atr14_pct"]

    _pv = df.pivot(index="date", columns="stock_id", values="close")
    _bb_mid = _pv.rolling(BB_PERIOD, min_periods=10).mean()
    _bb_std = _pv.rolling(BB_PERIOD, min_periods=10).std(ddof=0)
    _bb_upper = _bb_mid + BB_STD * _bb_std
    _bb_lower = _bb_mid - BB_STD * _bb_std
    _bb_width = (_bb_upper - _bb_lower).replace(0, np.nan)

    def _stack_col(df_wide, name):
        return df_wide.stack(future_stack=True).rename(name).reset_index()

    df = df.merge(_stack_col(_bb_lower, "bb_lower"), on=["date", "stock_id"], how="left")
    df = df.merge(_stack_col(_bb_mid, "bb_mid"), on=["date", "stock_id"], how="left")
    df = df.merge(_stack_col(_bb_upper, "bb_upper"), on=["date", "stock_id"], how="left")
    df = df.merge(_stack_col(_bb_width, "bb_width"), on=["date", "stock_id"], how="left")

    df["bb_pos_in_band"] = (df["close"] - df["bb_lower"]) / df["bb_width"]
    df["bb_width_pct"] = df["bb_width"] / df["bb_mid"].replace(0, np.nan)

    df["bb_daily_return"] = df["bb_daily_return"].clip(-0.3, 0.3)
    df["bb_pos_in_band"] = df["bb_pos_in_band"].clip(-0.5, 2.5)
    df["bb_width_pct"] = df["bb_width_pct"].clip(0, 0.5)
    df["bb_atr_pct"] = df["bb_atr_pct"].clip(0, 0.3)

    # ── 盤整信號（三條件）──
    # 1. ZONE_DAYS 天的 close 都在 bb_mid ~ bb_upper 之間
    # 2. 其中最低1根K的 low 有碰到（<= bb_mid）（下影線測中軌）
    # 3. ZONE_DAYS 根K的 max_high - min_low <= ZONE_RANGE_PCT（盤整不大幅震盪）
    def _consolidation_signal(grp):
        in_zone = (grp["close"] >= grp["bb_mid"]) & (grp["close"] <= grp["bb_upper"])
        all_in_zone = in_zone.rolling(ZONE_DAYS, min_periods=ZONE_DAYS).min() >= 1

        touched_mid = (grp["low"] <= grp["bb_mid"]).astype(int)
        any_touch_mid = touched_mid.rolling(ZONE_DAYS, min_periods=ZONE_DAYS).max() >= 1

        roll_high = grp["high"].rolling(ZONE_DAYS, min_periods=ZONE_DAYS).max()
        roll_low = grp["low"].rolling(ZONE_DAYS, min_periods=ZONE_DAYS).min()
        hl_range = (roll_high - roll_low) / grp["close"].replace(0, np.nan)
        range_ok = hl_range <= ZONE_RANGE_PCT

        return (all_in_zone & any_touch_mid & range_ok).astype(np.int8)

    df["bb_consolidation_signal"] = df.groupby("stock_id", group_keys=False).apply(
        _consolidation_signal, include_groups=False
    )

    cols = [
        "date",
        "stock_id",
        "high",
        "low",
        "close",
        "atr14_pct",
        "bb_upper",
        "bb_consolidation_signal",
    ] + BB_FEATURES
    print(f"  [BollingBreak] 特徵完成（{_time.time()-_t0:.1f}s），共 {len(df):,} 行")
    return df[cols].sort_values(["stock_id", "date"]).reset_index(drop=True)


# ── 標籤建構 + 時序窗口 ───────────────────────────────────────────────────── #


def build_break_dataset(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    hold_days: int = HOLD_DAYS,
    min_atr_pct: float = MIN_ATR_PCT,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    建構 BB上軌盤整後突破資料集。

    標籤（二元）：
        Y=1: 持有期（hold_days）內最大漲幅 >= PROFIT_TARGET
        Y=0: 未達目標

    回傳 (X, y, df_meta)：
        X      : (n, lookback, n_features) float32
        y      : (n,) int
        df_meta: 對應 DataFrame（含 date, stock_id, Y）
    """
    import pandas as pd

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    end_buf = (pd.Timestamp(end) + pd.DateOffset(days=hold_days * 2)).strftime("%Y-%m-%d")

    print(f"── 載入 BB 原始特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_bb_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    def _break_label(grp):
        close = grp["close"].values.astype(np.float32)
        n = len(close)
        padded = np.concatenate([close, np.full(hold_days, np.nan, dtype=np.float32)])
        stacked = np.stack([padded[h : h + n] for h in range(1, hold_days + 1)], axis=1)
        with np.errstate(all="ignore"):
            max_future = np.nanmax(stacked, axis=1)
        fwd_ret = (max_future - close) / np.clip(close, 1e-6, None)
        result = (fwd_ret >= PROFIT_TARGET).astype(np.float32)
        result[-hold_days:] = np.nan
        return pd.Series(result, index=grp.index)

    df_all["Y"] = df_all.groupby("stock_id", group_keys=False).apply(_break_label, include_groups=False)

    df_signal = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["date"] < pd.Timestamp(end))
            & (df_all["bb_consolidation_signal"] == 1)
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
        f"盤整信號：{len(df_signal):,}  "
        f"突破成功(Y=1)={n_up/len(df_signal):.1%}  未突破(Y=0)={n_dn/len(df_signal):.1%}"
    )

    df_signal["date_str"] = df_signal["date"].astype(str)
    signal_lookup = {(r["stock_id"], r["date_str"]): i for i, r in df_signal[["stock_id", "date_str"]].iterrows()}

    all_X, all_meta_idx = [], []
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
            all_meta_idx.append(signal_lookup[key])

    print(f"時序窗口（lookback={lookback}）：{len(all_X):,} 筆")
    df_valid = df_signal.iloc[all_meta_idx].reset_index(drop=True)
    X = np.array(all_X, dtype=np.float32)
    y = df_valid["Y"].values.astype(np.int64)
    return X, y, df_valid


# ── 模型架構 ──────────────────────────────────────────────────────────────── #


class _FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class _BreakCNN(nn.Module):
    """Conv1d 三層，input: (batch, n_features, seq)"""

    def __init__(self, n_features: int, n_classes: int = 2):
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


class _BreakLSTM(nn.Module):
    """雙層 LSTM，input: (batch, seq, n_features)，取最後一步輸出"""

    def __init__(self, n_features: int, n_classes: int = 2, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ── 分類器封裝 ────────────────────────────────────────────────────────────── #


class BollingBreakClassifier:
    """
    BB上軌盤整後突破 CNN/LSTM 分類器。
    model_type="cnn" | "lstm"
    """

    def __init__(
        self,
        lookback: int = 20,
        model_type: str = "cnn",
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 256,
        class_weight: str = "balanced",
    ):
        self.lookback = lookback
        self.model_type = model_type
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.class_weight = class_weight
        self._model: nn.Module | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._calibrator = None
        self._best_prec = -np.inf

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def _to_tensor(self, X_n: np.ndarray) -> torch.Tensor:
        """CNN 需要 (batch, feat, seq)；LSTM 保持 (batch, seq, feat)"""
        if self.model_type == "cnn":
            return torch.tensor(X_n.transpose(0, 2, 1).astype(np.float32))
        return torch.tensor(X_n.astype(np.float32))

    def _build_model(self, n_features: int, n_classes: int = 2) -> nn.Module:
        if self.model_type == "cnn":
            return _BreakCNN(n_features, n_classes)
        return _BreakLSTM(n_features, n_classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 10,
    ) -> "BollingBreakClassifier":
        self._feat_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._feat_std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        X_n = self._normalize(X)
        X_val_n = self._normalize(X_val) if X_val is not None else None

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  訓練裝置：{device}  模型：{self.model_type.upper()}")

        n_features = X.shape[2]
        self._model = self._build_model(n_features).to(device)

        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=2).astype(np.float32)
            weights = torch.tensor(counts.sum() / (2 * counts + 1e-9), dtype=torch.float32).to(device)
            print(f"  類別權重：{weights.tolist()}")
            criterion = _FocalLoss(weight=weights, gamma=2.0)
        else:
            criterion = _FocalLoss(gamma=2.0)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        X_t = self._to_tensor(X_n)
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
                        vt = self._to_tensor(X_val_n).to(device)
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
        assert self._model is not None
        X_n = self._normalize(X.astype(np.float32))
        X_t = self._to_tensor(X_n)
        with torch.no_grad():
            proba = F.softmax(self._model(X_t), dim=1).numpy()
        if self._calibrator is not None:
            proba[:, 1] = self._calibrator.predict(proba[:, 1])
        return proba

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "BollingBreakClassifier":
        from sklearn.isotonic import IsotonicRegression

        up_prob_raw = self.predict_proba(X_val)[:, 1]
        y_binary = (y_val == 1).astype(np.float64)
        self._calibrator = IsotonicRegression(out_of_bounds="clip").fit(up_prob_raw, y_binary)
        cal_prob = self._calibrator.predict(up_prob_raw)
        prec_after = (y_val[cal_prob >= PREC_THR] == 1).mean() if (cal_prob >= PREC_THR).sum() >= 10 else 0.0
        print(f"  校準完成（n={len(y_val):,}，校準後 prec@{PREC_THR}={prec_after:.3f}）")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_bolling_break(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lookback: int = 20,
    epochs: int = 60,
    n_runs: int = 3,
    model_type: str = "cnn",
) -> BollingBreakClassifier:
    """訓練 n_runs 次，取 prec@PREC_THR 最佳的模型。"""
    best_score = -np.inf
    best_clf = None
    for run in range(n_runs):
        print(f"\n── Run {run+1}/{n_runs} ({model_type.upper()}) ──")
        clf = BollingBreakClassifier(lookback=lookback, model_type=model_type, epochs=epochs, lr=1e-3, batch_size=256)
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
        best_clf.calibrate(X_val, y_val)
    print(f"訓練集 acc：{(best_clf.predict(X_train) == y_train).mean():.4f}")
    return best_clf


# ── 存取模型 ──────────────────────────────────────────────────────────────── #

BOLLING_BREAK_CNN_PATH = "db/models/bolling_break_cnn.joblib"
BOLLING_BREAK_LSTM_PATH = "db/models/bolling_break_lstm.joblib"


def _model_path(model_type: str) -> str:
    return BOLLING_BREAK_CNN_PATH if model_type == "cnn" else BOLLING_BREAK_LSTM_PATH


def save_bolling_break(clf: BollingBreakClassifier, path: str | None = None, hold_days: int = HOLD_DAYS) -> None:
    import joblib, os

    if path is None:
        path = _model_path(clf.model_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    extra = {}
    if clf.model_type == "lstm":
        extra = {"hidden_size": clf._model.lstm.hidden_size, "num_layers": clf._model.lstm.num_layers}

    joblib.dump(
        {
            "lookback": clf.lookback,
            "model_type": clf.model_type,
            "lr": clf.lr,
            "batch_size": clf.batch_size,
            "hold_days": hold_days,
            "feat_mean": clf._feat_mean,
            "feat_std": clf._feat_std,
            "n_features": (clf._model.conv1.in_channels if clf.model_type == "cnn" else clf._model.lstm.input_size),
            "n_classes": (clf._model.fc2.out_features if clf.model_type == "cnn" else clf._model.fc.out_features),
            "weights": {k: v.cpu().numpy() for k, v in clf._model.state_dict().items()},
            "calibrator": clf._calibrator,
            **extra,
        },
        path,
    )
    print(f"BollingBreak({clf.model_type.upper()}) 已存：{path}（hold_days={hold_days}）")


def load_bolling_break(path: str) -> tuple["BollingBreakClassifier", int]:
    import joblib

    state = joblib.load(path)
    model_type = state.get("model_type", "cnn")
    clf = BollingBreakClassifier(
        lookback=state["lookback"], model_type=model_type, lr=state["lr"], batch_size=state["batch_size"]
    )
    clf._feat_mean = state["feat_mean"]
    clf._feat_std = state["feat_std"]
    n_features = state["n_features"]
    n_classes = state.get("n_classes", 2)
    if model_type == "cnn":
        clf._model = _BreakCNN(n_features, n_classes)
    else:
        clf._model = _BreakLSTM(n_features, n_classes, state.get("hidden_size", 64), state.get("num_layers", 2))
    clf._model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state["weights"].items()})
    clf._model.eval()
    clf._calibrator = state.get("calibrator", None)
    hold_days = state.get("hold_days", HOLD_DAYS)
    cal = "已校準" if clf._calibrator else "未校準"
    print(f"BollingBreak({model_type.upper()}) 已載入：{path}（hold_days={hold_days}，{cal}）")
    return clf, hold_days


# ── 信號生成（回測用）────────────────────────────────────────────────────── #


def make_signal_bolling_break(
    clf: BollingBreakClassifier,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    min_atr_pct: float = MIN_ATR_PCT,
    up_prob_threshold: float = 0.4,
) -> "pd.DataFrame":
    """掃信號區間，在盤整信號 + 突破機率 >= 閾值時發出訊號。"""
    import pandas as pd

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    df_all = build_bb_daily_features(stocks, st_buf, end)
    df_all["date"] = pd.to_datetime(df_all["date"])

    df_scan = (
        df_all[
            (df_all["date"] >= pd.Timestamp(st))
            & (df_all["bb_consolidation_signal"] == 1)
            & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
        ]
        .copy()
        .reset_index(drop=True)
    )
    print(f"  盤整信號：{len(df_scan):,} 筆")

    df_scan["_date_str"] = df_scan["date"].astype(str)
    signal_lookup = {(r["stock_id"], r["_date_str"]): i for i, r in df_scan[["stock_id", "_date_str"]].iterrows()}

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
        print("  無符合條件的信號")
        return pd.DataFrame(columns=["date", "stock_id", "up_prob"])

    X = np.array(all_X, dtype=np.float32)
    proba = clf.predict_proba(X)
    up_prob = proba[:, 1]

    signal = df_scan.iloc[all_scan_idx].reset_index(drop=True)[["date", "stock_id"]].copy()
    signal["up_prob"] = up_prob
    signal["2"] = up_prob
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

    # ── 參數 ──────────────────────────────────────────────────────────────── #
    TRAIN_ST = "2015-01-01"
    EVAL_ST = "2024-01-01"
    LOOKBACK = 20
    MODEL_TYPE = "cnn"  # "cnn" | "lstm"
    MODE = "train"  # "train" | "eval" | "backtest"
    # ─────────────────────────────────────────────────────────────────────── #

    MODEL_PATH = _model_path(MODEL_TYPE)
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train":
        print("\n══ 建立訓練集 ══")
        X_train, y_train, df_tr = build_break_dataset(stocks, TRAIN_ST, EVAL_ST, lookback=LOOKBACK, hold_days=HOLD_DAYS)
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_break_dataset(stocks, EVAL_ST, lookback=LOOKBACK, hold_days=HOLD_DAYS)
        print(f"訓練集：{X_train.shape}  測試集：{X_test.shape}")

        print(f"\n══ 訓練 BollingBreak-{MODEL_TYPE.upper()}（{HOLD_DAYS}日突破分類）══")
        clf = train_bolling_break(
            X_train, y_train, X_val=X_test, y_val=y_test, lookback=LOOKBACK, n_runs=3, model_type=MODEL_TYPE
        )
        save_bolling_break(clf, hold_days=HOLD_DAYS)

    elif MODE == "eval":
        clf, hold_days_loaded = load_bolling_break(MODEL_PATH)
        print("\n══ 建立測試集 ══")
        X_test, y_test, df_te = build_break_dataset(stocks, EVAL_ST, lookback=LOOKBACK, hold_days=hold_days_loaded)

        proba = clf.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        up_prob = proba[:, 1]
        acc = (pred == y_test).mean()
        up_mask = y_test == 1

        print(f"\nOOS 整體準確率：{acc:.3f}")
        print(f"突破(Y=1) 召回率：{(pred[up_mask]==1).mean():.3f}  （真實突破={up_mask.sum()}筆）")
        print(f"未突破(Y=0) 召回率：{(pred[~up_mask]==0).mean():.3f}  （真實未突破={(~up_mask).sum()}筆）")

        print(f"\n{'─'*52}")
        print(f"{'up_prob >=':>12}  {'精準率':>10}  {'信號數':>7}  {'佔比':>6}")
        print(f"{'─'*52}")
        for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = up_prob >= thr
            n = mask.sum()
            if n == 0:
                continue
            prec = (y_test[mask] == 1).mean()
            print(f"{thr:>12.2f}  {prec:>10.1%}  {n:>7,}  {n/len(y_test):>6.1%}")
        print(f"{'─'*52}")

        from sklearn.metrics import classification_report

        print("\n分類報告：")
        print(classification_report(y_test, pred, target_names=["未突破", "突破"]))

        plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cls, label, color in [(1, "突破(Y=1)", "seagreen"), (0, "未突破(Y=0)", "tomato")]:
            axes[0].hist(up_prob[y_test == cls], bins=40, alpha=0.5, label=label, color=color, density=True)
        axes[0].set_xlabel("突破機率")
        axes[0].set_title("突破機率分布（按真實類別）")
        axes[0].legend()

        df_eval = pd.DataFrame({"up_prob": up_prob, "Y": y_test})
        df_eval["decile"] = pd.qcut(up_prob, q=10, labels=False, duplicates="drop")
        rate = df_eval.groupby("decile")["Y"].apply(lambda x: (x == 1).mean())
        axes[1].bar(rate.index, rate.values, color="seagreen", alpha=0.8)
        axes[1].axhline((y_test == 1).mean(), color="gray", linestyle="--", linewidth=0.8, label="Base rate")
        axes[1].set_xlabel("突破機率 decile (0=lowest)")
        axes[1].set_ylabel("真實突破比例")
        axes[1].set_title("Precision by decile (OOS)")
        axes[1].legend()
        plt.suptitle(f"BollingBreak-{clf.model_type.upper()} | {hold_days_loaded}d | BB上軌盤整突破")
        plt.tight_layout()
        plt.show()

    elif MODE == "backtest":
        clf, hold_days_loaded = load_bolling_break(MODEL_PATH)
        print("\n══ 獨立回測 ══")
        backtest_platform.IS_USE_CACHE = True
        sig = make_signal_bolling_break(clf, stocks, st=EVAL_ST, lookback=LOOKBACK, up_prob_threshold=THRESHOLD)

        print(f"\n{'='*60}\n【BollingBreak-{clf.model_type.upper()} 高機率突破 + 持有{hold_days_loaded}日】\n{'='*60}")
        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=5,
            threshold=THRESHOLD,
            max_positions=3,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=0.05,
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
