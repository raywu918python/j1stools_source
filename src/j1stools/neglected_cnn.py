"""
外資忽視股 CNN 選股模型（方案 A：獨立模型）

輸入：過去 SEQ_LEN 天 OHLCV 序列（4 channel）
輸出：10日內達到 +10% 的機率

與 RFC/XGB/LGBM 使用相同的訓練/測試分割和 profit_label 標籤，
作為第 4 個獨立模型，standalone AUC > 0.52 才有意義加入 ensemble。
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from j1stools import parquet_db
from j1stools.neglected_rfc import PROFIT_TARGET, build_dataset
from j1stools.neglected_stock_classify import NeglectedGMM

SEQ_LEN = 20
N_CHANNELS = 4  # close_ret, high_ret, low_ret, vol_ratio
CNN_MODEL_PATH = "db/models/neglected_cnn.pt"
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3


# ── Dataset ──────────────────────────────────────────────────────── #


class OHLCVDataset(Dataset):
    def __init__(self, seqs: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(seqs, dtype=torch.float32)  # (N, C, T)
        self.y = torch.tensor(labels, dtype=torch.float32)  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────── #


class NeglectedCNN(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, 128, T)
        x = self.pool(x).squeeze(-1)  # (B, 128)
        return self.fc(x).squeeze(-1)  # (B,) logits


# ── Sequence builder ─────────────────────────────────────────────── #


def build_cnn_sequences(
    df_labels: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    從 df_labels（含 date, stock_id, Y）建立 OHLCV 序列。

    序列窗口包含進場當天（不含未來），從原始 OHLCV 讀取，
    不受 ATR 過濾影響（連續完整時序）。

    Returns
    -------
    sequences   : (N, N_CHANNELS, seq_len)  float32
    labels      : (N,)  float32
    valid_idx   : df_labels 中成功建立序列的原始 index list（用於對齊 df_labels 行）
    """
    st_buf = (df_labels["date"].min() - pd.Timedelta(days=seq_len * 3)).strftime("%Y-%m-%d")
    end_str = df_labels["date"].max().strftime("%Y-%m-%d")
    stocks = df_labels["stock_id"].unique().tolist()

    df_price = parquet_db.query_price(stocks, st_buf, end_str)
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_price = df_price.sort_values(["stock_id", "date"])

    # 建立 {stock_id: DataFrame} 查找表
    price_lookup: dict[str, pd.DataFrame] = {}
    for sid, grp in df_price.groupby("stock_id"):
        price_lookup[sid] = grp.reset_index(drop=True)

    sequences, labels_out, valid_idx = [], [], []

    for sid, label_grp in df_labels.groupby("stock_id"):
        if sid not in price_lookup:
            continue
        price_grp = price_lookup[sid]
        price_dates = price_grp["date"].values

        for orig_idx, row in label_grp.iterrows():
            target_date = np.datetime64(row["date"])
            pos = int(np.searchsorted(price_dates, target_date))

            # pos 指向 target_date 在 price_dates 中的位置
            if pos >= len(price_grp) or price_dates[pos] != target_date:
                continue  # 日期不在價格表裡（非交易日）
            if pos < seq_len - 1:
                continue  # 歷史不夠

            window = price_grp.iloc[pos - seq_len + 1 : pos + 1]  # 含進場日
            base_close = window["close"].iloc[0]
            if base_close <= 0:
                continue

            close_ret = window["close"].values / base_close - 1
            high_ret = window["high"].values / base_close - 1
            low_ret = window["low"].values / base_close - 1
            mean_vol = window["volume"].mean()
            vol_ratio = window["volume"].values / max(mean_vol, 1.0)

            seq = np.stack([close_ret, high_ret, low_ret, vol_ratio], axis=0).astype(np.float32)
            seq = np.clip(seq, -2.0, 2.0)  # 排除除權除息等極端值

            sequences.append(seq)
            labels_out.append(float(row["Y"]))
            valid_idx.append(orig_idx)

    print(f"序列建立：{len(sequences):,} 筆（原始 {len(df_labels):,} 筆，覆蓋率 {len(sequences)/len(df_labels):.1%}）")
    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels_out, dtype=np.float32),
        valid_idx,
    )


# ── Train ─────────────────────────────────────────────────────────── #


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_cnn(
    seqs_train: np.ndarray,
    labels_train: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> NeglectedCNN:
    device = _get_device()
    print(f"訓練裝置：{device}")

    pos = labels_train.sum()
    neg = len(labels_train) - pos
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

    loader = DataLoader(
        OHLCVDataset(seqs_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = NeglectedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(seqs_train):.4f}")

    auc_in = roc_auc_score(labels_train, predict_cnn(model, seqs_train))
    print(f"訓練集 AUC：{auc_in:.4f}（in-sample，供參考）")
    return model


# ── Predict ───────────────────────────────────────────────────────── #


def predict_cnn(
    model: NeglectedCNN,
    seqs: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    device = _get_device()
    model.eval()
    model.to(device)
    loader = DataLoader(
        OHLCVDataset(seqs, np.zeros(len(seqs))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probs = []
    with torch.no_grad():
        for X_b, _ in loader:
            probs.append(torch.sigmoid(model(X_b.to(device))).cpu().numpy())
    return np.concatenate(probs)


# ── Eval ──────────────────────────────────────────────────────────── #


def eval_cnn(
    model: NeglectedCNN,
    df_test: pd.DataFrame,
    seqs_test: np.ndarray,
    labels_test: np.ndarray,
    valid_idx: list,
    top_n: int = 10,
    hold_days: int = 10,
    top_pct: float | None = None,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    prob = predict_cnn(model, seqs_test)

    auc = roc_auc_score(labels_test, prob)
    print(f"\n{'='*55}")
    print(f"OOS AUC：{auc:.4f}")
    print(f"{'='*55}")
    print(classification_report(labels_test, (prob >= 0.5).astype(int), target_names=["跌", "漲"]))

    # 信心度分層勝率
    df_valid = df_test.loc[valid_idx].copy()
    df_cal = pd.DataFrame({"prob": prob, "Y": labels_test, "future_return": df_valid["future_return"].values})
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    bucket_labels = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"]
    df_cal["bucket"] = pd.cut(df_cal["prob"], bins=bins, labels=bucket_labels, right=False)
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

    # 回測
    df_eval = df_valid.copy()
    df_eval["prob"] = prob
    df_eval["future_return_clip"] = df_eval["future_return"].clip(-0.5, 1.0)

    effective_min_prob = 0.0
    if top_pct is not None:
        effective_min_prob = float(np.quantile(prob, 1.0 - top_pct))
        print(f"top_pct={top_pct:.0%} → 有效門檻 prob >= {effective_min_prob:.4f}")

    rebal_dates = sorted(df_eval["date"].unique())[::hold_days]
    cnn_returns, base_returns = [], []
    for date in rebal_dates:
        g = df_eval[df_eval["date"] == date]
        if len(g) == 0:
            continue
        candidates = g[g["prob"] >= effective_min_prob]
        if len(candidates) == 0:
            continue
        top_cnn = candidates.nlargest(top_n, "prob")
        cnn_returns.append(
            {
                "date": date,
                "mean_return": top_cnn["future_return_clip"].mean(),
                "win_rate": (top_cnn["max_ret"] >= PROFIT_TARGET).mean(),
            }
        )
        top_base = g.sample(min(top_n, len(g)), random_state=42)
        base_returns.append(
            {
                "date": date,
                "mean_return": top_base["future_return_clip"].mean(),
                "win_rate": (top_base["max_ret"] >= PROFIT_TARGET).mean(),
            }
        )

    df_perf = pd.DataFrame(cnn_returns).set_index("date")
    df_base = pd.DataFrame(base_returns).set_index("date")
    cum_cnn = (1 + df_perf["mean_return"]).cumprod()
    cum_base = (1 + df_base["mean_return"]).cumprod()

    thr_str = f"  門檻 prob>={effective_min_prob:.4f}" if effective_min_prob > 0 else ""
    print(f"\n{'='*60}")
    print(f"非重疊換倉 Top-{top_n}（每 {hold_days} 日換一次{thr_str}）  共 {len(cnn_returns)} 期進場")
    print(f"{'='*60}")
    print(f"{'':15} {'CNN':>10} {'隨機 Baseline':>15}")
    print(f"  平均報酬   {df_perf['mean_return'].mean():>9.2%} {df_base['mean_return'].mean():>14.2%}")
    print(f"  勝率       {df_perf['win_rate'].mean():>9.1%} {df_base['win_rate'].mean():>14.1%}")
    print(f"  累積報酬   {cum_cnn.iloc[-1]-1:>9.2%} {cum_base.iloc[-1]-1:>14.2%}")

    _, axes = plt.subplots(2, 1, figsize=(12, 8))
    cum_cnn.plot(ax=axes[0], color="darkorange", label=f"CNN Top-{top_n}")
    cum_base.plot(ax=axes[0], color="tomato", linestyle="--", label="隨機 Baseline")
    axes[0].axhline(1, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"外資忽視股 CNN vs 隨機 Baseline — Top-{top_n} 每 {hold_days} 日換倉")
    axes[0].set_ylabel("累積報酬倍數")
    axes[0].legend()

    df_perf["win_rate"].rolling(5).mean().plot(ax=axes[1], color="darkorange", label="CNN")
    df_base["win_rate"].rolling(5).mean().plot(ax=axes[1], color="tomato", linestyle="--", label="隨機")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("滾動 5 期勝率")
    axes[1].set_ylabel("勝率")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return df_perf


# ── Save / Load ───────────────────────────────────────────────────── #


def save_cnn(model: NeglectedCNN, path: str = CNN_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"CNN 模型已儲存：{path}")


def load_cnn(path: str = CNN_MODEL_PATH) -> NeglectedCNN:
    model = NeglectedCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"CNN 模型已載入：{path}")
    return model


# ── Main ──────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_ST, TRAIN_END = "2015-01-01", "2023-12-31"
    EVAL_ST = "2024-01-01"
    HOLD_DAYS = 10
    TOP_N = 10
    MIN_ATR = 0.02
    TOP_PCT = 0.2

    # ── 切換模式 ───────────────────────────────────────────── #
    MODE = "train+eval"  # "train+eval" | "eval_only"
    # ────────────────────────────────────────────────────────── #

    clf_gmm = NeglectedGMM.load()
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    if MODE == "train+eval":
        print("建立訓練集標籤...")
        df_train = build_dataset(
            clf_gmm, stocks, TRAIN_ST, TRAIN_END, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR
        )
        print("\n建立訓練序列...")
        seqs_train, labels_train, _ = build_cnn_sequences(df_train, seq_len=SEQ_LEN)
        print(f"正樣本：{labels_train.mean():.1%}")

        print("\n訓練 CNN...")
        model = train_cnn(seqs_train, labels_train, epochs=EPOCHS)
        save_cnn(model)

        print("\n建立測試集標籤...")
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)
        print("\n建立測試序列...")
        seqs_test, labels_test, valid_idx = build_cnn_sequences(df_test, seq_len=SEQ_LEN)
        print(f"正樣本：{labels_test.mean():.1%}")

        print("\n評估 CNN...")
        eval_cnn(model, df_test, seqs_test, labels_test, valid_idx, top_n=TOP_N, hold_days=HOLD_DAYS, top_pct=TOP_PCT)

    elif MODE == "eval_only":
        model = load_cnn()
        df_test = build_dataset(clf_gmm, stocks, EVAL_ST, clusters=None, hold_days=HOLD_DAYS, min_atr_pct=MIN_ATR)
        seqs_test, labels_test, valid_idx = build_cnn_sequences(df_test, seq_len=SEQ_LEN)
        eval_cnn(model, df_test, seqs_test, labels_test, valid_idx, top_n=TOP_N, hold_days=HOLD_DAYS, top_pct=TOP_PCT)
