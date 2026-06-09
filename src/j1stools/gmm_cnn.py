"""
SeqCNN：20日時序 → 10日超額報酬預測

設計定位：獨立時序偵測器，輸出 cnn_alpha_p2（超越大盤 5% 的機率）。
此 prob 值作為樹模型的額外特徵（stacking），補充截面模型缺少的時序維度。

宇宙 : ATR14/close >= 0.05（無 GMM，無放量過濾）
標籤 : 10日後 stock_return - 0050_return >= 5% = 達標 (Y=2)
      10日後 stock_return - 0050_return <= -5% = 停損 (Y=1)
      其餘 = 盤整 (Y=0)

CNN_FEATURES (10):
  cnn_daily_return    日報酬率
  cnn_volume_ratio    成交量 / 20日均量
  cnn_hl_range        (high - low) / close
  cnn_close_pos       (close - low) / (high - low)
  cnn_stock_vs_mkt    daily_return - 0050_return（個股超額）
  cnn_foreign_net     外資淨買 / 成交量
  cnn_trust_net       投信淨買 / 成交量
  cnn_margin_chg      融資餘額日增率
  cnn_short_chg       融券餘額日增率
  cnn_dt_ratio        當沖成交量 / 總成交量
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CNN_FEATURES = [
    "cnn_daily_return",
    "cnn_volume_ratio",
    "cnn_hl_range",
    "cnn_close_pos",
    "cnn_stock_vs_mkt",
    "cnn_foreign_net",
    "cnn_trust_net",
    "cnn_margin_chg",
    "cnn_short_chg",
    "cnn_dt_ratio",
]

HOLD_DAYS = 10
ALPHA_TARGET = 0.05  # 超越大盤 5% = 達標
ALPHA_STOP = -0.05  # 落後大盤 5% = 停損
MIN_ATR_PCT = 0.05
MARKET_PROXY = "0050"


# ── 原始特徵計算 ──────────────────────────────────────────────────────────── #


def build_cnn_daily_features(stocks: list, st: str, end: str = "2099-01-01"):
    """
    載入原始資料，計算 CNN_FEATURES，回傳 DataFrame(date, stock_id, *CNN_FEATURES, atr14_pct)。
    不做 ATR 過濾（由呼叫端決定）。
    """
    import pandas as pd
    from j1stools import parquet_db

    # ── 大盤日報酬（0050）──
    df_mkt = parquet_db.query_price([MARKET_PROXY], st, end)
    df_mkt["date"] = pd.to_datetime(df_mkt["date"])
    df_mkt = df_mkt.sort_values("date")
    df_mkt["mkt_return"] = df_mkt["close"].pct_change()

    # ── 個股價格特徵 ──
    df_price = parquet_db.query_price(stocks, st, end)
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_price = df_price.sort_values(["stock_id", "date"])

    df_price["cnn_daily_return"] = df_price.groupby("stock_id")["close"].pct_change()
    df_price["cnn_volume_ratio"] = df_price.groupby("stock_id")["volume"].transform(
        lambda x: x / x.rolling(20, min_periods=5).mean()
    )
    hl = (df_price["high"] - df_price["low"]).replace(0, np.nan)
    df_price["cnn_hl_range"] = hl / df_price["close"].replace(0, np.nan)
    df_price["cnn_close_pos"] = (df_price["close"] - df_price["low"]) / hl

    # ATR14（用於過濾，不進入 CNN 特徵）
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

    df_price["atr14_pct"] = df_price.groupby("stock_id", group_keys=False).apply(_atr14)

    feat = df_price[
        [
            "date",
            "stock_id",
            "volume",
            "cnn_daily_return",
            "cnn_volume_ratio",
            "cnn_hl_range",
            "cnn_close_pos",
            "atr14_pct",
        ]
    ].copy()

    # ── 個股超額報酬 ──
    feat = feat.merge(df_mkt[["date", "mkt_return"]], on="date", how="left")
    feat["cnn_stock_vs_mkt"] = feat["cnn_daily_return"] - feat["mkt_return"].fillna(0)
    feat = feat.drop(columns=["mkt_return"])

    # ── 法人特徵 ──
    df_ib = parquet_db.query_ib(stocks, st, end)
    df_ib["date"] = pd.to_datetime(df_ib["date"])
    df_ib["net"] = df_ib["buy"] - df_ib["sell"]

    for col_name, inst in [("foreign_net_raw", "Foreign_Investor"), ("trust_net_raw", "Investment_Trust")]:
        tmp = (
            df_ib[df_ib["name"] == inst]
            .groupby(["date", "stock_id"])["net"]
            .sum()
            .reset_index()
            .rename(columns={"net": col_name})
        )
        feat = feat.merge(tmp, on=["date", "stock_id"], how="left")

    vol = feat["volume"].replace(0, np.nan)
    feat["cnn_foreign_net"] = feat["foreign_net_raw"].fillna(0) / vol
    feat["cnn_trust_net"] = feat["trust_net_raw"].fillna(0) / vol

    # ── 融資融券特徵 ──
    df_margin = parquet_db.query_margin(stocks, st, end)
    df_margin["date"] = pd.to_datetime(df_margin["date"])
    for today_col, yest_col, out_col in [
        ("margin_purchase_today_balance", "margin_purchase_yesterday_balance", "cnn_margin_chg"),
        ("short_sale_today_balance", "short_sale_yesterday_balance", "cnn_short_chg"),
    ]:
        base = df_margin[yest_col].replace(0, np.nan)
        df_margin[out_col] = (df_margin[today_col] - df_margin[yest_col]) / base

    feat = feat.merge(
        df_margin[["date", "stock_id", "cnn_margin_chg", "cnn_short_chg"]],
        on=["date", "stock_id"],
        how="left",
    )

    # ── 當沖特徵 ──
    df_dt = parquet_db.query_day_trade(stocks, st, end)
    df_dt["date"] = pd.to_datetime(df_dt["date"])
    feat = feat.merge(df_dt[["date", "stock_id", "volume"]].rename(columns={"volume": "dt_volume"}),
                      on=["date", "stock_id"], how="left")
    feat["cnn_dt_ratio"] = feat["dt_volume"].fillna(0) / feat["volume"].replace(0, np.nan)
    feat = feat.drop(columns=["volume", "foreign_net_raw", "trust_net_raw", "dt_volume"])

    # clip 極端值
    for col in ["cnn_foreign_net", "cnn_trust_net", "cnn_margin_chg", "cnn_short_chg"]:
        feat[col] = feat[col].clip(-5, 5)
    feat["cnn_daily_return"] = feat["cnn_daily_return"].clip(-0.3, 0.3)
    feat["cnn_stock_vs_mkt"] = feat["cnn_stock_vs_mkt"].clip(-0.3, 0.3)
    feat["cnn_volume_ratio"] = feat["cnn_volume_ratio"].clip(0, 10)
    feat["cnn_dt_ratio"] = feat["cnn_dt_ratio"].clip(0, 1)

    return feat.sort_values(["stock_id", "date"]).reset_index(drop=True)


# ── Alpha 標籤建構 ────────────────────────────────────────────────────────── #


def build_alpha_label_dataset(
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    hold_days: int = HOLD_DAYS,
    alpha_target: float = ALPHA_TARGET,
    alpha_stop: float = ALPHA_STOP,
    min_atr_pct: float = MIN_ATR_PCT,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    全市場（ATR > min_atr_pct）時序資料集。

    回傳 (X, y, df_meta)：
        X      : (n, lookback, n_features) float32
        y      : (n,) int，0=盤整, 1=落後大盤, 2=超越大盤
        df_meta: 對應的 DataFrame（含 date, stock_id, Y, alpha_10d）
    """
    import pandas as pd
    from j1stools import parquet_db

    # buffer：確保 st 前有足夠歷史
    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    # end_buf：確保 hold_days 後的報酬算得到
    end_buf = (pd.Timestamp(end) + pd.DateOffset(days=hold_days * 2)).strftime("%Y-%m-%d")

    print(f"── 載入 CNN 原始特徵（{st_buf} ~ {end}）──")
    df_all = build_cnn_daily_features(stocks, st_buf, end_buf)

    # ── 10日後股票報酬 ──
    df_all["future_return"] = df_all.groupby("stock_id")["cnn_daily_return"].transform(
        lambda x: (1 + x).rolling(hold_days).apply(np.prod, raw=True).shift(-hold_days) - 1
    )

    # ── 10日後大盤報酬（0050）──
    df_mkt_fut = parquet_db.query_price([MARKET_PROXY], st_buf, end_buf)
    df_mkt_fut["date"] = pd.to_datetime(df_mkt_fut["date"])
    df_mkt_fut = df_mkt_fut.sort_values("date")
    mkt_daily = df_mkt_fut["close"].pct_change()
    mkt_10d = (1 + mkt_daily).rolling(hold_days).apply(np.prod, raw=True).shift(-hold_days) - 1
    df_mkt_fut["mkt_future"] = mkt_10d
    df_all = df_all.merge(df_mkt_fut[["date", "mkt_future"]], on="date", how="left")
    df_all["alpha_10d"] = df_all["future_return"] - df_all["mkt_future"]

    # ── 標籤 ──
    df_all["Y"] = 0
    df_all.loc[df_all["alpha_10d"] >= alpha_target, "Y"] = 2
    df_all.loc[df_all["alpha_10d"] <= alpha_stop, "Y"] = 1

    # ── 過濾：只保留信號區間內 ATR 合格且有完整標籤的日期 ──
    df_signal = df_all[
        (df_all["date"] >= pd.Timestamp(st))
        & (df_all["date"] < pd.Timestamp(end))
        & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)
        & df_all["alpha_10d"].notna()
    ].copy()

    label_dist = df_signal["Y"].value_counts(normalize=True).sort_index()
    print(f"標籤分布：盤整={label_dist.get(0,0):.1%}  停損={label_dist.get(1,0):.1%}  達標={label_dist.get(2,0):.1%}")

    # ── 建立時序窗口（stride_tricks 向量化，避免 iterrows）──
    df_signal = df_signal.reset_index(drop=True)
    df_signal["date_str"] = df_signal["date"].astype(str)
    # signal lookup: (stock_id, date_str) → df_signal 行號
    signal_lookup: dict[tuple, int] = {
        (r["stock_id"], r["date_str"]): i for i, r in df_signal[["stock_id", "date_str"]].iterrows()
    }

    all_X, all_meta_idx = [], []

    for sid, grp in df_all.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[CNN_FEATURES].fillna(0).values.astype(np.float32)  # (T, F)
        dates_str = grp["date"].astype(str).values
        T = len(grp)
        if T < lookback:
            continue

        # 一次建立所有視窗：(T-lookback+1, lookback, F)
        windows = np.lib.stride_tricks.sliding_window_view(vals, (lookback, vals.shape[1]))
        windows = windows[:, 0, :, :].copy()  # copy 確保記憶體連續

        # 窗口對應的信號日（最後一天）
        for i, d in enumerate(dates_str[lookback - 1 :]):
            key = (sid, d)
            if key not in signal_lookup:
                continue
            all_X.append(windows[i])
            all_meta_idx.append(signal_lookup[key])

    kept = len(all_X)
    print(f"時序窗口（lookback={lookback}）：{kept:,} 筆")

    df_valid = df_signal.iloc[all_meta_idx].reset_index(drop=True)
    X = np.array(all_X, dtype=np.float32)
    y = df_valid["Y"].values.astype(np.int64)
    return X, y, df_valid


# ── Breakout 宇宙資料集（Option B）──────────────────────────────────────── #


def build_breakout_cnn_dataset(
    df_signals: "pd.DataFrame",
    lookback: int = 20,
    hold_days: int = 5,
    return_target: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    從 build_dataset_breakout 輸出（breakout 宇宙）建構 CNN 時序資料集。
    標籤從價格資料重新計算（hold_days 後漲幅 > return_target = 1），
    不依賴樹模型的 Y 欄位。

    df_signals: 含 stock_id, date 欄位的 breakout 訊號 df
    回傳 (X, y_binary)：
        X        : (n, lookback, n_features) float32
        y_binary : (n,) int，1=達標, 0=未達標
    """
    import pandas as pd

    stocks = df_signals["stock_id"].unique().tolist()
    dates = pd.to_datetime(df_signals["date"])
    st_buf = (dates.min() - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    end_buf = (dates.max() + pd.DateOffset(days=hold_days * 3)).strftime("%Y-%m-%d")

    print(f"── 載入 CNN 原始特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_cnn_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # 計算 hold_days 後複利報酬
    df_all["fwd_return"] = df_all.groupby("stock_id")["cnn_daily_return"].transform(
        lambda x: (1 + x).rolling(hold_days).apply(np.prod, raw=True).shift(-hold_days) - 1
    )

    df_sig = df_signals.copy()
    df_sig["date"] = pd.to_datetime(df_sig["date"])
    df_sig["date_str"] = df_sig["date"].astype(str)
    signal_lookup: dict[tuple, int] = {
        (r["stock_id"], r["date_str"]): i for i, r in df_sig[["stock_id", "date_str"]].iterrows()
    }

    all_X, all_sig_idx = [], []

    for sid, grp in df_all.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[CNN_FEATURES].fillna(0).values.astype(np.float32)
        fwd = grp["fwd_return"].values
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
            if np.isnan(fwd[lookback - 1 + i]):
                continue
            all_X.append(windows[i])
            all_sig_idx.append((signal_lookup[key], fwd[lookback - 1 + i]))

    coverage = len(all_X) / len(df_signals) if len(df_signals) > 0 else 0
    print(f"時序窗口（lookback={lookback}）：{len(all_X):,} 筆，覆蓋率：{coverage:.1%}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array([v for _, v in all_sig_idx], dtype=np.float32)

    print(
        f"標籤：{hold_days}日後報酬  mean={y.mean():.2%}  std={y.std():.2%}  "
        f">{return_target:.0%} 達標率：{(y >= return_target).mean():.1%}"
    )
    return X, y


# ── CNN 架構 ──────────────────────────────────────────────────────────────── #


class _SeqCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(4)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class SeqCNNRanker:
    """
    20日時序 SeqCNN Ranking 模型。
    輸出連續分數（regression），用 IC 評估，取 top-N 選股。
    """

    def __init__(self, lookback: int = 20, epochs: int = 50, lr: float = 1e-3, batch_size: int = 256):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._fitted_features: list = []
        self._model: _SeqCNN | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeqCNNRanker":
        self._feat_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._feat_std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        X = self._normalize(X)

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  訓練裝置：{device}")

        n_features = X.shape[2]
        self._model = _SeqCNN(n_features, n_classes=1).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        X_t = torch.tensor(X.transpose(0, 2, 1).astype(np.float32))
        y_t = torch.tensor(y.astype(np.float32)).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.6f}")

        self._model.eval().to("cpu")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳連續 score（越高越好），shape: (n,)"""
        assert self._model is not None
        X = self._normalize(X)
        X_t = torch.tensor(X.transpose(0, 2, 1).astype(np.float32))
        with torch.no_grad():
            return self._model(X_t).squeeze(1).numpy()


# 向下相容別名
SeqCNNClassifier = SeqCNNRanker


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_seq_cnn(X_train: np.ndarray, y_train: np.ndarray, lookback: int = 20, epochs: int = 50) -> SeqCNNRanker:
    """訓練 SeqCNN Ranker，X_train: (n, lookback, n_features)，y_train: continuous return"""
    ranker = SeqCNNRanker(lookback=lookback, epochs=epochs, lr=1e-3, batch_size=256)
    ranker.fit(X_train, y_train)
    ranker._fitted_features = CNN_FEATURES

    ic = np.corrcoef(ranker.predict(X_train), y_train)[0, 1]
    print(f"訓練集 IC：{ic:.4f}（in-sample）")
    return ranker


# ── Stacking：把 CNN alpha prob 加入樹模型特徵 ────────────────────────────── #

CNN_ALPHA_COL = "cnn_alpha_p2"


def add_cnn_alpha_feature(cnn: SeqCNNClassifier, df_signals, df_daily_all, lookback: int = 20):
    """
    對 df_signals 每一筆（stock_id, date），從 df_daily_all 抓 lookback 日窗口，
    計算 CNN alpha prob，加入 cnn_alpha_p2 欄位，回傳新 DataFrame。

    df_signals  : breakout 信號 DataFrame（含 stock_id, date）
    df_daily_all: build_cnn_daily_features 的輸出（含 CNN_FEATURES + atr14_pct）
    """
    import pandas as pd

    stock_feat = {sid: grp.sort_values("date").reset_index(drop=True) for sid, grp in df_daily_all.groupby("stock_id")}

    X_list, valid_mask = [], []

    for _, row in df_signals.iterrows():
        stock = row["stock_id"]
        date = pd.Timestamp(row["date"])

        grp = stock_feat.get(stock)
        if grp is None:
            valid_mask.append(False)
            continue

        pos_arr = grp.index[grp["date"] <= date]
        if len(pos_arr) < lookback:
            valid_mask.append(False)
            continue

        pos = pos_arr[-1]
        window = grp.loc[pos - lookback + 1 : pos, CNN_FEATURES].fillna(0).values
        if window.shape[0] != lookback:
            valid_mask.append(False)
            continue

        X_list.append(window)
        valid_mask.append(True)

    df_out = df_signals.copy()
    df_out[CNN_ALPHA_COL] = np.nan

    if X_list:
        X = np.array(X_list, dtype=np.float32)
        scores = cnn.predict(X)
        df_out.loc[valid_mask, CNN_ALPHA_COL] = scores

    coverage = sum(valid_mask) / len(valid_mask) * 100
    print(f"CNN alpha feature 覆蓋率：{coverage:.1f}%，填 NaN：{df_out[CNN_ALPHA_COL].isna().sum()} 筆")
    return df_out


# ── 獨立訊號生成（回測用）────────────────────────────────────────────────── #


def make_signal_seq_cnn(
    cnn: SeqCNNClassifier,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    min_atr_pct: float = MIN_ATR_PCT,
):
    """掃全市場（ATR 過濾）產生 CNN 訊號，回傳 backtest_platform 格式 signal df。"""
    import pandas as pd

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    df_all = build_cnn_daily_features(stocks, st_buf, end)
    df_all = df_all.sort_values(["stock_id", "date"])

    # 只對信號區間 + ATR 合格的日期建立窗口
    df_scan = df_all[(df_all["date"] >= pd.Timestamp(st)) & (df_all["atr14_pct"].fillna(0) >= min_atr_pct)]

    stock_feat = {sid: grp.sort_values("date").reset_index(drop=True) for sid, grp in df_all.groupby("stock_id")}

    X_list, valid_rows = [], []

    for _, row in df_scan.iterrows():
        stock = row["stock_id"]
        date = pd.Timestamp(row["date"])

        grp = stock_feat.get(stock)
        if grp is None:
            continue

        pos_arr = grp.index[grp["date"] <= date]
        if len(pos_arr) < lookback:
            continue

        pos = pos_arr[-1]
        window = grp.loc[pos - lookback + 1 : pos, CNN_FEATURES].fillna(0).values
        if window.shape[0] != lookback:
            continue

        X_list.append(window)
        valid_rows.append({"date": date, "stock_id": stock})

    X = np.array(X_list, dtype=np.float32)
    scores = cnn.predict(X)

    signal = pd.DataFrame(valid_rows)
    signal["2"] = scores
    signal["date"] = pd.to_datetime(signal["date"])
    print(f"訊號筆數：{len(signal):,}  日期：{signal['date'].min().date()} ~ {signal['date'].max().date()}")
    return signal


# ── 獨立回測 ──────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    import pandas as pd
    import matplotlib

    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    from j1stools import backtest_platform, j1s_chart, parquet_db
    from j1stools.gmm_model_plus import build_dataset_breakout
    from j1stools.ib_margin_classify import IBMarginGMM, BREAKOUT_GMM_MODEL_PATH

    TRAIN_ST = "2015-01-01"
    EVAL_ST = "2024-01-01"
    LOOKBACK = 30
    HOLD_DAYS_CNN = 5
    RETURN_TARGET = 0.03
    EXCLUDE_CLUSTERS = [6, 8]

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    clf_gmm = IBMarginGMM.load(BREAKOUT_GMM_MODEL_PATH)
    BREAKOUT_CLUSTERS = [c for c in range(clf_gmm.n_components) if c not in EXCLUDE_CLUSTERS]

    # ── Breakout 訊號資料集 ──
    print("\n══ Breakout 訓練集（GMM 過濾，2015-2024）══")
    df_train = build_dataset_breakout(clf_gmm, stocks, TRAIN_ST, EVAL_ST, clusters=BREAKOUT_CLUSTERS)
    print("\n══ Breakout 測試集（GMM 過濾，2024-）══")
    df_test = build_dataset_breakout(clf_gmm, stocks, EVAL_ST, clusters=BREAKOUT_CLUSTERS)

    # ── CNN 時序窗口 ──
    print("\n── CNN 窗口（訓練集）──")
    X_train, y_train = build_breakout_cnn_dataset(
        df_train, lookback=LOOKBACK, hold_days=HOLD_DAYS_CNN, return_target=RETURN_TARGET
    )
    print("\n── CNN 窗口（測試集）──")
    X_test, y_test = build_breakout_cnn_dataset(
        df_test, lookback=LOOKBACK, hold_days=HOLD_DAYS_CNN, return_target=RETURN_TARGET
    )

    print(f"訓練集：{X_train.shape}  測試集：{X_test.shape}")

    # ── 訓練 ──
    print(f"\n══ 訓練 SeqCNN Ranker（50 epochs，breakout宇宙，{HOLD_DAYS_CNN}日報酬）══")
    cnn = train_seq_cnn(X_train, y_train, lookback=LOOKBACK, epochs=50)

    # ── OOS 評估（IC）──
    print("\n══ OOS 評估 ══")
    scores_test = cnn.predict(X_test)
    ic_oos = np.corrcoef(scores_test, y_test)[0, 1]
    print(f"OOS IC：{ic_oos:.4f}（目標 > 0.05 有意義）")

    # top / bottom 比較（參考 margin_lgbm_main evaluate_selection）
    top_n = 20
    df_eval = pd.concat(
        [
            df_test.iloc[: len(scores_test)].reset_index(drop=True),
            pd.Series(scores_test, name="pred_score"),
            pd.Series(y_test, name="fwd_return"),
        ],
        axis=1,
    )

    top = df_eval.nlargest(top_n, "pred_score")
    bot = df_eval.nsmallest(top_n, "pred_score")
    print(f"\n{'':=<50}")
    print(f"【前 {top_n} 名（CNN 看好）】")
    print(
        f"  平均報酬：{top['fwd_return'].mean():.2%}  "
        f"勝率>0%：{(top['fwd_return']>0).mean():.1%}  "
        f"勝率>3%：{(top['fwd_return']>0.03).mean():.1%}"
    )
    print(f"【後 {top_n} 名（CNN 看壞）】")
    print(
        f"  平均報酬：{bot['fwd_return'].mean():.2%}  "
        f"勝率>0%：{(bot['fwd_return']>0).mean():.1%}  "
        f"勝率>3%：{(bot['fwd_return']>0.03).mean():.1%}"
    )
    print(f"【區別能力】報酬差：{top['fwd_return'].mean()-bot['fwd_return'].mean():.2%}")
    print(f"{'':=<50}")

    # ── 散點圖：pred score vs actual return ──
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(scores_test, y_test, alpha=0.2, s=5, color="steelblue")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    axes[0].set_xlabel("CNN score")
    axes[0].set_ylabel(f"{HOLD_DAYS_CNN}d return")
    axes[0].set_title(f"Score vs Return (IC={ic_oos:.3f})")

    # score 分十等分，看平均報酬
    df_eval["decile"] = pd.qcut(scores_test, q=10, labels=False)
    dec = df_eval.groupby("decile")["fwd_return"].mean()
    axes[1].bar(dec.index, dec.values, color="seagreen", alpha=0.8)
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", linewidth=0.8, label=f"Mean {y_test.mean():.2%}")
    axes[1].set_xlabel("Score decile (0=lowest)")
    axes[1].set_ylabel("Avg return")
    axes[1].set_title("Return by score decile (OOS)")
    axes[1].legend()
    plt.suptitle(f"SeqCNN Ranker | {HOLD_DAYS_CNN}d return | Breakout universe")
    plt.tight_layout()
    plt.show()

    # ── 回測（取 score top-N）──
    print("\n══ 獨立回測 ══")
    backtest_platform.IS_USE_CACHE = True
    sig = make_signal_seq_cnn(cnn, stocks, st=EVAL_ST, lookback=LOOKBACK)

    for label, thr, hold in [
        ("SeqCNN top score（無門檻）+ 持有5日", None, 5),
    ]:
        print(f"\n{'='*60}\n【{label}】\n{'='*60}")
        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=5,
            threshold=thr,
            max_positions=5,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=0.07,
            use_fixed_tp=True,
            tp_stop=0.07,
            use_hold_days=True,
            hold_days=hold,
            group_limit=2,
            min_volume=200,
        )
        j1s_chart.plot_performance(pv, td)
