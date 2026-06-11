"""
SeqCNN：20日時序 → 10日超額報酬預測

設計定位：獨立時序偵測器，輸出 cnn_alpha_p2（超越大盤 5% 的機率）。
此 prob 值作為樹模型的額外特徵（stacking），補充截面模型缺少的時序維度。

宇宙 : ATR14/close >= 0.05（無 GMM，無放量過濾）
標籤 : 10日後 stock_return - 0050_return >= 5% = 達標 (Y=2)
      10日後 stock_return - 0050_return <= -5% = 停損 (Y=1)
      其餘 = 盤整 (Y=0)

CNN_FEATURES (11):
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
  cnn_macd_hist       MACD histogram / close（動能方向）
  cnn_macd_div        MACD Hist背離
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


def _normalize_price_windows(X: np.ndarray) -> np.ndarray:
    return X


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
    import time as _time
    from j1stools import parquet_db

    _t0 = _time.time()
    print(f"  [CNN] 載入 price（{len(stocks)} 檔，{st}~{end}）...")

    # ── 大盤日報酬（0050）──
    df_mkt = parquet_db.query_price([MARKET_PROXY], st, end)
    df_mkt["date"] = pd.to_datetime(df_mkt["date"])
    df_mkt = df_mkt.sort_values("date")
    df_mkt["mkt_return"] = df_mkt["close"].pct_change(fill_method=None)

    # ── 個股價格特徵 ──
    df_price = parquet_db.query_price(stocks, st, end)
    print(f"  [CNN] price done ({_time.time()-_t0:.1f}s)，載入 ib...")
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_price = df_price.sort_values(["stock_id", "date"])

    df_price["cnn_daily_return"] = df_price.groupby("stock_id")["close"].pct_change(fill_method=None)
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

    df_price["atr14_pct"] = df_price.groupby("stock_id", group_keys=False).apply(_atr14, include_groups=False)

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
    print(f"  [CNN] ib... ({_time.time()-_t0:.1f}s)")
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
    print(f"  [CNN] margin... ({_time.time()-_t0:.1f}s)")
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
    print(f"  [CNN] day_trade... ({_time.time()-_t0:.1f}s)")
    df_dt = parquet_db.query_day_trade(stocks, st, end)
    df_dt["date"] = pd.to_datetime(df_dt["date"])
    feat = feat.merge(
        df_dt[["date", "stock_id", "volume"]].rename(columns={"volume": "dt_volume"}),
        on=["date", "stock_id"],
        how="left",
    )
    feat["cnn_dt_ratio"] = feat["dt_volume"].fillna(0) / feat["volume"].replace(0, np.nan)
    feat = feat.drop(columns=["volume", "foreign_net_raw", "trust_net_raw", "dt_volume"])

    # ── MACD histogram（pivot 向量化，比 groupby.apply 快）──
    _pv = df_price.pivot(index="date", columns="stock_id", values="close")
    _ema12 = _pv.ewm(span=12, adjust=False).mean()
    _ema26 = _pv.ewm(span=26, adjust=False).mean()
    _macd = _ema12 - _ema26
    _sig = _macd.ewm(span=9, adjust=False).mean()
    _hist = (_macd - _sig) / _pv.replace(0, np.nan)
    _hist_long = _hist.stack(future_stack=True).rename("cnn_macd_hist").reset_index()
    df_price = df_price.merge(_hist_long, on=["date", "stock_id"], how="left")
    # 背離：用已算好的 hist，不重複跑 EWM
    price_slope = df_price.groupby("stock_id")["close"].pct_change(5, fill_method=None)
    macd_slope = df_price.groupby("stock_id")["cnn_macd_hist"].diff(5)
    df_price["cnn_macd_div"] = price_slope * macd_slope
    feat = feat.merge(
        df_price[["date", "stock_id", "cnn_macd_hist", "cnn_macd_div"]], on=["date", "stock_id"], how="left"
    )

    # clip 極端值
    for col in ["cnn_foreign_net", "cnn_trust_net", "cnn_margin_chg", "cnn_short_chg"]:
        feat[col] = feat[col].clip(-5, 5)
    feat["cnn_daily_return"] = feat["cnn_daily_return"].clip(-0.3, 0.3)
    feat["cnn_stock_vs_mkt"] = feat["cnn_stock_vs_mkt"].clip(-0.3, 0.3)
    feat["cnn_volume_ratio"] = feat["cnn_volume_ratio"].clip(0, 10)
    feat["cnn_dt_ratio"] = feat["cnn_dt_ratio"].clip(0, 1)
    feat["cnn_macd_hist"] = feat["cnn_macd_hist"].clip(-0.05, 0.05)
    feat["cnn_macd_div"] = feat["cnn_macd_div"].clip(-0.005, 0.005)

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
    mkt_daily = df_mkt_fut["close"].pct_change(fill_method=None)
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
    X = _normalize_price_windows(np.array(all_X, dtype=np.float32))
    y = df_valid["Y"].values.astype(np.int64)
    return X, y, df_valid


# ── Breakout 宇宙資料集（Option B）──────────────────────────────────────── #


def _cs_normalize_features_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """每個交易日對 CNN_FEATURES 做截面 z-score（ddof=0），讓模型看相對排名而非絕對值。"""
    import pandas as pd

    df = df.copy()
    for col in CNN_FEATURES:
        if col in df.columns:
            df[col] = df.groupby("date")[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))
    return df


def build_breakout_cnn_dataset(
    df_signals: "pd.DataFrame",
    lookback: int = 20,
    hold_days: int = 5,
    return_target: float = 0.03,
    cs_normalize: bool = True,
    cs_normalize_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    從 build_dataset_breakout 輸出（breakout 宇宙）建構 CNN 時序資料集。

    df_signals            : 含 stock_id, date 欄位的 breakout 訊號 df
    cs_normalize          : True = label 做截面 z-score（訓練用）；False = 原始報酬率（評估用）
    cs_normalize_features : True = 特徵也做截面 z-score，讓模型學相對排名
    回傳 (X, y)：
        X : (n, lookback, n_features) float32
        y : (n,) float32
    """
    import pandas as pd

    stocks = df_signals["stock_id"].unique().tolist()
    dates = pd.to_datetime(df_signals["date"])
    st_buf = (dates.min() - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    end_buf = (dates.max() + pd.DateOffset(days=hold_days * 3)).strftime("%Y-%m-%d")

    print(f"── 載入 CNN 原始特徵（{st_buf} ~ {end_buf}）──")
    df_all = build_cnn_daily_features(stocks, st_buf, end_buf)
    df_all["date"] = pd.to_datetime(df_all["date"])

    if cs_normalize_features:
        print("  特徵截面 z-score...")
        df_all = _cs_normalize_features_df(df_all)

    # 計算 hold_days 後報酬（直接用 price，避免 clipped return 失真）
    df_all = df_all.merge(
        parquet_db.query_price(stocks, st_buf, end_buf)[["date", "stock_id", "close"]].assign(
            date=lambda d: pd.to_datetime(d["date"])
        ),
        on=["date", "stock_id"],
        how="left",
    )
    df_all["fwd_return"] = df_all.groupby("stock_id")["close"].transform(lambda x: x.shift(-hold_days) / x - 1)
    df_all = df_all.drop(columns=["close"])

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
            all_sig_idx.append((signal_lookup[key], fwd[lookback - 1 + i], d))

    coverage = len(all_X) / len(df_signals) if len(df_signals) > 0 else 0
    print(f"時序窗口（lookback={lookback}）：{len(all_X):,} 筆，覆蓋率：{coverage:.1%}")

    X = _normalize_price_windows(np.array(all_X, dtype=np.float32))
    y_raw = np.array([v for _, v, _ in all_sig_idx], dtype=np.float32)

    print(
        f"標籤：{hold_days}日後報酬  mean={y_raw.mean():.2%}  std={y_raw.std():.2%}  "
        f">{return_target:.0%} 達標率：{(y_raw >= return_target).mean():.1%}"
    )

    if cs_normalize:
        date_arr = [d for _, _, d in all_sig_idx]
        df_y = pd.DataFrame({"date": date_arr, "y": y_raw})
        df_y["y_cs"] = df_y.groupby("date")["y"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))
        y = df_y["y_cs"].values.astype(np.float32)
        print(f"截面 z-score 後：mean={y.mean():.4f}  std={y.std():.4f}")
    else:
        y = y_raw

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

    def __init__(
        self,
        lookback: int = 20,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        loss: str = "mse",  # "mse" | "correlation" | "listnet"
    ):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.loss = loss
        self._fitted_features: list = []
        self._model: _SeqCNN | None = None
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
        patience: int = 10,
    ) -> "SeqCNNRanker":
        self._feat_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._feat_std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        X = self._normalize(X)
        if X_val is not None:
            X_val_n = self._normalize(X_val)

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  訓練裝置：{device}")

        n_features = X.shape[2]
        self._model = _SeqCNN(n_features, n_classes=1).to(device)

        def _correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            p = pred.flatten() - pred.flatten().mean()
            t = target.flatten() - target.flatten().mean()
            return -torch.nn.functional.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0))

        def _listnet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            p = torch.nn.functional.softmax(pred.flatten(), dim=0)
            t = torch.nn.functional.softmax(target.flatten(), dim=0)
            return -(t * torch.log(p + 1e-9)).sum()

        _loss_map = {"mse": nn.MSELoss(), "correlation": _correlation_loss, "listnet": _listnet_loss}
        if self.loss not in _loss_map:
            raise ValueError(f"loss 必須是 {list(_loss_map)} 之一，收到：{self.loss!r}")
        criterion = _loss_map[self.loss]
        print(f"  Loss function：{self.loss}")
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        X_t = torch.tensor(X.transpose(0, 2, 1).astype(np.float32))
        y_t = torch.tensor(y.astype(np.float32)).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_val_ic = -np.inf
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
                msg = f"  epoch {epoch+1:3d}/{self.epochs}  loss={total_loss/len(loader):.6f}"
                if X_val is not None:
                    with torch.no_grad():
                        val_t = torch.tensor(X_val_n.transpose(0, 2, 1).astype(np.float32), dtype=torch.float32).to(
                            device
                        )
                        val_scores = self._model(val_t).squeeze(1).cpu().numpy()
                    _corr = np.corrcoef(val_scores, y_val)[0, 1]
                    val_ic = float(_corr) if np.isfinite(_corr) else 0.0
                    msg += f"  val_IC={val_ic:.4f}"
                    if val_ic > best_val_ic:
                        best_val_ic = val_ic
                        best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                        no_improve = 0
                        msg += " ✓"
                    else:
                        no_improve += 5
                print(msg)
                if X_val is not None and no_improve >= patience:
                    print(f"  Early stopping（val IC 連 {patience} epoch 未改善）")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"  最佳 val IC：{best_val_ic:.4f}")
        self._model.eval().to("cpu")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳連續 score（越高越好），shape: (n,)"""
        assert self._model is not None
        X = self._normalize(X.astype(np.float32))
        X_t = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        with torch.no_grad():
            return self._model(X_t).squeeze(1).numpy()


# 向下相容別名
SeqCNNClassifier = SeqCNNRanker


# ── 訓練入口 ──────────────────────────────────────────────────────────────── #


def train_seq_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lookback: int = 20,
    epochs: int = 50,
    n_runs: int = 3,
    loss: str = "mse",
) -> SeqCNNRanker:
    """訓練 n_runs 次取 val IC 最佳的模型。"""
    best_ic = -np.inf
    best_ranker = None
    for run in range(n_runs):
        print(f"\n── Run {run+1}/{n_runs} ──")
        ranker = SeqCNNRanker(lookback=lookback, epochs=epochs, lr=1e-3, batch_size=256, loss=loss)
        ranker.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        ranker._fitted_features = CNN_FEATURES
        if X_val is not None:
            _c = np.corrcoef(ranker.predict(X_val), y_val)[0, 1]
            val_ic = float(_c) if np.isfinite(_c) else 0.0
        else:
            val_ic = 0.0
        print(f"  Run {run+1} val IC：{val_ic:.4f}")
        if val_ic > best_ic:
            best_ic = val_ic
            best_ranker = ranker
    if best_ranker is None:
        best_ranker = ranker  # fallback：全部 run IC 都是 NaN 時取最後一個
    print(f"\n最佳 Run val IC：{best_ic:.4f}")
    ic = np.corrcoef(best_ranker.predict(X_train), y_train)[0, 1]
    print(f"訓練集 IC：{ic:.4f}（in-sample）")
    return best_ranker


# ── Stacking：把 CNN alpha prob 加入樹模型特徵 ────────────────────────────── #

CNN_ALPHA_COL = "cnn_score"
CNN_RANKER_PATH = "db/models/seq_cnn_ranker.pt"
CNN_SCORES_PATH = "db/models/seq_cnn_scores.parquet"


def save_seq_cnn(cnn: "SeqCNNRanker", path: str = CNN_RANKER_PATH) -> None:
    """state_dict tensor 全轉 numpy，用 joblib 存（避免 torch.load hang 問題）。"""
    import joblib, os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "lookback": cnn.lookback,
            "lr": cnn.lr,
            "batch_size": cnn.batch_size,
            "feat_mean": cnn._feat_mean,
            "feat_std": cnn._feat_std,
            "n_features": cnn._model.conv1.in_channels,
            "weights": {k: v.cpu().numpy() for k, v in cnn._model.state_dict().items()},
        },
        path,
    )
    print(f"CNN Ranker 已存：{path}")


def save_cnn_scores(
    cnn: "SeqCNNRanker",
    df_signals,
    lookback: int = 30,
    path: str = CNN_SCORES_PATH,
) -> None:
    """
    計算 df_signals 每筆訊號的 CNN score，存成 parquet（date, stock_id, cnn_score）。
    供 gmm_model_plus.py 直接讀取，避免在該 process 載入 torch（OpenMP 衝突問題）。
    """
    import pandas as pd, os

    stocks = df_signals["stock_id"].unique().tolist()
    dates = pd.to_datetime(df_signals["date"])
    st_buf = (dates.min() - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    end_buf = (dates.max() + pd.DateOffset(days=5)).strftime("%Y-%m-%d")

    print(f"  [save_cnn_scores] 載入日資料 {st_buf} ~ {end_buf}...")
    df_daily = build_cnn_daily_features(stocks, st_buf, end_buf)
    df_scored = add_cnn_alpha_feature(cnn, df_signals, df_daily, lookback=lookback)

    out = df_scored[["date", "stock_id", CNN_ALPHA_COL]].copy()
    out["date"] = pd.to_datetime(out["date"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"CNN scores 已存：{path}（{len(out):,} 筆，覆蓋率：{out[CNN_ALPHA_COL].notna().mean():.1%}）")


def load_seq_cnn(path: str = CNN_RANKER_PATH) -> "SeqCNNRanker":
    """joblib 載入，numpy weights 轉回 tensor 重建模型。"""
    import joblib
    import torch

    print(f"  [load_seq_cnn] joblib.load {path} ...")
    state = joblib.load(path)
    print(f"  [load_seq_cnn] 建立 SeqCNNRanker ...")
    cnn = SeqCNNRanker(lookback=state["lookback"], lr=state["lr"], batch_size=state["batch_size"])
    cnn._feat_mean = state["feat_mean"]
    cnn._feat_std = state["feat_std"]
    cnn._fitted_features = CNN_FEATURES
    print(f"  [load_seq_cnn] 建立 _SeqCNN ...")
    cnn._model = _SeqCNN(state["n_features"], n_classes=1)
    print(f"  [load_seq_cnn] 轉換 weights ({len(state['weights'])} 個)...")
    sd = {}
    for k, v in state["weights"].items():
        print(f"    {k}: {v.shape}")
        sd[k] = torch.from_numpy(v.copy())
    print(f"  [load_seq_cnn] load_state_dict ...")
    cnn._model.load_state_dict(sd)
    cnn._model.eval()
    print(f"  [load_seq_cnn] 完成")
    return cnn


def add_cnn_alpha_feature(cnn: "SeqCNNRanker", df_signals, df_daily_all, lookback: int = 30):
    """
    對 df_signals（breakout 訊號）加入 cnn_score 欄位（stride_tricks 向量化）。

    df_signals  : 含 stock_id, date 欄位
    df_daily_all: build_cnn_daily_features 輸出（含 CNN_FEATURES）
    """
    import pandas as pd

    df_sig = df_signals.copy()
    df_sig["date"] = pd.to_datetime(df_sig["date"])
    df_sig["_date_str"] = df_sig["date"].astype(str)
    signal_lookup: dict[tuple, int] = {
        (r["stock_id"], r["_date_str"]): i for i, r in df_sig[["stock_id", "_date_str"]].iterrows()
    }

    df_all = df_daily_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])

    all_X: list[np.ndarray] = []
    all_sig_idx: list[int] = []

    # 按 stock_id 分組 signal，避免對每檔股票掃全部日期
    sig_by_stock: dict[str, dict[str, int]] = {}
    for (sid, d), idx in signal_lookup.items():
        sig_by_stock.setdefault(sid, {})[d] = idx

    for sid, grp in df_all.groupby("stock_id"):
        if sid not in sig_by_stock:
            continue
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[CNN_FEATURES].fillna(0).values.astype(np.float32)
        dates_str = grp["date"].astype(str).values
        if len(grp) < lookback:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(vals, (lookback, vals.shape[1]))
        windows = windows[:, 0, :, :].copy()
        date_to_wi = {d: i for i, d in enumerate(dates_str[lookback - 1 :])}
        for d, sig_i in sig_by_stock[sid].items():
            wi = date_to_wi.get(d)
            if wi is None:
                continue
            all_X.append(windows[wi])
            all_sig_idx.append(sig_i)

    df_out = df_sig.drop(columns=["_date_str"])
    df_out[CNN_ALPHA_COL] = np.nan

    if all_X:
        X = _normalize_price_windows(np.array(all_X, dtype=np.float32))
        scores = cnn.predict(X)
        for arr_i, sig_i in enumerate(all_sig_idx):
            df_out.at[sig_i, CNN_ALPHA_COL] = scores[arr_i]

    coverage = df_out[CNN_ALPHA_COL].notna().mean() * 100
    print(f"CNN score 覆蓋率：{coverage:.1f}%，NaN：{df_out[CNN_ALPHA_COL].isna().sum()} 筆")
    return df_out


# ── 獨立訊號生成（回測用）────────────────────────────────────────────────── #


def make_signal_seq_cnn(
    cnn: SeqCNNClassifier,
    stocks: list,
    st: str,
    end: str = "2099-01-01",
    lookback: int = 20,
    min_atr_pct: float = MIN_ATR_PCT,
    cs_normalize_features: bool = False,
):
    """掃全市場（ATR 過濾）產生 CNN 訊號，回傳 backtest_platform 格式 signal df。"""
    import pandas as pd

    st_buf = (pd.Timestamp(st) - pd.DateOffset(days=lookback * 3)).strftime("%Y-%m-%d")
    df_all = build_cnn_daily_features(stocks, st_buf, end)
    if cs_normalize_features:
        df_all = _cs_normalize_features_df(df_all)
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
    from j1stools.gmm_classify import IBMarginGMM, BREAKOUT_GMM_MODEL_PATH, EXCLUDE_CLUSTERS

    # ── 參數 ─────────────────────────────────────────────────────────────────── #
    TRAIN_ST = "2015-01-01"
    EVAL_ST = "2024-01-01"
    LOOKBACK = 30
    HOLD_DAYS_CNN = 5
    RETURN_TARGET = 0.03

    # ── 切換模式 ──────────────────────────────────────────────────────────────── #
    #
    #  train     : 【重新訓練】建 CNN 時序窗口 → 訓練 SeqCNNRanker → 存模型 + scores parquet
    #              ⚠️  完成後需重跑 gmm_model_plus.py 以更新 cnn_score 特徵
    #
    #  scores    : 【只更新 scores】載現有模型 → 重算並存 scores parquet（不重訓）
    #              適合 GMM/訊號宇宙有變動，但 CNN 模型不動的情況
    #
    #  eval      : 【OOS 評估】載現有模型 → IC 散點圖 + decile 報酬圖
    #              ⚠️  y_test 使用原始報酬率（非 z-score），圖表數字為真實 %
    #
    #  backtest  : 【獨立回測】載現有模型 → make_signal → backtest_platform
    #
    MODE = "eval"  # "train" | "scores" | "eval" | "backtest"
    # ──────────────────────────────────────────────────────────────────────────── #

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    clf_gmm = IBMarginGMM.load(BREAKOUT_GMM_MODEL_PATH)
    BREAKOUT_CLUSTERS = [c for c in range(clf_gmm.n_components) if c not in EXCLUDE_CLUSTERS]

    print("\n══ Breakout 訓練集（GMM 過濾）══")
    df_train = build_dataset_breakout(clf_gmm, stocks, TRAIN_ST, EVAL_ST, clusters=BREAKOUT_CLUSTERS)
    print("\n══ Breakout 測試集（GMM 過濾，OOS）══")
    df_test = build_dataset_breakout(clf_gmm, stocks, EVAL_ST, clusters=BREAKOUT_CLUSTERS)

    if MODE == "train":
        print("\n── CNN 窗口（訓練集，截面 z-score）──")
        X_train, y_train = build_breakout_cnn_dataset(
            df_train,
            lookback=LOOKBACK,
            hold_days=HOLD_DAYS_CNN,
            return_target=RETURN_TARGET,
            cs_normalize=True,
            cs_normalize_features=False,
        )
        print("\n── CNN 窗口（測試集，原始報酬）──")
        X_test, y_test = build_breakout_cnn_dataset(
            df_test,
            lookback=LOOKBACK,
            hold_days=HOLD_DAYS_CNN,
            return_target=RETURN_TARGET,
            cs_normalize=False,
            cs_normalize_features=False,
        )
        print(f"訓練集：{X_train.shape}  測試集：{X_test.shape}")

        print(f"\n══ 訓練 SeqCNN Ranker（breakout宇宙，{HOLD_DAYS_CNN}日截面排名）══")
        cnn = train_seq_cnn(X_train, y_train, X_val=X_test, y_val=y_test, lookback=LOOKBACK, epochs=50)
        save_seq_cnn(cnn, CNN_RANKER_PATH)

        print("\n── 存儲 CNN scores parquet ──")
        _all = pd.concat([df_train, df_test], ignore_index=True)
        save_cnn_scores(cnn, _all, lookback=LOOKBACK, path=CNN_SCORES_PATH)
        print("\n完成。可直接跑 gmm_model_plus.py。")

    elif MODE == "scores":
        print(f"\n══ 載入現有模型 {CNN_RANKER_PATH} ══")
        cnn = load_seq_cnn(CNN_RANKER_PATH)
        print("── 重算並存儲 CNN scores parquet ──")
        _all = pd.concat([df_train, df_test], ignore_index=True)
        save_cnn_scores(cnn, _all, lookback=LOOKBACK, path=CNN_SCORES_PATH)
        print("\n完成。可直接跑 gmm_model_plus.py。")

    elif MODE == "eval":
        print(f"\n══ 載入現有模型 {CNN_RANKER_PATH} ══")
        cnn = load_seq_cnn(CNN_RANKER_PATH)

        print("\n── CNN 窗口（測試集，原始報酬）──")
        X_test, y_test = build_breakout_cnn_dataset(
            df_test,
            lookback=LOOKBACK,
            hold_days=HOLD_DAYS_CNN,
            return_target=RETURN_TARGET,
            cs_normalize=False,
            cs_normalize_features=False,
        )
        scores_test = cnn.predict(X_test)
        ic_oos = np.corrcoef(scores_test, y_test)[0, 1]
        print(f"\nOOS IC：{ic_oos:.4f}（目標 > 0.05）")

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

        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(scores_test, y_test, alpha=0.2, s=5, color="steelblue")
        axes[0].axhline(0, color="gray", linewidth=0.5)
        axes[0].axvline(0, color="gray", linewidth=0.5)
        axes[0].set_xlabel("CNN score")
        axes[0].set_ylabel(f"{HOLD_DAYS_CNN}d return")
        axes[0].set_title(f"Score vs Return (IC={ic_oos:.3f})")

        df_eval["decile"] = pd.qcut(scores_test, q=10, labels=False, duplicates="drop")
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

    elif MODE == "backtest":
        print(f"\n══ 載入現有模型 {CNN_RANKER_PATH} ══")
        cnn = load_seq_cnn(CNN_RANKER_PATH)

        print("\n══ 獨立回測 ══")
        backtest_platform.IS_USE_CACHE = True
        sig = make_signal_seq_cnn(cnn, stocks, st=EVAL_ST, lookback=LOOKBACK)

        print(f"\n{'='*60}\n【SeqCNN top score + 持有{HOLD_DAYS_CNN}日】\n{'='*60}")
        pv, td, _, _ = backtest_platform.prepare_data_backtest(
            sig,
            top_n=5,
            threshold=None,
            max_positions=5,
            use_sl_trail=False,
            use_fixed_sl=True,
            sl_stop=0.07,
            use_fixed_tp=True,
            tp_stop=0.07,
            use_hold_days=True,
            hold_days=HOLD_DAYS_CNN,
            group_limit=2,
            min_volume=200,
        )
        j1s_chart.plot_performance(pv, td)

    else:
        print(f"未知 MODE：{MODE!r}，請選 train / scores / eval / backtest")
