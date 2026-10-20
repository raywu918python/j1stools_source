"""W 底 + 拉回 特徵（只用當下已知的資料，不偷看未來）。

低點用「確認後才標記」：t-n 是前後各 n 天的最低點，到 t 才確認（同 macdh_divergences._confirmed_price_lows）。

特徵（都以第 t 天收盤時已知的資料計算）：
- f_w_bottom_30d       過去 30 個交易日內有成形且突破頸線的 W 底
- f_w_pullback_hold    W 底突破後拉回 >= PULLBACK_ATR 個 ATR，且突破後最低價沒跌破 W 底
- f_w_pullback_hist_div 拉回低點高於 W 底第二腳，但拉回期間 MACD 柱體最低值低於第二腳時的柱體（隱藏型背離）
- f_w_pullback_depth   (突破後高點 - 收盤) / ATR
- f_w_dist_to_low      (收盤 - W 底) / ATR
- f_w_setup            上面三個 0/1 同時成立
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

N = 5  # 低點前後各 N 天
LOOKBACK = 30  # W 底第一腳要在過去 30 個交易日內
LOW_TOL = 0.03  # 兩腳價差 ±3%
NECK_MIN = 0.05  # 頸線至少比低點高 5%
PULLBACK_ATR = 1.0  # 拉回至少 1 個 ATR

FEATURES = [
    "f_w_bottom_30d",
    "f_w_pullback_hold",
    "f_w_pullback_hist_div",
    "f_w_pullback_depth",
    "f_w_dist_to_low",
    "f_w_setup",
]


def _stock_features(close, high, low, hist, atr):
    size = len(close)
    out = np.zeros((size, len(FEATURES)), dtype="float32")
    lows = []  # 已確認低點 (idx, low, hist)
    for t in range(size):
        k = t - N
        if k >= N and low[k] == low[k - N : t + 1].min():
            lows.append((k, low[k], hist[k]))
        while lows and lows[0][0] < t - LOOKBACK:
            lows.pop(0)
        if len(lows) < 2 or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue

        # 由新到舊找最近一組成立的 W 底（相鄰兩個確認低點）
        for j in range(len(lows) - 1, 0, -1):
            i1, p1, _ = lows[j - 1]
            i2, p2, h2 = lows[j]
            w_low = min(p1, p2)
            if abs(p2 / p1 - 1) > LOW_TOL:
                continue
            neck = high[i1 : i2 + 1].max()
            if neck < w_low * (1 + NECK_MIN):
                continue
            after = close[i2 + 1 : t + 1]
            brk = np.flatnonzero(after > neck)
            if brk.size == 0:
                continue
            b = i2 + 1 + brk[0]  # 突破日

            peak = high[b : t + 1].max()
            pb_low = low[b : t + 1].min()
            depth = (peak - close[t]) / atr[t]
            hold = depth >= PULLBACK_ATR and low[i2 + 1 : t + 1].min() > w_low
            pb_hist = np.nanmin(hist[b : t + 1]) if np.isfinite(hist[b : t + 1]).any() else np.nan
            hidden_div = hold and pb_low > p2 and np.isfinite(h2) and pb_hist < h2

            out[t] = [1, hold, hidden_div, depth, (close[t] - w_low) / atr[t], hold and hidden_div]
            break
    return out


def f_w_bottom_pullback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["stock_id", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("stock_id", sort=False):
        atr = ta.atr(g["high"], g["low"], g["close"], length=14)
        macd = ta.macd(g["close"], fast=12, slow=26, signal=9)
        hist = macd.iloc[:, 1] if macd is not None else pd.Series(np.nan, index=g.index)
        feats = _stock_features(
            g["close"].to_numpy("float64"),
            g["high"].to_numpy("float64"),
            g["low"].to_numpy("float64"),
            hist.to_numpy("float64"),
            (atr if atr is not None else pd.Series(np.nan, index=g.index)).to_numpy("float64"),
        )
        parts.append(pd.DataFrame(feats, index=g.index, columns=FEATURES))
    feat = pd.concat(parts) if parts else pd.DataFrame(columns=FEATURES)
    return pd.concat([df, feat], axis=1).sort_values(by=["date", "stock_id"]).reset_index(drop=True)
