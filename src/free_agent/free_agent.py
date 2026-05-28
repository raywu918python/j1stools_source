"""
free_agent.py — Groq + Llama 3.3 70B 免費統計探索 Agent

流程：
  給定報價 + IB + Margin 資料
  → Agent 自動產生條件假設
  → analyze_signal 計算統計效果
  → 報告有效組合，供你訓練 / 驗證

安裝：pip install groq schedule
設定：.env 加入 GROQ_API_KEY=xxx
申請：https://console.groq.com → API Keys（免費，每天 14,400 次）
"""

import json
import os
import schedule
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 額度用完自動輪換，順序：最強 → 備用 → 量大
MODELS = [
    "llama-3.3-70b-versatile",  # 1,000 次/天，tool calling 最穩
    "llama-3.1-8b-instant",     # 14,400 次/天
]
_model_idx = 0


def _current_model() -> str:
    return MODELS[_model_idx]


def _next_model() -> bool:
    global _model_idx
    if _model_idx < len(MODELS) - 1:
        _model_idx += 1
        print(f"\n[模型輪換] 切換至 {MODELS[_model_idx]}")
        return True
    print("\n[模型輪換] 所有模型額度已用完，今天停止")
    return False


# ── 工具定義（OpenAI 格式）─────────────────────────────────
# 指標清單已在 SYSTEM_PROMPT 說明，description 只留用途
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "quick_rfc",
            "description": "訓練 RandomForestClassifier，用 predict_proba 評估各信心度門檻的命中率，回傳 feature importance。",
            "parameters": {
                "type": "object",
                "properties": {
                    "hold_days":     {"type": "integer", "description": "持有天數，預設 10"},
                    "profit_target": {"type": "number",  "description": "目標報酬，預設 0.15"},
                    "n_stocks":      {"type": "integer", "description": "抽樣股票數，預設 200"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_signal",
            "description": "測試條件字串，回傳 hit_rate/avg_return/win_rate/sample_count。條件格式見 SYSTEM_PROMPT。",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition":      {"type": "string",  "description": "條件字串，用 & | ~ "},
                    "hold_days":      {"type": "integer", "description": "持有天數，預設 10"},
                    "label_type":     {"type": "string",  "description": "hit/return/max_return/atr_move，預設 hit"},
                    "profit_target":  {"type": "number",  "description": "hit 的目標報酬，預設 0.15"},
                },
                "required": ["condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_correlations",
            "description": "掃描所有指標與目標（N天漲15%）的相關性，第一步必須呼叫。",
            "parameters": {
                "type": "object",
                "properties": {
                    "hold_days":     {"type": "integer", "description": "持有天數，預設 10"},
                    "profit_target": {"type": "number",  "description": "目標報酬，預設 0.15"},
                    "top_n":         {"type": "integer", "description": "回傳前幾名，預設 20"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "條件通過 analyze_signal 後執行回測，設停損停利，回傳 sharpe/win_rate/MDD/total_return。",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string",  "description": "進場條件字串"},
                    "hold_days": {"type": "integer", "description": "最大持有天數，預設 10"},
                    "sl_stop":   {"type": "number",  "description": "停損，預設 0.08"},
                    "tp_stop":   {"type": "number",  "description": "停利，預設 0.15"},
                },
                "required": ["condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_pattern_signal",
            "description": "偵測 ABCD 或三角收斂型態，統計型態後的報酬。pattern=abcd 或 triangle。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern":    {"type": "string", "description": "abcd 或 triangle"},
                    "hold_days":  {"type": "integer", "description": "持有天數，預設 5"},
                    "label_type": {"type": "string",  "description": "return/max_return/atr_move，預設 return"},
                },
                "required": ["pattern"],
            },
        },
    },
]


# ── 指標建構 ───────────────────────────────────────────────
def _rolling_rank(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return df.rolling(window, min_periods=max(1, window // 2)).rank(pct=True)


def _build_indicators(stocks: list, start: str, end: str):
    from j1stools import parquet_db

    df_p = parquet_db.query_price(stocks, start, end)
    df_p["date"] = pd.to_datetime(df_p["date"])

    piv = lambda col: df_p.pivot(index="date", columns="stock_id", values=col).sort_index()
    close = piv("close")
    high = piv("high")
    low = piv("low")
    vol = piv("volume")

    ind: dict = {
        "close": close,
        "high": high,
        "low": low,
        "open": piv("open"),
        "volume": vol,
    }

    for n in [5, 10, 20, 60, 120]:
        ind[f"ma{n}"] = close.rolling(n).mean()

    for fast, slow, name in [(5, 10, "ma5_x_ma10"), (5, 20, "ma5_x_ma20"), (10, 60, "ma10_x_ma60")]:
        f, s = ind[f"ma{fast}"], ind[f"ma{slow}"]
        ind[name] = (f > s) & (f.shift(1) <= s.shift(1))

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    ind["rsi"] = 100 - 100 / (1 + gain / loss)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"] = ema12 - ema26
    ind["signal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macdh"] = ind["macd"] - ind["signal"]

    ind["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ind["bb_upper"] = ind["bb_mid"] + 2 * bb_std
    ind["bb_lower"] = ind["bb_mid"] - 2 * bb_std
    ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_mid"]

    tr = (
        pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ]
        )
        .groupby(level=0)
        .max()
    )
    atr14 = tr.rolling(14).mean()
    ind["atr"] = atr14
    ind["atr_rank"] = _rolling_rank(atr14)  # 波動度百分位

    # ADX / +DI / -DI (14)
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    ind["adx"] = dx.rolling(14).mean()  # >25 趨勢明顯，>40 強趨勢
    ind["plus_di"] = plus_di  # 多方力道
    ind["minus_di"] = minus_di  # 空方力道

    # 成交量
    vol_ma20 = vol.rolling(20).mean()
    ind["vol_ratio"] = vol / (vol_ma20 + 1e-9)
    ind["vol_trend"] = vol_ma20 / vol.rolling(60).mean()  # 量能趨勢（>1 量增）
    # OBV
    direction = close.diff().apply(lambda x: x.map(lambda v: 1 if v > 0 else (-1 if v < 0 else 0)))
    ind["obv"] = (vol * direction).cumsum()
    ind["obv_ma"] = ind["obv"].rolling(20).mean()

    # ABCD N 字型特徵
    try:
        from j1stools.abcd_feature import detect_n_shape_features

        df_abcd = detect_n_shape_features(df_p.copy())
        for feat in [
            "f_n_confirmed",
            "f_n_structure_score",
            "f_n_bc_retracement",
            "f_n_ab_gain",
            "f_n_d_breakout_strength",
            "f_n_volume_confirm",
        ]:
            if feat in df_abcd.columns:
                ind[feat] = df_abcd.pivot(index="date", columns="stock_id", values=feat).reindex(close.index)
    except Exception:
        pass

    # IB
    try:
        df_ib = parquet_db.query_ib(stocks, start, end)
        df_ib["date"] = pd.to_datetime(df_ib["date"])
        df_ib["net"] = df_ib["buy"] - df_ib["sell"]
        nets = []
        for src, col, rank_col in [
            ("Foreign_Investor", "net_foreign", "foreign_rank"),
            ("Investment_Trust", "net_trust", "trust_rank"),
            ("Dealer_self", "net_dealer", "dealer_rank"),
        ]:
            sub = (
                df_ib[df_ib["name"] == src]
                .pivot(index="date", columns="stock_id", values="net")
                .reindex(close.index)
                .fillna(0)
            )
            ind[col] = sub
            ind[rank_col] = _rolling_rank(sub)
            nets.append(sub)
        inst = sum(nets)
        ind["net_inst"] = inst
        ind["inst_rank"] = _rolling_rank(inst)
    except Exception:
        pass

    # Margin
    try:
        df_m = parquet_db.query_margin(stocks, start, end)
        df_m["date"] = pd.to_datetime(df_m["date"])

        def _mpiv(col):
            return df_m.pivot(index="date", columns="stock_id", values=col).reindex(close.index).ffill()

        margin = _mpiv("margin_purchase_today_balance")
        short = _mpiv("short_sale_today_balance")
        ind["margin_bal"] = margin
        ind["short_bal"] = short
        ind["margin_chg"] = margin.pct_change().clip(-1, 1)
        ind["short_chg"] = short.pct_change().clip(-1, 1)
        ind["short_ratio"] = short / (margin + 1e-9)
        ind["margin_chg_rank"] = _rolling_rank(ind["margin_chg"])
        ind["short_ratio_rank"] = _rolling_rank(ind["short_ratio"])
    except Exception:
        pass

    return ind, close


# ── RFC 快速訓練 ───────────────────────────────────────────
def _build_rfc_features(stocks: list, start: str, end: str) -> pd.DataFrame:
    """長格式 feature table，每行一個 (date, stock_id)"""
    from j1stools import parquet_db

    df = parquet_db.query_price(stocks, start, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    rows = []
    for sid, g in df.groupby("stock_id"):
        g = g.set_index("date").sort_index()
        c = g["close"]
        h = g["high"]
        l = g["low"]
        v = g["volume"]

        # 均線距離
        for n in [5, 10, 20, 60]:
            ma = c.rolling(n).mean()
            g[f"f_dist_ma{n}"] = (c - ma) / (ma + 1e-9)

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        g["f_rsi"] = 100 - 100 / (1 + gain / loss)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        g["f_macdh"] = macd - macd.ewm(span=9, adjust=False).mean()

        # 布林寬度 & 位置
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        g["f_bb_width"] = (2 * 2 * bb_std) / (bb_mid + 1e-9)
        g["f_bb_pos"] = (c - bb_mid) / (bb_std + 1e-9)

        # ATR 正規化
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        g["f_atr_norm"] = atr / (c + 1e-9)

        # 成交量
        g["f_vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)

        g["stock_id"] = sid
        rows.append(g.reset_index())

    feat_df = pd.concat(rows, ignore_index=True)

    # 加 IB rank
    try:
        df_ib = parquet_db.query_ib(stocks, start, end)
        df_ib["date"] = pd.to_datetime(df_ib["date"])
        df_ib["net"] = df_ib["buy"] - df_ib["sell"]
        for src, col in [("Foreign_Investor", "f_foreign_rank"), ("Investment_Trust", "f_trust_rank")]:
            sub = (
                df_ib[df_ib["name"] == src]
                .pivot(index="date", columns="stock_id", values="net")
                .rolling(60, min_periods=10)
                .rank(pct=True)
                .stack()
                .reset_index()
            )
            sub.columns = ["date", "stock_id", col]
            feat_df = feat_df.merge(sub, on=["date", "stock_id"], how="left")
    except Exception:
        feat_df["f_foreign_rank"] = np.nan
        feat_df["f_trust_rank"] = np.nan

    # 加 Margin rank
    try:
        df_m = parquet_db.query_margin(stocks, start, end)
        df_m["date"] = pd.to_datetime(df_m["date"])
        margin = df_m.pivot(index="date", columns="stock_id", values="margin_purchase_today_balance")
        short = df_m.pivot(index="date", columns="stock_id", values="short_sale_today_balance")
        ratio = short / (margin + 1e-9)
        ratio_rank = ratio.rolling(60, min_periods=10).rank(pct=True).stack().reset_index()
        ratio_rank.columns = ["date", "stock_id", "f_short_ratio_rank"]
        feat_df = feat_df.merge(ratio_rank, on=["date", "stock_id"], how="left")
    except Exception:
        feat_df["f_short_ratio_rank"] = np.nan

    return feat_df


def _normalize_condition(cond: str) -> str:
    """把 a>1&b<2 補成 (a>1) & (b<2)，避免 pandas operator precedence 錯誤"""
    import re
    tokens = re.split(r'(\s*[&|]\s*)', cond.strip())
    result = []
    for tok in tokens:
        s = tok.strip()
        if s in ('&', '|'):
            result.append(f' {s} ')
        elif s:
            result.append(s if s.startswith('(') else f'({s})')
    return ''.join(result)


# ── 工具執行 ───────────────────────────────────────────────
def execute_tool(name: str, inputs: dict) -> dict:
    if name == "scan_correlations":
        hold_days = inputs.get("hold_days", 10)
        profit_target = inputs.get("profit_target", 0.15)
        top_n = inputs.get("top_n", 20)

        try:
            ind, close = _build_indicators(WATCH_STOCKS, SESSION_START, SESSION_END)
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}

        # target：N天最高點 >= profit_target（binary）
        future_high = pd.concat([close.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
        target = ((future_high / close - 1) >= profit_target).astype(float)
        target_flat = target.values.flatten().astype(float)

        # 跳過原始價格欄位，只看衍生指標
        skip = {
            "close",
            "high",
            "low",
            "open",
            "volume",
            "atr",
            "bb_upper",
            "bb_lower",
            "bb_mid",
            "macd",
            "signal",
            "net_foreign",
            "net_trust",
            "net_dealer",
            "net_inst",
            "margin_bal",
            "short_bal",
            "obv",
        }

        corr_list = []
        for feat_name, df in ind.items():
            if feat_name in skip:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            # 對齊 target 的 index 和 columns（確保 shape 一致）
            df_aligned = df.reindex(index=close.index, columns=close.columns)
            x = df_aligned.values.flatten().astype(float)
            y = target_flat
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 500:
                continue
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            if not np.isnan(corr):
                corr_list.append({"feature": feat_name, "correlation": round(corr, 4)})

        corr_list.sort(key=lambda r: abs(r["correlation"]), reverse=True)

        return {
            "hold_days": hold_days,
            "profit_target": profit_target,
            "top_correlations": corr_list[:top_n],
            "note": "correlation > 0 代表與上漲正相關，用高相關性指標組合 analyze_signal 條件",
        }

    if name == "quick_rfc":
        import random as _random
        from sklearn.ensemble import RandomForestClassifier
        from j1stools import parquet_db

        hold_days     = inputs.get("hold_days", 10)
        profit_target = inputs.get("profit_target", 0.15)
        n_stocks      = inputs.get("n_stocks", 200)

        stocks = _random.sample(WATCH_STOCKS, min(n_stocks, len(WATCH_STOCKS)))

        try:
            feat_df = _build_rfc_features(stocks, "2015-01-01", SESSION_END)
        except Exception as e:
            return {"error": f"特徵建構失敗：{e}"}

        # target：N天最高點 >= profit_target → 1，否則 0
        close_wide = feat_df.pivot(index="date", columns="stock_id", values="close")
        future_max  = pd.concat(
            [close_wide.shift(-i) for i in range(1, hold_days + 1)], axis=0
        ).groupby(level=0).max()
        target_wide = ((future_max / close_wide - 1) >= profit_target).astype(int)
        target_long = target_wide.stack().reset_index()
        target_long.columns = ["date", "stock_id", "target"]
        feat_df = feat_df.merge(target_long, on=["date", "stock_id"], how="left")

        # 把 agent 統計通過的條件加成 binary 特徵
        passing = [e for e in _load_signal_log() if e.get("hit_rate", 0) >= 0.35 and "condition" in e]
        added_signal_feats = []
        if passing:
            try:
                ind, _ = _build_indicators(stocks, "2015-01-01", SESSION_END)
                for i, entry in enumerate(passing[:20]):
                    col = f"f_signal_{i}"
                    try:
                        sig = eval(entry["condition"], {"__builtins__": {}}, ind)
                        if isinstance(sig, pd.DataFrame):
                            sig_long = sig.astype(float).stack().reset_index()
                            sig_long.columns = ["date", "stock_id", col]
                            feat_df = feat_df.merge(sig_long, on=["date", "stock_id"], how="left")
                            feat_df[col] = feat_df[col].fillna(0)
                            added_signal_feats.append(col)
                    except Exception:
                        pass
            except Exception:
                pass

        f_cols = [c for c in feat_df.columns if c.startswith("f_")]
        feat_df = feat_df.dropna(subset=[c for c in f_cols if c not in added_signal_feats] + ["target"])

        # 時間切分：前 80% 訓練，後 20% 測試
        dates  = sorted(feat_df["date"].unique())
        cutoff = dates[int(len(dates) * 0.8)]
        train  = feat_df[feat_df["date"] < cutoff]
        test   = feat_df[feat_df["date"] >= cutoff]

        if len(train) < 200 or len(test) < 100:
            return {"error": f"資料不足：train={len(train)}, test={len(test)}"}

        rfc = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
        rfc.fit(train[f_cols], train["target"])
        proba  = rfc.predict_proba(test[f_cols])[:, 1]
        actual = test["target"].values

        # 基準率：不用模型，直接看有多少筆命中
        base_rate = round(float(actual.mean()), 4)

        # 各信心度門檻：proba >= th 時的實際命中率
        results = {}
        for th in [0.5, 0.6, 0.7, 0.8]:
            mask = proba >= th
            n    = int(mask.sum())
            if n < 10:
                results[f"proba>={th}"] = {"n": n, "note": "樣本不足"}
                continue
            hit = float(actual[mask].mean())
            results[f"proba>={th}"] = {"n": n, "hit_rate": round(hit, 4)}

        # feature importance 排名
        importance   = sorted(zip(f_cols, rfc.feature_importances_), key=lambda x: x[1], reverse=True)
        top_features = [{"feature": f, "importance": round(float(v), 4)} for f, v in importance[:10]]

        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "quick_rfc.joblib")
        joblib.dump(rfc, model_path)

        return {
            "hold_days":          hold_days,
            "profit_target":      profit_target,
            "train_size":         len(train),
            "test_size":          len(test),
            "base_rate":          base_rate,
            "thresholds":         results,
            "top_features":       top_features,
            "signal_feats_added": len(added_signal_feats),
            "model_saved":        model_path,
        }

    if name == "run_backtest":
        from j1stools.backtest_engine import backtest_engine

        condition = _normalize_condition(inputs["condition"])
        hold_days = inputs.get("hold_days", 10)
        sl_stop   = inputs.get("sl_stop", 0.08)
        tp_stop   = inputs.get("tp_stop", 0.15)

        try:
            ind, close = _build_indicators(WATCH_STOCKS, SESSION_START, SESSION_END)
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}

        try:
            entries = eval(condition, {"__builtins__": {}}, ind)
        except Exception as e:
            return {"error": f"條件解析失敗：{e}"}

        if not isinstance(entries, pd.DataFrame):
            return {"error": "條件必須回傳 DataFrame"}

        entries     = entries.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        exits       = pd.Series(False, index=close.index)
        df_proba    = pd.DataFrame(1.0,   index=close.index, columns=close.columns)
        stock_group = {s: "default" for s in close.columns}

        try:
            portfolio_value, trades_df, _ = backtest_engine(
                close=close, entries=entries, exits=exits,
                df_proba=df_proba, stock_group=stock_group,
                hold_days=hold_days, sl_stop=sl_stop, tp_stop=tp_stop,
                use_fixed_sl=True, use_fixed_tp=True,
                use_sl_trail=False, use_hold_days=True,
            )
        except Exception as e:
            return {"error": f"回測失敗：{e}"}

        if len(trades_df) < 5:
            return {"error": f"交易筆數不足（{len(trades_df)} 筆）"}

        pv           = portfolio_value.dropna()
        rets         = pv.pct_change(fill_method=None).dropna()
        total_return = round(float(pv.iloc[-1] / pv.iloc[0] - 1), 4) if len(pv) > 1 else None
        sharpe       = round(float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0, 2)
        max_dd       = round(float(((pv / pv.cummax()) - 1).min()), 4)

        # trades_df 欄位：return_pct（百分比），pnl（絕對值）
        if "return_pct" in trades_df.columns:
            rp         = trades_df["return_pct"]
            win_rate   = round(float((rp > 0).mean()), 4)
            avg_return = round(float(rp.mean()), 4)
            avg_win    = round(float(rp[rp > 0].mean()), 4) if (rp > 0).any() else 0
            avg_loss   = round(float(rp[rp < 0].mean()), 4) if (rp < 0).any() else 0
        else:
            win_rate = avg_return = avg_win = avg_loss = None

        # 存 HTML 圖表
        chart_path = None
        try:
            from j1stools.j1s_chart import plot_performance
            import plotly.graph_objects as go, json as _json
            fig_json  = plot_performance(portfolio_value, trades_df, is_web=True)
            fig       = go.Figure(_json.loads(fig_json))
            os.makedirs(RESULTS_DIR, exist_ok=True)
            slug      = condition[:60].replace(" ", "").replace("(", "").replace(")", "").replace(">", "gt").replace("<", "lt").replace("&", "_")
            chart_path = os.path.join(RESULTS_DIR, f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.html")
            fig.write_html(chart_path)
            print(f"  圖表已存：{chart_path}")
        except Exception as ce:
            print(f"  圖表存檔失敗：{ce}")

        return {
            "condition":     condition,
            "hold_days":     hold_days,
            "sl_stop":       sl_stop,
            "tp_stop":       tp_stop,
            "total_trades":  len(trades_df),
            "total_return":  total_return,
            "sharpe":        sharpe,
            "max_drawdown":  max_dd,
            "win_rate":      win_rate,
            "avg_return":    avg_return,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "chart":         chart_path,
        }

    if name == "analyze_signal":
        hold_days = inputs.get("hold_days", 5)
        condition = _normalize_condition(inputs["condition"])
        label_type = inputs.get("label_type", "return")
        profit_target = inputs.get("profit_target", 0.15)

        try:
            ind, close = _build_indicators(WATCH_STOCKS, SESSION_START, SESSION_END)
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}

        try:
            signal = eval(condition, {"__builtins__": {}}, ind)
        except Exception as e:
            return {"error": f"條件解析失敗：{e}"}

        if not isinstance(signal, pd.DataFrame):
            return {"error": "條件必須回傳 DataFrame（每支股票一欄）"}

        mask = signal.fillna(False).astype(bool)

        # N 天最高點（max_return / atr_move / hit 都需要）
        if label_type in ("max_return", "atr_move", "hit"):
            future_high = pd.concat([close.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()

        if label_type == "max_return":
            labels = future_high / close - 1
        elif label_type == "atr_move":
            labels = (future_high - close) > ind["atr"]
        elif label_type == "hit":
            # 10天內最高點有沒有超過 profit_target
            labels = (future_high / close - 1) >= profit_target
        else:
            labels = close.shift(-hold_days) / close - 1

        values = labels[mask].values.flatten()
        if label_type in ("atr_move", "hit"):
            values = values[~pd.isnull(values)]
        else:
            values = values[~np.isnan(values.astype(float))]

        n = len(values)
        if n < 20:
            return {"error": f"樣本太少（{n} 筆），條件太嚴或資料不足"}

        if label_type in ("atr_move", "hit"):
            hit_rate = float(np.mean(values.astype(float)))
            key = "hit_rate" if label_type == "hit" else "atr_hit_rate"
            out = {
                "condition": condition,
                "label_type": label_type,
                "hold_days": hold_days,
                "sample_count": n,
                key: round(hit_rate, 4),
            }
            if label_type == "hit":
                out["profit_target"] = profit_target
            _append_signal_log({**out, "date": datetime.now().strftime("%Y-%m-%d")})
            return out

        values = values.astype(float)
        avg_r = float(np.mean(values))
        win = float((values > 0).mean())
        std_r = float(np.std(values))
        out = {
            "condition": condition,
            "label_type": label_type,
            "hold_days": hold_days,
            "sample_count": n,
            "avg_return": round(avg_r, 4),
            "win_rate": round(win, 4),
            "median_return": round(float(np.median(values)), 4),
            "std": round(std_r, 4),
            "sharpe_proxy": round(avg_r / (std_r + 1e-9), 4),
        }
        _append_signal_log({**out, "date": datetime.now().strftime("%Y-%m-%d")})
        return out

    if name == "analyze_pattern_signal":
        import random as _random

        pattern = inputs.get("pattern", "abcd")
        hold_days = inputs.get("hold_days", 5)
        label_type = inputs.get("label_type", "return")

        from j1stools import parquet_db

        if pattern == "abcd":
            stocks_sample = WATCH_STOCKS
        else:
            stocks_sample = _random.sample(WATCH_STOCKS, min(100, len(WATCH_STOCKS)))

        try:
            df_p = parquet_db.query_price(stocks_sample, SESSION_START, SESSION_END)
            df_p["date"] = pd.to_datetime(df_p["date"])
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}

        close_wide = df_p.pivot(index="date", columns="stock_id", values="close").sort_index()

        # 建 signal mask
        signal_mask = pd.DataFrame(False, index=close_wide.index, columns=close_wide.columns)

        if pattern == "abcd":
            try:
                from j1stools.abcd_feature import detect_n_shape_features

                df_abcd = detect_n_shape_features(df_p.copy())
                confirmed = (
                    df_abcd.pivot(index="date", columns="stock_id", values="f_n_confirmed")
                    .reindex(close_wide.index)
                    .fillna(0)
                )
                signal_mask = confirmed.astype(bool)
            except Exception as e:
                return {"error": f"ABCD 偵測失敗：{e}"}

        elif pattern == "triangle":
            try:
                from j1stools.pattern_triangle import find_human_triangle

                df_tri = find_human_triangle(df_p.copy())
                for _, row in df_tri.iterrows():
                    d, s = row["date"], row["stock_id"]
                    if d in signal_mask.index and s in signal_mask.columns:
                        signal_mask.at[d, s] = True
            except Exception as e:
                return {"error": f"三角收斂偵測失敗：{e}"}

        # 計算 label
        atr_wide = None
        if label_type == "atr_move":
            tr = (
                pd.concat(
                    [
                        df_p.pivot(index="date", columns="stock_id", values="high").sort_index()
                        - df_p.pivot(index="date", columns="stock_id", values="low").sort_index(),
                    ]
                )
                .groupby(level=0)
                .max()
            )
            atr_wide = tr.rolling(14).mean().reindex(close_wide.index)

        if label_type == "max_return":
            labels = (
                pd.concat([close_wide.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
                / close_wide
                - 1
            )
        elif label_type == "atr_move":
            future_high = (
                pd.concat([close_wide.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
            )
            labels = (future_high - close_wide) > atr_wide
        else:
            labels = close_wide.shift(-hold_days) / close_wide - 1

        values = labels[signal_mask].values.flatten()
        if label_type == "atr_move":
            values = values[~pd.isnull(values)]
        else:
            values = values[~np.isnan(values.astype(float))]

        n = len(values)
        if n < 20:
            return {"error": f"樣本太少（{n} 筆）"}

        if label_type == "atr_move":
            return {
                "pattern": pattern,
                "label_type": label_type,
                "hold_days": hold_days,
                "sample_count": n,
                "atr_hit_rate": round(float(np.mean(values)), 4),
            }

        values = values.astype(float)
        avg_r = float(np.mean(values))
        std_r = float(np.std(values))
        return {
            "pattern": pattern,
            "label_type": label_type,
            "hold_days": hold_days,
            "sample_count": n,
            "avg_return": round(avg_r, 4),
            "win_rate": round(float((values > 0).mean()), 4),
            "median_return": round(float(np.median(values)), 4),
            "std": round(std_r, 4),
            "sharpe_proxy": round(avg_r / (std_r + 1e-9), 4),
        }

    return {"error": f"未知工具：{name}"}


# ── Agent 主循環 ───────────────────────────────────────────
SYSTEM_PROMPT = """你是台灣股票量化研究助理，目標：找出 10 天內最高點能超過 +15% 的有效指標組合。

━━ 可用指標（analyze_signal 的 condition 字串）━━
【報價】close, high, low, open, volume
【均線】ma5, ma10, ma20, ma60, ma120；交叉: ma5_x_ma10, ma5_x_ma20, ma10_x_ma60
【動能】rsi, macd, signal, macdh, bb_upper, bb_lower, bb_mid, bb_width
【趨勢】adx(>25趨勢,>40強), plus_di, minus_di
【波動】atr, atr_rank(波動百分位 0~1)
【量能】vol_ratio(今量/20日均), vol_trend(量能趨勢), obv, obv_ma
【ABCD】f_n_confirmed(1=N字成立), f_n_structure_score, f_n_bc_retracement, f_n_ab_gain, f_n_d_breakout_strength, f_n_volume_confirm
【IB】net_foreign, net_trust, net_dealer, net_inst; foreign_rank, trust_rank, dealer_rank, inst_rank
【Margin】margin_bal, short_bal, margin_chg, short_chg, short_ratio; margin_chg_rank, short_ratio_rank

label_type 選擇：
  return      — 持有 N 天固定報酬
  max_return  — N 天內最高點報酬
  atr_move    — N 天內是否漲超過 1 ATR（命中率）
  hit         — N 天內最高點 >= profit_target 的命中率 ← 主要用這個

━━ 工作流程（三步驟）━━

【步驟一：相關性掃描】
1. 第一步固定呼叫 scan_correlations(hold_days=10, profit_target=0.15)
2. 記住 top 10 相關性指標，之後的 analyze_signal 條件以這些指標為主軸

【步驟二：統計篩選（1年資料）】
3. 根據相關性排名，組合 6~8 個條件，用高相關性指標為核心，再加其他過濾
4. 每個條件用 analyze_signal(label_type="hit", hold_days=10, profit_target=0.15)
5. 篩選標準：hit_rate >= 0.35 且 sample_count >= 30
6. 不通過 → 參考相關性結果修改（最多調整 3 次）

【步驟三：回測驗證（含停損停利）】
7. 每個通過的條件執行 run_backtest(condition=..., hold_days=10, sl_stop=0.08, tp_stop=0.15)
8. 評估標準：win_rate > 0.5 且 sharpe > 0.5 且 avg_loss > -0.09（停損有效）
9. 通過回測的條件才算真正有效

【步驟四：RFC 模型（全期 2015-2026）】
10. 回測通過後，執行 quick_rfc(hold_days=10, profit_target=0.15)
11. 看 top_features 和 thresholds，對比 base_rate

━━ 報告格式 ━━
通過條件：條件字串 + hit_rate + sample_count + 分析說明
RFC 結果：各門檻 hit_rate vs base_rate，結論
建議：下一步特徵設計方向

請用繁體中文回覆。"""


def _compress_result(fn_name: str, result: dict) -> str:
    """把 tool 結果壓縮成短字串，減少 context token 用量"""
    if "error" in result:
        return f"error: {result['error']}"

    if fn_name == "analyze_signal":
        cond = result.get("condition", "")[-40:]  # 只保留條件尾段
        if "hit_rate" in result:
            flag = "✅" if result["hit_rate"] >= 0.35 else "❌"
            return f"hit={result['hit_rate']} n={result.get('sample_count')} {flag} ...{cond}"
        return f"avg={result.get('avg_return')} win={result.get('win_rate')} n={result.get('sample_count')} ...{cond}"

    if fn_name == "run_backtest":
        return (
            f"return={result.get('total_return')} sharpe={result.get('sharpe')} "
            f"mdd={result.get('max_drawdown')} win={result.get('win_rate')} "
            f"avg_win={result.get('avg_win')} avg_loss={result.get('avg_loss')} "
            f"trades={result.get('total_trades')}"
        )

    if fn_name == "quick_rfc":
        base = result.get("base_rate")
        ths  = result.get("thresholds", {})
        th_str = " | ".join(
            f"{k}: n={v['n']} hit={v.get('hit_rate','?')}"
            for k, v in ths.items() if "hit_rate" in v
        )
        feats = [f["feature"] for f in result.get("top_features", [])[:3]]
        return f"base={base} | {th_str} | top:{','.join(feats)} signal_added={result.get('signal_feats_added')}"

    if fn_name == "scan_correlations":
        top = result.get("top_correlations", [])[:8]
        items = " ".join(f"{r['feature']}({r['correlation']})" for r in top)
        return f"top相關: {items}"

    if fn_name == "analyze_pattern_signal":
        if "hit_rate" in result:
            return f"pattern={result.get('pattern')} hit={result['hit_rate']} n={result.get('sample_count')}"
        return f"pattern={result.get('pattern')} avg={result.get('avg_return')} win={result.get('win_rate')} n={result.get('sample_count')}"

    # 其他 tool 直接壓縮成單行
    return json.dumps(result, ensure_ascii=False)[:200]


def _trim_messages(messages: list, keep_last: int = 20) -> list:
    """只送 system + 第一條 user（含歷史記憶）+ 最近 keep_last 條工具呼叫。
    確保不從 tool 訊息開頭切入（避免 API 格式錯誤）。"""
    if len(messages) <= 2:
        return messages
    fixed = messages[:2]   # [system, user+history]
    rest  = messages[2:]   # 工具呼叫來回
    if len(rest) <= keep_last:
        return messages
    trimmed = rest[-keep_last:]
    # 若切到一半的 tool 訊息，往後移到第一個 assistant
    while trimmed and trimmed[0].get("role") == "tool":
        trimmed = trimmed[1:]
    print(f"  [trim] 保留最近 {len(trimmed)} 條（共 {len(rest)} 條）")
    return fixed + trimmed


def run_agent(task: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    print(f"\n任務：{task}\n{'─' * 60}")
    final_text = ""
    _rate_limit_count = 0

    while True:
        try:
            response = client.chat.completions.create(
                model=_current_model(),
                messages=_trim_messages(messages, keep_last=20),
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            err = str(e)
            # 速率限制或額度耗盡
            if "429" in err or "rate_limit" in err.lower() or "quota" in err.lower() or "exceeded" in err.lower():
                if _rate_limit_count < 1:
                    _rate_limit_count += 1
                    print(f"\n[速率限制] 等待 65 秒後重試 {_current_model()}...")
                    time.sleep(65)
                    continue
                else:
                    _rate_limit_count = 0
                    print(f"\n[速率限制持續] 切換模型")
                    if not _next_model():
                        break
                    continue
            if "400" in err and ("tool_use_failed" in err or "model_decommissioned" in err):
                reason = "模型格式錯誤" if "tool_use_failed" in err else "模型已下架"
                print(f"\n[{reason}] 切換模型重試")
                if not _next_model():
                    break
                continue
            raise

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            final_text = msg.content or ""
            print(f"\n{'=' * 60}\nAgent 報告：\n{final_text}")
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"→ {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")
            result = execute_tool(fn_name, fn_args)
            print(f"  {json.dumps(result, ensure_ascii=False)}")
            compressed = _compress_result(fn_name, result)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      compressed,
            })

    return final_text


# ── 全域設定 ───────────────────────────────────────────────
from j1stools import parquet_db as _pdb

WATCH_STOCKS = _pdb.activate_stocks()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SIGNAL_LOG = os.path.join(RESULTS_DIR, "signal_log.jsonl")
SESSION_START = ""
SESSION_END = ""


def _append_signal_log(entry: dict):
    """把一筆 analyze_signal 結果寫入 JSONL log"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SIGNAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_signal_log(max_entries: int = 200) -> list[dict]:
    """讀取最近 N 筆 log"""
    if not os.path.exists(SIGNAL_LOG):
        return []
    with open(SIGNAL_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = []
    for line in lines[-max_entries:]:
        try:
            entries.append(json.loads(line.strip()))
        except Exception:
            pass
    return entries


def _shorten_cond(cond: str, max_len: int = 60) -> str:
    cond = cond.strip()
    return cond if len(cond) <= max_len else "..." + cond[-max_len:]


def _build_history_prompt(entries: list[dict]) -> str:
    """把 log 整理成 prompt 片段，告訴 agent 之前試了什麼"""
    if not entries:
        return ""

    passed = [e for e in entries if e.get("hit_rate", 0) >= 0.35]
    failed = [e for e in entries if e.get("hit_rate", 0) < 0.35]

    lines = ["\n━━ 歷史紀錄（避免重複，探索新方向）━━"]

    if passed:
        lines.append(f"\n✅ 通過({len(passed)}筆，可變種)：")
        for e in passed[-5:]:
            lines.append(f"  hit={e['hit_rate']:.3f} n={e.get('sample_count','?')} {_shorten_cond(e['condition'], 70)}")

    if failed:
        lines.append(f"\n❌ 未通過({len(failed)}筆，勿重複)：")
        seen = set()
        for e in failed[-15:]:
            cond = e["condition"]
            if cond not in seen:
                seen.add(cond)
                lines.append(f"  hit={e.get('hit_rate','?')} {_shorten_cond(cond, 50)}")

    return "\n".join(lines)


def daily_task():
    global SESSION_START, SESSION_END

    today = datetime.now()
    SESSION_END = today.strftime("%Y-%m-%d")
    SESSION_START = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    # 讀取歷史 log，讓 agent 知道之前試了什麼
    history = _load_signal_log()
    history_text = _build_history_prompt(history)

    task = (
        f"今天是 {SESSION_END}。\n"
        f"股票池：{len(WATCH_STOCKS)} 支台灣上市股\n"
        f"統計期間（1年）：{SESSION_START} 到 {SESSION_END}\n"
        "\n"
        "目標：找出 10 天內最高點能超過 +15% 的指標組合。\n"
        "請按照兩階段工作流程執行：\n"
        "第一步：analyze_signal(label_type='hit', hold_days=10, profit_target=0.15) 統計篩選，\n"
        "hit_rate >= 0.35 且 sample_count >= 30 的條件才通過。\n"
        "第二步：通過的條件執行 run_backtest(sl_stop=0.08, tp_stop=0.15)，驗證停損後實際勝率。\n"
        "第三步：回測通過後執行 quick_rfc(hold_days=10, profit_target=0.15) 訓練全期模型。\n"
        "產出完整報告。" + history_text
    )

    result = run_agent(task)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = os.path.join(RESULTS_DIR, f"signal_report_{today.strftime('%Y%m%d_%H%M')}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"生成時間：{today}\n期間：{SESSION_START}~{SESSION_END}\n\n{'=' * 60}\n\n{result}")
    print(f"\n報告已存：{fname}")


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("--once", "-o"):
        daily_task()
    else:
        RUN_TIME = "08:30"
        schedule.every().day.at(RUN_TIME).do(daily_task)
        print(f"Free Agent 已啟動，每天 {RUN_TIME} 自動探索指標組合")
        print(f"報告存至：{RESULTS_DIR}/")
        print("加 --once 或 -o 參數立刻執行\n")
        daily_task()
        while True:
            schedule.run_pending()
            time.sleep(30)
