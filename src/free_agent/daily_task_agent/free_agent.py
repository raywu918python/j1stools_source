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
from openai import OpenAI

load_dotenv()

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
_gemini_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
_ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
_deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)

# 額度用完自動輪換，格式：(model_name, client)
MODELS = [
    ("deepseek-v4-flash", _deepseek_client),  # 主力：便宜、1M context、穩定
    ("qwen2.5:14b", _ollama_client),  # 備用：本機無限制
    ("llama-3.3-70b-versatile", _groq_client),
    # ("llama-3.1-8b-instant", _groq_client),
]
_model_idx = 0


def select_model(model: str = "") -> str:
    """用名稱片段指定模型，例如 qwen / deepseek / llama。"""
    global _model_idx
    if not model:
        return _current_model()
    needle = model.lower()
    for i, (model_name, _) in enumerate(MODELS):
        if needle in model_name.lower():
            _model_idx = i
            return model_name
    raise ValueError(f"找不到模型：{model}，可用：{[m for m, _ in MODELS]}")


def _current_model() -> str:
    return MODELS[_model_idx][0]


def _current_client():
    return MODELS[_model_idx][1]


def _use_full_tool_results() -> bool:
    """本機模型不怕 token 成本，保留完整 tool result 讓探索資訊更完整。"""
    model = _current_model().lower()
    return "qwen" in model or _current_client() is _ollama_client


def _next_model() -> bool:
    global _model_idx
    if _model_idx < len(MODELS) - 1:
        _model_idx += 1
        print(f"\n[模型輪換] 切換至 {MODELS[_model_idx][0]}")
        return True
    print("\n[模型輪換] 所有模型額度已用完，今天停止")
    return False


# ── 工具定義（OpenAI 格式）─────────────────────────────────
# Llama 只負責探索統計，run_backtest/quick_rfc 由 validate_and_train() 純 Python 執行
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_signal",
            "description": "測試條件字串，回傳 hit_rate/sample_count。條件格式見 SYSTEM_PROMPT。",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "條件字串，用 & | ~，每個子句加括號"},
                    "hold_days": {"type": "integer", "description": "持有天數，預設 10"},
                    "label_type": {"type": "string", "description": "hit/return/max_return/atr_move，預設 hit"},
                    "profit_target": {"type": "number", "description": "hit 的目標報酬，預設 0.15"},
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
                    "hold_days": {"type": "integer", "description": "持有天數，預設 10"},
                    "profit_target": {"type": "number", "description": "目標報酬，預設 0.15"},
                    "top_n": {"type": "integer", "description": "回傳前幾名，預設 20"},
                },
                "required": [],
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
                    "pattern": {"type": "string", "description": "abcd 或 triangle"},
                    "hold_days": {"type": "integer", "description": "持有天數，預設 5"},
                    "label_type": {"type": "string", "description": "return/max_return/atr_move，預設 return"},
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

    tokens = re.split(r"(\s*[&|]\s*)", cond.strip())
    result = []
    for tok in tokens:
        s = tok.strip()
        if s in ("&", "|"):
            result.append(f" {s} ")
        elif s:
            result.append(s if s.startswith("(") else f"({s})")
    return "".join(result)


# ── 工具執行 ───────────────────────────────────────────────
def execute_tool(name: str, inputs: dict) -> dict:
    # 累計 tool 呼叫次數
    _tool_usage_path = os.path.join(RESULTS_DIR, "tool_usage.json")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        _usage = json.load(open(_tool_usage_path)) if os.path.exists(_tool_usage_path) else {}
        _usage[name] = _usage.get(name, 0) + 1
        with open(_tool_usage_path, "w", encoding="utf-8") as _f:
            json.dump(_usage, _f, ensure_ascii=False)
    except Exception:
        pass

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

        # 存完整相關係數供報表使用（每次覆蓋，保留最新一次）
        corr_path = os.path.join(RESULTS_DIR, "scan_correlations.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(corr_path, "w", encoding="utf-8") as _f:
            json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "correlations": corr_list}, _f, ensure_ascii=False, indent=2)

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

        hold_days = inputs.get("hold_days", 10)
        profit_target = inputs.get("profit_target", 0.15)
        n_stocks = inputs.get("n_stocks", 200)
        random_state = inputs.get("random_state", 42)
        model_suffix = inputs.get("model_suffix", "")
        model_label = inputs.get("model_label", "全歷史通過條件")

        rng = _random.Random(random_state)
        stocks = rng.sample(WATCH_STOCKS, min(n_stocks, len(WATCH_STOCKS)))

        try:
            feat_df = _build_rfc_features(stocks, "2015-01-01", SESSION_END)
        except Exception as e:
            return {"error": f"特徵建構失敗：{e}"}

        # target：N天最高點 >= profit_target → 1，否則 0
        close_wide = feat_df.pivot(index="date", columns="stock_id", values="close")
        future_max = pd.concat([close_wide.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
        target_wide = ((future_max / close_wide - 1) >= profit_target).astype(int)
        target_long = target_wide.stack().reset_index()
        target_long.columns = ["date", "stock_id", "target"]
        feat_df = feat_df.merge(target_long, on=["date", "stock_id"], how="left")

        # 把 agent 統計通過的條件加成 binary 特徵（按 hit_rate 排序取前20）
        signal_entries = inputs.get("signal_entries")
        raw_passing = signal_entries if signal_entries is not None else _load_signal_log()
        passing = []
        seen_conditions = set()
        for entry in sorted(
            [e for e in raw_passing if e.get("hit_rate", 0) >= 0.35 and "condition" in e],
            key=lambda x: x["hit_rate"],
            reverse=True,
        ):
            key = _canonical_condition(entry.get("condition", ""))
            if key and key not in seen_conditions:
                seen_conditions.add(key)
                passing.append(entry)
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

        tech_cols = [c for c in feat_df.columns if c.startswith("f_") and c not in added_signal_feats]
        all_cols  = tech_cols + added_signal_feats
        feat_df = feat_df.dropna(subset=tech_cols + ["target"])

        # 時間切分：前 80% 訓練，後 20% 測試
        dates = sorted(feat_df["date"].unique())
        cutoff = dates[int(len(dates) * 0.8)]
        train = feat_df[feat_df["date"] < cutoff]
        test  = feat_df[feat_df["date"] >= cutoff]

        if len(train) < 200 or len(test) < 100:
            return {"error": f"資料不足：train={len(train)}, test={len(test)}"}

        actual = test["target"].values
        base_rate = round(float(actual.mean()), 4)

        import joblib

        def _eval_rfc(feat_cols, model_path):
            rfc = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
            rfc.fit(train[feat_cols], train["target"])
            proba = rfc.predict_proba(test[feat_cols])[:, 1]
            thresholds = {}
            for th in [0.5, 0.6, 0.7, 0.8]:
                mask = proba >= th
                n = int(mask.sum())
                if n < 10:
                    thresholds[f"proba>={th}"] = {"n": n, "note": "樣本不足"}
                else:
                    thresholds[f"proba>={th}"] = {"n": n, "hit_rate": round(float(actual[mask].mean()), 4)}
            importance = sorted(zip(feat_cols, rfc.feature_importances_), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": f, "importance": round(float(v), 4)} for f, v in importance[:10]]
            joblib.dump(rfc, model_path)
            return thresholds, top_features

        safe_suffix = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(model_suffix)).strip("_")
        suffix = f"_{safe_suffix}" if safe_suffix else ""

        # 模型A：只用技術指標
        model_a_path = os.path.join(os.path.dirname(__file__), f"quick_rfc_tech{suffix}.joblib")
        th_a, feat_a = _eval_rfc(tech_cols, model_a_path)

        # 模型B：技術指標 + top20 f_signal
        model_b_path = os.path.join(os.path.dirname(__file__), f"quick_rfc{suffix}.joblib")
        th_b, feat_b = _eval_rfc(all_cols, model_b_path)

        return {
            "model_label": model_label,
            "hold_days": hold_days,
            "profit_target": profit_target,
            "train_size": len(train),
            "test_size": len(test),
            "base_rate": base_rate,
            "model_a": {"name": "技術指標", "thresholds": th_a, "top_features": feat_a, "model_saved": model_a_path},
            "model_b": {"name": f"技術指標 + top{len(added_signal_feats)} f_signal", "thresholds": th_b, "top_features": feat_b, "model_saved": model_b_path},
            "signal_condition_count": len(passing),
            "signal_feats_added": len(added_signal_feats),
        }

    if name == "run_backtest":
        from j1stools.backtest_engine import backtest_engine

        if "condition" not in inputs:
            return {"error": "缺少必填參數 condition，請提供條件字串後再呼叫 run_backtest"}
        condition = _normalize_condition(inputs["condition"])
        hold_days = inputs.get("hold_days", 10)
        sl_stop = inputs.get("sl_stop", 0.08)
        tp_stop = inputs.get("tp_stop", 0.15)

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

        entries = entries.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        exits = pd.Series(False, index=close.index)
        df_proba = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        stock_group = {s: "default" for s in close.columns}

        try:
            portfolio_value, trades_df, _ = backtest_engine(
                close=close,
                entries=entries,
                exits=exits,
                df_proba=df_proba,
                stock_group=stock_group,
                hold_days=hold_days,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                use_fixed_sl=True,
                use_fixed_tp=True,
                use_sl_trail=False,
                use_hold_days=True,
            )
        except Exception as e:
            return {"error": f"回測失敗：{e}"}

        if len(trades_df) < 5:
            return {"error": f"交易筆數不足（{len(trades_df)} 筆）"}

        pv = portfolio_value.dropna()
        rets = pv.pct_change(fill_method=None).dropna()
        total_return = round(float(pv.iloc[-1] / pv.iloc[0] - 1), 4) if len(pv) > 1 else None
        sharpe = round(float(rets.mean() / rets.std() * (252**0.5)) if rets.std() > 0 else 0, 2)
        max_dd = round(float(((pv / pv.cummax()) - 1).min()), 4)

        # trades_df 欄位：return_pct（百分比），pnl（絕對值）
        if "return_pct" in trades_df.columns:
            rp = trades_df["return_pct"]
            win_rate = round(float((rp > 0).mean()), 4)
            avg_return = round(float(rp.mean()), 4)
            avg_win = round(float(rp[rp > 0].mean()), 4) if (rp > 0).any() else 0
            avg_loss = round(float(rp[rp < 0].mean()), 4) if (rp < 0).any() else 0
        else:
            win_rate = avg_return = avg_win = avg_loss = None

        # 存 HTML 圖表
        chart_path = None
        try:
            from j1stools.j1s_chart import plot_performance
            import plotly.graph_objects as go, json as _json

            pv_clean = portfolio_value.replace([np.inf, -np.inf], np.nan).dropna()
            fig_json = plot_performance(pv_clean, trades_df, is_web=True)
            fig = go.Figure(_json.loads(fig_json))
            os.makedirs(RESULTS_DIR, exist_ok=True)
            slug = (
                condition[:60]
                .replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace(">", "gt")
                .replace("<", "lt")
                .replace("&", "_")
            )
            chart_path = os.path.join(RESULTS_DIR, f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.html")
            fig.write_html(chart_path)
            print(f"  圖表已存：{chart_path}")
        except Exception as ce:
            print(f"  圖表存檔失敗：{ce}")

        return {
            "condition": condition,
            "hold_days": hold_days,
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
            "total_trades": len(trades_df),
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "chart": chart_path,
        }

    if name == "analyze_signal":
        hold_days = inputs.get("hold_days", 5)
        if "condition" not in inputs:
            return {"error": "缺少必填參數 condition，請提供條件字串後再呼叫 analyze_signal"}
        condition = _normalize_condition(inputs["condition"])
        label_type = inputs.get("label_type", "hit")
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
SYSTEM_PROMPT = """你是台灣股票量化研究助理，專門探索「10天最高點超過+15%」的有效指標組合。

━━ 可用指標（condition 用 & | ~，每個子句加括號）━━
報價: close,high,low,open,volume
均線: ma5,ma10,ma20,ma60,ma120；交叉: ma5_x_ma10,ma5_x_ma20,ma10_x_ma60
動能: rsi,macd,macdh,bb_upper,bb_lower,bb_mid,bb_width
趨勢: adx(>25有趨勢),plus_di,minus_di
波動: atr,atr_rank(0~1百分位)
量能: vol_ratio(今量/20均),vol_trend,obv,obv_ma
ABCD: f_n_confirmed,f_n_structure_score,f_n_ab_gain,f_n_d_breakout_strength,f_n_volume_confirm
IB: net_foreign,net_trust,net_dealer,net_inst,foreign_rank,trust_rank,dealer_rank,inst_rank
Margin: margin_bal,short_bal,margin_chg,short_chg,short_ratio,margin_chg_rank,short_ratio_rank

━━ 工作流程 ━━
1. 先呼叫 scan_correlations(hold_days=10,profit_target=0.15)，記住 top 相關指標
2. 用高相關指標組合條件，呼叫 analyze_signal(label_type="hit",hold_days=10,profit_target=0.15)
3. 篩選標準：hit_rate>=0.35 且 sample_count>=30
4. 不通過→換方向，不重複已試過的組合
5. 每 5 次 analyze_signal 測試中，至少 2 次必須包含 scan_correlations 前三名以外的指標
   （也就是條件不能只由前三名指標組成，需加入第4名以後或其他類別指標）
6. 找到越多通過條件越好，盡量多試

━━ 關鍵技巧 ━━
- sample_count 太大（>5000）通常 hit_rate 低，試更嚴格的門檻來縮小樣本
- 例如：f_n_structure_score > 0.1 → 試 > 0.3、> 0.5、> 0.7
- 越嚴格的條件（sample_count 500~3000）反而 hit_rate 更高
- 相關係數只是起點，門檻要自己往上調整才能找到真正有效的條件
- 不要長時間只使用前三名相關指標；要輪流加入趨勢、量能、籌碼、融資券等非前三名指標探索

回測與模型訓練由外部程式處理，你只負責找條件。
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
        ths = result.get("thresholds", {})
        th_str = " | ".join(f"{k}: n={v['n']} hit={v.get('hit_rate','?')}" for k, v in ths.items() if "hit_rate" in v)
        feats = [f["feature"] for f in result.get("top_features", [])[:3]]
        return f"base={base} | {th_str} | top:{','.join(feats)} signal_added={result.get('signal_feats_added')}"

    if fn_name == "scan_correlations":
        top = result.get("top_correlations", [])
        top3 = top[:3]
        others = top[3:12]
        top3_items = " ".join(f"{r['feature']}({r['correlation']})" for r in top3)
        other_items = " ".join(f"{r['feature']}({r['correlation']})" for r in others)
        return f"前三名: {top3_items} | 非前三候選: {other_items}"

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
    fixed = messages[:2]  # [system, user+history]
    rest = messages[2:]  # 工具呼叫來回
    if len(rest) <= keep_last:
        return messages
    trimmed = rest[-keep_last:]
    # 若切到一半的 tool 訊息，往後移到第一個 assistant
    while trimmed:
        msg = trimmed[0]
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "tool":
            trimmed = trimmed[1:]
        else:
            break
    print(f"  [trim] 保留最近 {len(trimmed)} 條（共 {len(rest)} 條）")
    return fixed + trimmed


def run_agent(task: str, max_signal_calls: int = 0) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    print(f"\n任務：{task}\n{'─' * 60}")
    final_text = ""
    _rate_limit_count = 0
    _signal_call_count = 0

    while True:
        try:
            # Gemini 免費版 15 RPM，每次請求前等 5 秒
            if "gemini" in _current_model():
                time.sleep(5)
            response = _current_client().chat.completions.create(
                model=_current_model(),
                messages=_trim_messages(messages, keep_last=20),
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            err = str(e)
            print(f"\n[API錯誤] {_current_model()}: {err[:200]}")
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
            tool_content = (
                json.dumps(result, ensure_ascii=False)
                if _use_full_tool_results()
                else _compress_result(fn_name, result)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                }
            )
            if fn_name == "analyze_signal":
                _signal_call_count += 1
                if max_signal_calls and _signal_call_count >= max_signal_calls:
                    final_text = f"[測試停止] 已完成 {max_signal_calls} 次 analyze_signal。"
                    print(f"\n{final_text}")
                    return final_text

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

    hit_entries = [e for e in entries if "hit_rate" in e]
    passed = [e for e in hit_entries if e.get("hit_rate", 0) >= 0.35]
    failed = [e for e in hit_entries if e.get("hit_rate", 0) < 0.35]

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


BACKTEST_LOG = os.path.join(RESULTS_DIR, "backtest_log.jsonl")


def _canonical_condition(cond: str) -> str:
    """排序 & 分隔的子句，讓順序不同的同義條件視為同一個"""
    parts = [p.strip() for p in cond.split("&")]
    return " & ".join(sorted(parts))


def _load_backtested_conditions() -> set:
    if not os.path.exists(BACKTEST_LOG):
        return set()
    done = set()
    with open(BACKTEST_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                cond = json.loads(line.strip()).get("condition", "")
                done.add(_canonical_condition(cond))
            except Exception:
                pass
    return done


def validate_and_train():
    """
    Agent 統計篩選後的驗證與訓練流程（不用 LLM）

    流程：
        Agent 統計篩選 (hit_rate >= 0.35, n >= 100)
            ↓
        1. 回測驗證：用歷史價格模擬真實進出場（停損8%、停利15%、持10天）
                    確認統計上有效的條件，在真實交易情境下是否真的賺錢
            ↓
        2. RFC 訓練：把通過的條件轉成 binary 特徵（0/1），
                    加上技術指標，訓練模型學「哪些特徵組合未來10天有機會漲15%」
            ↓
        3. 產報表：回測結果整理成 report.html
    """
    passing = [
        e for e in _load_signal_log()
        if e.get("hit_rate", 0) >= 0.35 and e.get("sample_count", 0) >= 100
    ]
    if not passing:
        print("沒有通過的條件，跳過驗證")
        return

    done = _load_backtested_conditions()
    seen_canonical = set()
    new_passing = []
    for e in passing:
        key = _canonical_condition(e.get("condition", ""))
        if key not in done and key not in seen_canonical:
            seen_canonical.add(key)
            new_passing.append(e)

    if not new_passing:
        print(f"[驗證] 所有 {len(passing)} 個通過條件已回測，跳過")
        return

    print(f"\n[驗證] 開始回測 {len(new_passing)} 個新條件...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for entry in new_passing:
        cond = entry["condition"]
        print(f"\n→ 回測：{cond}")
        bt = execute_tool(
            "run_backtest",
            {
                "condition": cond,
                "hold_days": 10,
                "sl_stop": 0.08,
                "tp_stop": 0.15,
            },
        )
        print(f"  {json.dumps(bt, ensure_ascii=False)}")
        with open(BACKTEST_LOG, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "condition": cond,
                        "hit_rate": entry.get("hit_rate"),
                        "sample_count": entry.get("sample_count"),
                        "backtest": bt,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n[驗證] 訓練 RFC（全歷史通過條件）...")
    rfc_all = execute_tool(
        "quick_rfc",
        {
            "hold_days": 10,
            "profit_target": 0.15,
            "model_suffix": "all",
            "model_label": "全歷史通過條件",
        },
    )
    print(f"  {json.dumps(rfc_all, ensure_ascii=False)}")

    print("\n[驗證] 訓練 RFC（本次新增條件）...")
    rfc_latest = execute_tool(
        "quick_rfc",
        {
            "hold_days": 10,
            "profit_target": 0.15,
            "signal_entries": new_passing,
            "model_suffix": "latest",
            "model_label": "本次新增條件",
        },
    )
    print(f"  {json.dumps(rfc_latest, ensure_ascii=False)}")

    # 存 RFC 結果供報表使用
    rfc_log = os.path.join(RESULTS_DIR, "rfc_result.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(rfc_log, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "runs": {
                    "all": rfc_all,
                    "latest": rfc_latest,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _generate_report()


def _generate_report():
    """讀 backtest_log + rfc_result.json，產 HTML 報表到 results/report.html"""
    import re as _re

    # 所有 Agent 可用 tool 定義（固定清單，讓 0 次也能顯示）
    _ALL_TOOLS = {
        "scan_correlations":    "掃描指標與未來報酬的相關係數，作為選指標的依據",
        "analyze_signal":       "測試任意條件的統計勝率（hit_rate）與樣本數",
        "analyze_pattern_signal":"測試型態訊號（ABCD、三角收斂等）的勝率",
    }

    # 讀 tool 呼叫次數
    tool_usage: dict = {}
    tool_usage_path = os.path.join(RESULTS_DIR, "tool_usage.json")
    if os.path.exists(tool_usage_path):
        with open(tool_usage_path, encoding="utf-8") as f:
            tool_usage = json.load(f)

    # 讀 RFC 結果
    rfc_data = {}
    rfc_log = os.path.join(RESULTS_DIR, "rfc_result.json")
    if os.path.exists(rfc_log):
        with open(rfc_log, encoding="utf-8") as f:
            rfc_data = json.load(f)

    # 讀 scan_correlations 結果
    corr_map: dict = {}
    corr_date = ""
    corr_path = os.path.join(RESULTS_DIR, "scan_correlations.json")
    if os.path.exists(corr_path):
        with open(corr_path, encoding="utf-8") as f:
            corr_raw = json.load(f)
            corr_date = corr_raw.get("date", "")
            for item in corr_raw.get("correlations", []):
                corr_map[item["feature"]] = item["correlation"]

    rows = []
    if not os.path.exists(BACKTEST_LOG):
        return
    with open(BACKTEST_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            bt = e.get("backtest", {})
            if "error" in bt or bt.get("total_trades", 0) < 50:
                continue
            rows.append({
                "condition": e["condition"],
                "hit_rate": e.get("hit_rate", 0),
                "sample_count": e.get("sample_count", 0),
                "trades": bt.get("total_trades"),
                "total_return": bt.get("total_return", 0),
                "sharpe": bt.get("sharpe", 0),
                "max_drawdown": bt.get("max_drawdown", 0),
                "win_rate": bt.get("win_rate", 0),
                "avg_win": bt.get("avg_win", 0),
                "avg_loss": bt.get("avg_loss", 0),
                "chart": bt.get("chart"),
            })

    if not rows:
        print("[報表] 無有效回測資料，跳過")
        return

    rows.sort(key=lambda x: x["sharpe"], reverse=True)

    _ALL_AGENT_INDICATORS = [
        "close", "high", "low", "open", "volume",
        "ma5", "ma10", "ma20", "ma60", "ma120",
        "ma5_x_ma10", "ma5_x_ma20", "ma10_x_ma60",
        "rsi", "macd", "macdh", "bb_upper", "bb_lower", "bb_mid", "bb_width",
        "adx", "plus_di", "minus_di",
        "atr", "atr_rank",
        "vol_ratio", "vol_trend", "obv", "obv_ma",
        "f_n_confirmed", "f_n_structure_score", "f_n_ab_gain",
        "f_n_d_breakout_strength", "f_n_volume_confirm",
        "net_foreign", "net_trust", "net_dealer", "net_inst",
        "foreign_rank", "trust_rank", "dealer_rank", "inst_rank",
        "margin_bal", "short_bal", "margin_chg", "short_chg",
        "short_ratio", "margin_chg_rank", "short_ratio_rank",
    ]
    _IND_ORDER = {name: i for i, name in enumerate(_ALL_AGENT_INDICATORS)}

    # 指標使用頻率：先列出所有可用指標，沒被用到也顯示 0
    all_ind: dict = {name: 0 for name in _ALL_AGENT_INDICATORS}
    for r in rows:
        cond = r["condition"]
        tokens = set(_re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cond))
        for t in tokens:
            if t in all_ind:
                all_ind[t] += 1
    all_ind_sorted = sorted(all_ind.items(), key=lambda x: (-x[1], _IND_ORDER.get(x[0], 9999)))

    sharpes = [r["sharpe"] for r in rows]
    wrs = [r["win_rate"] for r in rows]
    rets = [r["total_return"] for r in rows]

    def _sc(s):
        if s >= 2.0: return "#1a7a3a"
        if s >= 1.5: return "#2ecc71"
        if s >= 1.0: return "#f39c12"
        if s >= 0:   return "#e67e22"
        return "#e74c3c"

    rows_html = ""
    for i, r in enumerate(rows):
        chart_link = f'<a href="{r["chart"]}" target="_blank">📊</a>' if r.get("chart") else ""
        dd_color = "#e74c3c" if r["max_drawdown"] < -0.15 else "#e67e22" if r["max_drawdown"] < -0.10 else "#2ecc71"
        cond = r["condition"]
        n_ind = len(set(_re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cond)) & set(_ALL_AGENT_INDICATORS))
        rows_html += (
            f'<tr>'
            f'<td style="text-align:center">{i+1}</td>'
            f'<td style="font-family:monospace;font-size:12px;white-space:nowrap">{cond}</td>'
            f'<td style="text-align:center">{n_ind}</td>'
            f'<td style="text-align:center;color:{_sc(r["sharpe"])};font-weight:bold">{r["sharpe"]:.2f}</td>'
            f'<td style="text-align:center">{r["win_rate"]:.1%}</td>'
            f'<td style="text-align:center">{r["total_return"]:+.2f}</td>'
            f'<td style="text-align:center;color:{dd_color}">{r["max_drawdown"]:.1%}</td>'
            f'<td style="text-align:center">{r["trades"]}</td>'
            f'<td style="text-align:center">{r["hit_rate"]:.1%}</td>'
            f'<td style="text-align:center">{r["sample_count"]}</td>'
            f'<td style="text-align:center">{chart_link}</td>'
            f'</tr>'
        )

    _IND_DESC = {
        "adx":                    "趨勢強度（>25 代表趨勢明確）",
        "atr":                    "平均真實波幅（ATR）",
        "atr_rank":               "波動率排名（ATR 在歷史中的百分位）",
        "bb_lower":               "布林通道下緣",
        "bb_mid":                 "布林通道中線",
        "bb_upper":               "布林通道上緣",
        "bb_width":               "布林通道寬度（越大代表波動越劇烈）",
        "close":                  "收盤價",
        "dealer_rank":            "自營商買超排名",
        "f_n_ab_gain":            "ABCD型態 AB段漲幅",
        "f_n_confirmed":          "ABCD型態確認度",
        "f_n_d_breakout_strength":"ABCD型態 D點突破強度",
        "f_n_structure_score":    "ABCD型態結構分數",
        "f_n_volume_confirm":     "ABCD型態成交量確認",
        "foreign_rank":           "外資買超排名",
        "high":                   "最高價",
        "inst_rank":              "三大法人合計買超排名",
        "low":                    "最低價",
        "ma5":                    "5日均線",
        "ma10":                   "10日均線",
        "ma20":                   "20日均線",
        "ma60":                   "60日均線",
        "ma120":                  "120日均線",
        "ma5_x_ma20":             "5日均線向上穿越20日均線",
        "ma10_x_ma60":            "10日均線向上穿越60日均線",
        "macd":                   "MACD線（快線-慢線）",
        "macdh":                  "MACD柱（多頭時 > 0）",
        "margin_bal":             "融資餘額",
        "margin_chg":             "融資增減量",
        "margin_chg_rank":        "融資增減排名（越高代表散戶加碼越積極）",
        "minus_di":               "DMI負向指標（下跌力道）",
        "net_dealer":             "自營商買賣超（正值買超）",
        "net_foreign":            "外資買賣超（正值買超）",
        "net_inst":               "三大法人合計買賣超",
        "net_trust":              "投信買賣超（正值買超）",
        "obv":                    "能量潮（累計成交量）",
        "open":                   "開盤價",
        "plus_di":                "DMI正向指標（上漲力道）",
        "rsi":                    "相對強弱指數（0~1，>0.5 偏強勢）",
        "short_bal":              "融券餘額（正值有融券）",
        "short_chg":              "融券增減量",
        "short_ratio":            "融券比率（融券/融資）",
        "short_ratio_rank":       "融券比率排名",
        "trust_rank":             "投信買超排名",
        "volume":                 "成交量",
        "vol_ratio":              "今日成交量 / 近期均量",
        "vol_trend":              "成交量趨勢（正值量增）",
        "ma5_x_ma10":             "5日均線向上穿越10日均線（黃金交叉）",
        "plus_di_gt_minus_di":    "上漲力道 > 下跌力道",
        "obv_ma":                 "OBV均線（能量潮均值）",
    }

    max_cnt = all_ind_sorted[0][1] if all_ind_sorted else 1
    ind_rows = ""
    for ind, cnt in all_ind_sorted:
        pct = cnt / max_cnt * 100
        corr = corr_map.get(ind)
        corr_str = f'{corr:+.4f}' if corr is not None else "—"
        corr_color = "#1a7a3a" if corr and corr > 0.1 else "#e74c3c" if corr and corr < -0.1 else "#888"
        desc = _IND_DESC.get(ind, "")
        ind_rows += (
            f'<tr>'
            f'<td style="font-family:monospace">{ind}</td>'
            f'<td style="font-size:12px;color:#666">{desc}</td>'
            f'<td style="text-align:center">{cnt}</td>'
            f'<td><div style="background:#3498db;height:14px;width:{pct:.0f}%;border-radius:3px;min-width:4px"></div></td>'
            f'<td style="text-align:center;color:{corr_color};font-weight:{"bold" if corr and abs(corr)>0.1 else "normal"}">{corr_str}</td>'
            f'</tr>'
        )

    # Agent 工具使用區塊
    total_calls = sum(tool_usage.get(t, 0) for t in _ALL_TOOLS)
    tool_rows = ""
    for tool_name, tool_desc in _ALL_TOOLS.items():
        cnt = tool_usage.get(tool_name, 0)
        pct = cnt / max(total_calls, 1) * 100
        bar_color = "#2ecc71" if cnt > 0 else "#ddd"
        cnt_color = "#2c3e50" if cnt > 0 else "#bbb"
        tool_rows += (
            f'<tr>'
            f'<td style="font-family:monospace">{tool_name}</td>'
            f'<td style="font-size:12px;color:#666">{tool_desc}</td>'
            f'<td style="text-align:center;color:{cnt_color};font-weight:{"bold" if cnt>0 else "normal"}">{cnt}</td>'
            f'<td><div style="background:{bar_color};height:14px;width:{max(pct,2):.0f}%;border-radius:3px"></div></td>'
            f'</tr>'
        )
    tool_section = f"""
  <div class="sec">
    <h2>Agent 可用工具清單（累計呼叫次數）</h2>
    <p style="font-size:12px;color:#888;margin:0 0 10px">以下是 Agent 可以使用的所有工具。呼叫次數 = 歷史累計實際使用次數，0 次代表 Agent 從未呼叫過（工具存在但未被使用）</p>
    <table style="width:auto;min-width:500px">
      <thead><tr><th>工具名稱</th><th>說明</th><th style="text-align:center">呼叫次數</th><th style="min-width:200px">佔比</th></tr></thead>
      <tbody>{tool_rows}</tbody>
    </table>
  </div>"""

    # RFC 區塊 HTML
    def _fmt_int(value) -> str:
        try:
            return f"{int(value):,}"
        except Exception:
            return str(value)

    def _th_rows(thresholds):
        html = ""
        for th, v in thresholds.items():
            if "note" in v:
                html += f'<tr><td>{th}</td><td style="color:#aaa">{v["n"]} 筆（樣本不足）</td><td>—</td></tr>'
            else:
                hit = v["hit_rate"]
                color = "#1a7a3a" if hit >= 0.4 else "#f39c12" if hit >= 0.35 else "#e74c3c"
                html += f'<tr><td>{th}</td><td>{v["n"]} 筆</td><td style="color:{color};font-weight:bold">{hit:.1%}</td></tr>'
        return html

    def _feat_bars(top_features, color):
        max_imp = top_features[0]["importance"] if top_features else 1
        html = ""
        for f in top_features:
            pct = f["importance"] / max_imp * 100
            html += (
                f'<div style="margin:3px 0;display:flex;align-items:center;gap:6px">'
                f'<div style="width:140px;font-family:monospace;font-size:11px;text-align:right">{f["feature"]}</div>'
                f'<div style="background:{color};height:14px;width:{pct:.0f}%;border-radius:3px"></div>'
                f'<div style="font-size:11px;color:#666">{f["importance"]:.4f}</div>'
                f'</div>'
            )
        return html

    def _render_rfc_section(data: dict, title: str, rfc_date: str = "") -> str:
        if not data or "error" in data:
            err = data.get("error", "無資料") if isinstance(data, dict) else "無資料"
            return f"""
  <div class="sec">
    <h2>{title}</h2>
    <p style="font-size:13px;color:#e74c3c;margin:0">{err}</p>
  </div>"""

        base_rate = data.get("base_rate", 0)
        label = data.get("model_label", title)
        signal_count = data.get("signal_condition_count", data.get("signal_feats_added", 0))

        # 相容舊格式（只有 thresholds/top_features）和新格式（model_a/model_b）
        if "model_a" in data and "model_b" in data:
            ma, mb = data["model_a"], data["model_b"]
            col_a = _th_rows(ma.get("thresholds", {}))
            col_b = _th_rows(mb.get("thresholds", {}))
            bar_a = _feat_bars(ma.get("top_features", []), "#3498db")
            bar_b = _feat_bars(mb.get("top_features", []), "#9b59b6")
            model_tables = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:12px">
      <div>
        <p style="font-size:13px;font-weight:600;margin:0 0 8px">A｜{ma.get('name','技術指標')}</p>
        <table style="width:auto"><thead><tr><th>門檻</th><th>篩出</th><th>命中率</th></tr></thead>
        <tbody>{col_a}</tbody></table>
        <p style="font-size:12px;color:#555;margin:10px 0 6px">特徵重要性 Top 10</p>
        {bar_a}
      </div>
      <div>
        <p style="font-size:13px;font-weight:600;margin:0 0 8px">B｜{mb.get('name','技術指標 + f_signal')}</p>
        <table style="width:auto"><thead><tr><th>門檻</th><th>篩出</th><th>命中率</th></tr></thead>
        <tbody>{col_b}</tbody></table>
        <p style="font-size:12px;color:#555;margin:10px 0 6px">特徵重要性 Top 10</p>
        {bar_b}
      </div>
    </div>"""
        else:
            thresholds = data.get("thresholds", {})
            top_features = data.get("top_features", [])
            model_tables = f"""
    <table style="width:auto;margin-top:12px">
      <thead><tr><th>門檻</th><th>篩出</th><th>命中率</th></tr></thead>
      <tbody>{_th_rows(thresholds)}</tbody>
    </table>
    <p style="font-size:12px;color:#555;margin:10px 0 6px">特徵重要性 Top 10</p>
    {_feat_bars(top_features, "#9b59b6")}"""

        return f"""
  <div class="sec">
    <h2>{title}{f"（訓練時間：{rfc_date}）" if rfc_date else ""}</h2>
    <p style="font-size:13px;color:#555;margin:0">
      訊號範圍：<strong>{label}</strong>｜
      候選條件：{signal_count} 筆｜
      訓練集：{_fmt_int(data.get('train_size','?'))} 筆｜
      測試集：{_fmt_int(data.get('test_size','?'))} 筆｜
      自然命中率：<strong>{base_rate:.1%}</strong>｜
      命中 = 進場後10天內最高點 ≥ +15%
    </p>
    {model_tables}
  </div>"""

    rfc_section = ""
    if rfc_data:
        rfc_date = rfc_data.get("date", "")
        if "runs" in rfc_data:
            runs = rfc_data.get("runs", {})
            if "latest" in runs:
                rfc_section += _render_rfc_section(runs["latest"], "RFC 模型比較｜本次新增條件", rfc_date)
            if "all" in runs:
                rfc_section += _render_rfc_section(runs["all"], "RFC 模型比較｜全歷史條件", rfc_date)
        else:
            rfc_section = _render_rfc_section(rfc_data, "RFC 模型比較", rfc_date)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>Free Agent 回測報表 {now_str[:10]}</title>
<style>
  body{{font-family:-apple-system,Arial,sans-serif;margin:0;background:#f5f6fa;color:#2c3e50}}
  .hd{{background:linear-gradient(135deg,#2c3e50,#3498db);color:white;padding:28px 40px}}
  .hd h1{{margin:0 0 6px;font-size:24px}}.hd p{{margin:0;opacity:.8;font-size:14px}}
  .ct{{max-width:1400px;margin:24px auto;padding:0 24px}}
  .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
  .card{{background:white;border-radius:10px;padding:18px 22px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  .card .lb{{font-size:12px;color:#888;margin-bottom:6px}}
  .card .vl{{font-size:28px;font-weight:700;color:#2c3e50}}
  .card .sb{{font-size:12px;color:#aaa;margin-top:4px}}
  .sec{{background:white;border-radius:10px;padding:20px 24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  .sec h2{{margin:0 0 16px;font-size:16px;border-left:4px solid #3498db;padding-left:10px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#f8f9fa;padding:10px 8px;text-align:left;border-bottom:2px solid #e0e0e0;white-space:nowrap}}
  td{{padding:8px;border-bottom:1px solid #f0f0f0}}
  tr:hover td{{background:#fafbff}}
</style>
</head>
<body>
<div class="hd">
  <h1>Free Agent 回測報表</h1>
  <p>統計期間：{SESSION_START} ~ {SESSION_END}（3年）｜股票池：{len(WATCH_STOCKS)}支｜生成：{now_str}</p>
</div>
<div class="ct">
  <div class="cards">
    <div class="card"><div class="lb">有效條件數</div><div class="vl">{len(rows)}</div><div class="sb">trades ≥ 50 筆</div></div>
    <div class="card"><div class="lb">Sharpe ≥ 1.5</div><div class="vl" style="color:#1a7a3a">{sum(1 for r in rows if r["sharpe"]>=1.5)}</div><div class="sb">最高 {max(sharpes):.2f}</div></div>
    <div class="card"><div class="lb">平均勝率</div><div class="vl">{sum(wrs)/len(wrs):.1%}</div><div class="sb">最高 {max(wrs):.1%}</div></div>
    <div class="card"><div class="lb">平均報酬（回測3年）</div><div class="vl" style="color:#2980b9">{sum(rets)/len(rets):+.2f}</div><div class="sb">最高 {max(rets):+.2f}</div></div>
  </div>
  <div class="sec">
    <h2>指標使用統計（共 {len(all_ind_sorted)} 個）{f"｜相關係數來源：{corr_date}" if corr_date else ""}</h2>
    <p style="font-size:12px;color:#888;margin:0 0 12px">相關係數 = Agent 呼叫 scan_correlations 時與未來漲跌的線性相關（正值＝正相關）</p>
    <div style="overflow-x:auto">
    <table style="width:auto;min-width:500px">
      <thead><tr><th>指標名稱</th><th>說明</th><th style="text-align:center">出現次數</th><th style="min-width:200px">頻率</th><th style="text-align:center">相關係數</th></tr></thead>
      <tbody>{ind_rows}</tbody>
    </table>
    </div>
  </div>
  {tool_section}
  {rfc_section}
  <div class="sec">
    <h2>條件回測結果（{len(rows)} 筆，依 Sharpe 排序）</h2>
    <p style="font-size:12px;color:#888;margin:0 0 12px">持有10天｜停損8%｜停利15%</p>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>#</th><th>進場條件</th><th>指標數</th><th>Sharpe</th><th>勝率</th><th>報酬</th>
        <th>最大回撤</th><th>交易筆數</th><th>統計勝率</th><th>統計樣本</th><th>圖表</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
  </div>
</div>
</body>
</html>"""

    report_path = os.path.join(RESULTS_DIR, "report.html")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[報表] 已存：{report_path}（{len(rows)} 筆條件）")


def daily_task(model: str = "", max_signal_calls: int = 0, validate: bool = True):
    global SESSION_START, SESSION_END
    if model:
        selected = select_model(model)
        print(f"[model] {selected}")

    today = datetime.now()
    SESSION_END = today.strftime("%Y-%m-%d")
    SESSION_START = (today - timedelta(days=1095)).strftime("%Y-%m-%d")

    # Agent 探索新條件（只用 scan_correlations + analyze_signal）
    history = _load_signal_log()
    history_text = _build_history_prompt(history)

    task = (
        f"今天是 {SESSION_END}，統計期間：{SESSION_START} 到 {SESSION_END}，股票池 {len(WATCH_STOCKS)} 支。\n"
        "目標：找出更多 hit_rate >= 0.35 且 sample_count >= 30 的新條件。\n"
        "第一步呼叫 scan_correlations，再用高相關指標與非前三名候選指標組合 analyze_signal 測試。\n"
        "每 5 次 analyze_signal 至少 2 次要加入 scan_correlations 前三名以外的指標。\n"
        "回測與模型訓練由外部處理，你只需找到盡量多的通過條件。" + history_text
    )

    result = run_agent(task, max_signal_calls=max_signal_calls)

    # Agent 結束後，再驗證本次新找到的條件
    if validate:
        validate_and_train()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = os.path.join(RESULTS_DIR, f"signal_report_{today.strftime('%Y%m%d_%H%M')}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"生成時間：{today}\n期間：{SESSION_START}~{SESSION_END}\n\n{'=' * 60}\n\n{result}")
    print(f"\n報告已存：{fname}")


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("--schedule", "-s"):
        RUN_TIME = "08:30"
        model = sys.argv[2] if len(sys.argv) > 2 else ""
        max_signal_calls = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        schedule.every().day.at(RUN_TIME).do(daily_task, model=model, max_signal_calls=max_signal_calls)
        print(f"Free Agent 已啟動，每天 {RUN_TIME} 自動探索指標組合")
        print(f"報告存至：{RESULTS_DIR}/")
        print("Ctrl+C 停止\n")
        daily_task(model=model, max_signal_calls=max_signal_calls)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        model = sys.argv[1] if len(sys.argv) > 1 else ""
        max_signal_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        daily_task(model=model, max_signal_calls=max_signal_calls, validate=max_signal_calls == 0)
