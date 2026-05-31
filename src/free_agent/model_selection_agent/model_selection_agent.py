"""
model_selection_agent.py — Target-driven feature discovery Agent

Design:
  Each session focuses on a (hold_days, profit_target, sl_stop) target.
  Agent freely explores all indicators to find features that predict the target.
  -> validate_and_train() trains target-specific RandomForest models.
  -> Output: multiple small, precise feature sets per target.

Time split:
  Training: 2015-01-01 ~ 2023-12-31
  Validation: 2024-01-01 ~ today
"""

import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
_ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
_deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)
_gemini_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODELS = [
    # ("qwen3:14b", _ollama_client),
    ("qwen3-coder:latest", _ollama_client),
    ("llama-3.3-70b-versatile", _groq_client),
    ("deepseek-v4-flash", _deepseek_client),
    (os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), _gemini_client),
]
_model_idx = 0

# ── 目前目標參數（session 全域） ──────────────────────────────────
_current_target: dict = {"task": "return", "hold_days": 10, "threshold": 0.15, "sl_stop": 0.08}

# task 說明
TASK_DEFS = {
    "return":        "未來 {hold_days} 天內最高漲幅 >= {threshold:.0%}",
    "breakout":      "未來 {hold_days} 天突破近 {lookback} 天高點",
    "consolidation": "未來 {hold_days} 天振幅 < {threshold:.0%}（盤整）",
    "trend_up":      "持有 {hold_days} 天後漲幅 >= {threshold:.0%}（期末報酬）",
    "trend_down":    "持有 {hold_days} 天後跌幅 >= {threshold:.0%}（做空）",
}


def _target_name(tgt: dict) -> str:
    task = tgt.get("task", "return")
    hd = tgt["hold_days"]
    th = int(tgt.get("threshold", tgt.get("profit_target", 0.15)) * 100)
    return f'{hd}d_{task}_{th}'


def _current_model() -> str:
    return MODELS[_model_idx][0]


def _current_client():
    return MODELS[_model_idx][1]


def _use_full_tool_results() -> bool:
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


# ── 指標說明對照表 ────────────────────────────────────────────────
INDICATOR_DESCRIPTIONS: dict[str, str] = {
    "rsi": "RSI 相對強弱指數（>50 偏多，>70 超買）",
    "macd": "MACD 快慢線差值（正值偏多）",
    "macdh": "MACD 柱狀圖（正值=多頭柱，負值=空頭柱）",
    "adx": "ADX 趨勢強度（>25 有明確趨勢，>40 強趨勢）",
    "plus_di": "+DI 多方方向指標（>minus_di 代表多頭主導）",
    "minus_di": "-DI 空方方向指標",
    "bb_width": "布林通道寬度（>0.5 通道擴張，突破前兆）",
    "bb_upper": "布林通道上軌",
    "bb_lower": "布林通道下軌",
    "bb_mid": "布林通道中線（20日均線）",
    "atr_rank": "ATR 歷史百分位（值域0~1，>0.7 波動處於歷史高位）",
    "atr_pct": "ATR 占股價比例（atr/close，>0.05 代表每日波動超過 5%，跨股票可比）",
    "ma5": "5日均線",
    "ma10": "10日均線",
    "ma20": "20日均線",
    "ma60": "60日均線",
    "ma120": "120日均線",
    "ma5_x_ma10": "5日均線在10日均線之上（多頭排列）",
    "ma5_x_ma20": "5日均線在20日均線之上",
    "ma10_x_ma60": "10日均線在60日均線之上（中期多頭）",
    "vol_ratio": "量比（當日量/20日均量，>2 爆量）",
    "vol_trend": "量能趨勢（20日均量/60日均量，>1 量能放大）",
    "obv": "OBV 能量潮（累計成交量方向）",
    "obv_ma": "OBV 20日均線",
    "net_foreign": "外資買賣超張數（>0 外資買超）",
    "net_trust": "投信買賣超張數（>0 投信買超）",
    "net_dealer": "自營商買賣超張數（>0 自營買超）",
    "net_inst": "三大法人合計買賣超（>0 法人合計買超）",
    "foreign_rank": "外資買超歷史百分位（值域0~1，>0.7 外資強力買入）",
    "trust_rank": "投信買超歷史百分位（值域0~1，>0.7 投信強力買入）",
    "dealer_rank": "自營商買超歷史百分位（值域0~1，>0.7 強買）",
    "inst_rank": "三大法人合計歷史百分位（值域0~1，>0.7 法人整體強買）",
    "net_foreign_pct": "外資買超張數/30日均量（跨股票可比，>0.05 外資強力買入）",
    "net_inst_pct": "三大法人合計買超/30日均量（跨股票可比，>0.05 法人整體強買）",
    "net_trust_pct": "投信買超/30日均量（跨股票可比，>0.02 投信積極買入）",
    "net_dealer_pct": "自營商買超/30日均量（跨股票可比）",
    "margin_chg": "融資變化率（pct_change，>0 融資增加，<0 融資減少）",
    "short_chg": "融券變化率（<0 融券減少，回補中）",
    "short_ratio": "融券/融資比率（>0.3 融券壓力大）",
    "short_ratio_rank": "融券比率歷史百分位（值域0~1，>0.9 處於歷史極高位）",
    "margin_chg_rank": "融資變化率歷史百分位（值域0~1，>0.7 融資大幅增加）",
    "f_n_confirmed": "ABCD N字型態確認（>0 表示型態成立）",
    "f_n_structure_score": "ABCD 結構品質分數（0~1，>0.6 高品質）",
    "f_n_bc_retracement": "BC 段回撤比例（黃金比例 0.382~0.618 最佳）",
    "f_n_ab_gain": "AB 段漲幅（>0.1 代表第一波有力）",
    "f_n_d_breakout_strength": "D 點突破強度（>0.5 突破有力）",
    "f_n_volume_confirm": "D 點量能確認（>0.5 突破時有量）",
    "is_human_triangle": "是否為 human 三角收斂型態（refined+fuzzy 合併）",
    "human_only": "是否為 fuzzy 三角形（比 strict 更能抓模糊收斂）",
    "is_refined_triangle": "是否為嚴格三角收斂型態",
    "triangle_score": "三角收斂品質分數（0~1，>0.65 高品質）",
}

# close/high/low/open 只能用於相對比較（close > ma20、close > bb_upper），不可與固定數字比較
AGENT_INDICATORS = list(INDICATOR_DESCRIPTIONS.keys()) + [
    "close",
    "high",
    "low",
    "open",
]


# ── Grid Search 門檻定義 ─────────────────────────────────────────
_GRID_THRESHOLDS: dict[str, list[float]] = {
    "rsi": [50, 55, 60, 65],
    "macdh": [0],
    "adx": [20, 25, 30],
    "bb_width": [0.3, 0.5, 0.8],
    "atr_rank": [0.6, 0.7, 0.8],
    "ma5_x_ma10": [True],
    "ma5_x_ma20": [True],
    "ma10_x_ma60": [True],
    "vol_ratio": [1.0, 1.5, 2.0, 2.5],
    "vol_trend": [1.0, 1.2, 1.5],
    "net_foreign": [0, 100, 500],
    "net_trust": [0, 100],
    "net_dealer": [0, 100],
    "net_inst": [0, 200, 500],
    "foreign_rank": [0.6, 0.7, 0.8],
    "trust_rank": [0.6, 0.7, 0.8],
    "inst_rank": [0.6, 0.7, 0.8],
    "short_chg": [0, -0.05, -0.1],
    "short_ratio_rank": [0.7, 0.8, 0.9],
    "short_ratio": [0.2, 0.3, 0.5],
    "f_n_confirmed": [True],
    "f_n_structure_score": [0.3, 0.5, 0.7],
    "f_n_ab_gain": [0.05, 0.1, 0.15],
    "f_n_d_breakout_strength": [0.3, 0.5, 0.7],
    "f_n_volume_confirm": [0.3, 0.5],
    "human_only": [True],
    "triangle_score": [0.5, 0.65, 0.8],
}


# 純比較基準，不計入飽和（不代表任何市場信號）
_SATURATE_EXEMPT = {
    "close",
    "high",
    "low",
    "open",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ma120",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "obv_ma",
}


def _condition_indicators(condition: str) -> set[str]:
    tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", condition or ""))
    return tokens & set(AGENT_INDICATORS)


def _saturable_indicators(condition: str) -> set[str]:
    """只回傳應計入飽和的指標（排除純比較基準）。"""
    return _condition_indicators(condition) - _SATURATE_EXEMPT


# ── 工具定義 ────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_signal",
            "description": "測試條件字串，回傳 hit_rate 和 sample_count。hold_days/profit_target 使用 session 設定值（不需傳入）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "條件字串，用 & | ~，每個子句加括號"},
                    "label_type": {"type": "string", "description": "hit/return/max_return，預設 hit"},
                },
                "required": ["condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_history",
            "description": (
                "查詢本 target 已測試過的條件記錄。"
                "可按指標名稱篩選、按最低命中率篩選、只看通過的條件。"
                "用途：避免重複測試、了解目前最高 hit_rate、確認哪些指標已試過。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "篩選包含此指標名稱的條件（可選）"},
                    "min_hit_rate": {"type": "number", "description": "只回傳 hit_rate 大於此值的條件（可選，預設 0）"},
                    "passed_only": {"type": "boolean", "description": "只回傳通過門檻的條件，預設 false"},
                    "top_n": {"type": "integer", "description": "回傳筆數上限，預設 20"},
                },
                "required": [],
            },
        },
    },
]


# ── 指標建構 ─────────────────────────────────────────────────────
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

    ind: dict = {"close": close, "high": high, "low": low, "open": piv("open"), "volume": vol}

    for n in [5, 10, 20, 60, 120]:
        ind[f"ma{n}"] = close.rolling(n).mean()

    for fast, slow, name in [(5, 10, "ma5_x_ma10"), (5, 20, "ma5_x_ma20"), (10, 60, "ma10_x_ma60")]:
        f, s = ind[f"ma{fast}"], ind[f"ma{slow}"]
        ind[name] = f > s

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

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()]).groupby(level=0).max()
    atr14 = tr.rolling(14).mean()
    ind["atr_rank"] = _rolling_rank(atr14)
    ind["atr_pct"] = atr14 / (close + 1e-9)  # ATR 占股價比例，跨股票可比

    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    ind["adx"] = dx.rolling(14).mean()
    ind["plus_di"] = plus_di
    ind["minus_di"] = minus_di

    vol_ma20 = vol.rolling(20).mean()
    ind["vol_ratio"] = vol / (vol_ma20 + 1e-9)
    ind["vol_trend"] = vol_ma20 / vol.rolling(60).mean()
    direction = close.diff().apply(lambda x: x.map(lambda v: 1 if v > 0 else (-1 if v < 0 else 0)))
    ind["obv"] = (vol * direction).cumsum()
    ind["obv_ma"] = ind["obv"].rolling(20).mean()

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

    # 三角收斂（有快取就用快取，沒有就跳過）
    try:
        from j1stools.pattern_triangle import add_triangle_compare_columns

        _tri_cache_paths = [
            os.path.join(os.path.dirname(__file__), "../../..", "triangle_human_cache_2020_2025.pkl"),
            os.path.join(os.path.dirname(__file__), "../../..", "triangle_human_cache.pkl"),
        ]
        df_tri = None
        for cp in _tri_cache_paths:
            cp = os.path.normpath(cp)
            if os.path.exists(cp):
                df_tri = pd.read_pickle(cp)
                df_tri["date"] = pd.to_datetime(df_tri["date"])
                df_tri = add_triangle_compare_columns(df_tri)
                break
        if df_tri is not None:
            for feat in ["is_human_triangle", "human_only", "is_refined_triangle", "triangle_score"]:
                if feat in df_tri.columns:
                    piv_tri = (
                        df_tri[df_tri["stock_id"].isin(stocks)]
                        .pivot(index="date", columns="stock_id", values=feat)
                        .reindex(close.index)
                    )
                    ind[feat] = piv_tri
    except Exception:
        pass

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
        # 法人張數 / 30日均量 → 跨股票可比的相對買賣力道
        vol_ma30 = vol.rolling(30, min_periods=10).mean()
        for col in ["net_foreign", "net_trust", "net_dealer", "net_inst"]:
            ind[f"{col}_pct"] = ind[col] / (vol_ma30 + 1e-9)
    except Exception:
        pass

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


# ── RFC 特徵建構 ─────────────────────────────────────────────────
def _build_rfc_features(stocks: list, start: str, end: str) -> pd.DataFrame:
    from j1stools import parquet_db

    df = parquet_db.query_price(stocks, start, end)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    rows = []
    for sid, g in df.groupby("stock_id"):
        g = g.set_index("date").sort_index()
        c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
        for n in [5, 10, 20, 60]:
            ma = c.rolling(n).mean()
            g[f"f_dist_ma{n}"] = (c - ma) / (ma + 1e-9)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        g["f_rsi"] = 100 - 100 / (1 + gain / loss)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        g["f_macdh"] = macd - macd.ewm(span=9, adjust=False).mean()
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        g["f_bb_width"] = (2 * 2 * bb_std) / (bb_mid + 1e-9)
        g["f_bb_pos"] = (c - bb_mid) / (bb_std + 1e-9)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        g["f_atr_norm"] = atr / (c + 1e-9)
        g["f_vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)
        g["f_vol_trend"] = v.rolling(20).mean() / (v.rolling(60).mean() + 1e-9)
        # ADX
        h_diff = h.diff()
        l_diff = -l.diff()
        plus_dm = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
        minus_dm = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-9)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        g["f_adx"] = dx.rolling(14).mean()
        g["f_plus_di_ratio"] = plus_di / (plus_di + minus_di + 1e-9)  # 多空方向比例
        # ATR 百分位
        g["f_atr_rank"] = atr.rolling(60, min_periods=20).rank(pct=True)
        # OBV 動能（相對 MA）
        direction = c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (v * direction).cumsum()
        obv_ma20 = obv.rolling(20).mean()
        g["f_obv_above_ma"] = (obv > obv_ma20).astype(float)
        # 均線排列
        ma5 = c.rolling(5).mean()
        ma10 = c.rolling(10).mean()
        ma60 = c.rolling(60).mean()
        g["f_ma5_x_ma10"] = (ma5 > ma10).astype(float)
        g["f_ma10_x_ma60"] = (ma10 > ma60).astype(float)
        g["f_close_vs_ma60"] = (c - ma60) / (ma60 + 1e-9)  # 股價偏離 MA60 百分比
        g["stock_id"] = sid
        rows.append(g.reset_index())

    feat_df = pd.concat(rows, ignore_index=True)

    try:
        df_ib = parquet_db.query_ib(stocks, start, end)
        df_ib["date"] = pd.to_datetime(df_ib["date"])
        df_ib["net"] = df_ib["buy"] - df_ib["sell"]
        nets = []
        for src, col, rank_col in [
            ("Foreign_Investor", "f_net_foreign", "f_foreign_rank"),
            ("Investment_Trust", "f_net_trust", "f_trust_rank"),
            ("Dealer_self", "f_net_dealer", "f_dealer_rank"),
        ]:
            sub = df_ib[df_ib["name"] == src].pivot(index="date", columns="stock_id", values="net").fillna(0)
            nets.append(sub)
            rank = sub.rolling(60, min_periods=10).rank(pct=True).stack().reset_index()
            rank.columns = ["date", "stock_id", rank_col]
            feat_df = feat_df.merge(rank, on=["date", "stock_id"], how="left")
        inst = sum(nets)
        inst_rank = inst.rolling(60, min_periods=10).rank(pct=True).stack().reset_index()
        inst_rank.columns = ["date", "stock_id", "f_inst_rank"]
        feat_df = feat_df.merge(inst_rank, on=["date", "stock_id"], how="left")
        # 法人買超 / 成交量（標準化，跨股票可比）
        vol_wide = (
            feat_df.pivot(index="date", columns="stock_id", values="volume") if "volume" in feat_df.columns else None
        )
        if vol_wide is not None:
            vol_ma30 = vol_wide.rolling(30, min_periods=10).mean()
            for src_net, pct_col in zip(nets + [inst], ["f_foreign_pct", "f_trust_pct", "f_dealer_pct", "f_inst_pct"]):
                pct = (src_net / (vol_ma30 + 1e-9)).stack().reset_index()
                pct.columns = ["date", "stock_id", pct_col]
                feat_df = feat_df.merge(pct, on=["date", "stock_id"], how="left")
    except Exception:
        for col in ["f_foreign_rank", "f_trust_rank", "f_dealer_rank", "f_inst_rank"]:
            feat_df[col] = np.nan

    try:
        df_m = parquet_db.query_margin(stocks, start, end)
        df_m["date"] = pd.to_datetime(df_m["date"])
        margin = df_m.pivot(index="date", columns="stock_id", values="margin_purchase_today_balance")
        short = df_m.pivot(index="date", columns="stock_id", values="short_sale_today_balance")
        ratio = short / (margin + 1e-9)
        margin_chg = margin.pct_change().clip(-1, 1)
        short_chg = short.pct_change().clip(-1, 1)
        for ser, col in [
            (ratio.rolling(60, min_periods=10).rank(pct=True), "f_short_ratio_rank"),
            (margin_chg.rolling(60, min_periods=10).rank(pct=True), "f_margin_chg_rank"),
            (short_chg, "f_short_chg"),
            (margin_chg, "f_margin_chg"),
        ]:
            stacked = ser.stack().reset_index()
            stacked.columns = ["date", "stock_id", col]
            feat_df = feat_df.merge(stacked, on=["date", "stock_id"], how="left")
    except Exception:
        for col in ["f_short_ratio_rank", "f_margin_chg_rank", "f_short_chg", "f_margin_chg"]:
            feat_df[col] = np.nan

    return feat_df


def _normalize_condition(cond: str) -> str:
    tokens = re.split(r"(\s*[&|]\s*)", cond.strip())
    result = []
    for tok in tokens:
        s = tok.strip()
        if s in ("&", "|"):
            result.append(f" {s} ")
        elif s:
            result.append(s if s.startswith("(") else f"({s})")
    return "".join(result)


# ── 全域設定 ─────────────────────────────────────────────────────
from j1stools import parquet_db as _pdb

WATCH_STOCKS = _pdb.activate_stocks()
PREDICT_START = "2024-01-01"
TRAIN_END = "2023-12-31"
TRAIN_YEARS = 8
TRAIN_START = f"{int(PREDICT_START[:4]) - TRAIN_YEARS}-01-01"
SESSION_END = datetime.now().strftime("%Y-%m-%d")
BASE_RESULTS = os.path.join(os.path.dirname(__file__), "results")
_INDICATOR_CACHE: dict = {}


def _get_indicators(start: str, end: str):
    key = (start, end)
    if key not in _INDICATOR_CACHE:
        print(f"  [載入指標] {start} ~ {end}（首次，之後從快取）")
        _INDICATOR_CACHE[key] = _build_indicators(WATCH_STOCKS, start, end)
    return _INDICATOR_CACHE[key]


def _results_dir(target: dict) -> str:
    d = os.path.join(BASE_RESULTS, _target_name(target))
    os.makedirs(d, exist_ok=True)
    return d


def _signal_log_path(target: dict) -> str:
    return os.path.join(_results_dir(target), "signal_log.jsonl")


def _grid_log_path(target: dict) -> str:
    return os.path.join(_results_dir(target), "grid_log.jsonl")


def _backtest_log_path(target: dict) -> str:
    return os.path.join(_results_dir(target), "backtest_log.jsonl")


def _append_signal_log(entry: dict, target: dict):
    path = _signal_log_path(target)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_signal_log(target: dict, max_entries: int = 300) -> list[dict]:
    path = _signal_log_path(target)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = []
    for line in lines[-max_entries:]:
        try:
            entries.append(json.loads(line.strip()))
        except Exception:
            pass
    return entries


# ── Grid Search ─────────────────────────────────────────────────────
def _grid_condition(ind_a: str, th_a, ind_b: str, th_b) -> str:
    def _fmt(ind: str, th) -> str:
        if th is True or isinstance(th, bool):
            return f"({ind})"
        if isinstance(th, str):
            return f"({ind} {th})"
        return f"({ind} > {th})"

    return f"{_fmt(ind_a, th_a)} & {_fmt(ind_b, th_b)}"


def _load_grid_log(target: dict) -> list[dict]:
    path = _grid_log_path(target)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                pass
    return entries


def _run_grid_search(target: dict, force: bool = False):
    """暴力測試 2 指標組合，結果永久存於 grid_log.jsonl（只跑一次）。"""
    global _current_target
    _current_target = target
    name = _target_name(target)

    grid_path = _grid_log_path(target)
    existing_entries = _load_grid_log(target)
    existing_set = {_normalize_condition(e.get("condition", "")) for e in existing_entries}

    today = datetime.now().strftime("%Y-%m-%d")
    already_today = any(e.get("date") == today for e in existing_entries)
    if existing_entries and already_today and not force:
        passed = [e for e in existing_entries if e.get("hit_rate", 0) >= _pass_threshold(target)]
        print(f"\n  [Grid Search] {name}: 今日已跑（{len(existing_entries)} 筆，{len(passed)} 筆通過），跳過")
        return len(passed)

    candidate_pool = sorted(set(_GRID_THRESHOLDS.keys()))
    if len(candidate_pool) < 2:
        return 0

    import itertools

    # 重跑時清掉舊結果
    if os.path.exists(grid_path):
        os.remove(grid_path)
    existing_set = set()

    combos_tested = 0
    combos_passed = 0

    # 全部指標對 × 全部門檻組合（exhaustive，只跑一次所以要完整）
    all_combos = [
        (ind_a, th_a, ind_b, th_b)
        for ind_a, ind_b in itertools.combinations(candidate_pool, 2)
        for th_a in _GRID_THRESHOLDS[ind_a]
        for th_b in _GRID_THRESHOLDS[ind_b]
    ]
    print(f"\n  [Grid Search] {name}: exhaustive 測試 {len(all_combos)} 組組合...")
    ind_cache, close_cache = _get_indicators(TRAIN_START, TRAIN_END)
    hd = target["hold_days"]
    pt = target["profit_target"]
    future_high = pd.concat([close_cache.shift(-i) for i in range(1, hd + 1)], axis=0).groupby(level=0).max()

    for ind_a, th_a, ind_b, th_b in all_combos:
        if ind_a not in ind_cache or ind_b not in ind_cache:
            continue
        condition = _normalize_condition(_grid_condition(ind_a, th_a, ind_b, th_b))
        if condition in existing_set:
            continue
        try:
            signal = eval(condition, {"__builtins__": {}}, ind_cache)
            mask = signal.astype("boolean").fillna(False).astype(bool)
            labels = (future_high / close_cache - 1) >= pt
            values = labels[mask].values.flatten()
            values = values[~pd.isnull(values)].astype(float)
            combos_tested += 1
            n = len(values)
            if n < 20:
                continue
            hr = round(float(np.mean(values)), 4)
            entry = {
                "condition": condition,
                "hit_rate": hr,
                "sample_count": n,
                "hold_days": hd,
                "profit_target": pt,
                "source": "grid",
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            with open(grid_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing_set.add(condition)
            if hr >= _pass_threshold(target) and n >= 100:
                combos_passed += 1
                print(f"    ✅ {condition}: hit={hr:.2%} n={n}")
        except Exception:
            continue

    print(f"  [Grid Search] {name}: 測試 {combos_tested} 組，通過 {combos_passed} 組")
    return combos_passed


# ── 工具執行 ─────────────────────────────────────────────────────
def execute_tool(name: str, inputs: dict) -> dict:
    target = _current_target
    results_d = _results_dir(target)

    tool_usage_path = os.path.join(results_d, "tool_usage.json")
    try:
        _usage = json.load(open(tool_usage_path)) if os.path.exists(tool_usage_path) else {}
        _usage[name] = _usage.get(name, 0) + 1
        with open(tool_usage_path, "w", encoding="utf-8") as _f:
            json.dump(_usage, _f, ensure_ascii=False)
    except Exception:
        pass

    if name == "check_history":
        top_n = inputs.get("top_n", 20)
        ind_limit = 5
        # 合併：本 session + .bak + grid_log（永久）
        all_entries = _load_signal_log(target, max_entries=9999)
        bak_path = _signal_log_path(target) + ".bak"
        if os.path.exists(bak_path):
            try:
                with open(bak_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines[-300:]:
                    try:
                        all_entries.append(json.loads(line.strip()))
                    except Exception:
                        pass
            except Exception:
                pass
        all_entries.extend(_load_grid_log(target))
        pass_th = _pass_threshold(target)
        llm_entries = [e for e in all_entries if e.get("source") != "grid"]
        grid_entries = [e for e in all_entries if e.get("source") == "grid"]
        passed_llm = [e for e in llm_entries if e.get("hit_rate", 0) >= pass_th and e.get("sample_count", 0) >= 100]
        passed_grid = [e for e in grid_entries if e.get("hit_rate", 0) >= pass_th and e.get("sample_count", 0) >= 100]

        ind_usage: dict = {}
        for e in llm_entries:  # grid 條目不計入飽和
            for t in _saturable_indicators(e.get("condition", "")):
                ind_usage[t] = ind_usage.get(t, 0) + 1
        saturated = [k for k, v in ind_usage.items() if v >= ind_limit]

        ind_filter = inputs.get("indicator", "")

        def _match(e):
            return not ind_filter or ind_filter in e.get("condition", "")

        # 無 indicator 篩選時，補充分維度摘要（避免 LLM 反覆查詢）
        dimension_summary = {}
        if not ind_filter:
            _dim_map = {
                "技術面": ["adx", "rsi", "macdh", "bb_width", "atr_rank", "ma5_x_ma10", "ma10_x_ma60"],
                "量能面": ["vol_ratio", "vol_trend", "obv"],
                "籌碼面": ["net_foreign", "net_inst", "inst_rank", "foreign_rank", "short_chg", "short_ratio_rank"],
                "型態面": [
                    "f_n_structure_score",
                    "f_n_ab_gain",
                    "f_n_d_breakout_strength",
                    "triangle_score",
                    "human_only",
                ],
            }
            all_passed = sorted(passed_llm + passed_grid, key=lambda x: -x.get("hit_rate", 0))
            for dim, keywords in _dim_map.items():
                top = [
                    {"condition": e["condition"], "hit_rate": e["hit_rate"]}
                    for e in all_passed
                    if any(k in e.get("condition", "") for k in keywords)
                ][:3]
                dim_done = any(any(k in e.get("condition", "") for k in keywords) for e in passed_llm)
                dimension_summary[dim] = {"top3": top, "llm_found": dim_done}

        return {
            "total_tested": len(all_entries),
            "passed": len(passed_llm) + len(passed_grid),
            "saturated_indicators": saturated,
            "dimension_summary": dimension_summary,  # 各維度 top3（無 indicator 時才有）
            "passed_conditions": [
                {"condition": e.get("condition", ""), "hit_rate": e.get("hit_rate")}
                for e in sorted(filter(_match, passed_llm), key=lambda x: -x.get("hit_rate", 0))[:top_n]
            ],
            "grid_passed": [
                {"condition": e.get("condition", ""), "hit_rate": e.get("hit_rate")}
                for e in sorted(filter(_match, passed_grid), key=lambda x: -x.get("hit_rate", 0))[:top_n]
            ],
            "grid_summary": (
                f"Grid Search 共測 {len(grid_entries)} 組，{len(passed_grid)} 組通過（hit>={pass_th:.2f}）"
                + (
                    f"，其中含 '{ind_filter}' 的通過條件：{sum(1 for e in passed_grid if ind_filter in e.get('condition',''))} 組"
                    if ind_filter
                    else ""
                )
            ),
        }

    if name == "analyze_signal":
        hold_days = target["hold_days"]
        profit_target = target["profit_target"]
        condition = _normalize_condition(inputs["condition"])
        label_type = inputs.get("label_type", "hit")
        used_inds = _condition_indicators(condition)
        if len(used_inds) < 3 and not inputs.get("_bypass_min_indicators"):
            return {
                "error": (
                    f"需要 3 個指標（目前 {len(used_inds)} 個）。"
                    "請取 grid_passed 任一條件，再加入量能（vol_ratio/vol_trend）、"
                    "型態（f_n_structure_score/triangle_score）或籌碼（short_ratio_rank/trust_rank）之一。"
                )
            }
        # 飽和硬擋：合併 session + bak + grid_log 計算，防止 LLM 靠記憶繞過 tool description
        if not inputs.get("_bypass_min_indicators"):
            _all = _load_signal_log(target, max_entries=9999)
            _bak = _signal_log_path(target) + ".bak"
            if os.path.exists(_bak):
                try:
                    with open(_bak, encoding="utf-8") as _f:
                        for _l in _f:
                            try:
                                _all.append(json.loads(_l.strip()))
                            except Exception:
                                pass
                except Exception:
                    pass
            _all.extend(_load_grid_log(target))
            _iu: dict = {}
            for _e in _all:
                if _e.get("source") != "grid":  # grid 條目不計入飽和
                    for _t in _saturable_indicators(_e.get("condition", "")):
                        _iu[_t] = _iu.get(_t, 0) + 1
            all_saturated = {k for k, v in _iu.items() if v >= 5}
            saturated_used = _saturable_indicators(condition) & all_saturated
            if saturated_used:
                available_now = sorted(set(AGENT_INDICATORS) - all_saturated)
                return {
                    "error": f"飽和指標 {sorted(saturated_used)} 已出現 3+ 次，不可再用。",
                    "現在可用的指標": available_now,
                    "建議": "請從上面的可用指標中，選擇尚未探索的維度（量能、籌碼、型態）重新組合。",
                }
        try:
            ind, close = _get_indicators(TRAIN_START, TRAIN_END)
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}
        try:
            signal = eval(condition, {"__builtins__": {}}, ind)
        except Exception as e:
            return {"error": f"條件解析失敗：{e}"}
        if not isinstance(signal, pd.DataFrame):
            return {"error": "條件必須回傳 DataFrame"}
        mask = signal.astype("boolean").fillna(False).astype(bool)
        future_high = pd.concat([close.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
        future_low = pd.concat([close.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).min()
        task = target.get("task", "return")
        threshold = target.get("threshold", target.get("profit_target", 0.15))

        if task == "return":
            labels = (future_high / close - 1) >= threshold
        elif task == "breakout":
            lookback = target.get("lookback", 60)
            rolling_high = close.rolling(lookback, min_periods=lookback // 2).max()
            labels = future_high > rolling_high
        elif task == "consolidation":
            labels = (future_high - future_low) / (close + 1e-9) < threshold
        elif task == "trend_up":
            labels = (close.shift(-hold_days) / close - 1) >= threshold
        elif task == "trend_down":
            labels = (close / close.shift(-hold_days) - 1) >= threshold
        elif label_type == "max_return":
            labels = future_high / close - 1
        else:
            labels = close.shift(-hold_days) / close - 1
        values = labels[mask].values.flatten()
        values = values[~pd.isnull(values)]
        n = len(values)
        if n < 20:
            # 樣本太少也記錄，讓稀疏指標計入飽和，避免 LLM 一直重試
            _append_signal_log(
                {
                    "condition": condition,
                    "hit_rate": -1,
                    "sample_count": n,
                    "hold_days": hold_days,
                    "profit_target": profit_target,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "note": "too_few_samples",
                },
                target,
            )
            return {"error": f"樣本太少（{n} 筆），此指標組合觸發頻率極低，建議改用其他指標"}
        if label_type == "hit":
            hit_rate = float(np.mean(values.astype(float)))
            out = {
                "condition": condition,
                "label_type": label_type,
                "hold_days": hold_days,
                "sample_count": n,
                "hit_rate": round(hit_rate, 4),
                "profit_target": profit_target,
            }
            _append_signal_log({**out, "date": datetime.now().strftime("%Y-%m-%d")}, target)
            return out
        values = values.astype(float)
        out = {
            "condition": condition,
            "label_type": label_type,
            "hold_days": hold_days,
            "sample_count": n,
            "avg_return": round(float(np.mean(values)), 4),
            "win_rate": round(float((values > 0).mean()), 4),
            "std": round(float(np.std(values)), 4),
        }
        _append_signal_log({**out, "date": datetime.now().strftime("%Y-%m-%d")}, target)
        return out

    return {"error": f"未知工具：{name}"}


def _pass_threshold(target: dict) -> float:
    hd = target["hold_days"]
    task = target.get("task", "return")
    if task == "consolidation":
        return 0.50  # 盤整任務：一半以上時間維持低波動才算
    if hd <= 3:
        return 0.30
    return 0.35


# ── 系統 Prompt（根據 target 動態生成）────────────────────────────
def _build_system_prompt(target: dict) -> str:
    name = _target_name(target)
    hd = target["hold_days"]
    task = target.get("task", "return")
    threshold = target.get("threshold", target.get("profit_target", 0.15))
    sl = target.get("sl_stop", 0.08)
    task_desc = TASK_DEFS.get(task, task).format(**{**target, "threshold": threshold})

    # 根據目標時長給予提示
    if hd <= 3:
        hint = (
            "短線目標（3天內），重點關注：\n"
            "  1. 動能指標（rsi、macdh）— 短線慣性\n"
            "  2. 量能指標（vol_ratio、vol_trend）— 短線資金動能\n"
            "  3. 籌碼指標（short_ratio_rank、foreign_rank）— 短線軋空/法人買盤\n"
            "  4. 型態（f_n_d_breakout_strength）— 突破後的短期延續性"
        )
    elif hd <= 7:
        hint = (
            "中短線目標（5~7天），重點關注：\n"
            "  1. 趨勢指標（adx、plus_di）— 中期趨勢明確度\n"
            "  2. 量能趨勢（vol_trend）— 量能是否持續\n"
            "  3. 法人籌碼（net_foreign、inst_rank）— 法人佈局\n"
            "  4. 型態（f_n_structure_score、triangle_score）— 型態完成後的動能"
        )
    else:
        hint = (
            "中長期目標（10天），重點關注：\n"
            "  1. 趨勢強度（adx、bb_width）— 中長期趨勢\n"
            "  2. 型態品質（f_n_structure_score、f_n_ab_gain）— 型態完整度\n"
            "  3. 法人持續買超（inst_rank、foreign_rank）— 中長期資金\n"
            "  4. 均線排列（ma5_x_ma10、ma10_x_ma60）— 多頭排列確認"
        )

    pass_th = _pass_threshold(target)

    return f"""你是台灣股票量化研究助理。

━━ 本次任務 ━━
目標代號：{name}
預測任務：{task_desc}
持有天數：{hd} 天　停損：{sl:.0%}
建立一組**來自不同維度**的特徵候選條件，供後續 RFC 模型使用。

**完成標準：量能、籌碼、型態三個維度各至少 1 個通過條件（技術面自然會有，不算在完成條件內）。**
- 量能 / 籌碼 / 型態各維度找到 1~2 個即可，找到後立刻換下一個維度
- 技術面可自由探索，但不應成為唯一重心
- 通過門檻：hit_rate >= {pass_th:.2f}，sample_count >= 100

━━ 工作流程 ━━
1. 先呼叫 check_history()，看 grid_passed（2指標基礎）和 passed_conditions（已通過條件）
2. 確認量能 / 籌碼 / 型態哪個維度還沒有通過條件，優先去那個維度探索
3. 取 grid_passed 裡好的組合，加入目標維度的指標，測試 3+ 指標條件
4. condition 參數的可用指標清單已即時更新，只能使用清單內的指標

━━ 四個維度 ━━
- 技術面：adx, rsi, macdh, bb_width, atr_rank, ma5_x_ma10, ma10_x_ma60
- 量能面：vol_ratio, vol_trend, obv, obv_ma
- 籌碼面：net_foreign, net_inst, inst_rank, foreign_rank, short_chg, short_ratio_rank
- 型態面：f_n_structure_score, f_n_ab_gain, triangle_score, human_only

━━ 探索提示 ━━
{hint}

━━ 技巧 ━━
- sample_count 500~3000 最佳；> 5000 應加嚴門檻
- 統計資料期間：{TRAIN_START} ~ {TRAIN_END}

請用繁體中文回覆。"""


# ── 訊息壓縮 ─────────────────────────────────────────────────────
def _compress_result(fn_name: str, result: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if fn_name == "analyze_signal":
        cond = result.get("condition", "")[-40:]
        if "hit_rate" in result:
            flag = "✅" if result["hit_rate"] >= 0.35 else "❌"
            return f"hit={result['hit_rate']} n={result.get('sample_count')} {flag} ...{cond}"
        return f"avg={result.get('avg_return')} win={result.get('win_rate')} n={result.get('sample_count')} ...{cond}"
    if fn_name == "check_history":
        lines = [
            f"total={result.get('total_tested')} passed={result.get('passed')}",
            f"saturated={result.get('saturated_indicators',[])}",
            f"grid_summary={result.get('grid_summary','')}",
        ]
        dim = result.get("dimension_summary", {})
        if dim:
            for d_name, d_info in dim.items():
                done = "✅已有LLM條件" if d_info.get("llm_found") else "❌尚未找到"
                lines.append(f"  [{d_name}] {done}")
                for e in d_info.get("top3", []):
                    lines.append(f"    hit={e['hit_rate']} {e['condition']}")
        else:
            for e in result.get("grid_passed", [])[:10]:
                lines.append(f"  grid✅ hit={e['hit_rate']} {e['condition']}")
            for e in result.get("passed_conditions", [])[:10]:
                lines.append(f"  llm✅  hit={e['hit_rate']} {e['condition']}")
        return "\n".join(lines)
    if fn_name == "run_backtest":
        return (
            f"return={result.get('total_return')} sharpe={result.get('sharpe')} "
            f"mdd={result.get('max_drawdown')} win={result.get('win_rate')} "
            f"trades={result.get('total_trades')}"
        )
    return json.dumps(result, ensure_ascii=False)[:300]


def _trim_messages(messages: list, keep_last: int = 20) -> list:
    if len(messages) <= 2:
        return messages
    fixed = messages[:2]
    rest = messages[2:]
    if len(rest) <= keep_last:
        return messages
    trimmed = rest[-keep_last:]
    while trimmed:
        msg = trimmed[0]
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "tool":
            trimmed = trimmed[1:]
        else:
            break
    return fixed + trimmed


# ── Agent 主循環 ─────────────────────────────────────────────────
def run_agent(target: dict, max_calls: int = 0, model_override: str = "") -> str:
    """
    target: {"hold_days": 10, "profit_target": 0.15, "sl_stop": 0.08}
    max_calls: 最多幾次 tool 呼叫就停（0 = 不限）
    """
    global _current_target, _model_idx
    _current_target = target
    _model_idx = 0

    if model_override:
        for i, (m, _) in enumerate(MODELS):
            if model_override in m:
                _model_idx = i
                break

    # 備份舊 signal_log
    sig_path = _signal_log_path(target)
    if os.path.exists(sig_path):
        bak_path = sig_path + ".bak"
        import shutil

        shutil.copy2(sig_path, bak_path)
        os.remove(sig_path)
        print(f"  [session] 舊 signal_log 已備份為 .bak，重新開始記錄")

    tgt_name = _target_name(target)
    task = (
        f"為目標「{tgt_name}」蒐集特徵候選條件。\n"
        f"目標：持有 {target['hold_days']} 天，報酬 {target['profit_target']:.0%}。\n"
        f"門檻：hit_rate >= {_pass_threshold(target):.2f}，sample_count >= 100（{TRAIN_START}~{TRAIN_END}）。\n"
        f"{'（你必須做滿 ' + str(max_calls) + ' 次工具呼叫才能停止，不要提早結束）' if max_calls else ''}\n"
        f"提示：可先呼叫 check_history() 查看已測試記錄，避免重複。"
    )

    messages = [
        {"role": "system", "content": _build_system_prompt(target)},
        {"role": "user", "content": task},
    ]
    final_text = ""
    _rate_limit_count = 0
    _call_count = 0
    _tool_usage: dict = {}

    def _get_tools_with_usage() -> list:
        import copy as _copy

        sig_entries = _load_signal_log(target, max_entries=9999)
        n_tested = len(sig_entries)
        n_passed = len([e for e in sig_entries if e.get("hit_rate", 0) >= _pass_threshold(target)])

        # 計算歷史（含 bak + grid_log）飽和指標
        all_entries = list(sig_entries)
        _bak = _signal_log_path(target) + ".bak"
        if os.path.exists(_bak):
            try:
                with open(_bak, encoding="utf-8") as _bf:
                    for _line in _bf:
                        try:
                            all_entries.append(json.loads(_line.strip()))
                        except Exception:
                            pass
            except Exception:
                pass
        # grid 條目不計入飽和，只計算 LLM 生成的條目
        _ind_usage: dict = {}
        for _e in all_entries:
            if _e.get("source") != "grid":
                for _t in _saturable_indicators(_e.get("condition", "")):
                    _ind_usage[_t] = _ind_usage.get(_t, 0) + 1
        saturated = sorted(k for k, v in _ind_usage.items() if v >= 3)

        available = sorted(set(AGENT_INDICATORS) - set(saturated))

        result = _copy.deepcopy(TOOLS)
        for tool in result:
            name = tool["function"]["name"]
            count = _tool_usage.get(name, 0)
            notes = []
            if count > 0:
                notes.append(f"已呼叫 {count} 次")
            if name == "analyze_signal":
                notes.append(f"本 session 已測 {n_tested} 個條件，{n_passed} 個通過")
                # 直接把可用指標列表注入 condition 參數 description
                tool["function"]["parameters"]["properties"]["condition"]["description"] = (
                    f"條件字串，用 & | ~，每個子句加括號。"
                    f"【只能使用以下 {len(available)} 個指標，其餘已飽和禁用】：{available}"
                )
            if notes:
                tool["function"]["description"] += f"（{'；'.join(notes)}）"
        return result

    print(
        f"\n[{tgt_name}] 開始探索... {'（測試：最多 ' + str(max_calls) + ' 次工具呼叫）' if max_calls else ''}\n{'─' * 60}"
    )

    while True:
        try:
            if "gemini" in _current_model():
                time.sleep(5)
            response = _current_client().chat.completions.create(
                model=_current_model(),
                messages=_trim_messages(messages, keep_last=20),
                tools=_get_tools_with_usage(),
                tool_choice="auto",
            )
        except Exception as e:
            err = str(e)
            print(f"\n[API錯誤] {_current_model()}: {err[:200]}")
            if "429" in err or "rate_limit" in err.lower() or "quota" in err.lower() or "exceeded" in err.lower():
                if _rate_limit_count < 1:
                    _rate_limit_count += 1
                    time.sleep(65)
                    continue
                else:
                    _rate_limit_count = 0
                    if model_override:
                        break
                    if not _next_model():
                        break
                    continue
            raise

        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            try:
                _think_path = os.path.join(_results_dir(target), "thinking_log.jsonl")
                with open(_think_path, "a", encoding="utf-8") as _tf:
                    _tf.write(
                        json.dumps(
                            {
                                "call": _call_count,
                                "model": _current_model(),
                                "thinking": msg.content,
                                "tool_calls": [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else [],
                                "time": datetime.now().strftime("%H:%M:%S"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:
                pass

        if not msg.tool_calls:
            final_text = msg.content or ""
            print(f"\n{'=' * 60}\nAgent 報告：\n{final_text}")
            try:
                agent_report_path = os.path.join(_results_dir(target), "agent_report.txt")
                with open(agent_report_path, "w", encoding="utf-8") as _f:
                    _f.write(final_text)
            except Exception:
                pass
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"→ {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

            result = execute_tool(fn_name, fn_args)

            tool_content = (
                json.dumps(result, ensure_ascii=False)
                if _use_full_tool_results()
                else _compress_result(fn_name, result)
            )
            print(f"  {tool_content}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})
            _tool_usage[fn_name] = _tool_usage.get(fn_name, 0) + 1
            _call_count += 1
            if max_calls and _call_count >= max_calls:
                print(f"\n[測試] 已達 {max_calls} 次工具呼叫上限，停止")
                return final_text

    return final_text


# ── 驗證與訓練 ────────────────────────────────────────────────────
def validate_and_train(target: dict):
    """先訓練 RFC → 用模型信心度排序進場做回測。"""
    global _current_target
    _current_target = target
    tgt_name = _target_name(target)
    pass_th = _pass_threshold(target)
    hd = target["hold_days"]
    pt = target["profit_target"]
    sl = target["sl_stop"]

    passing = [
        e
        for e in _load_signal_log(target, max_entries=9999)
        if e.get("hit_rate", 0) >= pass_th and e.get("sample_count", 0) >= 100
    ]

    # 讀取歷史
    prev_passing = []
    bak_path = _signal_log_path(target) + ".bak"
    if os.path.exists(bak_path):
        with open(bak_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("hit_rate", 0) >= pass_th and e.get("sample_count", 0) >= 100:
                        prev_passing.append(e)
                except:
                    pass

    hist_passing = []
    bt_path = _backtest_log_path(target)
    if os.path.exists(bt_path):
        with open(bt_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("hit_rate", 0) >= pass_th and e.get("sample_count", 0) >= 100:
                        hist_passing.append(e)
                except:
                    pass

    if not passing and not hist_passing:
        print(f"[{tgt_name}] 沒有通過的條件，跳過驗證")
        return

    print(f"\n[{tgt_name}] 訓練 RFC（用於模型回測）...")
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    bt_result = None
    try:
        stocks = WATCH_STOCKS
        feat_df = _build_rfc_features(stocks, TRAIN_START, SESSION_END)

        close_wide = feat_df.pivot(index="date", columns="stock_id", values="close")
        future_max = pd.concat([close_wide.shift(-i) for i in range(1, hd + 1)], axis=0).groupby(level=0).max()
        target_wide = ((future_max / close_wide - 1) >= pt).astype(int)
        target_long = target_wide.stack().reset_index()
        target_long.columns = ["date", "stock_id", "target"]
        feat_df = feat_df.merge(target_long, on=["date", "stock_id"], how="left")

        ind, _ = _build_indicators(stocks, TRAIN_START, SESSION_END)

        all_sources = passing + prev_passing + hist_passing
        all_conds = {e["condition"]: e for e in sorted(all_sources, key=lambda x: -x.get("hit_rate", 0))}

        # 從所有通過條件提取唯一信號指標（去重，排除純比較基準）
        used_inds: set[str] = set()
        for cond in all_conds:
            used_inds |= _saturable_indicators(cond)

        # 把指標連續值從 ind 合併進 feat_df
        llm_feat_cols: list[str] = []
        for ind_name in sorted(used_inds):
            f_col = f"f_{ind_name}"
            if f_col in feat_df.columns:
                llm_feat_cols.append(f_col)
            elif ind_name in feat_df.columns:
                llm_feat_cols.append(ind_name)
            elif ind_name in ind:
                try:
                    wide = ind[ind_name]
                    if isinstance(wide, pd.DataFrame):
                        long = wide.astype(float).stack().reset_index()
                        long.columns = ["date", "stock_id", f_col]
                        feat_df = feat_df.merge(long, on=["date", "stock_id"], how="left")
                        feat_df[f_col] = feat_df[f_col].fillna(0)
                        llm_feat_cols.append(f_col)
                except Exception as _e:
                    print(f"  [指標合併失敗] {ind_name}: {_e}")

        tech_cols = [c for c in feat_df.columns if c.startswith("f_") and c not in llm_feat_cols]
        feat_df = feat_df.dropna(subset=tech_cols + ["target"])
        print(f"  LLM 選出指標（{len(llm_feat_cols)} 個）：{llm_feat_cols}")

        cutoff = pd.Timestamp(PREDICT_START)
        train = feat_df[feat_df["date"] < cutoff]
        test = feat_df[feat_df["date"] >= cutoff]

        if len(train) < 200 or len(test) < 50:
            print(f"  資料不足：train={len(train)}, test={len(test)}")
        else:
            actual = test["target"].values
            base_rate = round(float(actual.mean()), 4)

            def _train_rfc(feat_cols, model_name):
                rfc = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
                rfc.fit(train[feat_cols], train["target"])
                proba = rfc.predict_proba(test[feat_cols])[:, 1]
                thresholds = {}
                for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    mask = proba >= th
                    n = int(mask.sum())
                    thresholds[f"proba>={th}"] = (
                        {"n": n, "hit_rate": round(float(actual[mask].mean()), 4)}
                        if n >= 10
                        else {"n": n, "note": "樣本不足"}
                    )
                importance = sorted(zip(feat_cols, rfc.feature_importances_), key=lambda x: x[1], reverse=True)
                top_feats = [{"feature": f, "importance": round(float(v), 4)} for f, v in importance[:15]]
                model_path = os.path.join(_results_dir(target), f"rfc_{model_name}.joblib")
                joblib.dump(rfc, model_path)
                print(f"  [{model_name}] base={base_rate} {thresholds}")
                return {"thresholds": thresholds, "top_features": top_feats, "model_path": model_path, "rfc": rfc}

            model_a = _train_rfc(tech_cols, "tech_only")
            model_new = _train_rfc(llm_feat_cols, "llm_selected") if llm_feat_cols else {}

            bt_result = None
            if model_new:
                rfc_new = model_new["rfc"]
                test_proba = rfc_new.predict_proba(test[llm_feat_cols])[:, 1]
                test_c = test.copy()
                test_c["proba"] = test_proba
                proba_wide = test_c.pivot(index="date", columns="stock_id", values="proba").fillna(0)
                close_wide_bt = test_c.pivot(index="date", columns="stock_id", values="close")

                from j1stools.backtest_engine import backtest_engine

                entries = proba_wide >= 0.5
                exits = pd.Series(False, index=proba_wide.index)
                stock_group_map = {s: "default" for s in proba_wide.columns}

                try:
                    portfolio_value, trades_df, _ = backtest_engine(
                        close=close_wide_bt,
                        entries=entries,
                        exits=exits,
                        df_proba=proba_wide,
                        stock_group=stock_group_map,
                        hold_days=hd,
                        sl_stop=sl,
                        tp_stop=pt,
                        use_fixed_sl=True,
                        use_fixed_tp=True,
                        use_sl_trail=False,
                        use_hold_days=True,
                        max_positions=5,
                    )
                    pv = portfolio_value.dropna()
                    rets = pv.pct_change(fill_method=None).dropna()
                    total_return = round(float(pv.iloc[-1] / pv.iloc[0] - 1), 4) if len(pv) > 1 else None
                    sharpe = round(float(rets.mean() / rets.std() * (252**0.5)) if rets.std() > 0 else 0, 2)
                    max_dd = round(float(((pv / pv.cummax()) - 1).min()), 4)
                    rp = trades_df.get("return_pct", pd.Series(dtype=float))
                    win_rate = round(float((rp > 0).mean()), 4) if len(rp) else None
                    avg_return = round(float(rp.mean()), 4) if len(rp) else None
                    avg_win = round(float(rp[rp > 0].mean()), 4) if (rp > 0).any() else 0
                    avg_loss = round(float(rp[rp < 0].mean()), 4) if (rp < 0).any() else 0

                    bt_result = {
                        "type": "model_backtest",
                        "model": "target_model",
                        "period": f"{PREDICT_START}~{SESSION_END}",
                        "total_trades": len(trades_df),
                        "total_return": total_return,
                        "sharpe": sharpe,
                        "max_drawdown": max_dd,
                        "win_rate": win_rate,
                        "avg_return": avg_return,
                        "avg_win": avg_win,
                        "avg_loss": avg_loss,
                    }

                    with open(_backtest_log_path(target), "w", encoding="utf-8") as f:
                        f.write(json.dumps(bt_result, ensure_ascii=False) + "\n")

                    wr_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"
                    ret_str = f"{total_return:+.2%}" if total_return is not None else "N/A"
                    print(
                        f"\n  [模型回測] Sharpe={sharpe}  總報酬={ret_str}  "
                        f"交易={len(trades_df)} 筆  獲利={wr_str}  最大回撤={max_dd:.1%}"
                    )
                except Exception as e:
                    print(f"  [模型回測] 失敗：{e}")
                    import traceback

                    traceback.print_exc()

            def _json_safe(m: dict) -> dict:
                return {k: v for k, v in m.items() if k != "rfc"}

            rfc_result = {
                "target": tgt_name,
                "hold_days": hd,
                "profit_target": pt,
                "sl_stop": sl,
                "pass_threshold": pass_th,
                "base_rate": base_rate,
                "train_size": len(train),
                "test_size": len(test),
                "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
                "test_period": f"{PREDICT_START} ~ {SESSION_END}",
                "n_conditions": len(all_conds),
                "n_tech_features": len(tech_cols),
                "n_llm_features": len(llm_feat_cols),
                "llm_features": llm_feat_cols,
                "model_a": {"name": "技術指標（基準線）", **_json_safe(model_a)},
                "model_new": (
                    {"name": f"LLM選指標（{len(llm_feat_cols)}個）", **_json_safe(model_new)} if model_new else {}
                ),
                "backtest": bt_result,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            rfc_path = os.path.join(_results_dir(target), "rfc_result.json")
            tmp_path = rfc_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(rfc_result, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, rfc_path)
            print(f"  RFC 結果已存：{rfc_path}")

    except Exception as e:
        print(f"  RFC 訓練失敗：{e}")
        import traceback

        traceback.print_exc()

    _generate_report(target)


# ── 報表 ─────────────────────────────────────────────────────────
def _generate_report(target: dict):
    results_d = _results_dir(target)
    tgt_name = _target_name(target)
    hd = target["hold_days"]
    pt = target["profit_target"]

    rfc_path = os.path.join(results_d, "rfc_result.json")
    if not os.path.exists(rfc_path):
        print(f"[報表] 無 RFC 結果")
        return

    try:
        with open(rfc_path, encoding="utf-8") as f:
            rfc = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[報表] rfc_result.json 損壞（{e}），刪除後重跑")
        os.remove(rfc_path)
        return

    backtest = rfc.get("backtest") or {}
    ma = rfc.get("model_a", {})
    mn = rfc.get("model_new", {})

    def _sc(s):
        if s >= 2.0:
            return "#1a7a3a"
        if s >= 1.5:
            return "#2ecc71"
        if s >= 1.0:
            return "#f39c12"
        if s >= 0:
            return "#e67e22"
        return "#e74c3c"

    def _row_html(m, name, feat_count):
        if not m:
            return f"<tr><td style='border:1px solid #333;padding:6px'>{name}</td><td colspan='5' style='color:#888;text-align:center'>未訓練</td></tr>"
        th = m.get("thresholds", {})

        def _fmt(v):
            return f"{v:.1%}" if isinstance(v, (int, float)) else "-"

        return f"""<tr>
      <td style="border:1px solid #333;padding:6px;font-weight:bold">{name}</td>
      <td style="text-align:center;border:1px solid #333;padding:6px">{feat_count}</td>
      <td style="text-align:center;border:1px solid #333;padding:6px">{_fmt(th.get("proba>=0.5",{}).get("hit_rate","-"))}</td>
      <td style="text-align:center;border:1px solid #333;padding:6px">{_fmt(th.get("proba>=0.6",{}).get("hit_rate","-"))}</td>
      <td style="text-align:center;border:1px solid #333;padding:6px">{_fmt(th.get("proba>=0.7",{}).get("hit_rate","-"))}</td>
      <td style="text-align:center;border:1px solid #333;padding:6px">{_fmt(th.get("proba>=0.8",{}).get("hit_rate","-"))}</td>
      <td style="text-align:center;border:1px solid #333;padding:6px">{_fmt(th.get("proba>=0.9",{}).get("hit_rate","-"))}</td>
    </tr>"""

    comparison_html = f"""
    <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
    <tr>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">模型</th>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">特徵數</th>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">proba>=0.5</th>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">proba>=0.6</th>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">proba>=0.7</th>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">proba>=0.8</th>
      <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">proba>=0.9</th>
    </tr>
    {_row_html(ma, "技術指標（基準線）", rfc.get("n_tech_features", 0))}
    {_row_html(mn, f"LLM選指標（{rfc.get('n_llm_features', 0)}個）", rfc.get("n_llm_features", 0))}
    </table>"""

    bt_html = ""
    if backtest:
        trades = backtest.get("total_trades", 0)
        sharpe = backtest.get("sharpe", 0) or 0
        total_ret = backtest.get("total_return")
        mdd = backtest.get("max_drawdown", 0) or 0
        wr = backtest.get("win_rate")
        avg_r = backtest.get("avg_return")
        avg_w = backtest.get("avg_win", 0) or 0
        avg_l = backtest.get("avg_loss", 0) or 0
        sharpe_color = _sc(sharpe)
        dd_color = "#e74c3c" if mdd < -0.15 else "#e67e22" if mdd < -0.10 else "#2ecc71"
        _pct = lambda v: f"{v:+.2%}" if v is not None else "N/A"
        _wpct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        bt_html = f"""
        <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
        <tr>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">交易次數</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center;color:{sharpe_color}">Sharpe</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">總報酬</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center;color:{dd_color}">最大回撤</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">獲利率</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">平均報酬</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">平均贏</th>
          <th style="background:#0d47a1;color:#fff;padding:8px;text-align:center">平均輸</th>
        </tr>
        <tr>
          <td style="text-align:center;border:1px solid #333;padding:6px">{trades}</td>
          <td style="text-align:center;border:1px solid #333;padding:6px;font-weight:bold;color:{sharpe_color}">{sharpe:.2f}</td>
          <td style="text-align:center;border:1px solid #333;padding:6px">{_pct(total_ret)}</td>
          <td style="text-align:center;border:1px solid #333;padding:6px;color:{dd_color}">{mdd:.1%}</td>
          <td style="text-align:center;border:1px solid #333;padding:6px">{_wpct(wr)}</td>
          <td style="text-align:center;border:1px solid #333;padding:6px">{f"{avg_r:+.2f}%" if avg_r is not None else "N/A"}</td>
          <td style="text-align:center;border:1px solid #333;padding:6px;color:#2ecc71">{avg_w:+.2f}%</td>
          <td style="text-align:center;border:1px solid #333;padding:6px;color:#e74c3c">{avg_l:+.2f}%</td>
        </tr>
        </table>"""

    feat_rows = ""
    if mn:
        for i, ft in enumerate(mn.get("top_features", [])[:15]):
            bar_w = min(int(ft["importance"] * 100), 100)
            feat_rows += (
                f"<tr><td style='text-align:center;border:1px solid #333;padding:6px'>{i+1}</td>"
                f"<td style='font-family:monospace;font-size:11px;border:1px solid #333;padding:6px'>{ft['feature']}</td>"
                f"<td style='text-align:center;border:1px solid #333;padding:6px'>{ft['importance']:.4f}</td>"
                f"<td style='border:1px solid #333;padding:6px'><div style='background:#4fc3f7;height:12px;width:{bar_w}%;border-radius:3px;min-width:4px'></div></td></tr>"
            )

    # 歷史報表導覽
    import glob

    archive_files = sorted(glob.glob(os.path.join(results_d, "report_*.html")), reverse=True)
    nav_links = " | ".join(
        f'<a href="{os.path.basename(p)}" style="color:#4fc3f7">{os.path.basename(p).replace("report_","").replace(".html","")}</a>'
        for p in archive_files[:10]
    )
    nav_html = (
        f'<div style="background:#0d1b2a;padding:10px;border-radius:6px;margin-bottom:20px;font-size:13px">'
        f'歷史報表：{nav_links if nav_links else "（尚無歷史）"}</div>'
    )

    # Agent 報告
    agent_report_html = ""
    agent_report_path = os.path.join(results_d, "agent_report.txt")
    if os.path.exists(agent_report_path):
        with open(agent_report_path, encoding="utf-8") as f:
            agent_text = f.read().strip()
        if agent_text:
            agent_report_html = f'<pre style="white-space:pre-wrap;background:#0d1b2a;padding:16px;border-radius:8px;font-size:13px;line-height:1.6">{agent_text}</pre>'

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8">
<title>目標 {tgt_name} 模型報表</title>
<style>
  body{{font-family:sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#1a1a2e;color:#e0e0e0}}
  h1,h2,h3,h4{{color:#4fc3f7}}
  table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
  th{{background:#0d47a1;color:#fff;padding:8px;text-align:center}}
  td{{border:1px solid #333;padding:6px}}
  tr:nth-child(even){{background:#16213e}}
</style>
</head>
<body>
<h1>🎯 目標 {tgt_name} — 專屬模型報表</h1>
<p>持有 {hd} 天，目標報酬 {pt:.0%}，停損 {target['sl_stop']:.0%}</p>
<p>統計期：{TRAIN_START} ~ {TRAIN_END}　回測期：{PREDICT_START} ~ {SESSION_END}</p>
{nav_html}
<h2>三個 RFC 模型比較</h2>
<p style="color:#aaa;font-size:13px">驗證期不同 proba 門檻的命中率比較</p>
{comparison_html}

<h2>模型回測結果（out-of-sample）</h2>
{bt_html if bt_html else '<p style="color:#666">無回測資料</p>'}

<h2>模型 C 重要特徵排名</h2>
<table style="width:80%">
<tr><th>#</th><th>特徵</th><th>重要性</th><th>視覺化</th></tr>
{feat_rows if feat_rows else '<tr><td colspan="4" style="color:#888;text-align:center">無資料</td></tr>'}
</table>

<h2>Agent 探索報告</h2>
{agent_report_html if agent_report_html else '<p style="color:#666">無文字報告</p>'}

<p style="color:#666;font-size:12px">產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""

    report_path = os.path.join(results_d, "report.html")

    # 備份舊報表（帶時間戳），保留歷史
    if os.path.exists(report_path):
        import shutil

        ts = datetime.fromtimestamp(os.path.getmtime(report_path)).strftime("%Y%m%d_%H%M")
        archive_path = os.path.join(results_d, f"report_{ts}.html")
        shutil.copy2(report_path, archive_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[報表] 已存：{report_path}")


# ── 入口 ─────────────────────────────────────────────────────────
def run_session(
    task: str = "return",
    hold_days: int = 10,
    threshold: float = 0.15,
    sl_stop: float = 0.08,
    lookback: int = 60,
    model: str = "",
    max_calls: int = 0,
):
    """
    task: return / breakout / consolidation / trend_up / trend_down
    hold_days: 持有天數
    threshold: 任務門檻（return=漲幅, consolidation=振幅, trend_up/down=漲跌幅）
    sl_stop: 停損幅度
    lookback: breakout 任務用的回看天數（預設 60）
    model: 指定模型（"deepseek"、"gemini"、"qwen"，空字串 = 預設第一個）
    max_calls: 測試模式最多幾次工具呼叫（0 = 不限）
    """
    target = {
        "task": task,
        "hold_days": hold_days,
        "threshold": threshold,
        "sl_stop": sl_stop,
        "lookback": lookback,
    }
    tgt_name = _target_name(target)
    task_desc = TASK_DEFS.get(task, task).format(**target)
    print(
        f"\n{'=' * 60}\n開始目標 {tgt_name}：{task_desc}\n{'=' * 60}"
    )

    _run_grid_search(target)  # 第一次跑完後永久快取，之後直接跳過
    run_agent(target, max_calls=max_calls, model_override=model)
    validate_and_train(target)


if __name__ == "__main__":
    import sys

    hold_days = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    profit_target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    sl_stop = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
    model = sys.argv[4] if len(sys.argv) > 4 else "qwen"
    max_calls = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    model = "qwen"
    max_calls = 30
    run_session(hold_days=hold_days, profit_target=profit_target, sl_stop=sl_stop, model=model, max_calls=max_calls)
