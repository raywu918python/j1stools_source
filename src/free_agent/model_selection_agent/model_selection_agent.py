"""
model_selection_agent.py — 假設驅動的專屬模型建立 Agent

設計思路：
  每次 session 聚焦一個「市場假設」（軋空、籌碼堆積、型態突破...）
  → Agent 在該假設空間內找條件
  → validate_and_train() 訓練該假設的專屬模型
  → 最終產出多個小而準的專屬模型，每個有明確的市場解釋

時間切分：
  訓練：2015-01-01 ~ 2023-12-31
  驗證：2024-01-01 ~ 今天

持有參數（固定）：
  hold_days=10, profit_target=0.15, sl_stop=0.08
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()

_groq_client   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
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
    ("deepseek-v4-flash", _deepseek_client),                                    # 主力
    (os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), _gemini_client),       # 備用1：免費（每天20次）
    # ("llama-3.3-70b-versatile", _groq_client),                                # 備用2：tool_use 完全不穩，停用
    ("qwen2.5:14b", _ollama_client),                                            # 備用3：本機
]
_model_idx = 0


def _current_model() -> str:
    return MODELS[_model_idx][0]


def _current_client():
    return MODELS[_model_idx][1]


def _next_model() -> bool:
    global _model_idx
    if _model_idx < len(MODELS) - 1:
        _model_idx += 1
        print(f"\n[模型輪換] 切換至 {MODELS[_model_idx][0]}")
        return True
    print("\n[模型輪換] 所有模型額度已用完，今天停止")
    return False


# ── 假設設定 ───────────────────────────────────────────────────
# 每個假設定義：描述、重點指標、探索提示
HYPOTHESIS_CONFIGS: dict[str, dict] = {
    "squeeze": {
        "name": "軋空型",
        "description": "融券部位被迫回補，配合外資或法人買超，形成短期強烈上漲",
        "indicator_categories": {
            "融券端": ["short_bal", "short_chg", "short_ratio", "short_ratio_rank"],
            "法人端": ["net_foreign", "net_trust", "net_dealer", "foreign_rank", "trust_rank", "inst_rank"],
            "量能端": ["vol_ratio", "vol_trend", "obv_ma"],
            "型態確認": ["f_n_confirmed", "f_n_structure_score", "f_n_ab_gain", "f_n_d_breakout_strength", "f_n_volume_confirm"],
            "技術確認": ["rsi", "adx", "bb_width", "atr_rank"],
        },
        "explore_hint": (
            "融券回補信號：short_chg < 0（融券減少），short_bal > 0（有融券部位），short_ratio_rank > 0.8（融券比率歷史高位）。"
            "法人承接信號：net_foreign > 0 或 net_trust > 0（法人買超），inst_rank > 0.7。"
            "型態確認：f_n_structure_score > 0.7（ABCD高品質型態），f_n_d_breakout_strength > 0.5（D點突破強）。"
            "量能確認：vol_ratio > 2（爆量）。"
            "重要提示：試試「融券高位 + ABCD型態 + 法人買超」的三角組合，例如 short_ratio_rank > 0.9 & f_n_structure_score > 0.7 & net_foreign > 0。"
        ),
    },
    "chip": {
        "name": "籌碼堆積型",
        "description": "外資、投信、自營商同步買超，籌碼集中到法人手中，後續動能強",
        "indicator_categories": {
            "外資端": ["net_foreign", "foreign_rank"],
            "投信端": ["net_trust", "trust_rank"],
            "自營端": ["net_dealer", "dealer_rank"],
            "合計法人": ["net_inst", "inst_rank"],
            "技術確認": ["rsi", "adx", "ma5_x_ma10", "bb_width"],
        },
        "explore_hint": (
            "三大法人同步買超：net_inst > 0，foreign_rank > 0.6，trust_rank > 0.6。"
            "持續買超幾天比單日更強。"
            "技術面：均線多頭排列（ma5_x_ma10），趨勢明確（adx > 25）。"
        ),
    },
    "breakout": {
        "name": "趨勢突破型",
        "description": "股價突破整理區間，配合量能放大和趨勢指標，形成新的上升趨勢",
        "indicator_categories": {
            "突破強度": ["bb_width", "adx", "plus_di"],
            "量能確認": ["vol_ratio", "vol_trend"],
            "均線排列": ["ma5_x_ma10", "ma5_x_ma20", "ma10_x_ma60"],
            "動能位置": ["rsi", "macd", "macdh", "atr_rank"],
        },
        "explore_hint": (
            "突破訊號：bb_width > 0.5（通道擴張），adx > 25（趨勢明確），plus_di > minus_di。"
            "量能確認：vol_ratio > 1.5（量增）。"
            "均線：ma5_x_ma10（黃金交叉），ma10_x_ma60（中期多頭）。"
        ),
    },
    "pattern": {
        "name": "型態突破型",
        "description": "ABCD N字型或三角收斂型態確認後，利用型態特徵預測後續走勢",
        "indicator_categories": {
            "ABCD確認": ["f_n_confirmed", "f_n_structure_score"],
            "ABCD強度": ["f_n_ab_gain", "f_n_d_breakout_strength", "f_n_volume_confirm"],
            "三角收斂": ["is_human_triangle", "human_only", "triangle_score"],
            "量能技術": ["vol_ratio", "rsi", "adx"],
        },
        "explore_hint": (
            "ABCD型態：f_n_confirmed > 0.15，f_n_structure_score > 0.3，f_n_d_breakout_strength > 0.5。"
            "三角收斂（fuzzy偵測，比strict更準）："
            "  - (human_only) 表示 fuzzy 三角形成立（True/False 指標）"
            "  - (is_human_triangle) 表示 refined + fuzzy 合併"
            "  - (triangle_score > 0.65) 表示高品質三角形"
            "組合範例：(human_only) & (vol_ratio > 1.5) & (adx > 20)。"
            "組合範例：(human_only) & (f_n_d_breakout_strength > 0.4) & (net_foreign > 0)。"
            "也可先呼叫 analyze_pattern_signal(pattern='triangle') 了解三角形基礎勝率。"
        ),
    },
    "momentum": {
        "name": "動能延續型",
        "description": "近期已有漲勢，動能指標顯示強勢，預測動能繼續延伸",
        "indicator_categories": {
            "動能強度": ["rsi", "macd", "macdh"],
            "趨勢確認": ["adx", "plus_di", "minus_di"],
            "量能趨勢": ["vol_trend", "vol_ratio", "obv_ma"],
            "波動位置": ["bb_width", "atr_rank"],
        },
        "explore_hint": (
            "動能剛起步：rsi > 55 且 < 80（強勢但未超買），macdh > 0（多頭柱）。"
            "趨勢確認：adx > 25，plus_di > minus_di。"
            "量能支撐：vol_trend > 1（量能趨勢向上）。"
        ),
    },
}

# ── 指標說明對照表 ────────────────────────────────────────────────
INDICATOR_DESCRIPTIONS: dict[str, str] = {
    # 技術指標
    "rsi":           "RSI 相對強弱指數（>50 偏多，>70 超買）",
    "macd":          "MACD 快慢線差值（正值偏多）",
    "macdh":         "MACD 柱狀圖（正值=多頭柱，負值=空頭柱）",
    "adx":           "ADX 趨勢強度（>25 有明確趨勢，>40 強趨勢）",
    "plus_di":       "+DI 多方方向指標（>minus_di 代表多頭主導）",
    "minus_di":      "-DI 空方方向指標",
    "bb_width":      "布林通道寬度（>0.5 通道擴張，突破前兆）",
    "bb_upper":      "布林通道上軌",
    "bb_lower":      "布林通道下軌",
    "bb_mid":        "布林通道中線（20日均線）",
    "atr":           "ATR 真實波幅（波動度絕對值）",
    "atr_rank":      "ATR 歷史百分位（>0.7 波動處於歷史高位）",
    # 均線
    "ma5":           "5日均線",
    "ma10":          "10日均線",
    "ma20":          "20日均線",
    "ma60":          "60日均線",
    "ma120":         "120日均線",
    "ma5_x_ma10":    "5日上穿10日均線（黃金交叉信號）",
    "ma5_x_ma20":    "5日上穿20日均線",
    "ma10_x_ma60":   "10日上穿60日均線（中期多頭確認）",
    # 量能
    "vol_ratio":     "量比（當日量/20日均量，>2 爆量）",
    "vol_trend":     "量能趨勢（20日均量/60日均量，>1 量能放大）",
    "obv":           "OBV 能量潮（累計成交量方向）",
    "obv_ma":        "OBV 20日均線",
    # 法人籌碼
    "net_foreign":   "外資買賣超張數（>0 外資買超）",
    "net_trust":     "投信買賣超張數（>0 投信買超）",
    "net_dealer":    "自營商買賣超張數（>0 自營買超）",
    "net_inst":      "三大法人合計買賣超（>0 法人合計買超）",
    "foreign_rank":  "外資買超歷史百分位（>0.7 外資強力買入）",
    "trust_rank":    "投信買超歷史百分位（>0.7 投信強力買入）",
    "dealer_rank":   "自營商買超歷史百分位",
    "inst_rank":     "三大法人合計歷史百分位（>0.7 法人整體強買）",
    # 融資融券
    "margin_bal":    "融資餘額（融資張數）",
    "short_bal":     "融券餘額（空頭張數，>0 有融券部位）",
    "margin_chg":    "融資變化率（pct_change）",
    "short_chg":     "融券變化率（<0 融券減少，回補中）",
    "short_ratio":   "融券/融資比率（>0.3 融券壓力大）",
    "short_ratio_rank": "融券比率歷史百分位（>0.9 處於歷史極高位）",
    "margin_chg_rank":  "融資變化率歷史百分位",
    # ABCD 型態
    "f_n_confirmed":          "ABCD N字型態確認（>0 表示型態成立）",
    "f_n_structure_score":    "ABCD 結構品質分數（0~1，>0.6 高品質）",
    "f_n_bc_retracement":     "BC 段回撤比例（黃金比例 0.382~0.618 最佳）",
    "f_n_ab_gain":            "AB 段漲幅（>0.1 代表第一波有力）",
    "f_n_d_breakout_strength":"D 點突破強度（>0.5 突破有力）",
    "f_n_volume_confirm":     "D 點量能確認（>0.5 突破時有量）",
    # 三角收斂
    "is_human_triangle":  "是否為 human 三角收斂型態（refined+fuzzy 合併）",
    "human_only":         "是否為 fuzzy 三角形（比 strict 更能抓模糊收斂）",
    "is_refined_triangle":"是否為嚴格三角收斂型態",
    "triangle_score":     "三角收斂品質分數（0~1，>0.65 高品質）",
}


# ── 工具定義 ────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_signal",
            "description": "測試條件字串，回傳 hit_rate（10天最高點漲15%的機率）和 sample_count。",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "條件字串，用 & | ~，每個子句加括號"},
                    "hold_days": {"type": "integer", "description": "持有天數，預設 10"},
                    "label_type": {"type": "string", "description": "hit/return/max_return，預設 hit"},
                    "profit_target": {"type": "number", "description": "目標報酬，預設 0.15"},
                },
                "required": ["condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_pattern_signal",
            "description": "偵測 ABCD 或三角收斂型態，統計型態後的報酬。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "abcd 或 triangle"},
                    "hold_days": {"type": "integer", "description": "持有天數，預設 10"},
                    "label_type": {"type": "string", "description": "hit/return/max_return，預設 hit"},
                    "profit_target": {"type": "number", "description": "目標報酬，預設 0.15"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_correlations",
            "description": "掃描指標與目標的相關係數（可選用，不強制）。",
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
            "name": "check_history",
            "description": (
                "查詢本次 session 已測試過的條件記錄（從本機讀取，不耗 token）。"
                "可按指標名稱篩選、按最低命中率篩選、只看通過的條件。"
                "用途：避免重複測試、了解目前最高 hit_rate、確認哪些指標已試過。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "篩選包含此指標名稱的條件（可選）"},
                    "min_hit_rate": {"type": "number", "description": "只回傳 hit_rate 大於此值的條件（可選，預設 0）"},
                    "passed_only": {"type": "boolean", "description": "只回傳通過門檻（hit_rate>=0.35）的條件，預設 false"},
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
    high  = piv("high")
    low   = piv("low")
    vol   = piv("volume")

    ind: dict = {"close": close, "high": high, "low": low, "open": piv("open"), "volume": vol}

    for n in [5, 10, 20, 60, 120]:
        ind[f"ma{n}"] = close.rolling(n).mean()

    for fast, slow, name in [(5, 10, "ma5_x_ma10"), (5, 20, "ma5_x_ma20"), (10, 60, "ma10_x_ma60")]:
        f, s = ind[f"ma{fast}"], ind[f"ma{slow}"]
        ind[name] = (f > s) & (f.shift(1) <= s.shift(1))

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    ind["rsi"] = 100 - 100 / (1 + gain / loss)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"]   = ema12 - ema26
    ind["signal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macdh"]  = ind["macd"] - ind["signal"]

    ind["bb_mid"]   = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    ind["bb_upper"] = ind["bb_mid"] + 2 * bb_std
    ind["bb_lower"] = ind["bb_mid"] - 2 * bb_std
    ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_mid"]

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()]).groupby(level=0).max()
    atr14 = tr.rolling(14).mean()
    ind["atr"]      = atr14
    ind["atr_rank"] = _rolling_rank(atr14)

    high_diff = high.diff()
    low_diff  = -low.diff()
    plus_dm   = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm  = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    plus_di   = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-9)
    minus_di  = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    ind["adx"]      = dx.rolling(14).mean()
    ind["plus_di"]  = plus_di
    ind["minus_di"] = minus_di

    vol_ma20       = vol.rolling(20).mean()
    ind["vol_ratio"] = vol / (vol_ma20 + 1e-9)
    ind["vol_trend"] = vol_ma20 / vol.rolling(60).mean()
    direction        = close.diff().apply(lambda x: x.map(lambda v: 1 if v > 0 else (-1 if v < 0 else 0)))
    ind["obv"]       = (vol * direction).cumsum()
    ind["obv_ma"]    = ind["obv"].rolling(20).mean()

    try:
        from j1stools.abcd_feature import detect_n_shape_features
        df_abcd = detect_n_shape_features(df_p.copy())
        for feat in ["f_n_confirmed", "f_n_structure_score", "f_n_bc_retracement",
                     "f_n_ab_gain", "f_n_d_breakout_strength", "f_n_volume_confirm"]:
            if feat in df_abcd.columns:
                ind[feat] = df_abcd.pivot(index="date", columns="stock_id", values=feat).reindex(close.index)
    except Exception:
        pass

    if _current_hypothesis == "pattern":
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
                    print(f"  [三角偵測] 讀取快取：{os.path.basename(cp)}")
                    df_tri = pd.read_pickle(cp)
                    df_tri["date"] = pd.to_datetime(df_tri["date"])
                    df_tri = add_triangle_compare_columns(df_tri)
                    break
            if df_tri is None:
                print("  [三角偵測] 無快取，重新偵測（較慢）...")
                from j1stools.pattern_triangle import detect_human_triangle
                df_tri = detect_human_triangle(df_p.copy())
                df_tri = add_triangle_compare_columns(df_tri)
            for feat in ["is_human_triangle", "human_only", "is_refined_triangle", "triangle_score"]:
                if feat in df_tri.columns:
                    piv_tri = (
                        df_tri[df_tri["stock_id"].isin(stocks)]
                        .pivot(index="date", columns="stock_id", values=feat)
                        .reindex(close.index)
                    )
                    ind[feat] = piv_tri
        except Exception as e:
            print(f"  [三角偵測] 失敗：{e}")

    try:
        df_ib  = parquet_db.query_ib(stocks, start, end)
        df_ib["date"] = pd.to_datetime(df_ib["date"])
        df_ib["net"]  = df_ib["buy"] - df_ib["sell"]
        nets = []
        for src, col, rank_col in [
            ("Foreign_Investor", "net_foreign", "foreign_rank"),
            ("Investment_Trust", "net_trust",   "trust_rank"),
            ("Dealer_self",      "net_dealer",  "dealer_rank"),
        ]:
            sub = df_ib[df_ib["name"] == src].pivot(index="date", columns="stock_id", values="net").reindex(close.index).fillna(0)
            ind[col]      = sub
            ind[rank_col] = _rolling_rank(sub)
            nets.append(sub)
        inst = sum(nets)
        ind["net_inst"]  = inst
        ind["inst_rank"] = _rolling_rank(inst)
    except Exception:
        pass

    try:
        df_m = parquet_db.query_margin(stocks, start, end)
        df_m["date"] = pd.to_datetime(df_m["date"])
        def _mpiv(col):
            return df_m.pivot(index="date", columns="stock_id", values=col).reindex(close.index).ffill()
        margin = _mpiv("margin_purchase_today_balance")
        short  = _mpiv("short_sale_today_balance")
        ind["margin_bal"]       = margin
        ind["short_bal"]        = short
        ind["margin_chg"]       = margin.pct_change().clip(-1, 1)
        ind["short_chg"]        = short.pct_change().clip(-1, 1)
        ind["short_ratio"]      = short / (margin + 1e-9)
        ind["margin_chg_rank"]  = _rolling_rank(ind["margin_chg"])
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
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        g["f_rsi"]      = 100 - 100 / (1 + gain / loss)
        ema12           = c.ewm(span=12, adjust=False).mean()
        ema26           = c.ewm(span=26, adjust=False).mean()
        macd            = ema12 - ema26
        g["f_macdh"]    = macd - macd.ewm(span=9, adjust=False).mean()
        bb_mid          = c.rolling(20).mean()
        bb_std          = c.rolling(20).std()
        g["f_bb_width"] = (2 * 2 * bb_std) / (bb_mid + 1e-9)
        g["f_bb_pos"]   = (c - bb_mid) / (bb_std + 1e-9)
        tr              = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr             = tr.rolling(14).mean()
        g["f_atr_norm"] = atr / (c + 1e-9)
        g["f_vol_ratio"]= v / (v.rolling(20).mean() + 1e-9)
        g["stock_id"]   = sid
        rows.append(g.reset_index())

    feat_df = pd.concat(rows, ignore_index=True)

    try:
        df_ib = parquet_db.query_ib(stocks, start, end)
        df_ib["date"] = pd.to_datetime(df_ib["date"])
        df_ib["net"]  = df_ib["buy"] - df_ib["sell"]
        for src, col in [("Foreign_Investor", "f_foreign_rank"), ("Investment_Trust", "f_trust_rank")]:
            sub = (
                df_ib[df_ib["name"] == src]
                .pivot(index="date", columns="stock_id", values="net")
                .rolling(60, min_periods=10).rank(pct=True)
                .stack().reset_index()
            )
            sub.columns = ["date", "stock_id", col]
            feat_df = feat_df.merge(sub, on=["date", "stock_id"], how="left")
    except Exception:
        feat_df["f_foreign_rank"] = np.nan
        feat_df["f_trust_rank"]   = np.nan

    try:
        df_m = parquet_db.query_margin(stocks, start, end)
        df_m["date"] = pd.to_datetime(df_m["date"])
        margin = df_m.pivot(index="date", columns="stock_id", values="margin_purchase_today_balance")
        short  = df_m.pivot(index="date", columns="stock_id", values="short_sale_today_balance")
        ratio  = short / (margin + 1e-9)
        ratio_rank = ratio.rolling(60, min_periods=10).rank(pct=True).stack().reset_index()
        ratio_rank.columns = ["date", "stock_id", "f_short_ratio_rank"]
        feat_df = feat_df.merge(ratio_rank, on=["date", "stock_id"], how="left")
    except Exception:
        feat_df["f_short_ratio_rank"] = np.nan

    return feat_df


def _normalize_condition(cond: str) -> str:
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


# ── 全域設定 ─────────────────────────────────────────────────────
from j1stools import parquet_db as _pdb

WATCH_STOCKS  = _pdb.activate_stocks()
PREDICT_START = "2024-01-01"
TRAIN_END     = "2023-12-31"
TRAIN_YEARS   = 8                                          # 可改：測試用 1，正式用 8
TRAIN_START   = f"{int(PREDICT_START[:4]) - TRAIN_YEARS}-01-01"  # 2024-5 = 2019
SESSION_END   = datetime.now().strftime("%Y-%m-%d")
BASE_RESULTS  = os.path.join(os.path.dirname(__file__), "results")

_current_hypothesis: str = "breakout"
_INDICATOR_CACHE: dict = {}  # (start, end) -> (ind, close)，session 內只載一次


def _get_indicators(start: str, end: str):
    key = (start, end)
    if key not in _INDICATOR_CACHE:
        print(f"  [載入指標] {start} ~ {end}（首次，之後從快取）")
        _INDICATOR_CACHE[key] = _build_indicators(WATCH_STOCKS, start, end)
    return _INDICATOR_CACHE[key]


def _results_dir(hypothesis: str) -> str:
    d = os.path.join(BASE_RESULTS, hypothesis)
    os.makedirs(d, exist_ok=True)
    return d


def _signal_log_path(hypothesis: str) -> str:
    return os.path.join(_results_dir(hypothesis), "signal_log.jsonl")


def _backtest_log_path(hypothesis: str) -> str:
    return os.path.join(_results_dir(hypothesis), "backtest_log.jsonl")


def _append_signal_log(entry: dict, hypothesis: str):
    path = _signal_log_path(hypothesis)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_signal_log(hypothesis: str, max_entries: int = 300) -> list[dict]:
    path = _signal_log_path(hypothesis)
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


# ── 工具執行 ─────────────────────────────────────────────────────
def execute_tool(name: str, inputs: dict) -> dict:
    hypothesis = _current_hypothesis
    tool_usage_path = os.path.join(_results_dir(hypothesis), "tool_usage.json")
    try:
        _usage = json.load(open(tool_usage_path)) if os.path.exists(tool_usage_path) else {}
        _usage[name] = _usage.get(name, 0) + 1
        with open(tool_usage_path, "w", encoding="utf-8") as _f:
            json.dump(_usage, _f, ensure_ascii=False)
    except Exception:
        pass

    if name == "scan_correlations":
        hold_days     = inputs.get("hold_days", 10)
        profit_target = inputs.get("profit_target", 0.15)
        top_n         = inputs.get("top_n", 20)
        try:
            ind, close = _get_indicators(TRAIN_START, TRAIN_END)
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}
        future_high  = pd.concat([close.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
        target       = ((future_high / close - 1) >= profit_target).astype(float)
        target_flat  = target.values.flatten().astype(float)
        skip = {"close","high","low","open","volume","atr","bb_upper","bb_lower","bb_mid",
                "macd","signal","net_foreign","net_trust","net_dealer","net_inst","margin_bal","short_bal","obv"}
        corr_list = []
        for feat_name, df in ind.items():
            if feat_name in skip or not isinstance(df, pd.DataFrame):
                continue
            df_aligned = df.reindex(index=close.index, columns=close.columns)
            x    = df_aligned.values.flatten().astype(float)
            mask = ~(np.isnan(x) | np.isnan(target_flat))
            if mask.sum() < 500:
                continue
            corr = float(np.corrcoef(x[mask], target_flat[mask])[0, 1])
            if not np.isnan(corr):
                corr_list.append({"feature": feat_name, "correlation": round(corr, 4)})
        corr_list.sort(key=lambda r: abs(r["correlation"]), reverse=True)
        corr_path = os.path.join(_results_dir(hypothesis), "scan_correlations.json")
        with open(corr_path, "w", encoding="utf-8") as _f:
            json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "correlations": corr_list}, _f, ensure_ascii=False, indent=2)
        return {"hold_days": hold_days, "profit_target": profit_target, "top_correlations": corr_list[:top_n]}

    if name == "check_history":
        indicator   = inputs.get("indicator", "")
        min_hit     = inputs.get("min_hit_rate", 0)
        passed_only = inputs.get("passed_only", False)
        top_n       = inputs.get("top_n", 20)
        entries     = _load_signal_log(hypothesis, max_entries=9999)
        if passed_only:
            entries = [e for e in entries if e.get("hit_rate", 0) >= 0.35]
        if min_hit > 0:
            entries = [e for e in entries if e.get("hit_rate", 0) >= min_hit]
        if indicator:
            entries = [e for e in entries if indicator in e.get("condition", "")]
        entries.sort(key=lambda x: -x.get("hit_rate", 0))
        total_tested = len(_load_signal_log(hypothesis, max_entries=9999))
        passed_count = len([e for e in _load_signal_log(hypothesis, max_entries=9999) if e.get("hit_rate", 0) >= 0.35])
        return {
            "total_tested": total_tested,
            "passed_035": passed_count,
            "filtered_count": len(entries),
            "conditions": [
                {
                    "condition": e.get("condition", ""),
                    "hit_rate": e.get("hit_rate"),
                    "sample_count": e.get("sample_count"),
                }
                for e in entries[:top_n]
            ],
        }

    if name == "analyze_signal":
        hold_days     = inputs.get("hold_days", 10)
        condition     = _normalize_condition(inputs["condition"])
        label_type    = inputs.get("label_type", "hit")
        profit_target = inputs.get("profit_target", 0.15)
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
        mask = signal.fillna(False).astype(bool)
        future_high = pd.concat([close.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
        if label_type == "hit":
            labels = (future_high / close - 1) >= profit_target
        elif label_type == "max_return":
            labels = future_high / close - 1
        else:
            labels = close.shift(-hold_days) / close - 1
        values = labels[mask].values.flatten()
        values = values[~pd.isnull(values)]
        n = len(values)
        if n < 20:
            return {"error": f"樣本太少（{n} 筆）"}
        if label_type == "hit":
            hit_rate = float(np.mean(values.astype(float)))
            out = {"condition": condition, "label_type": label_type, "hold_days": hold_days,
                   "sample_count": n, "hit_rate": round(hit_rate, 4), "profit_target": profit_target}
            _append_signal_log({**out, "date": datetime.now().strftime("%Y-%m-%d")}, hypothesis)
            return out
        values = values.astype(float)
        out = {"condition": condition, "label_type": label_type, "hold_days": hold_days, "sample_count": n,
               "avg_return": round(float(np.mean(values)), 4), "win_rate": round(float((values > 0).mean()), 4),
               "std": round(float(np.std(values)), 4)}
        _append_signal_log({**out, "date": datetime.now().strftime("%Y-%m-%d")}, hypothesis)
        return out

    if name == "analyze_pattern_signal":
        import random as _random
        pattern    = inputs.get("pattern", "abcd")
        hold_days  = inputs.get("hold_days", 10)
        label_type = inputs.get("label_type", "hit")
        profit_target = inputs.get("profit_target", 0.15)
        from j1stools import parquet_db
        stocks_sample = WATCH_STOCKS if pattern == "abcd" else _random.sample(WATCH_STOCKS, min(100, len(WATCH_STOCKS)))
        try:
            df_p = parquet_db.query_price(stocks_sample, TRAIN_START, TRAIN_END)
            df_p["date"] = pd.to_datetime(df_p["date"])
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}
        close_wide  = df_p.pivot(index="date", columns="stock_id", values="close").sort_index()
        signal_mask = pd.DataFrame(False, index=close_wide.index, columns=close_wide.columns)
        if pattern == "abcd":
            try:
                from j1stools.abcd_feature import detect_n_shape_features
                df_abcd   = detect_n_shape_features(df_p.copy())
                confirmed = df_abcd.pivot(index="date", columns="stock_id", values="f_n_confirmed").reindex(close_wide.index).fillna(0)
                signal_mask = confirmed.astype(bool)
            except Exception as e:
                return {"error": f"ABCD 偵測失敗：{e}"}
        elif pattern == "triangle":
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
                if df_tri is None:
                    from j1stools.pattern_triangle import find_human_triangle
                    df_tri = find_human_triangle(df_p.copy())
                df_tri_sig = df_tri[df_tri["human_only"].fillna(False)]
                for _, row in df_tri_sig.iterrows():
                    d, s = row["date"], row["stock_id"]
                    if d in signal_mask.index and s in signal_mask.columns:
                        signal_mask.at[d, s] = True
            except Exception as e:
                return {"error": f"三角收斂偵測失敗：{e}"}
        future_high = pd.concat([close_wide.shift(-i) for i in range(1, hold_days + 1)], axis=0).groupby(level=0).max()
        if label_type == "hit":
            labels = (future_high / close_wide - 1) >= profit_target
        else:
            labels = close_wide.shift(-hold_days) / close_wide - 1
        values = labels[signal_mask].values.flatten()
        values = values[~pd.isnull(values)]
        n = len(values)
        if n < 20:
            return {"error": f"樣本太少（{n} 筆）"}
        if label_type == "hit":
            return {"pattern": pattern, "hold_days": hold_days, "sample_count": n,
                    "hit_rate": round(float(np.mean(values.astype(float))), 4)}
        values = values.astype(float)
        return {"pattern": pattern, "hold_days": hold_days, "sample_count": n,
                "avg_return": round(float(np.mean(values)), 4),
                "win_rate": round(float((values > 0).mean()), 4)}

    if name == "run_backtest":
        from j1stools.backtest_engine import backtest_engine
        condition = _normalize_condition(inputs["condition"])
        hold_days = inputs.get("hold_days", 10)
        sl_stop   = inputs.get("sl_stop", 0.08)
        tp_stop   = inputs.get("tp_stop", 0.15)
        start     = inputs.get("start", PREDICT_START)
        end       = inputs.get("end", SESSION_END)
        # 往前多載 120 天讓指標有 warmup，避免 MA60/RSI 等計算錯誤觸發假信號
        warmup_start = (pd.Timestamp(start) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
        signal_start  = pd.Timestamp(start)
        try:
            ind, close = _build_indicators(WATCH_STOCKS, warmup_start, end)
        except Exception as e:
            return {"error": f"資料載入失敗：{e}"}
        try:
            entries = eval(condition, {"__builtins__": {}}, ind)
        except Exception as e:
            return {"error": f"條件解析失敗：{e}"}
        if not isinstance(entries, pd.DataFrame):
            return {"error": "條件必須回傳 DataFrame"}
        entries = entries.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        # warmup 期間不允許開倉
        entries.loc[entries.index < signal_start] = False
        close   = close.loc[close.index >= signal_start]
        entries = entries.loc[entries.index >= signal_start]
        exits       = pd.Series(False, index=close.index)
        df_proba    = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        stock_group = {s: "default" for s in close.columns}
        try:
            portfolio_value, trades_df, _ = backtest_engine(
                close=close, entries=entries, exits=exits, df_proba=df_proba,
                stock_group=stock_group, hold_days=hold_days, sl_stop=sl_stop, tp_stop=tp_stop,
                use_fixed_sl=True, use_fixed_tp=True, use_sl_trail=False, use_hold_days=True,
            )
        except Exception as e:
            return {"error": f"回測失敗：{e}"}
        if len(trades_df) < 5:
            return {"error": f"交易筆數不足（{len(trades_df)} 筆）"}
        pv           = portfolio_value.dropna()
        rets         = pv.pct_change(fill_method=None).dropna()
        total_return = round(float(pv.iloc[-1] / pv.iloc[0] - 1), 4) if len(pv) > 1 else None
        sharpe       = round(float(rets.mean() / rets.std() * (252**0.5)) if rets.std() > 0 else 0, 2)
        max_dd       = round(float(((pv / pv.cummax()) - 1).min()), 4)
        rp           = trades_df.get("return_pct", pd.Series(dtype=float))
        win_rate     = round(float((rp > 0).mean()), 4) if len(rp) else None
        avg_return   = round(float(rp.mean()), 4) if len(rp) else None
        avg_win      = round(float(rp[rp > 0].mean()), 4) if (rp > 0).any() else 0
        avg_loss     = round(float(rp[rp < 0].mean()), 4) if (rp < 0).any() else 0
        chart_path   = None
        try:
            from j1stools.j1s_chart import plot_performance
            import plotly.graph_objects as go, json as _json
            pv_clean   = portfolio_value.replace([np.inf, -np.inf], np.nan).dropna()
            fig_json   = plot_performance(pv_clean, trades_df, is_web=True)
            fig        = go.Figure(_json.loads(fig_json))
            results_d  = _results_dir(hypothesis)
            slug       = condition[:60].replace(" ", "").replace("(","").replace(")","").replace(">","gt").replace("<","lt").replace("&","_")
            chart_path = os.path.join(results_d, f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.html")
            fig.write_html(chart_path)
        except Exception:
            pass
        return {"condition": condition, "period": f"{start}~{end}", "hold_days": hold_days,
                "total_trades": len(trades_df), "total_return": total_return, "sharpe": sharpe,
                "max_drawdown": max_dd, "win_rate": win_rate, "avg_return": avg_return,
                "avg_win": avg_win, "avg_loss": avg_loss, "chart": chart_path}

    return {"error": f"未知工具：{name}"}


# ── 系統 Prompt（根據假設動態生成）────────────────────────────────
def _build_system_prompt(hypothesis: str) -> str:
    cfg        = HYPOTHESIS_CONFIGS.get(hypothesis, HYPOTHESIS_CONFIGS["breakout"])
    categories = cfg.get("indicator_categories", {})

    # 把分類格式化成 prompt 區塊
    cat_lines = []
    for cat_name, indicators in categories.items():
        cat_lines.append(f"  【{cat_name}】{', '.join(indicators)}")
    cat_block = "\n".join(cat_lines)
    n_cats = len(categories)

    return f"""你是台灣股票量化研究助理，這次的任務是為「{cfg['name']}」建立專屬選股模型的「特徵候選集」。

━━ 你的角色 ━━
你不是在找「最強的預測條件」，而是在做「特徵工程」。
每個通過篩選的條件，最後會變成模型的一個 binary 特徵（今天符合 = 1，不符合 = 0）。
模型用這些特徵的組合來辨識「{cfg['name']}」的機會。

━━ 本次假設 ━━
{cfg['description']}

━━ 指標分類（共 {n_cats} 類，每類各找 2~3 個通過條件）━━
{cat_block}

工作規則：
1. 依序從每個類別選 2~3 個指標，組合成條件
2. 呼叫 analyze_signal(label_type="hit", hold_days=10, profit_target=0.15)
3. 篩選標準：hit_rate >= 0.35 且 sample_count >= 100
4. 某個類別找到 2~3 個通過條件後，立刻換下一個類別
5. 所有類別都探索完後才可以停止

━━ 探索提示 ━━
{cfg['explore_hint']}

━━ 可用指標（完整清單）━━
報價: close,high,low,open,volume
均線: ma5,ma10,ma20,ma60,ma120；交叉: ma5_x_ma10,ma5_x_ma20,ma10_x_ma60
動能: rsi,macd,macdh,bb_width,bb_upper,bb_lower,bb_mid
趨勢: adx(>25有趨勢),plus_di,minus_di
波動: atr,atr_rank(0~1百分位)
量能: vol_ratio(今量/20均),vol_trend,obv,obv_ma
ABCD: f_n_confirmed,f_n_structure_score,f_n_ab_gain,f_n_d_breakout_strength,f_n_volume_confirm
IB: net_foreign,net_trust,net_dealer,net_inst,foreign_rank,trust_rank,dealer_rank,inst_rank
Margin: margin_bal,short_bal,margin_chg,short_chg,short_ratio,margin_chg_rank,short_ratio_rank

━━ 關鍵技巧 ━━
- sample_count > 5000 時 hit_rate 通常低，加嚴門檻縮小樣本（500~3000 最佳）
- 統計資料期間：{TRAIN_START} ~ {TRAIN_END}（近 5 年）
- 不需要先呼叫 scan_correlations

回測與模型訓練由外部程式處理，你只負責蒐集特徵候選條件。
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
    if fn_name == "run_backtest":
        return (f"return={result.get('total_return')} sharpe={result.get('sharpe')} "
                f"mdd={result.get('max_drawdown')} win={result.get('win_rate')} "
                f"trades={result.get('total_trades')} period={result.get('period')}")
    if fn_name == "scan_correlations":
        top = result.get("top_correlations", [])[:8]
        return "top: " + " ".join(f"{r['feature']}({r['correlation']})" for r in top)
    if fn_name == "analyze_pattern_signal":
        if "hit_rate" in result:
            return f"pattern={result.get('pattern')} hit={result['hit_rate']} n={result.get('sample_count')}"
        return f"pattern={result.get('pattern')} avg={result.get('avg_return')} win={result.get('win_rate')} n={result.get('sample_count')}"
    return json.dumps(result, ensure_ascii=False)[:200]


def _trim_messages(messages: list, keep_last: int = 20) -> list:
    if len(messages) <= 2:
        return messages
    fixed   = messages[:2]
    rest    = messages[2:]
    if len(rest) <= keep_last:
        return messages
    trimmed = rest[-keep_last:]
    while trimmed:
        msg  = trimmed[0]
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "tool":
            trimmed = trimmed[1:]
        else:
            break
    return fixed + trimmed


def _shorten_cond(cond: str, max_len: int = 70) -> str:
    return cond if len(cond) <= max_len else "..." + cond[-max_len:]


def _build_history_prompt(entries: list[dict]) -> str:
    if not entries:
        return ""
    passed = [e for e in entries if e.get("hit_rate", 0) >= 0.35]
    failed = [e for e in entries if e.get("hit_rate", 0) < 0.35 and "condition" in e]
    lines  = ["\n━━ 歷史紀錄（避免重複，探索新方向）━━"]
    if passed:
        lines.append(f"\n✅ 通過({len(passed)}筆)：")
        for e in passed[-5:]:
            lines.append(f"  hit={e['hit_rate']:.3f} n={e.get('sample_count','?')} {_shorten_cond(e['condition'])}")
    if failed:
        lines.append(f"\n❌ 未通過({len(failed)}筆，勿重複)：")
        seen = set()
        for e in failed[-15:]:
            cond = e["condition"]
            if cond not in seen:
                seen.add(cond)
                lines.append(f"  {_shorten_cond(cond, 50)}")
    return "\n".join(lines)


# ── Agent 主循環 ─────────────────────────────────────────────────
def run_agent(hypothesis: str = "breakout", max_calls: int = 0, model_override: str = "") -> str:
    """
    max_calls: 最多幾次 tool 呼叫就停（0 = 不限，用於測試）
    model_override: 強制使用指定模型（用於測試切換免費模型）
    """
    global _current_hypothesis, _model_idx
    _current_hypothesis = hypothesis
    _model_idx = 0

    if model_override:
        for i, (m, _) in enumerate(MODELS):
            if model_override in m:
                _model_idx = i
                break

    # 新 session 開始前備份舊 signal_log，避免新舊條件混入
    sig_path = _signal_log_path(hypothesis)
    if os.path.exists(sig_path):
        bak_path = sig_path + ".bak"
        import shutil
        shutil.copy2(sig_path, bak_path)
        os.remove(sig_path)
        print(f"  [session] 舊 signal_log 已備份為 .bak，重新開始記錄")

    cfg  = HYPOTHESIS_CONFIGS.get(hypothesis, HYPOTHESIS_CONFIGS["breakout"])
    task = (
        f"為「{cfg['name']}」假設蒐集特徵候選條件（用於建立選股模型）。\n"
        f"目標：hit_rate >= 0.35 且 sample_count >= 100（{TRAIN_START}~{TRAIN_END}）。\n"
        f"{'（測試模式：快速探索幾個維度即可）' if max_calls else ''}\n"
        f"提示：可先呼叫 check_history() 查看已測試記錄，避免重複。"
    )

    messages      = [{"role": "system", "content": _build_system_prompt(hypothesis)},
                     {"role": "user",   "content": task}]
    final_text    = ""
    _rate_limit_count = 0
    _call_count   = 0
    _tool_usage: dict = {}   # tool 名稱 → 呼叫次數

    def _get_tools_with_usage() -> list:
        import copy as _copy
        sig_entries  = _load_signal_log(hypothesis, max_entries=9999)
        n_tested     = len(sig_entries)
        n_passed     = len([e for e in sig_entries if e.get("hit_rate", 0) >= 0.35])
        result = _copy.deepcopy(TOOLS)
        for tool in result:
            name  = tool["function"]["name"]
            count = _tool_usage.get(name, 0)
            notes = []
            if count > 0:
                notes.append(f"已呼叫 {count} 次")
            if name == "analyze_signal":
                notes.append(f"本 session 已測 {n_tested} 個條件，{n_passed} 個通過")
            if name == "scan_correlations" and count >= 1:
                notes.append("本 session 無需再呼叫")
            if notes:
                tool["function"]["description"] += f"（{'；'.join(notes)}）"
        return result

    print(f"\n[{cfg['name']}] 開始探索... {'（測試：最多 ' + str(max_calls) + ' 次工具呼叫）' if max_calls else ''}\n{'─' * 60}")

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
                    if not _next_model():
                        break
                    continue
            # llama 的舊格式 tool call：<function=name,{args}> 或 tool name 混入 args
            if "tool_use_failed" in err or "tool call validation failed" in err:
                import re as _re
                # 嘗試從錯誤訊息解析出 tool name 和 args
                m = _re.search(r"attempted to call tool '([^,{]+)[,{]([^']*)'", err)
                if not m:
                    m = _re.search(r"<function=([^,>]+)[,>](\{[^<]*\})?", err)
                if m:
                    _fn  = m.group(1).strip()
                    _raw = m.group(2) or "{}"
                    try:
                        _args = json.loads(_raw if _raw.startswith("{") else "{}")
                    except Exception:
                        _args = {}
                    print(f"\n[llama修復] 解析舊格式 tool call：{_fn}({_args})")
                    _result = execute_tool(_fn, _args)
                    _compressed = _compress_result(_fn, _result)
                    print(f"  {_compressed}")
                    # 補一個假的 assistant + tool 訊息讓對話繼續
                    import uuid as _uuid
                    _fake_id = f"fix_{_uuid.uuid4().hex[:8]}"
                    messages.append({"role": "assistant", "content": None, "tool_calls": [
                        {"id": _fake_id, "type": "function", "function": {"name": _fn, "arguments": json.dumps(_args)}}
                    ]})
                    messages.append({"role": "tool", "tool_call_id": _fake_id, "content": _compressed})
                    _call_count += 1
                    if max_calls and _call_count >= max_calls:
                        print(f"\n[測試] 已達 {max_calls} 次工具呼叫上限，停止")
                        return final_text
                    continue
                if not _next_model():
                    break
                continue
            raise

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            final_text = msg.content or ""
            print(f"\n{'=' * 60}\nAgent 報告：\n{final_text}")
            # 存 Agent 最終報告供報表使用
            try:
                agent_report_path = os.path.join(_results_dir(hypothesis), "agent_report.txt")
                with open(agent_report_path, "w", encoding="utf-8") as _f:
                    _f.write(final_text)
            except Exception:
                pass
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"→ {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")
            result     = execute_tool(fn_name, fn_args)
            compressed = _compress_result(fn_name, result)
            print(f"  {compressed}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": compressed})
            _tool_usage[fn_name] = _tool_usage.get(fn_name, 0) + 1
            _call_count += 1
            if max_calls and _call_count >= max_calls:
                print(f"\n[測試] 已達 {max_calls} 次工具呼叫上限，停止")
                return final_text

    return final_text


# ── 驗證與訓練 ────────────────────────────────────────────────────
def _canonical_condition(cond: str) -> str:
    parts = [p.strip() for p in cond.split("&")]
    return " & ".join(sorted(parts))


def _load_backtested(hypothesis: str) -> set:
    path = _backtest_log_path(hypothesis)
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(_canonical_condition(json.loads(line.strip()).get("condition", "")))
            except Exception:
                pass
    return done


def validate_and_train(hypothesis: str = "breakout"):
    """
    回測 + RFC 快速驗證，使用 2024-2026 資料（out-of-sample）。

    流程：
      signal_log（訓練期統計通過） → 回測（預測期 2024~今）→ RFC 驗證 → 報表
    """
    global _current_hypothesis
    _current_hypothesis = hypothesis

    passing = [
        e for e in _load_signal_log(hypothesis)
        if e.get("hit_rate", 0) >= 0.35 and e.get("sample_count", 0) >= 100
    ]
    if not passing:
        print(f"[{hypothesis}] 沒有通過的條件，跳過驗證")
        return

    done     = _load_backtested(hypothesis)
    seen     = set()
    new_pass = []
    for e in passing:
        key = _canonical_condition(e.get("condition", ""))
        if key not in done and key not in seen:
            seen.add(key)
            new_pass.append(e)

    if not new_pass:
        print(f"[{hypothesis}] 所有 {len(passing)} 個條件已回測，跳過回測，直接進入 RFC 訓練")

    print(f"\n[{hypothesis}] 回測 {len(new_pass)} 個新條件（預測期：{PREDICT_START}~{SESSION_END}）...")
    backtest_log = _backtest_log_path(hypothesis)

    # 預先載入預測期指標（計算 hit_rate_predict 用）
    try:
        ind_pred, close_pred = _get_indicators(PREDICT_START, SESSION_END)
        future_high_pred = pd.concat(
            [close_pred.shift(-i) for i in range(1, 11)], axis=0
        ).groupby(level=0).max()
    except Exception:
        ind_pred, close_pred, future_high_pred = None, None, None

    for entry in new_pass:
        cond = entry["condition"]
        print(f"\n→ 回測：{cond}")
        bt = execute_tool("run_backtest", {
            "condition": cond,
            "hold_days": 10,
            "sl_stop":   0.08,
            "tp_stop":   0.15,
            "start":     PREDICT_START,
            "end":       SESSION_END,
        })
        # 計算預測期 hit_rate（同訓練期邏輯，純統計無停損）
        hit_rate_predict = None
        if ind_pred is not None:
            try:
                sig  = eval(cond, {"__builtins__": {}}, ind_pred)
                mask = sig.fillna(False).astype(bool)
                vals = ((future_high_pred / close_pred - 1) >= 0.15)[mask].values.flatten()
                vals = vals[~pd.isnull(vals)]
                if len(vals) >= 10:
                    hit_rate_predict = round(float(np.mean(vals.astype(float))), 4)
            except Exception:
                pass
        print(f"  hit_predict={hit_rate_predict} {json.dumps({k: v for k, v in bt.items() if k != 'chart'}, ensure_ascii=False)}")
        with open(backtest_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "condition":       cond,
                "hit_rate":        entry.get("hit_rate"),
                "hit_rate_predict": hit_rate_predict,
                "sample_count":    entry.get("sample_count"),
                "backtest":        bt,
                "date":            datetime.now().strftime("%Y-%m-%d"),
            }, ensure_ascii=False) + "\n")

    # RFC 快速驗證
    print(f"\n[{hypothesis}] 訓練 RFC（快速驗證用）...")
    import random as _random
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    try:
        n_stocks = 200
        stocks   = _random.sample(WATCH_STOCKS, min(n_stocks, len(WATCH_STOCKS)))
        # 載完整資料（train+predict），才能做 train/test split
        feat_df  = _build_rfc_features(stocks, TRAIN_START, SESSION_END)

        close_wide  = feat_df.pivot(index="date", columns="stock_id", values="close")
        future_max  = pd.concat([close_wide.shift(-i) for i in range(1, 11)], axis=0).groupby(level=0).max()
        target_wide = ((future_max / close_wide - 1) >= 0.15).astype(int)
        target_long = target_wide.stack().reset_index()
        target_long.columns = ["date", "stock_id", "target"]
        feat_df = feat_df.merge(target_long, on=["date", "stock_id"], how="left")

        # 把通過條件加成 binary 特徵（ind 需要完整期間才能 eval）
        ind, _ = _build_indicators(stocks, TRAIN_START, SESSION_END)
        added  = []
        for i, e in enumerate(sorted(passing, key=lambda x: x["hit_rate"], reverse=True)[:20]):
            col = f"f_signal_{i}"
            try:
                sig     = eval(e["condition"], {"__builtins__": {}}, ind)
                sig_long = sig.astype(float).stack().reset_index()
                sig_long.columns = ["date", "stock_id", col]
                feat_df = feat_df.merge(sig_long, on=["date", "stock_id"], how="left")
                feat_df[col] = feat_df[col].fillna(0)
                added.append(col)
            except Exception:
                pass

        tech_cols = [c for c in feat_df.columns if c.startswith("f_") and c not in added]
        all_cols  = tech_cols + added
        feat_df   = feat_df.dropna(subset=tech_cols + ["target"])

        # 固定切點：2024-01-01
        cutoff = pd.Timestamp(PREDICT_START)
        train  = feat_df[feat_df["date"] < cutoff]
        test   = feat_df[feat_df["date"] >= cutoff]

        if len(train) < 200 or len(test) < 50:
            print(f"  資料不足：train={len(train)}, test={len(test)}")
        else:
            actual    = test["target"].values
            base_rate = round(float(actual.mean()), 4)

            def _eval_rfc(feat_cols, model_name):
                rfc = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
                rfc.fit(train[feat_cols], train["target"])
                proba = rfc.predict_proba(test[feat_cols])[:, 1]
                thresholds = {}
                for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    mask = proba >= th
                    n    = int(mask.sum())
                    thresholds[f"proba>={th}"] = (
                        {"n": n, "hit_rate": round(float(actual[mask].mean()), 4)} if n >= 10
                        else {"n": n, "note": "樣本不足"}
                    )
                importance = sorted(zip(feat_cols, rfc.feature_importances_), key=lambda x: x[1], reverse=True)
                top_feats  = [{"feature": f, "importance": round(float(v), 4)} for f, v in importance[:10]]
                model_path = os.path.join(_results_dir(hypothesis), f"rfc_{model_name}.joblib")
                joblib.dump(rfc, model_path)
                print(f"  [{model_name}] base={base_rate} {thresholds}")
                return {"thresholds": thresholds, "top_features": top_feats, "model_path": model_path}

            rfc_a = _eval_rfc(tech_cols, "tech_only")
            rfc_b = _eval_rfc(all_cols,  "tech_signal")

            rfc_result = {
                "hypothesis":    hypothesis,
                "base_rate":     base_rate,
                "train_size":    len(train),
                "test_size":     len(test),
                "train_period":  f"{TRAIN_START} ~ {TRAIN_END}",
                "test_period":   f"{PREDICT_START} ~ {SESSION_END}",
                "model_a":       {"name": "技術指標",                    **rfc_a},
                "model_b":       {"name": f"技術指標+top{len(added)}信號", **rfc_b},
                "date":          datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            rfc_path = os.path.join(_results_dir(hypothesis), "rfc_result.json")
            with open(rfc_path, "w", encoding="utf-8") as f:
                json.dump(rfc_result, f, ensure_ascii=False, indent=2)
            print(f"  RFC 結果已存：{rfc_path}")

    except Exception as e:
        print(f"  RFC 訓練失敗：{e}")

    _generate_report(hypothesis)


# ── 報表 ─────────────────────────────────────────────────────────
def _generate_report(hypothesis: str):
    results_d    = _results_dir(hypothesis)
    backtest_log = _backtest_log_path(hypothesis)
    cfg          = HYPOTHESIS_CONFIGS.get(hypothesis, {})
    hyp_name     = cfg.get("name", hypothesis)

    if not os.path.exists(backtest_log):
        print(f"[報表] 無回測資料")
        return

    rows = []
    with open(backtest_log, encoding="utf-8") as f:
        for line in f:
            try:
                e  = json.loads(line)
                bt = e.get("backtest", {})
                if "error" in bt or bt.get("total_trades", 0) < 10:
                    continue
                rows.append({
                    "condition":        e["condition"],
                    "hit_rate":         e.get("hit_rate", 0),
                    "hit_rate_predict": e.get("hit_rate_predict"),
                    "sample_count":     e.get("sample_count", 0),
                    "trades":           bt.get("total_trades"),
                    "total_return":     bt.get("total_return", 0),
                    "sharpe":           bt.get("sharpe", 0),
                    "max_drawdown":     bt.get("max_drawdown", 0),
                    "win_rate":         bt.get("win_rate", 0),
                    "avg_win":          bt.get("avg_win", 0),
                    "avg_loss":         bt.get("avg_loss", 0),
                    "chart":            bt.get("chart"),
                })
            except Exception:
                continue

    if not rows:
        print("[報表] 無有效資料")
        return

    rows.sort(key=lambda x: x["sharpe"], reverse=True)

    def _sc(s):
        if s >= 2.0: return "#1a7a3a"
        if s >= 1.5: return "#2ecc71"
        if s >= 1.0: return "#f39c12"
        if s >= 0:   return "#e67e22"
        return "#e74c3c"

    rows_html = ""
    for i, r in enumerate(rows):
        chart_link = f'<a href="{r["chart"]}" target="_blank">📊</a>' if r.get("chart") else ""
        dd_color   = "#e74c3c" if r["max_drawdown"] < -0.15 else "#e67e22" if r["max_drawdown"] < -0.10 else "#2ecc71"
        hr_pred    = r.get("hit_rate_predict")
        hr_pred_str = f"{hr_pred:.1%}" if hr_pred is not None else "—"
        # 若訓練命中率大幅高於驗證命中率，標紅警示
        hr_pred_color = ""
        if hr_pred is not None and r["hit_rate"] - hr_pred > 0.10:
            hr_pred_color = 'style="color:#e74c3c"'
        rows_html += (
            f'<tr>'
            f'<td style="text-align:center">{i+1}</td>'
            f'<td style="font-family:monospace;font-size:12px;white-space:nowrap">{r["condition"]}</td>'
            f'<td style="text-align:center;color:{_sc(r["sharpe"])};font-weight:bold">{r["sharpe"]:.2f}</td>'
            f'<td style="text-align:center">{r["win_rate"]:.1%}</td>'
            f'<td style="text-align:center">{r["total_return"]:+.2%}</td>'
            f'<td style="text-align:center;color:{dd_color}">{r["max_drawdown"]:.1%}</td>'
            f'<td style="text-align:center">{r["trades"]}</td>'
            f'<td style="text-align:center">{r["hit_rate"]:.1%}</td>'
            f'<td style="text-align:center" {hr_pred_color}>{hr_pred_str}</td>'
            f'<td style="text-align:center">{r["sample_count"]}</td>'
            f'<td style="text-align:center">{chart_link}</td>'
            f'</tr>'
        )

    # RFC 區塊
    rfc_html = ""
    rfc_path = os.path.join(results_d, "rfc_result.json")
    if os.path.exists(rfc_path):
        with open(rfc_path, encoding="utf-8") as f:
            rfc = json.load(f)
        train_p = rfc.get("train_period", "")
        test_p  = rfc.get("test_period", "")
        base    = rfc.get("base_rate", "?")
        rfc_html = f'<p>訓練期：{train_p}　驗證期：{test_p}　基礎命中率：{base:.1%}</p>'
        for model_key in ["model_a", "model_b"]:
            m = rfc.get(model_key, {})
            if not m:
                continue
            th_rows = ""
            for th, v in m.get("thresholds", {}).items():
                n = v.get("n", 0)
                if "hit_rate" in v:
                    hr = v["hit_rate"]
                    hr_color = "#2ecc71" if hr >= 0.5 else "#f39c12" if hr >= 0.35 else "#e74c3c"
                    th_rows += (
                        f'<tr>'
                        f'<td style="text-align:center">{th}</td>'
                        f'<td style="text-align:center">{n}</td>'
                        f'<td style="text-align:center;color:{hr_color};font-weight:bold">{hr:.1%}</td>'
                        f'</tr>'
                    )
                else:
                    th_rows += (
                        f'<tr>'
                        f'<td style="text-align:center">{th}</td>'
                        f'<td style="text-align:center;color:#555">{n}</td>'
                        f'<td style="text-align:center;color:#555">樣本不足</td>'
                        f'</tr>'
                    )
            feat_str = ", ".join(f['feature'] for f in m.get("top_features", [])[:5])
            rfc_html += (
                f'<h4>{m.get("name","")}</h4>'
                f'<table border="1" cellpadding="4" style="width:300px">'
                f'<tr><th>門檻</th><th>樣本數</th><th>命中率</th></tr>'
                f'{th_rows}</table>'
                f'<p>重要特徵：{feat_str}</p>'
            )

    # ── 指標使用統計（from signal_log，含未通過的）────────────────
    import re as _re
    all_entries  = _load_signal_log(hypothesis, max_entries=9999)
    ind_counter: dict = {}
    for e in all_entries:
        cond = e.get("condition", "")
        for t in _re.findall(r"([a-z_]+)\s*[><!=]", cond):
            ind_counter[t] = ind_counter.get(t, 0) + 1
        for t in _re.findall(r"\(\s*([a-z_]+)\s*\)", cond):
            ind_counter[t] = ind_counter.get(t, 0) + 1
    ind_sorted   = sorted(ind_counter.items(), key=lambda x: -x[1])
    max_ind_cnt  = ind_sorted[0][1] if ind_sorted else 1
    total_tested = len(all_entries)
    passed_count = len([e for e in all_entries if e.get("hit_rate", 0) >= 0.35])

    ind_rows_html = ""
    key_inds = set(cfg.get("key_indicators", []))
    for ind_name, cnt in ind_sorted:
        pct      = cnt / max_ind_cnt * 100
        is_key   = ind_name in key_inds
        color    = "#4fc3f7" if is_key else "#aaa"
        badge    = ' <span style="font-size:10px;color:#f39c12">★重點</span>' if is_key else ""
        desc     = INDICATOR_DESCRIPTIONS.get(ind_name, "")
        desc_html = f'<span style="color:#666;font-size:12px;margin-left:8px">{desc}</span>' if desc else ""
        ind_rows_html += (
            f'<tr>'
            f'<td style="font-family:monospace;color:{color}">{ind_name}{badge}{desc_html}</td>'
            f'<td style="text-align:center">{cnt}</td>'
            f'<td><div style="background:#3498db;height:12px;width:{pct:.0f}%;border-radius:3px;min-width:4px"></div></td>'
            f'</tr>'
        )

    # ── 類別覆蓋摘要（通過條件中各類別出現幾次）──────────────────
    categories     = cfg.get("indicator_categories", {})
    passing_entries = [e for e in all_entries if e.get("hit_rate", 0) >= 0.35]
    cat_coverage: dict = {cat: 0 for cat in categories}
    for e in passing_entries:
        cond = e.get("condition", "")
        for cat, inds in categories.items():
            if any(ind in cond for ind in inds):
                cat_coverage[cat] += 1
    dim_rows_html = ""
    target_per_cat = 2
    for cat, cnt in cat_coverage.items():
        inds_str   = ", ".join(categories[cat])
        status     = "✅" if cnt >= target_per_cat else "⚠️" if cnt >= 1 else "⬜"
        bar_color  = "#2ecc71" if cnt >= target_per_cat else "#f39c12" if cnt >= 1 else "#555"
        bar_w      = min(cnt * 25, 100)
        dim_rows_html += (
            f'<tr>'
            f'<td style="font-weight:bold">{status} {cat}</td>'
            f'<td style="font-family:monospace;font-size:11px;color:#888">{inds_str}</td>'
            f'<td style="text-align:center">{cnt} 個</td>'
            f'<td><div style="background:{bar_color};height:12px;width:{bar_w}%;border-radius:3px;min-width:4px"></div></td>'
            f'</tr>'
        )

    # ── Agent 最終報告文字 ─────────────────────────────────────────
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
<title>{hyp_name} 模型報表</title>
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
<h1>🎯 {hyp_name} — 專屬模型報表</h1>
<p>假設：{cfg.get('description','')}</p>
<p>統計期：{TRAIN_START} ~ {TRAIN_END}　回測期：{PREDICT_START} ~ {SESSION_END}</p>
<p>Agent 共測試 <b>{total_tested}</b> 個條件，通過 <b>{passed_count}</b> 個（{passed_count/max(total_tested,1):.1%}），回測有效 <b>{len(rows)}</b> 個</p>

<h2>類別覆蓋摘要</h2>
<p style="color:#aaa;font-size:13px">✅ = 達標（≥2個通過條件）　⚠️ = 部分（1個）　⬜ = 未覆蓋</p>
<table style="width:80%">
<tr><th>類別</th><th>指標</th><th>通過條件數</th><th>覆蓋度</th></tr>
{dim_rows_html}
</table>

<h2>指標使用統計（Agent 探索的指標）</h2>
<p style="color:#aaa;font-size:13px">★重點 = 本假設的重點指標；統計含未通過的 {total_tested} 個測試條件</p>
<table style="width:60%">
<tr><th>指標</th><th>使用次數</th><th>頻率</th></tr>
{ind_rows_html}
</table>

<h2>回測結果（2024-2026 out-of-sample）</h2>
<p style="color:#aaa;font-size:13px">
  <b>獲利率</b>＝回測中出場報酬 &gt; 0 的比例（含停損限制）
  <b>命中率</b>＝訓練期條件成立後 10 天最高點漲 15% 的比例（純統計，無停損）
</p>
<table>
<tr>
  <th>#</th><th>條件</th><th>Sharpe</th>
  <th title="回測出場報酬>0的比例（含停損）">獲利率<br><small>回測</small></th>
  <th title="總累積報酬">總報酬</th>
  <th>最大回撤</th><th>交易次數</th>
  <th title="訓練期：條件成立後10天最高點漲15%的比例（純統計，無停損）">命中率<br><small>訓練期</small></th>
  <th title="驗證期：同樣統計方式（紅色=比訓練期低10%以上，可能過擬合）">命中率<br><small>驗證期</small></th>
  <th title="訓練期樣本數">樣本數</th><th>圖</th>
</tr>
{rows_html}
</table>

<h2>RFC 快速驗證</h2>
{rfc_html if rfc_html else '<p>尚未訓練</p>'}

<h2>Agent 探索報告</h2>
{agent_report_html if agent_report_html else '<p style="color:#666">無文字報告</p>'}

<p style="color:#666;font-size:12px">產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""

    report_path = os.path.join(results_d, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[報表] 已存：{report_path}（{len(rows)} 筆條件）")


# ── 入口 ─────────────────────────────────────────────────────────
def run_session(hypothesis: str = "breakout"):
    """跑一個假設的完整 session：Agent 探索 → 驗證 → 訓練 → 報表"""
    if hypothesis not in HYPOTHESIS_CONFIGS:
        print(f"未知假設：{hypothesis}，可用：{list(HYPOTHESIS_CONFIGS.keys())}")
        return
    run_agent(hypothesis)
    validate_and_train(hypothesis)


if __name__ == "__main__":
    import sys
    hyp = sys.argv[1] if len(sys.argv) > 1 else "breakout"
    run_session(hyp)
