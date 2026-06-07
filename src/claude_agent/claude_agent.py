import anthropic
import json
import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from j1stools import parquet_db
from j1stools.backtest_engine import backtest_engine

client = anthropic.Anthropic()

# ── 工具定義 ──────────────────────────────────────────────
tools = [
    {
        "name": "get_price_data",
        "description": "取得股票歷史K線資料，回傳筆數與欄位資訊",
        "input_schema": {
            "type": "object",
            "properties": {
                "stocks": {"type": "array", "items": {"type": "string"}, "description": "股票代碼列表"},
                "start": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                "end": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
            },
            "required": ["stocks", "start", "end"],
        },
    },
    {
        "name": "run_custom_strategy",
        "description": (
            "Agent 自己設計進場條件跑回測。\n"
            "【技術指標】close, open, high, low, volume\n"
            "  均線: ma5, ma10, ma20, ma60, ma120\n"
            "  交叉: ma5_cross_ma10, ma5_cross_ma20, ma10_cross_ma60\n"
            "  RSI: rsi（14期）\n"
            "  MACD: macd, signal, macdh\n"
            "  布林: bb_upper, bb_mid, bb_lower, bb_width\n"
            "  ATR: atr（14期）\n"
            "  成交量: vol_ma5, vol_ma20, vol_ratio（今量/20日均量）\n"
            "【融資融券指標】需設 use_margin=true\n"
            "  margin_balance（融資餘額）, short_balance（融券餘額）\n"
            "  margin_change（融資增減率）, short_change（融券增減率）\n"
            "  short_vs_margin（空多比）\n"
            "【三大法人指標】需設 use_ib=true\n"
            "  net_foreign（外資買賣超）, net_trust（投信買賣超）\n"
            "  net_dealer（自營商）, net_institutional（三大法人合計）\n"
            "條件用 & | ~ 而非 and or not\n"
            "範例：'(ma5_cross_ma10) & (rsi < 70) & (close > ma60)'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_name": {"type": "string", "description": "策略名稱"},
                "entry_condition": {"type": "string", "description": "進場條件字串"},
                "stocks": {"type": "array", "items": {"type": "string"}, "description": "股票代碼列表"},
                "start": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                "end": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                "hold_days": {"type": "integer", "description": "持有天數，預設 5"},
                "sl_stop": {"type": "number", "description": "停損比例，預設 0.08"},
                "tp_stop": {"type": "number", "description": "停利比例，預設 0.15"},
                "use_margin": {"type": "boolean", "description": "是否加入融資融券指標，預設 false"},
                "use_ib": {"type": "boolean", "description": "是否加入三大法人指標，預設 false"},
            },
            "required": ["strategy_name", "entry_condition", "stocks", "start", "end"],
        },
    },
    {
        "name": "run_lgbm_backtest",
        "description": (
            "用已訓練好的 LightGBM 模型預測個股漲跌機率，以機率 > threshold 為進場訊號，跑回測。"
            "模型檔需存在 models/ 目錄下。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_name": {"type": "string", "description": "策略名稱"},
                "stocks": {"type": "array", "items": {"type": "string"}, "description": "股票代碼列表"},
                "start": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                "end": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                "model_file": {"type": "string", "description": "模型檔名，例如 lgbm_model.joblib"},
                "threshold": {"type": "number", "description": "進場機率門檻，預設 0.6"},
                "hold_days": {"type": "integer", "description": "持有天數，預設 5"},
                "sl_stop": {"type": "number", "description": "停損比例，預設 0.08"},
                "tp_stop": {"type": "number", "description": "停利比例，預設 0.15"},
            },
            "required": ["strategy_name", "stocks", "start", "end", "model_file"],
        },
    },
]


# ── 指標計算 ──────────────────────────────────────────────
def _calc_indicators(df_raw: pd.DataFrame) -> dict:
    close = df_raw.pivot(index="date", columns="stock_id", values="close").sort_index()
    high = df_raw.pivot(index="date", columns="stock_id", values="high").sort_index()
    low = df_raw.pivot(index="date", columns="stock_id", values="low").sort_index()
    vol = df_raw.pivot(index="date", columns="stock_id", values="volume").sort_index()

    ind = {"close": close, "high": high, "low": low, "volume": vol}

    for n in [5, 10, 20, 60, 120]:
        ind[f"ma{n}"] = close.rolling(n).mean()

    for fast, slow in [(5, 10), (5, 20), (10, 60)]:
        f_ma, s_ma = ind[f"ma{fast}"], ind[f"ma{slow}"]
        ind[f"ma{fast}_cross_ma{slow}"] = (f_ma > s_ma) & (f_ma.shift(1) <= s_ma.shift(1))

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    ind["rsi"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"] = ema12 - ema26
    ind["signal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macdh"] = ind["macd"] - ind["signal"]

    # 布林通道
    ind["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ind["bb_upper"] = ind["bb_mid"] + 2 * bb_std
    ind["bb_lower"] = ind["bb_mid"] - 2 * bb_std
    ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_mid"]

    # ATR(14)
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
    ind["atr"] = tr.rolling(14).mean()

    # 成交量
    ind["vol_ma5"] = vol.rolling(5).mean()
    ind["vol_ma20"] = vol.rolling(20).mean()
    ind["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-9)

    return ind, close


def _calc_margin_indicators(stocks, start, end, close_index):
    df = parquet_db.query_margin(stocks, start, end)
    df["date"] = pd.to_datetime(df["date"])

    def _pivot(col):
        return df.pivot(index="date", columns="stock_id", values=col).reindex(close_index).ffill()

    margin = _pivot("margin_purchase_today_balance")
    short = _pivot("short_sale_today_balance")

    return {
        "margin_balance": margin,
        "short_balance": short,
        "margin_change": margin.pct_change().clip(-1, 1),
        "short_change": short.pct_change().clip(-1, 1),
        "short_vs_margin": short / (margin + 1e-9),
    }


def _calc_ib_indicators(stocks, start, end, close_index):
    df = parquet_db.query_ib(stocks, start, end)
    df["date"] = pd.to_datetime(df["date"])
    df["net"] = df["buy"] - df["sell"]

    name_map = {
        "Foreign_Investor": "net_foreign",
        "Investment_Trust": "net_trust",
        "Dealer_self": "net_dealer",
    }
    result = {}
    for ib_name, col_name in name_map.items():
        sub = df[df["name"] == ib_name].pivot(index="date", columns="stock_id", values="net")
        result[col_name] = sub.reindex(close_index).fillna(0)

    result["net_institutional"] = sum(result.values())
    return result


def _run_backtest_with_entries(close, entries, hold_days, sl_stop, tp_stop):
    exits = pd.Series(False, index=close.index)
    df_proba = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    stock_group = {s: "default" for s in close.columns}

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

    returns = portfolio_value.pct_change().dropna()
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    sharpe = returns.mean() / returns.std() * (252**0.5) if returns.std() > 0 else 0
    max_drawdown = ((portfolio_value / portfolio_value.cummax()) - 1).min()

    return {
        "total_return": round(float(total_return), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_drawdown), 4),
        "total_trades": len(trades_df),
    }


# ── 工具執行 ──────────────────────────────────────────────
def execute_tool(name: str, inputs: dict):
    if name == "get_price_data":
        df = parquet_db.query_price(inputs["stocks"], inputs["start"], inputs["end"])
        df["date"] = pd.to_datetime(df["date"])
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": f"{df['date'].min().date()} ~ {df['date'].max().date()}",
        }

    if name == "run_custom_strategy":
        df_raw = parquet_db.query_price(inputs["stocks"], inputs["start"], inputs["end"])
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        ind, close = _calc_indicators(df_raw)

        if inputs.get("use_margin"):
            ind.update(_calc_margin_indicators(inputs["stocks"], inputs["start"], inputs["end"], close.index))

        if inputs.get("use_ib"):
            ind.update(_calc_ib_indicators(inputs["stocks"], inputs["start"], inputs["end"], close.index))

        try:
            entries = eval(inputs["entry_condition"], {"__builtins__": {}}, ind)
        except Exception as e:
            return {"error": f"進場條件解析失敗：{e}"}

        if not isinstance(entries, pd.DataFrame):
            return {"error": "進場條件必須回傳 DataFrame（每支股票一欄）"}

        result = _run_backtest_with_entries(
            close,
            entries,
            hold_days=inputs.get("hold_days", 5),
            sl_stop=inputs.get("sl_stop", 0.08),
            tp_stop=inputs.get("tp_stop", 0.15),
        )
        result["strategy"] = inputs["strategy_name"]
        result["entry_condition"] = inputs["entry_condition"]
        return result

    if name == "run_lgbm_backtest":
        model_path = os.path.join("models", inputs["model_file"])
        if not os.path.exists(model_path):
            return {"error": f"找不到模型檔：{model_path}"}

        df_raw = parquet_db.query_price(inputs["stocks"], inputs["start"], inputs["end"])
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        _, close = _calc_indicators(df_raw)

        models = joblib.load(model_path)
        threshold = inputs.get("threshold", 0.6)

        features = [col for col in df_raw.columns if col.startswith("f_")]
        if not features:
            return {"error": "資料中沒有 f_ 開頭的特徵欄位，請先跑 feature_builder"}

        df_raw["proba"] = np.mean([m.predict_proba(df_raw[features])[:, 1] for m in models], axis=0)

        signal = df_raw.pivot(index="date", columns="stock_id", values="proba").reindex(close.index)
        entries = signal > threshold

        result = _run_backtest_with_entries(
            close,
            entries,
            hold_days=inputs.get("hold_days", 5),
            sl_stop=inputs.get("sl_stop", 0.08),
            tp_stop=inputs.get("tp_stop", 0.15),
        )
        result["strategy"] = inputs["strategy_name"]
        result["model_file"] = inputs["model_file"]
        result["threshold"] = threshold
        return result

    return {"error": f"未知工具：{name}"}


# ── Agent 主循環 ──────────────────────────────────────────
def run_agent(task: str):
    messages = [{"role": "user", "content": task}]
    print(f"\n任務：{task}\n{'─'*50}")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent 結論：\n{block.text}")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"→ {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                    result = execute_tool(block.name, block.input)
                    print(f"  結果：{json.dumps(result, ensure_ascii=False)}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    run_agent(
        "幫我用 2330 2317 2454，時間 2023-01-01 到 2024-12-31，"
        "自己發想 3 種不同的進場策略並比較績效，"
        "至少一種要用到融資融券或三大法人資料，找出最好的那個"
    )
