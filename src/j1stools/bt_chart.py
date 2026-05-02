import pandas as pd
import plotly.graph_objects as go

import plotly.graph_objects as go
import pandas as pd


import plotly.graph_objects as go
import pandas as pd
import backtrader as bt

import parquet_db


def plot_performance(strat, st, end, initial_cash=1_000_000):
    """
    backtrader 版本的績效圖
    對應 vbt 的 pk0050_vbt_1chat_ai
    """
    trades_df = pd.DataFrame(strat.closed_trades)

    # ── 1. 取得策略每日資產價值 ──────────────────────────
    # backtrader 用 analyzer 取得每日價值
    daily_values = pd.Series(strat.analyzers.time_return.get_analysis())
    # 轉成累計資產價值
    strategy_value = initial_cash * (1 + daily_values).cumprod()
    strategy_value.index = pd.to_datetime(strategy_value.index)

    # ── 2. 取得 0050 基準 ─────────────────────────────────
    df0050 = parquet_db.query_price(["0050"], st, end)
    df0050.set_index("date", inplace=True)
    benchmark_close = df0050["close"]
    benchmark_returns = benchmark_close.pct_change().fillna(0)
    benchmark_value = initial_cash * (1 + benchmark_returns).cumprod()
    benchmark_value.index = pd.to_datetime(benchmark_value.index)

    # ── 3. 對齊資料 ───────────────────────────────────────
    strategy_value, benchmark_value = strategy_value.align(benchmark_value, join="outer")

    # ── 4. 取得交易紀錄 ───────────────────────────────────
    trades_df = pd.DataFrame(strat.closed_trades)

    # ── 5. 對齊買賣點到策略價值 ───────────────────────────
    if not trades_df.empty:
        trades_df["entry_value"] = trades_df["entry_date"].apply(lambda x: strategy_value.asof(x))
        trades_df["exit_value"] = trades_df["exit_date"].apply(lambda x: strategy_value.asof(x))

    # ── 6. 建立圖表 ───────────────────────────────────────
    fig = go.Figure()

    # 策略價值曲線
    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value.values,
            name="策略價值",
            line=dict(color="green"),
        )
    )

    # 0050 基準曲線
    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name="0050 基準",
            line=dict(color="blue", dash="dash"),
        )
    )

    # 買入點
    if not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_date"],
                y=trades_df["entry_value"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="blue"),
                name="買入",
                text=(
                    "標的: "
                    + trades_df["stock_id"].astype(str)
                    + "<br>買入價: "
                    + trades_df["entry_price"].round(2).astype(str)
                    + "<br>股數: "
                    + trades_df["size"].round(0).astype(str)
                ),
                hoverinfo="text+name",
            )
        )

        # 賣出點
        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"],
                y=trades_df["exit_value"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="賣出",
                text=(
                    "標的: "
                    + trades_df["stock_id"].astype(str)
                    + "<br>賣出價: "
                    + trades_df["exit_price"].round(2).astype(str)
                    + "<br>獲利: "
                    + trades_df["pnl"].round(2).astype(str)
                    + "<br>報酬率: "
                    + trades_df["return_pct"].round(2).astype(str)
                    + "%"
                ),
                hoverinfo="text+name",
            )
        )

    # ── 7. Layout ─────────────────────────────────────────
    fig.update_layout(
        hovermode="x unified",
        hoverdistance=1000,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#333333", zerolinecolor="#444444"),
        yaxis=dict(gridcolor="#333333", zerolinecolor="#444444", title="資產價值"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=None,
        height=None,
        hoverlabel=dict(
            bgcolor="#444444",
            font_size=14,
            font_family="Rockwell",
            font_color="#e0e0e0",
            bordercolor="#333333",
        ),
    )

    # ── 8. 輸出 ───────────────────────────────────────────
    fig.show()  # 本機用

    with open("chart/chart_data.json", "w", encoding="utf-8") as f:
        f.write(fig.to_json())

    return fig


def show_interactive_chart(strat, strategy_value, trades_df):
    fig = go.Figure()

    # --- 1. 畫資產線 ---
    # 確保 strategy_value 是 Series 且沒有空值
    strategy_value = strategy_value.dropna()

    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value.values,
            mode="lines",
            name="資產總值",
            line=dict(color="#10b981", width=2),
        )
    )

    # --- 2. 畫交易點 ---
    if not trades_df.empty:
        # 轉換交易時間，確保跟 strategy_value 的 index 類型一致
        exit_times = pd.to_datetime(trades_df["Exit Timestamp"])

        # 修正：直接用 trades_df 裡的價格或 PnL，暫時不要用 asof 避免對齊失敗
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=[strategy_value.asof(d) for d in exit_times],  # 嘗試對齊資產線高度
                mode="markers",
                name="賣出點",
                marker=dict(symbol="triangle-down", size=12, color="#f87171"),
                customdata=trades_df[["Column", "PnL", "Return"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>損益: %{customdata[1]}<br>報酬: %{customdata[2]:.2%}<extra></extra>",
            )
        )

    # --- 3. 強制設定座標軸範圍 ---
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(type="date", autorange=True),
        yaxis=dict(autorange=True, tickformat="$,.0f"),
        hovermode="x unified",
    )

    fig.show()


def generate_plotly_chart(strat, strategy_value, benchmark_df):
    # --- 1. 取得資產價值線 ---
    # strat.observers.broker.value 紀錄了每一根 K 線結束後的總資產
    equity_data = strat.observers.broker.value.array
    time_data = [bt.num2date(x) for x in strat.observers.broker.datetime.array]
    strategy_value = pd.Series(equity_data, index=pd.to_datetime(time_data))

    # --- 2. 處理基準線 (0050) ---
    benchmark_close = benchmark_df["close"]
    initial_cash = strategy_value.iloc[0]
    benchmark_value = initial_cash * (benchmark_close / benchmark_close.iloc[0])

    # 對齊
    strategy_value, benchmark_value = strategy_value.align(benchmark_value, join="inner")

    # --- 3. 處理交易點 ---
    trades = pd.DataFrame(strat.trade_list)

    import plotly.graph_objects as go

    fig = go.Figure()

    # 畫策略線
    fig.add_trace(
        go.Scatter(x=strategy_value.index, y=strategy_value, name="Just 1 stock 策略", line=dict(color="#10b981"))
    )

    # 畫基準線
    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index, y=benchmark_value, name="0050 基準", line=dict(color="#3b82f6", dash="dash")
        )
    )

    if not trades.empty:
        # 畫買入點
        fig.add_trace(
            go.Scatter(
                x=trades["Entry Timestamp"],
                y=[strategy_value.asof(d) for d in trades["Entry Timestamp"]],
                mode="markers",
                name="買入",
                marker=dict(symbol="triangle-up", size=10, color="#60a5fa"),
                text=trades["Column"],
                hovertemplate="買入: %{text}<extra></extra>",
            )
        )

        # 畫賣出點
        fig.add_trace(
            go.Scatter(
                x=trades["Exit Timestamp"],
                y=[strategy_value.asof(d) for d in trades["Exit Timestamp"]],
                mode="markers",
                name="賣出",
                marker=dict(symbol="triangle-down", size=10, color="#f87171"),
                text=trades["Column"],
                hovertemplate="賣出: %{text}<br>報酬: %{customdata:.2%}<extra></extra>",
                customdata=trades["Return"],
            )
        )

    # 套用你之前漂亮的 Layout 設定...
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    return fig.to_json()
