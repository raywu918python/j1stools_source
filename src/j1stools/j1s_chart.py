import numpy as np
import pandas as pd
import plotly.graph_objects as go

from j1stools import parquet_db


import pandas as pd
import numpy as np
import plotly.graph_objects as go


import pandas as pd
import numpy as np
import plotly.graph_objects as go


import pandas as pd
import numpy as np
import plotly.graph_objects as go


import pandas as pd
import numpy as np
import plotly.graph_objects as go


def plot_performance(portfolio_value, trades_df, initial_cash=1_000_000, is_web=False):
    """
    接收 simple_backtest 回傳的 portfolio_value 和 trades_df
    """

    # ── 1. 策略每日資產價值 ──────────────────────────────
    strategy_value = portfolio_value.copy()
    strategy_value.index = pd.to_datetime(strategy_value.index)
    st = strategy_value.index.min()
    end = strategy_value.index.max()

    # ── 2. 0050 基準 ──────────────────────────────────────
    df0050 = parquet_db.query_price(["0050"], st, end)
    df0050.set_index("date", inplace=True)
    benchmark_close = df0050["close"]
    benchmark_returns = benchmark_close.pct_change().fillna(0)

    benchmark_value = initial_cash * (1 + benchmark_returns).cumprod()
    benchmark_value.index = pd.to_datetime(benchmark_value.index)

    # ── 3. 對齊資料 ────────────────────────────────────
    combined_index = strategy_value.index.union(benchmark_value.index)
    strategy_value = strategy_value.reindex(combined_index).ffill()
    benchmark_value = benchmark_value.reindex(combined_index).ffill()

    # ── 4. 對齊買賣點資料 ──────────────────────────────
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

        trades_df["entry_value"] = trades_df["entry_date"].apply(
            lambda x: strategy_value.asof(x) if x in combined_index else np.nan
        )
        trades_df["exit_value"] = trades_df["exit_date"].apply(
            lambda x: strategy_value.asof(x) if x in combined_index else np.nan
        )

    # ── 5. 建立圖表 ─────────────────────────────────────
    fig = go.Figure()

    # (A) 0050 基準線 (不顯示 Hover)
    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name="0050 基準",
            line=dict(color="#666666", width=1.5, dash="dash"),
            opacity=0.7,
            hoverinfo="skip",
        )
    )

    # (B) 策略價值曲線 (整合策略值與基準值)
    strategy_text = [
        f"日期: {d.strftime('%Y-%m-%d')}<br>策略價值: NT$ {s:,.0f}<br>大盤價值: NT$ {b:,.0f}"
        for d, s, b in zip(strategy_value.index, strategy_value.values, benchmark_value.values)
    ]

    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value.values,
            name="策略價值",
            line=dict(color="#00E676", width=2.5),
            text=strategy_text,
            hoverinfo="text",  # ⭐ 僅顯示純文字，避開 HTML 複雜渲染
        )
    )

    if not trades_df.empty:
        # (C) 買進標記
        entry_text = [
            f"標的: {sid}<br>日期: {d.strftime('%Y-%m-%d')}<br>買入價: {p:,.2f}"
            for d, p, sid in zip(trades_df["entry_date"], trades_df["entry_price"], trades_df["stock_id"])
        ]

        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_date"],
                y=trades_df["entry_value"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00B0FF"),
                name="買入",
                text=entry_text,
                hoverinfo="text",
            )
        )

        # (D) 賣出標記
        exit_text = [
            f"標的: {sid}<br>日期: {d.strftime('%Y-%m-%d')}<br>賣出價: {p:,.2f}<br>獲利金額: NT$ {pf:,.0f}<br>報酬率: {ret:.0f}%"
            for d, p, pf, ret, sid in zip(
                trades_df["exit_date"],
                trades_df["exit_price"],
                trades_df["pnl"],
                trades_df["return_pct"],
                trades_df["stock_id"],
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"],
                y=trades_df["exit_value"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#FF1744"),
                name="賣出",
                text=exit_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(text="<b>策略績效與交易點位分析</b>", font=dict(color="#FFFFFF", size=18), x=0.05),
        hovermode="x",
        hoverdistance=20,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(color="#e0e0e0", family="Arial"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A", title="資產價值 (NTD)", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        height=650,
        hoverlabel=dict(bgcolor="#2A2A2A", font_size=12, font_color="#FFFFFF", bordercolor="#444444"),
    )

    if is_web:
        return fig.to_json()
    else:
        fig.show()


def plot_performance_local(portfolio_value, trades_df, initial_cash=1_000_000, is_web=False):
    """
    接收 simple_backtest 回傳的 portfolio_value 和 trades_df
    """

    # ── 1. 策略每日資產價值 ──────────────────────────────
    strategy_value = portfolio_value.copy()
    strategy_value.index = pd.to_datetime(strategy_value.index)
    st = strategy_value.index.min()
    end = strategy_value.index.max()

    # ── 2. 0050 基準 ──────────────────────────────────────
    # 假設 parquet_db 已在外部正確初始化
    df0050 = parquet_db.query_price(["0050"], st, end)
    df0050.set_index("date", inplace=True)
    benchmark_close = df0050["close"]
    benchmark_returns = benchmark_close.pct_change().fillna(0)

    # 確保基準值大小與策略資產對齊 (使用 cumprod 計算累積價值)
    benchmark_value = initial_cash * (1 + benchmark_returns).cumprod()
    benchmark_value.index = pd.to_datetime(benchmark_value.index)

    # ── 3. 對齊資料 (處理不同交易日，確保索引完全一致，避免缺失) ─────
    combined_index = strategy_value.index.union(benchmark_value.index)
    strategy_value = strategy_value.reindex(combined_index).ffill()  # 策略值補齊
    benchmark_value = benchmark_value.reindex(combined_index).ffill()  # 0050值補齊

    # ── 4. 對齊買賣點資料 ───────────────────────────────
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

        # 確保對應到正確的 index 位置，避免無效資料 (改用 .asof 強制對齊)
        trades_df["entry_value"] = trades_df["entry_date"].apply(
            lambda x: strategy_value.asof(x) if x in combined_index else np.nan
        )
        trades_df["exit_value"] = trades_df["exit_date"].apply(
            lambda x: strategy_value.asof(x) if x in combined_index else np.nan
        )

    # ── 5. 建立圖表 ───────────────────────────────────────
    fig = go.Figure()

    # (A) 0050 基準線 (次要)
    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name="0050 基準",
            line=dict(color="#666666", width=1.5, dash="dash"),
            opacity=0.7,
            hoverinfo="skip",  # ⭐ 完全移除這條線上的 Hover
        )
    )

    # (B) 策略價值曲線 (主要)
    # ⭐⭐⭐ 新增客製化 Hover：整合策略值與基準值 ⭐⭐⭐
    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value.values,
            name="策略價值",
            line=dict(color="#00E676", width=2.5),
            customdata=benchmark_value.values.round(0),  # 綁定同一天的 0050 價值 (取整)
            hovertemplate=(
                "日期: %{x|%Y-%m-%d}<br>"
                "策略價值: NT$ %{y:,.0f}<br>"
                "大盤價值: NT$ %{customdata:,.0f}"  # 從 customdata 讀取
                "<extra></extra>"
            ),
        )
    )

    if not trades_df.empty:
        # (C) 買進標記 (分離資訊)
        # ⭐⭐⭐ 客製化 Hover：只顯示買入資訊 ⭐⭐⭐
        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_date"],
                y=trades_df["entry_value"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#00B0FF", line=dict(width=1, color="white")),
                name="買入",
                customdata=np.stack(
                    [
                        trades_df["entry_price"].round(0),  # 買入價 (取整)
                        trades_df["stock_id"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>買入標的: %{customdata[1]}</b><br>"
                    "日期: %{x|%Y-%m-%d}<br>"
                    "買入價: %{customdata[0]:,.2f}"  # 只顯示買入價
                    "<extra></extra>"
                ),
            )
        )

        # (D) 賣出標記 (分離資訊)
        # ⭐⭐⭐ 客製化 Hover：顯示完整賣出與績效 ⭐⭐⭐
        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"],
                y=trades_df["exit_value"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#FF1744", line=dict(width=1, color="white")),
                name="賣出",
                customdata=np.stack(
                    [
                        trades_df["exit_date"].dt.strftime("%Y-%m-%d"),
                        trades_df["exit_price"].round(0),  # 賣出價 (取整)
                        trades_df["pnl"].round(0),  # 獲利 (取整)
                        trades_df["return_pct"].round(0),  # 報酬率 (取整)
                        trades_df["stock_id"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>賣出標的: %{customdata[4]}</b><br>"
                    "日期: %{x|%Y-%m-%d}<br>"
                    "賣出價: %{customdata[1]:,.2f}<br>"
                    "獲利金額: NT$ %{customdata[2]:,.0f}<br>"
                    "報酬率: %{customdata[3]}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="<b>策略績效與交易點位分析</b>", font=dict(color="#FFFFFF", size=18), x=0.05),
        # ⭐⭐⭐ 關鍵修改 ⭐⭐⭐：改回 "x" 模式，以便對「線」和「點」分別進行 Hover
        hovermode="x",
        hoverdistance=30,  # 設定合理的 hover 距離
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(color="#e0e0e0", family="Arial"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A", title="資產價值 (NTD)", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        width=None,
        height=650,
        hoverlabel=dict(
            bgcolor="#2A2A2A",
            font_size=13,
            font_color="#FFFFFF",
            bordercolor="#444444",
        ),
    )


def plot_performance_old(portfolio_value, trades_df, initial_cash=1_000_000, is_web=False):
    """
    接收 simple_backtest 回傳的 portfolio_value 和 trades_df
    """

    # ── 1. 策略每日資產價值 ──────────────────────────────
    strategy_value = portfolio_value.copy()
    strategy_value.index = pd.to_datetime(strategy_value.index)
    st = strategy_value.index.min()
    end = strategy_value.index.max()
    # ── 2. 0050 基準 ──────────────────────────────────────
    df0050 = parquet_db.query_price(["0050"], st, end)
    df0050.set_index("date", inplace=True)
    benchmark_close = df0050["close"]
    benchmark_returns = benchmark_close.pct_change().fillna(0)
    benchmark_value = initial_cash * (1 + benchmark_returns).cumprod()
    benchmark_value.index = pd.to_datetime(benchmark_value.index)

    # ── 3. 對齊資料 ───────────────────────────────────────
    strategy_value, benchmark_value = strategy_value.align(benchmark_value, join="outer")

    # ── 4. 對齊買賣點到策略價值 ───────────────────────────
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        trades_df["entry_value"] = trades_df["entry_date"].apply(lambda x: strategy_value.asof(x))
        trades_df["exit_value"] = trades_df["exit_date"].apply(lambda x: strategy_value.asof(x))

    # ── 5. 建立圖表 ───────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value.values,
            name="策略價值",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name="0050 基準",
            line=dict(color="blue", dash="dash"),
        )
    )

    if not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_date"].dt.strftime("%Y-%m-%d"),
                y=trades_df["entry_value"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="blue"),
                name="買入",
                customdata=np.stack(
                    [
                        trades_df["exit_date"].dt.strftime("%Y-%m-%d"),  # ✅ 格式化日期
                        trades_df["exit_price"].round(2),
                        trades_df["pnl"].round(2),
                        trades_df["return_pct"].round(2),
                        trades_df["stock_id"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "標的: %{customdata[4]}<br>"
                    "賣出日期: %{customdata[0]}<br>"  # ✅ 顯示 2024-01-19 格式
                    "賣出價: %{customdata[1]}<br>"
                    "獲利: %{customdata[2]}<br>"
                    "報酬率: %{customdata[3]}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"],
                y=trades_df["exit_value"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="賣出",
                customdata=np.stack(
                    [
                        trades_df["exit_date"].dt.strftime("%Y-%m-%d"),  # ✅ 格式化日期
                        trades_df["exit_price"].round(2),
                        trades_df["pnl"].round(2),
                        trades_df["return_pct"].round(2),
                        trades_df["stock_id"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "標的: %{customdata[4]}<br>"
                    "賣出日期: %{customdata[0]}<br>"  # ✅ 顯示 2024-01-19 格式
                    "賣出價: %{customdata[1]}<br>"
                    "獲利: %{customdata[2]}<br>"
                    "報酬率: %{customdata[3]}%"
                    "<extra></extra>"
                ),
            )
        )

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

    if is_web:
        return fig.to_json()
    else:
        fig.show()
    # with open("chart/chart_data.json", "w", encoding="utf-8") as f:
    # f.write(fig.to_json())
    # return fig


def chart_gantt(trades_df):
    import plotly.express as px

    if trades_df.empty:
        print("沒有交易紀錄")
        return

    df_plot = trades_df.copy()
    df_plot["entry_date"] = pd.to_datetime(df_plot["entry_date"])
    df_plot["exit_date"] = pd.to_datetime(df_plot["exit_date"])

    stock_dict = parquet_db.query_group_by_ids(df_plot["stock_id"])
    df_plot["name"] = df_plot["stock_id"].map(lambda x: stock_dict.get(x, {}).get("name"))
    df_plot["group"] = df_plot["stock_id"].map(lambda x: stock_dict.get(x, {}).get("group"))
    df_plot["merge_name"] = df_plot["stock_id"] + "|" + df_plot["name"] + "|" + df_plot["group"]
    # hover 用格式化日期
    df_plot["Entry_Day"] = df_plot["entry_date"].dt.strftime("%Y-%m-%d")
    df_plot["Exit_Day"] = df_plot["exit_date"].dt.strftime("%Y-%m-%d")

    df_plot["Status"] = df_plot["pnl"].apply(lambda x: "Profit" if x > 0 else "Loss")

    fig = px.timeline(
        df_plot,
        x_start="entry_date",
        x_end="exit_date",
        y="merge_name",
        color="Status",
        color_discrete_map={"Profit": "#26a69a", "Loss": "#ef5350"},
        hover_data={
            "entry_date": False,
            "exit_date": False,
            "Entry_Day": True,
            "Exit_Day": True,
            "pnl": ":.0f",
            "return_pct": ":.2f",
        },
        title="Trade Duration & Timeline",
    )

    fig.update_yaxes(categoryorder="min ascending", title="Stock ID")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )

    fig.show()

    with open("chart/chart_gantt.json", "w", encoding="utf-8") as f:
        f.write(fig.to_json())


def chart_allocation(portfolio_value, trades_df, close):
    import plotly.express as px
    import plotly.graph_objects as go

    if trades_df.empty:
        print("沒有交易紀錄")
        return

    # ── 1. 計算每日各標的市值 ────────────────────────────
    dates = portfolio_value.index

    # 重建每日持倉
    positions = {}  # {sid: {shares, entry_bar}}
    daily_allocation = []

    entries_by_date = trades_df.groupby("entry_date")
    exits_by_date = trades_df.groupby("exit_date")

    for dt in dates:
        # 出場
        if dt in exits_by_date.groups:
            for _, row in exits_by_date.get_group(dt).iterrows():
                positions.pop(row["stock_id"], None)

        # 進場
        if dt in entries_by_date.groups:
            for _, row in entries_by_date.get_group(dt).iterrows():
                positions[row["stock_id"]] = row["size"]

        # 計算各標的市值
        day_alloc = {}
        for sid, shares in positions.items():
            if sid in close.columns and dt in close.index:
                day_alloc[sid] = close.loc[dt, sid] * shares

        day_alloc["現金"] = portfolio_value.loc[dt] - sum(day_alloc.values())
        daily_allocation.append(day_alloc)

    alloc_df = pd.DataFrame(daily_allocation, index=dates).fillna(0)

    # 轉成比例
    proportions = alloc_df.div(portfolio_value, axis=0)

    # ── 2. 持倉配置堆疊圖 ────────────────────────────────
    fig1 = px.area(
        proportions,
        title="Asset Allocation Over Time",
        labels={"value": "比例", "index": "日期"},
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333", title="比例"),
    )
    fig1.show()

    with open("chart/chart_allocation.json", "w", encoding="utf-8") as f:
        f.write(fig1.to_json())

    # ── 3. 相關性熱力圖 ───────────────────────────────────
    # 只算有交易過的標的
    traded_stocks = trades_df["stock_id"].unique().tolist()
    traded_close = close[[s for s in traded_stocks if s in close.columns]]

    corr_matrix = traded_close.pct_change().corr()

    fig2 = go.Figure(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="相關係數"),
        )
    )
    fig2.update_layout(
        title="Asset Correlation Matrix",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    fig2.show()

    with open("chart/chart_correlation.json", "w", encoding="utf-8") as f:
        f.write(fig2.to_json())
