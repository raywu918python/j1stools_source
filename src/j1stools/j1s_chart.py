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
            hovertemplate="策略價值: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name="0050 基準",
            line=dict(color="blue", dash="dash"),
            hovertemplate="大盤基準: %{y:,.0f}<extra></extra>",
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
                        trades_df["stock_id"],
                        trades_df["entry_date"].dt.strftime("%Y-%m-%d"),
                        trades_df["entry_price"].round(2),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "標的: %{customdata[0]}<br>"
                    "買入日期: %{customdata[1]}<br>"
                    "買入價: %{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_date"].dt.strftime("%Y-%m-%d"),
                y=trades_df["exit_value"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="賣出",
                customdata=np.stack(
                    [
                        trades_df["stock_id"],
                        trades_df["exit_date"].dt.strftime("%Y-%m-%d"),
                        trades_df["exit_price"].round(2),
                        trades_df["pnl"].round(0).astype(int),
                        trades_df["return_pct"].round(0).astype(int),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "標的: %{customdata[0]}<br>"
                    "賣出日期: %{customdata[1]}<br>"
                    "賣出價: %{customdata[2]:.2f}<br>"  # ← 小數2位
                    "獲利: %{customdata[3]:,.0f}<br>"  # ← 整數+千分位
                    "報酬率: %{customdata[4]}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        hovermode="x unified",
        hoverdistance=1000,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color="#e0e0e0"),
        xaxis=dict(
            gridcolor="#333333",
            zerolinecolor="#444444",
            rangeslider=dict(visible=True, thickness=0.06),
            type="date",
        ),
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
