# ============================================================
# 輕量回測引擎
# ============================================================

import pandas as pd


def backtest_engine(
    close,
    entries,
    exits,
    df_proba,
    stock_group,
    max_positions=10,
    sl_trail=0.1,
    use_hold_days=True,  # ✅ 新增開關
    use_sl_trail=True,
    sl_stop=0.08,  # ✅ 固定停損 8%
    tp_stop=0.15,  # ✅ 固定停利 15%
    use_fixed_sl=False,  # ✅ 固定停損開關
    use_fixed_tp=False,  # ✅ 固定停利開關
    hold_days=5,
    init_cash=1_000_000,
    fee=0.001,
    group_limit=3,  # ✅ 每族群最多 3 支
    use_fixed_sl_tp=False,
    use_proba_sizing=False,
):
    dates = close.index
    cash = float(init_cash)
    positions = {}
    portfolio_records = []
    trades = []

    # ✅ 按機率排序
    entries_dict = {}
    for dt, row in entries.iterrows():
        if row.any():
            true_cols = row[row].index.tolist()
            true_cols = sorted(true_cols, key=lambda x: df_proba.loc[dt, x], reverse=True)
            entries_dict[dt] = true_cols

    exits_arr = exits.values

    for i, dt in enumerate(dates):
        is_danger = bool(exits_arr[i]) if i < len(exits_arr) else False

        # ── 出場 ──────────────────────────────────────────
        for sid in list(positions.keys()):
            if sid not in close.columns:
                continue
            price = close.loc[dt, sid]
            pos = positions[sid]
            pos["highest"] = max(pos["highest"], price)

            should_exit = (
                is_danger
                # 移動停損
                or (use_sl_trail and price <= pos["highest"] * (1 - sl_trail))
                # 固定天數
                or (use_hold_days and (i - pos["entry_bar"]) >= hold_days)
                # 固定停損
                or (use_fixed_sl and price <= pos["entry_price"] * (1 - sl_stop))
                # 固定停利
                or (use_fixed_tp and price >= pos["entry_price"] * (1 + tp_stop))
            )

            if should_exit:
                if use_fixed_sl_tp and use_fixed_sl and price <= pos["entry_price"] * (1 - tp_stop):
                    sell_value = pos["entry_price"] * (1 - sl_stop) * pos["shares"] * (1 - fee)
                elif use_fixed_sl_tp and use_fixed_tp and price >= pos["entry_price"] * (1 + tp_stop):
                    sell_value = pos["entry_price"] * (1 + tp_stop) * pos["shares"] * (1 - fee)
                else:
                    sell_value = price * pos["shares"] * (1 - fee)

                if is_danger:
                    reason = "market_exit"
                elif use_fixed_tp and price >= pos["entry_price"] * (1 + tp_stop):
                    reason = "tp_stop"
                elif use_fixed_sl and price <= pos["entry_price"] * (1 - sl_stop):
                    reason = "sl_stop"
                elif use_sl_trail and price <= pos["highest"] * (1 - sl_trail):
                    reason = "sl_trail"
                elif use_hold_days and (i - pos["entry_bar"]) >= hold_days:
                    reason = "hold_days"
                else:
                    reason = "exit"

                pnl = sell_value - pos["cost"]
                cash += sell_value
                trades.append(
                    {
                        "stock_id": sid,
                        "entry_date": pos["entry_date"],
                        "entry_price": pos["entry_price"],
                        "stop": pos["stop"],
                        "target": pos["target"],
                        "exit_date": dt,
                        "exit_price": price,
                        "exit_reason": reason,
                        "highest": pos["highest"],
                        "cost": pos["cost"],
                        "pnl": pnl,
                        "pnl_pct": (price - pos["entry_price"]) / pos["entry_price"] * 100,
                        "return_pct": pnl / pos["cost"] * 100,
                        "hold_days": i - pos["entry_bar"],
                    }
                )
                del positions[sid]

        # ── 進場（市場危險時不進場）──────────────────────
        if not is_danger:
            slots = max_positions - len(positions)
            if slots > 0:
                candidates = [sid for sid in entries_dict.get(dt, []) if sid not in positions and sid in close.columns][
                    :slots
                ]
                if candidates:
                    per_slot = cash / slots
                    for sid in candidates:
                        # ✅ 族群限制
                        if stock_group is not None:
                            group = stock_group.get(sid, "未知")
                            current_group_count = sum(1 for s in positions if stock_group.get(s, "未知") == group)
                            if current_group_count >= group_limit:
                                # print(f"🚫 {sid} 族群 {group} 已滿 {group_limit} 支")
                                continue
                        price = close.loc[dt, sid]
                        if price <= 0:
                            continue
                        # ✅ 依信心度調整倍數
                        if use_proba_sizing:
                            proba = df_proba.loc[dt, sid]
                            multiplier = get_proba_multiplier(proba)
                        else:
                            multiplier = 1.0

                        cost = min(per_slot * multiplier, cash)  # 不超過剩餘現金
                        shares = cost * (1 - fee) / price
                        cash -= cost
                        positions[sid] = {
                            "shares": shares,
                            "entry_price": price,
                            "entry_date": dt,
                            "entry_bar": i,
                            "highest": price,
                            "cost": cost,
                            "stop": round(price * (1 - sl_stop), 4) if use_fixed_sl else None,
                            "target": round(price * (1 + tp_stop), 4) if use_fixed_tp else None,
                        }

        # ── 每日資產價值 ──────────────────────────────────
        pos_value = sum(close.loc[dt, sid] * pos["shares"] for sid, pos in positions.items() if sid in close.columns)
        portfolio_records.append({"date": dt, "cash": round(cash, 2), "market_value": round(pos_value, 2), "total": round(cash + pos_value, 2)})

    _COLS = [
        "stock_id", "entry_date", "entry_price", "stop", "target",
        "exit_date", "exit_price", "exit_reason", "highest",
        "cost", "pnl", "pnl_pct", "return_pct", "hold_days",
    ]
    portfolio_df = pd.DataFrame(portfolio_records)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=_COLS)
    return portfolio_df, trades_df, positions


def get_proba_multiplier(proba):
    """依信心度回傳倍數"""
    if proba >= 0.9:
        return 1.5
    elif proba >= 0.8:
        return 1.2
    else:
        return 1
