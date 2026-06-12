import os
from pathlib import Path

import shioaji as sj
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".trade_env")


def login() -> sj.Shioaji:
    api = sj.Shioaji()
    api.login(
        api_key=os.environ["SINOPAC_API_KEY"],
        secret_key=os.environ["SINOPAC_SECRET_KEY"],
    )
    api.activate_ca(
        ca_path=os.environ["SINOPAC_CA_PATH"],
        ca_passwd=os.environ["SINOPAC_CA_PASSWD"],
        person_id=os.environ["SINOPAC_PERSON_ID"],
    )
    return api


def test():
    api = sj.Shioaji(simulation=True)  # 是否用虛擬主機登入
    api.login(
        api_key=os.environ["SINOPAC_API_KEY"],
        secret_key=os.environ["SINOPAC_SECRET_KEY"],
    )
    return api


def _tick(price: float) -> float:
    if price < 10:
        return 0.01
    elif price < 50:
        return 0.05
    elif price < 100:
        return 0.1
    elif price < 500:
        return 0.5
    elif price < 1000:
        return 1.0
    return 5.0


def close(api: sj.Shioaji, stock_id: str, quantity: int):
    contract = api.Contracts.Stocks[stock_id]
    price = api.snapshots([contract])[0].close
    raw = price * 0.995
    tick = _tick(raw)
    limit_price = round(int(raw / tick) * tick, 2)  # floor，確保賣得掉

    trade = api.place_order(
        contract,
        api.Order(
            price=limit_price,
            quantity=quantity,
            action=sj.Action.Sell,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
        ),
    )
    return trade


def open(api: sj.Shioaji, stock_id: str, quantity: int, action: sj.Action = sj.Action.Buy):
    contract = api.Contracts.Stocks[stock_id]
    close = api.snapshots([contract])[0].close
    raw = close * 1.01
    tick = _tick(raw)
    limit_price = round(round(raw / tick) * tick, 2)

    trade = api.place_order(
        contract,
        api.Order(
            price=limit_price,
            quantity=quantity,
            action=action,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
        ),
    )
    return trade


def get_positions(api: sj.Shioaji) -> dict[str, int]:
    """回傳 {stock_id: quantity_lots} 目前持倉"""
    return {p.code: p.quantity for p in api.list_positions(api.stock_account)}


def list_orders(api: sj.Shioaji) -> list:
    api.update_status()
    trades = sorted(api.list_trades(), key=lambda t: t.status.order_datetime)
    return [
        {
            "order_id": t.order.id,
            "time": t.status.order_datetime,
            "stock_id": t.contract.code,
            "action": t.order.action,
            "price": t.order.price,
            "quantity": t.order.quantity,
            "filled": t.status.deal_quantity,
            "status": t.status.status,
            "msg": t.status.msg,
        }
        for t in trades
    ]


def get_settlements(api: sj.Shioaji) -> list:
    """交割明細（對帳單）"""
    print("交割明細（對帳單）")
    return [
        {
            "date": s.t_date,
            "stock_id": s.code,
            "action": s.action,
            "quantity": s.quantity,
            "price": s.price,
            "amount": s.amount,
        }
        for s in api.settlements(api.stock_account)
    ]


if __name__ == "__main__":
    api = test()
    # print(open(api, "0050", 1))
    # for i in list_orders(api):
    # print(i)

    # for s in get_settlements(api):
    # print(s)

    api.logout()
    # api = login()
    # api.logout()
