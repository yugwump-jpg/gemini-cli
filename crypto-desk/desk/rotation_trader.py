"""Weekly relative-strength rotation: PAPER EXPERIMENT ONLY.

Status: did NOT pass the desk's 18/25 bar (scored 15/25 on 2026-11-11). Owner approved running it
on paper as a labeled experiment. Results do not count toward the promotion ladder.

Rules (same as the backtest, research/strategies.py rs_rotation, final params lookback=56, top_k=2, sma=100):
- Universe: BTC ETH LTC LINK BCH UNI AVAX AAVE (SOL excluded: data gap).
- Decide on the Sunday UTC daily close (runs Monday ~00:15 UTC = Sunday evening Mountain Time).
- Hold the top 2 coins by 56-day return among those above their 100-day average; otherwise cash.
- Each position sized so a 3x ATR(20) stop loses 2% of desk equity; total exposure <= 100%. No leverage.
- Stop placed on Alpaca as a stop-limit sell right after the buy fills.
- Kill switch: stale data (last bar > 3h old) or 3 losing trades in a row -> no new entries.

Desk equity = $200 + (paper account equity - $100,000). This assumes nothing else trades in the paper account.

Usage (from crypto-desk/):  python -I desk/rotation_trader.py            # dry run, prints the plan
                            python -I desk/rotation_trader.py --paper    # submit paper orders
                            add --force to rebalance on a day other than Monday UTC
"""
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "research"))
import alpaca  # noqa: E402
from fetch_data import fetch  # noqa: E402

UNIVERSE = ["BTC", "ETH", "LTC", "LINK", "BCH", "UNI", "AVAX", "AAVE"]
LOOKBACK, TOP_K, SMA, ATR_N, STOP_K = 56, 2, 100, 20, 3
RISK, ROUND_TRIP_COST = 0.02, 0.006  # cost is an estimate: 0.25% fee + 0.05% slippage per side
START_CAPITAL, PAPER_START = 200.0, 100_000.0
TAG = "desk-rot"
MIN_ORDER = 10.0  # Alpaca rejects crypto orders with cost basis under $10 (seen 2026-11-11)


def daily_bars(symbol, days=200):
    """Alpaca 1Day crypto bars are UTC days (checked 2026-11-11: identical to resampled hourly bars)."""
    now = dt.datetime.now(dt.timezone.utc)
    b = pd.DataFrame(fetch(symbol, timeframe="1Day", start=(now - dt.timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")))
    b["t"] = pd.to_datetime(b["t"])
    d = b.set_index("t")[["o", "h", "l", "c", "v"]]
    recent = fetch(symbol, timeframe="1Hour", start=(now - dt.timedelta(hours=12)).strftime("%Y-%m-%dT%H:00:00Z"))
    last_bar = pd.Timestamp(recent[-1]["t"]) if recent else d.index[-1]
    today = pd.Timestamp.now(tz="UTC").floor("1D")
    return d[d.index < today], last_bar  # drop the incomplete current day


def plan():
    bars, last = {}, {}
    for s in UNIVERSE:
        bars[s], last[s] = daily_bars(s)
    closes = pd.DataFrame({s: d["c"] for s, d in bars.items()}).dropna()
    mom = closes.iloc[-1] / closes.iloc[-1 - LOOKBACK] - 1
    above = closes.iloc[-1] > closes.iloc[-SMA:].mean()
    ranked = mom[above].sort_values(ascending=False)
    hold = list(ranked.index[:TOP_K])
    atr = {}
    for s, d in bars.items():
        prev = d["c"].shift(1)
        tr = pd.concat([d["h"] - d["l"], (d["h"] - prev).abs(), (d["l"] - prev).abs()], axis=1).max(axis=1)
        atr[s] = float(tr.iloc[-ATR_N:].mean())
    now = pd.Timestamp.now(tz="UTC")
    stale = {s: round((now - t).total_seconds() / 3600, 1) for s, t in last.items() if now - t > pd.Timedelta(hours=3)}
    return {
        "decision_close": str(closes.index[-1].date()),
        "momentum_56d": {s: round(float(v), 4) for s, v in mom.sort_values(ascending=False).items()},
        "above_100d_avg": {s: bool(v) for s, v in above.items()},
        "hold": hold,
        "price": {s: float(closes[s].iloc[-1]) for s in UNIVERSE},
        "atr20": atr,
        "stale_hours": stale,
    }


def losing_streak():
    """Consecutive losing round trips, most recent first, from the account's fill history."""
    fills = sorted(alpaca.activities("FILL") or [], key=lambda f: f["transaction_time"])
    cost, results = {}, []
    for f in fills:
        sym, qty, px = f["symbol"], float(f["qty"]), float(f["price"])
        if f["side"] == "buy":
            c = cost.setdefault(sym, [0.0, 0.0])
            c[0] += qty * px
            c[1] += qty
        elif sym in cost and cost[sym][1] > 0:
            avg = cost[sym][0] / cost[sym][1]
            results.append(px > avg)
            cost[sym][1] = max(0.0, cost[sym][1] - qty)
            cost[sym][0] = avg * cost[sym][1]
    streak = 0
    for won in reversed(results):
        if won:
            break
        streak += 1
    return streak


def wait_filled(order_id, timeout=60):
    for _ in range(timeout // 2):
        o = alpaca.get_order(order_id)
        if o["status"] == "filled":
            return o
        if o["status"] in ("canceled", "rejected", "expired"):
            return o
        time.sleep(2)
    return alpaca.get_order(order_id)


def main():
    submit = "--paper" in sys.argv
    force = "--force" in sys.argv
    p = plan()
    acct = alpaca.account()
    desk_equity = START_CAPITAL + float(acct["equity"]) - PAPER_START
    held = {x["symbol"].replace("USD", ""): x for x in alpaca.positions() or [] if x["asset_class"] == "crypto"}
    is_rebalance_day = pd.Timestamp.now(tz="UTC").dayofweek == 0  # Monday UTC = after Sunday close
    streak = losing_streak()
    blocked = []
    if p["stale_hours"]:
        blocked.append(f"stale data: {p['stale_hours']}")
    if streak >= 3:
        blocked.append(f"{streak} losing trades in a row")

    actions = []
    if is_rebalance_day or force:
        for s, pos in held.items():
            if s not in p["hold"]:
                actions.append({"action": "SELL", "symbol": s, "qty": pos["qty"], "why": "dropped out of top 2"})
        exposure = sum(float(x["market_value"]) for x in held.values() if x["symbol"].replace("USD", "") in p["hold"])
        for s in p["hold"]:
            if s in held:
                continue
            if blocked:
                actions.append({"action": "SKIP BUY", "symbol": s, "why": "; ".join(blocked)})
                continue
            px, sd = p["price"][s], STOP_K * p["atr20"][s]
            f = min(1.0, RISK / (sd / px + ROUND_TRIP_COST))
            notional = round(min(desk_equity * f, desk_equity - exposure), 2)
            if notional < MIN_ORDER:
                actions.append({"action": "SKIP BUY", "symbol": s, "notional": notional,
                                "why": f"below Alpaca's ${MIN_ORDER:.0f} crypto minimum (or no room under 100% exposure)"})
                continue
            if (px - sd) / px * notional < MIN_ORDER:
                actions.append({"action": "SKIP BUY", "symbol": s, "notional": notional,
                                "why": f"stop order would be below Alpaca's ${MIN_ORDER:.0f} minimum, so the position can't be protected"})
                continue
            exposure += notional
            actions.append({"action": "BUY", "symbol": s, "notional": notional, "size_pct": round(f * 100, 1),
                            "stop": round(px - sd, 4), "why": f"top 2 by 56d momentum ({p['momentum_56d'][s]:+.1%})"})
    report = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "mode": "PAPER" if submit else "DRY RUN",
              "desk_equity": round(desk_equity, 2), "paper_account_equity": acct["equity"],
              "rebalance_day": is_rebalance_day, "kill_switch": blocked or "clear", "losing_streak": streak,
              "positions": {s: {"qty": x["qty"], "value": x["market_value"], "unrealized_pl": x["unrealized_pl"]} for s, x in held.items()},
              "plan": p, "actions": actions, "results": []}

    if submit:
        for a in actions:
            sym = f"{a['symbol']}/USD"
            if a["action"] == "SELL":
                for o in alpaca.orders("open") or []:
                    if o["symbol"] in (sym, sym.replace("/", "")):
                        alpaca.cancel(o["id"])
                o = alpaca.submit(symbol=sym, qty=a["qty"], side="sell", type="market", time_in_force="gtc",
                                  client_order_id=f"{TAG}-sell-{a['symbol']}-{int(time.time())}")
                report["results"].append({"sell": a["symbol"], "status": wait_filled(o["id"])["status"]})
            elif a["action"] == "BUY":
                o = wait_filled(alpaca.submit(symbol=sym, notional=str(a["notional"]), side="buy", type="market",
                                              time_in_force="gtc", client_order_id=f"{TAG}-buy-{a['symbol']}-{int(time.time())}")["id"])
                r = {"buy": a["symbol"], "status": o["status"], "fill": o.get("filled_avg_price")}
                if o["status"] == "filled":
                    pos = next((x for x in alpaca.positions() if x["symbol"].replace("USD", "") == a["symbol"]), None)
                    if pos:
                        stop = a["stop"]
                        try:
                            so = alpaca.submit(symbol=sym, qty=pos["qty"], side="sell", type="stop_limit",
                                               stop_price=str(round(stop, 6)), limit_price=str(round(stop * 0.99, 6)),
                                               time_in_force="gtc", client_order_id=f"{TAG}-stop-{a['symbol']}-{int(time.time())}")
                            r["stop_order"] = so["status"]
                        except RuntimeError as e:
                            # Missing stop = rule violation. Flatten rather than hold an unprotected position.
                            r["stop_order"] = f"FAILED ({e}); position closed"
                            alpaca.submit(symbol=sym, qty=pos["qty"], side="sell", type="market", time_in_force="gtc")
                report["results"].append(r)

    out = os.path.join(HERE, "..", "journal", f"{dt.date.today()}_rotation.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("mode", "desk_equity", "rebalance_day", "kill_switch", "positions", "actions", "results")}, indent=1, default=str))
    print("hold:", p["hold"], "| momentum:", p["momentum_56d"])


if __name__ == "__main__":
    main()
