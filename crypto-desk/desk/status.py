"""Facts for the morning brief: paper account, desk positions, market moves, latest research verdicts.
Usage (from crypto-desk/): python -I desk/status.py
"""
import datetime as dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "research"))
import alpaca  # noqa: E402
from fetch_data import fetch  # noqa: E402

START_CAPITAL, PAPER_START = 200.0, 100_000.0


def market():
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=8)).strftime("%Y-%m-%dT00:00:00Z")
    out = {}
    for s in ["BTC", "ETH", "AVAX", "BCH", "UNI", "AAVE", "LTC", "LINK"]:
        b = fetch(s, timeframe="1Hour", start=start)
        if len(b) < 169:
            continue
        last = b[-1]["c"]
        out[s] = {"price": last, "chg_24h": round(last / b[-25]["c"] - 1, 4), "chg_7d": round(last / b[-169]["c"] - 1, 4),
                  "last_bar_utc": b[-1]["t"]}
    return out


def main():
    acct = alpaca.account()
    pos = alpaca.positions() or []
    research = sorted(glob.glob(os.path.join(HERE, "..", "journal", "research_*.json")))
    verdicts = {}
    if research:
        r = json.load(open(research[-1]))
        verdicts = {"file": os.path.basename(research[-1]), "n_trials": r["n_trials"],
                    "families": {f: {"score": v["score"]["total"], "qualifies": v["qualifies"],
                                     "wf_sharpe": v["walk_forward"]["sharpe"], "wf_max_dd": v["walk_forward"]["max_dd"]}
                                 for f, v in r["families"].items()}}
    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "desk_equity": round(START_CAPITAL + float(acct["equity"]) - PAPER_START, 2),
        "desk_pl_today": round(float(acct["equity"]) - float(acct["last_equity"]), 2),
        "positions": [{"symbol": p["symbol"], "qty": p["qty"], "value": p["market_value"], "unrealized_pl": p["unrealized_pl"],
                       "entry": p["avg_entry_price"]} for p in pos],
        "open_orders": [{"symbol": o["symbol"], "type": o["type"], "side": o["side"], "stop": o.get("stop_price")}
                        for o in alpaca.orders("open") or []],
        "market": market(),
        "research": verdicts,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
