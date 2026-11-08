"""Nightly research run. Usage: python -I research/run_research.py  (from crypto-desk/)

For each strategy family:
1. Run every parameter variant over the full history (each one is a trial).
2. In-sample vs out-of-sample: pick the best variant on 2021-01..2024-06, report it on 2024-07..now.
3. Walk-forward: 24-month train / 6-month test, re-picking the variant each fold; stitch test windows.
4. Stress: walk-forward picks re-run one bar late with double slippage.
5. Deflated Sharpe on the walk-forward result, counting every trial across all families.
6. Monte Carlo (1,000 resamples) worst drawdown on walk-forward trades.
7. Score with the desk rubric (18+ of 25 and >= 30 OOS trades to qualify).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (FEE, SLIP, deflated_sharpe, full_daily, load_hourly, longest_gap_hours, max_dd, monte_carlo_dd,  # noqa: E402
                      run, score, sharpe)
from strategies import GRIDS, variants  # noqa: E402

UNIVERSE = ["BTC", "ETH", "SOL", "LTC", "LINK", "BCH", "UNI", "AVAX", "AAVE"]
ROUND_TRIP = 2 * (FEE + SLIP)
OUT = os.path.join(os.path.dirname(__file__), "..", "journal")


def folds(start, end, train_m=24, test_m=6):
    t0 = start
    while True:
        tr_end = t0 + pd.DateOffset(months=train_m)
        te_end = min(tr_end + pd.DateOffset(months=test_m), end)
        if tr_end >= end:
            return
        yield (t0, tr_end, tr_end, te_end)
        t0 = t0 + pd.DateOffset(months=test_m)


def window(d, a, b):
    return d[(d.index >= a) & (d.index < b)]


def trades_in(trades, a, b):
    return [t for t in trades if a <= t["entry_t"] < b]


def main():
    present = [s for s in UNIVERSE if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", f"{s}_1h.csv"))]
    gaps = {s: longest_gap_hours(s) for s in present}
    excluded = {s: f"longest data gap {g:.0f}h" for s, g in gaps.items() if g > 24}
    data = {s: load_hourly(s) for s in present if s not in excluded}
    print("excluded:", excluded)
    start = max(pd.Timestamp("2021-01-01", tz="UTC"), min(d.index[0] for d in data.values()).floor("1D"))
    end = min(d.index[-1] for d in data.values()).floor("1D")
    split = pd.Timestamp("2024-07-01", tz="UTC")
    print("universe", list(data), start.date(), "->", end.date())

    results, all_trial_srs = {}, []
    for fam, (fn, grid) in GRIDS.items():
        vs = list(variants(grid))
        runs = []
        for p in vs:
            daily, trades = run(fn, p, data)
            d = full_daily(daily, start, end - pd.Timedelta(days=1))
            runs.append((p, d, trades))
            all_trial_srs.append(d.mean() / d.std() if d.std() > 0 else 0.0)
        results[fam] = {"fn": fn, "runs": runs}
        print(fam, "variants:", len(vs))

    n_trials = len(all_trial_srs)
    sr_var = float(np.var(all_trial_srs))
    report = {"universe": list(data), "excluded": excluded, "start": str(start.date()), "end": str(end.date()), "n_trials": n_trials,
              "fee_per_side_est": FEE, "slip_per_side_est": SLIP, "families": {}}

    for fam, r in results.items():
        runs, fn = r["runs"], r["fn"]
        # In-sample vs out-of-sample
        is_best = max(runs, key=lambda x: sharpe(window(x[1], start, split)))
        is_sr = sharpe(window(is_best[1], start, split))
        oos_sr_fixed = sharpe(window(is_best[1], split, end))

        # Walk-forward
        wf_daily, wf_trades, picks, stress_daily = [], [], [], []
        stress_cache = {}
        for a, b, c, e in folds(start, end):
            best = max(runs, key=lambda x: sharpe(window(x[1], a, b)) if len(trades_in(x[2], a, b)) >= 10 else -9)
            picks.append({"test": f"{c.date()}..{e.date()}", "params": best[0]})
            wf_daily.append(window(best[1], c, e))
            wf_trades += trades_in(best[2], c, e)
            key = json.dumps(best[0], sort_keys=True)
            if key not in stress_cache:
                sd, _ = run(fn, best[0], data, delay=1, slip=2 * SLIP)
                stress_cache[key] = full_daily(sd, start, end - pd.Timedelta(days=1))
            stress_daily.append(window(stress_cache[key], c, e))
        wf = pd.concat(wf_daily)
        st = pd.concat(stress_daily)
        n = len(wf_trades)
        avg_net = float(np.mean([t["ret"] for t in wf_trades])) if n else 0.0
        win = float(np.mean([t["ret"] > 0 for t in wf_trades])) if n else 0.0
        dsr = deflated_sharpe(wf, n_trials, sr_var)
        mc = monte_carlo_dd(wf_trades)
        # Edge after costs, size-weighted: net P&L per unit of round-trip cost actually paid.
        edge = (sum(t["f"] * t["ret"] for t in wf_trades) / sum(t["f"] * ROUND_TRIP for t in wf_trades)) if n else 0.0
        sc = score(sharpe(wf), dsr, max_dd(wf), n, edge)
        final = max(runs, key=lambda x: sharpe(window(x[1], end - pd.DateOffset(months=24), end)))
        y22 = window(final[1], pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))
        report["families"][fam] = {
            "in_sample": {"window": f"{start.date()}..{split.date()}", "best_params": is_best[0], "sharpe": round(is_sr, 2)},
            "out_of_sample_fixed": {"window": f"{split.date()}..{end.date()}", "sharpe": round(oos_sr_fixed, 2)},
            "walk_forward": {"window": f"{wf.index[0].date()}..{wf.index[-1].date()}", "sharpe": round(sharpe(wf), 2),
                             "total_return": round(float((1 + wf).prod() - 1), 4), "max_dd": round(max_dd(wf), 4),
                             "trades": n, "win_rate": round(win, 3), "avg_net_trade_ret": round(avg_net, 4),
                             "deflated_sharpe_prob": round(dsr, 3), "picks": picks},
            "stress_1bar_late_2x_slip": {"sharpe": round(sharpe(st), 2), "total_return": round(float((1 + st).prod() - 1), 4)},
            "monte_carlo_1000": {"median_dd": round(mc[0], 4), "p95_dd": round(mc[1], 4), "worst_dd": round(mc[2], 4)},
            "year_2022_with_final_params_in_sample": {"params": final[0], "return": round(float((1 + y22).prod() - 1), 4),
                                                       "max_dd": round(max_dd(y22), 4)},
            "final_params_for_paper": final[0],
            "score": sc,
            "edge_in_round_trip_costs": round(edge, 2),
            "qualifies": sc["total"] >= 18 and n >= 30 and edge > 0 and sharpe(wf) > 0,
        }
        print(fam, json.dumps(report["families"][fam]["score"]), "qualifies:", report["families"][fam]["qualifies"])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"research_{end.date()}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("saved", path)


if __name__ == "__main__":
    main()
