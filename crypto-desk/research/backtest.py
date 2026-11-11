"""Strategist's backtest engine for the crypto desk.

Rules this file enforces (see DESK_RULES.md):
- Real data only: bars from Alpaca's crypto feed, downloaded by fetch_data.py.
- No lookahead: signals use the close of bar i; fills happen at the open of bar i+1
  (i+2 when testing "one bar late").
- Costs on every fill: fee + slippage per side.
- Position size from an ATR stop so a full stop-out loses RISK of equity.
- Long only, no leverage: total open exposure never exceeds 100% of equity.

COST ASSUMPTIONS (estimates, check against Alpaca's current fee schedule):
- FEE = 0.25% per side (Alpaca's lowest-volume taker tier).
- SLIP = 0.05% per side (estimate). Stress test doubles it.
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FEE = 0.0025
SLIP = 0.0005
RISK = 0.02


# ----------------------------------------------------------------------------- data
def load_hourly(symbol):
    df = pd.read_csv(os.path.join(DATA, f"{symbol}_1h.csv"), parse_dates=["t"]).set_index("t")
    full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    df = df.reindex(full)
    # Hours with no trades on the venue: carry the last close forward, zero volume.
    df["c"] = df["c"].ffill()
    for col in ("o", "h", "l"):
        df[col] = df[col].fillna(df["c"])
    df["v"] = df["v"].fillna(0.0)
    return df


def longest_gap_hours(symbol):
    """Longest stretch with no bars in the raw file. Long gaps mean the venue didn't list the coin."""
    t = pd.read_csv(os.path.join(DATA, f"{symbol}_1h.csv"), usecols=["t"], parse_dates=["t"])["t"]
    return float(t.diff().max() / pd.Timedelta(hours=1))


def resample(df, rule):
    if rule == "1h":
        return df
    return df.resample(rule, label="left", closed="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()


def atr(df, n):
    prev = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - prev).abs(), (df["l"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ----------------------------------------------------------------------------- engine
def simulate_symbol(df, entry, exit_sig, stop_dist, max_hold, delay=0, slip=SLIP, fee=FEE, tp_dist=None):
    """Return a list of trades for one symbol.

    entry, exit_sig: boolean arrays decided at bar close.
    stop_dist: price distance of the stop, known at the signal bar.
    tp_dist: optional take-profit distance (same units). Stop is checked first in a bar (conservative).
    """
    o, h, l, c = (df[k].to_numpy() for k in "ohlc")
    idx = df.index
    n = len(df)
    ent, ex, sd = np.asarray(entry), np.asarray(exit_sig), np.asarray(stop_dist)
    tpd = None if tp_dist is None else np.asarray(tp_dist)
    trades = []
    i = 0
    while i < n:
        if not ent[i] or not np.isfinite(sd[i]) or sd[i] <= 0:
            i += 1
            continue
        j = i + 1 + delay
        if j >= n:
            break
        fill = o[j] * (1 + slip)
        stop = o[j] - sd[i]
        if stop <= 0:
            i += 1
            continue
        tp = o[j] + tpd[i] if tpd is not None and np.isfinite(tpd[i]) else np.inf
        k, exit_px, reason = j, None, None
        while k < n:
            if l[k] <= stop:
                exit_px = min(o[k], stop) * (1 - slip) if k > j else stop * (1 - slip)
                reason = "stop"
                break
            if h[k] >= tp:
                exit_px = (max(o[k], tp) if k > j else tp) * (1 - slip)
                reason = "target"
                break
            if (ex[k] or k - j + 1 >= max_hold) and k + 1 + delay < n:
                k_out = k + 1 + delay
                exit_px = o[k_out] * (1 - slip)
                reason = "signal" if ex[k] else "time"
                # Bars k+1..k_out-1 can still hit the stop while we wait (delay > 0).
                for m in range(k + 1, k_out):
                    if l[m] <= stop:
                        exit_px, k_out, reason = min(o[m], stop) * (1 - slip), m, "stop"
                        break
                k = k_out
                break
            k += 1
        if exit_px is None:  # still open at end of data: mark at last close
            k, exit_px, reason = n - 1, c[n - 1], "open"
        # Bar-by-bar return path from fill to exit, fees charged on entry and exit bars.
        path_px = np.concatenate([[fill], c[j:k], [exit_px]])
        bar_ret = path_px[1:] / path_px[:-1] - 1
        bar_ret[0] -= fee
        bar_ret[-1] -= fee
        net = exit_px * (1 - fee) / (fill * (1 + fee)) - 1
        risk_pct = (fill - stop) / fill + 2 * (fee + slip)
        trades.append({
            "entry_t": idx[j], "exit_t": idx[k], "f": min(1.0, RISK / risk_pct),
            "ret": net, "reason": reason, "bar_t": idx[j:k + 1], "bar_ret": bar_ret,
        })
        i = k if k > i else i + 1
    return trades


def portfolio(trades_by_symbol):
    """Accept trades in time order while total exposure stays <= 100%. Return daily returns + trades."""
    allt = sorted((t for ts in trades_by_symbol.values() for t in ts), key=lambda t: t["entry_t"])
    accepted, open_ = [], []
    for t in allt:
        open_ = [x for x in open_ if x["exit_t"] > t["entry_t"]]
        if sum(x["f"] for x in open_) + t["f"] <= 1.0 + 1e-9:
            accepted.append(t)
            open_.append(t)
    if not accepted:
        return pd.Series(dtype=float), []
    parts = [pd.Series(t["f"] * t["bar_ret"], index=t["bar_t"]) for t in accepted]
    bar = pd.concat(parts).groupby(level=0).sum()
    daily = bar.groupby(bar.index.floor("1D")).sum()
    return daily, accepted


def run(strategy, params, data, delay=0, slip=SLIP):
    """strategy(df_by_symbol, **params) -> {symbol: (df, entry, exit, stop_dist, max_hold)}"""
    sigs = strategy(data, **params)
    tbs = {s: simulate_symbol(*v[:5], delay=delay, slip=slip, tp_dist=v[5] if len(v) > 5 else None)
           for s, v in sigs.items()}
    return portfolio(tbs)


# ----------------------------------------------------------------------------- metrics
def full_daily(daily, start, end):
    utc = lambda x: pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x).tz_convert("UTC")
    days = pd.date_range(utc(start).floor("1D"), utc(end).floor("1D"), freq="1D")
    return daily.reindex(days, fill_value=0.0)


def sharpe(d):
    return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else 0.0


def max_dd(d):
    eq = (1 + d).cumprod()
    return float((eq / eq.cummax() - 1).min())


def deflated_sharpe(d, n_trials, trial_sr_var):
    """Bailey & Lopez de Prado (2014). Daily (non-annualized) Sharpe units. Returns a probability."""
    from statistics import NormalDist
    norm = NormalDist()
    x = np.asarray(d, dtype=float)
    sd = x.std()
    sr = x.mean() / sd if sd > 0 else 0.0
    g = 0.5772156649
    sr0 = np.sqrt(trial_sr_var) * ((1 - g) * norm.inv_cdf(1 - 1 / n_trials) + g * norm.inv_cdf(1 - 1 / (n_trials * np.e)))
    t = len(x)
    z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    sk, ku = float((z ** 3).mean()), float((z ** 4).mean())
    denom = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))


def monte_carlo_dd(trades, runs=1000, seed=7):
    """Resample the trade sequence with replacement; return (median, 95th pct, worst) max drawdown."""
    rng = np.random.default_rng(seed)
    r = np.array([t["f"] * t["ret"] for t in trades])
    if len(r) == 0:
        return (0.0, 0.0, 0.0)
    dds = []
    for _ in range(runs):
        eq = np.cumprod(1 + rng.choice(r, size=len(r), replace=True))
        dds.append((eq / np.maximum.accumulate(eq) - 1).min())
    dds = np.array(dds)
    return float(np.median(dds)), float(np.percentile(dds, 5)), float(dds.min())


def score(oos_sr, dsr, mdd, n_trades, edge_in_costs):
    """Desk rubric, each 1-5, fixed BEFORE looking at results."""
    s1 = 1 if oos_sr < 0 else 2 if oos_sr < 0.5 else 3 if oos_sr < 1 else 4 if oos_sr < 1.5 else 5
    s2 = 1 if dsr < 0.5 else 2 if dsr < 0.8 else 3 if dsr < 0.9 else 4 if dsr < 0.95 else 5
    a = abs(mdd)
    s3 = 1 if a > 0.4 else 2 if a > 0.3 else 3 if a > 0.2 else 4 if a > 0.1 else 5
    s4 = 1 if n_trades < 30 else 2 if n_trades < 50 else 3 if n_trades < 100 else 4 if n_trades < 200 else 5
    s5 = 1 if edge_in_costs < 0 else 2 if edge_in_costs < 1 else 3 if edge_in_costs < 2 else 4 if edge_in_costs < 3 else 5
    return {"oos_sharpe": s1, "deflated_sharpe": s2, "max_dd": s3, "trades": s4, "edge_after_costs": s5,
            "total": s1 + s2 + s3 + s4 + s5}
