"""Tonight's four strategy families. Every parameter combination listed in GRIDS counts as a trial
for the deflated Sharpe, whether it looks good or not.

Each strategy returns {symbol: (df, entry, exit, stop_distance, max_hold_bars)}.
All lookbacks use shift(1) so a bar never sees its own future.
"""
import itertools

import numpy as np
import pandas as pd

from backtest import atr, resample


def trend_breakout(data, n_high, n_low, stop_k):
    """S1 Daily trend breakout. Other side: traders selling into strength too early."""
    out = {}
    for s, h1 in data.items():
        df = resample(h1, "1D")
        a = atr(df, 20)
        entry = (df["c"] > df["h"].rolling(n_high).max().shift(1)) & (df["c"] > df["c"].rolling(100).mean())
        exit_ = df["c"] < df["l"].rolling(n_low).min().shift(1)
        out[s] = (df, entry.fillna(False).to_numpy(), exit_.fillna(False).to_numpy(), (stop_k * a).to_numpy(), 10_000)
    return out


def flush_reversion(data, z, hold_h, trend_filter):
    """S2 Hourly mean reversion after a sharp 24h drop. Other side: forced sellers (liquidations, panic)."""
    out = {}
    for s, df in data.items():
        r1 = np.log(df["c"]).diff()
        vol24 = r1.rolling(24 * 30).std().shift(1) * np.sqrt(24)
        r24 = np.log(df["c"] / df["c"].shift(24))
        entry = r24 < -z * vol24
        if trend_filter:
            entry &= df["c"] > df["c"].rolling(24 * 200).mean()
        a = atr(df, 24)
        no_exit = np.zeros(len(df), dtype=bool)
        out[s] = (df, entry.fillna(False).to_numpy(), no_exit, (3 * a).to_numpy(), hold_h)
    return out


def volume_breakout(data, n_high, vol_mult, n_low):
    """S3 4h breakout on expanded volume. Other side: range traders whose stops get run.
    Caveat: volume is Alpaca's own venue volume, not global volume."""
    out = {}
    for s, h1 in data.items():
        df = resample(h1, "4h")
        a = atr(df, 30)
        vol_avg = df["v"].rolling(30).mean().shift(1)
        entry = (df["c"] > df["h"].rolling(n_high).max().shift(1)) & (df["v"] > vol_mult * vol_avg)
        exit_ = df["c"] < df["l"].rolling(n_low).min().shift(1)
        out[s] = (df, entry.fillna(False).to_numpy(), exit_.fillna(False).to_numpy(), (2 * a).to_numpy(), 10_000)
    return out


def rs_rotation(data, lookback, top_k, sma):
    """S4 Weekly relative-strength rotation among majors. Other side: holders slow to rotate out of laggards."""
    daily = {s: resample(h1, "1D") for s, h1 in data.items()}
    closes = pd.DataFrame({s: d["c"] for s, d in daily.items()})
    mom = closes / closes.shift(lookback) - 1
    above = closes > closes.rolling(sma).mean()
    rank = mom.where(above).rank(axis=1, ascending=False)
    hold = (rank <= top_k)
    rebal = pd.Series(closes.index.dayofweek == 6, index=closes.index)  # decide on Sunday close
    out = {}
    for s, df in daily.items():
        hs = hold[s].reindex(df.index).fillna(False)
        rb = rebal.reindex(df.index).fillna(False)
        entry = rb & hs
        exit_ = rb & ~hs
        a = atr(df, 20)
        out[s] = (df, entry.to_numpy(), exit_.to_numpy(), (3 * a).to_numpy(), 10_000)
    return out


def snatch_and_sell(data, entry_mode, target, max_hold_h):
    """S5 Quick in-and-out ("snatch and sell"): small profit target, equal-size stop, short time limit.
    dip: buy after a sharp 3-hour drop (other side: panicked short-term sellers).
    pop: buy a fresh 24h high on 2x volume (other side: late sellers / shorts squeezed).
    Fees (~0.6% round trip, estimate) take a big bite out of a 1-3% target."""
    out = {}
    for s, df in data.items():
        r1 = np.log(df["c"]).diff()
        if entry_mode == "dip":
            vol3 = r1.rolling(24 * 30).std().shift(1) * np.sqrt(3)
            entry = np.log(df["c"] / df["c"].shift(3)) < -2.0 * vol3
        else:
            vol_avg = df["v"].rolling(24).mean().shift(1)
            entry = (df["c"] > df["h"].rolling(24).max().shift(1)) & (df["v"] > 2 * vol_avg)
        dist = (target * df["c"]).to_numpy()
        no_exit = np.zeros(len(df), dtype=bool)
        out[s] = (df, entry.fillna(False).to_numpy(), no_exit, dist, max_hold_h, dist)
    return out


GRIDS = {
    "S1 trend breakout (1D)": (trend_breakout, {"n_high": [20, 55], "n_low": [10, 20], "stop_k": [2, 3]}),
    "S2 flush reversion (1h)": (flush_reversion, {"z": [2.0, 2.5, 3.0], "hold_h": [12, 48], "trend_filter": [False, True]}),
    "S3 volume breakout (4h)": (volume_breakout, {"n_high": [30, 90], "vol_mult": [2, 3], "n_low": [10, 20]}),
    "S4 RS rotation (1W)": (rs_rotation, {"lookback": [28, 56], "top_k": [1, 2], "sma": [50, 100]}),
    "S5 snatch and sell (1h)": (snatch_and_sell, {"entry_mode": ["dip", "pop"], "target": [0.01, 0.02, 0.03],
                                                  "max_hold_h": [6, 24]}),
}


def variants(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))
