"""Download hourly crypto bars from Alpaca's public crypto data API (no key needed).

Source: https://data.alpaca.markets/v1beta3/crypto/us/bars
Saves one CSV per symbol to crypto-desk/data/.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

UNIVERSE = ["BTC", "ETH", "SOL", "LTC", "LINK", "BCH", "UNI", "AVAX", "AAVE"]
URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch(symbol, timeframe="1Hour", start="2021-01-01T00:00:00Z"):
    rows, token = [], None
    while True:
        q = {"symbols": f"{symbol}/USD", "timeframe": timeframe, "start": start, "limit": 10000}
        if token:
            q["page_token"] = token
        with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            d = json.load(r)
        rows += d["bars"].get(f"{symbol}/USD", [])
        token = d.get("next_page_token")
        if not token:
            return rows
        time.sleep(0.3)


def last_timestamp(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200))
        return f.read().decode().strip().splitlines()[-1].split(",")[0]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in sys.argv[1:] or UNIVERSE:
        path = os.path.join(OUT, f"{s}_1h.csv")
        if os.path.exists(path):
            # Incremental: fetch from the last saved hour onward, append only newer bars.
            last = last_timestamp(path)
            bars = [b for b in fetch(s, start=last) if b["t"] > last]
            mode = "a"
        else:
            bars, mode = fetch(s), "w"
        with open(path, mode) as f:
            if mode == "w":
                f.write("t,o,h,l,c,v\n")
            for b in bars:
                f.write(f'{b["t"]},{b["o"]},{b["h"]},{b["l"]},{b["c"]},{b["v"]}\n')
        print(s, "new bars:", len(bars), "last:", bars[-1]["t"] if bars else "unchanged", flush=True)
