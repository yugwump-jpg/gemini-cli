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


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in sys.argv[1:] or UNIVERSE:
        bars = fetch(s)
        with open(os.path.join(OUT, f"{s}_1h.csv"), "w") as f:
            f.write("t,o,h,l,c,v\n")
            for b in bars:
                f.write(f'{b["t"]},{b["o"]},{b["h"]},{b["l"]},{b["c"]},{b["v"]}\n')
        print(s, len(bars), bars[0]["t"], bars[-1]["t"])
