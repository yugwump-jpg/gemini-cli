"""Minimal Alpaca PAPER client. Refuses any host other than the paper endpoint.

Auth: the Key ID comes from the APCA_API_KEY_ID environment variable. The Secret is never in code
or env: the session's network proxy injects the APCA-API-SECRET-KEY header for paper-api.alpaca.markets.
"""
import json
import os
import urllib.error
import urllib.request

PAPER = "https://paper-api.alpaca.markets"


def _req(method, path, body=None):
    url = PAPER + path
    assert url.startswith(PAPER + "/"), "paper endpoint only"
    key_id = os.environ.get("APCA_API_KEY_ID")
    if not key_id:
        raise SystemExit("APCA_API_KEY_ID is not set")
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": "proxy-injected",
               "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Alpaca {method} {path} -> {e.code}: {e.read().decode()[:300]}") from None


def account():
    return _req("GET", "/v2/account")


def positions():
    return _req("GET", "/v2/positions")


def orders(status="open", limit=100):
    return _req("GET", f"/v2/orders?status={status}&limit={limit}&nested=false")


def get_order(order_id):
    return _req("GET", f"/v2/orders/{order_id}")


def submit(**order):
    return _req("POST", "/v2/orders", order)


def cancel(order_id):
    return _req("DELETE", f"/v2/orders/{order_id}")


def activities(kind="FILL", page_size=100):
    return _req("GET", f"/v2/account/activities/{kind}?page_size={page_size}")
