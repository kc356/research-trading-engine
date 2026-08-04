import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from requests import Response

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
WINDOW_S = 300
NS = 1_000_000_000

def slug_for_window(start_unix: int) -> str:
    assert start_unix % WINDOW_S == 0, "window start must be on the 300s grid"
    return f"btc-updown-5m-{start_unix}"

def utc_date(ts_unix: int) -> str:
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d")

class HttpClient:
    def __init__(self, min_interval_s: float = 0.15, max_retries: int = 5) -> None:
        self.session = requests.Session()
        self.min_interval = min_interval_s
        self.max_retries = max_retries
        self._last_call = 0.0

    def get_json(self, url: str, params: dict | None = None) -> list | None:
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                r = self.session.get(url, params=params, timeout=15)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        raise RuntimeError(f"gave up after {self.max_retries} retries: {url}")

@dataclass(frozen=True)
class MarketMeta:
    window_start: int
    window_end: int
    slug: str
    condition_id: str
    token_up: str
    token_down: str
    closed: bool
    outcome: str | None

class GammaClient:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def market_for_window(self, start_unix: int) -> MarketMeta | None:
        slug = slug_for_window(start_unix)
        events = self.http.get_json(f"{GAMMA}/events", params={"slug": slug})
        if not events:
            return None
        markets = events[0].get("markets") or []
        if not markets:
            return None
        m = markets[0]
        outcomes = json.loads(m["outcomes"])
        token_ids = json.loads(m["clobTokenIds"])
        by_name = dict(zip(outcomes, token_ids))
        closed = bool(m.get("closed"))
        outcome = None
        if closed and m.get("outcomePrices"):
            prices = [float(x) for x in json.loads(m["outcomePrices"])]
            outcome = outcomes[prices.index(max(prices))]
        return MarketMeta(
            window_start=start_unix,
            window_end=start_unix + WINDOW_S,
            slug=slug,
            condition_id=m["conditionId"],
            token_up=by_name["Up"],
            token_down=by_name["Down"],
            closed=closed,
            outcome=outcome,
        )



