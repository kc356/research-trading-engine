import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import polars as pl

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

MARKETS_SCHEMA = {
    "window_start_ns": pl.Int64,
    "window_end_ns": pl.Int64,
    "slug": pl.Utf8,
    "condition_id": pl.Utf8,
    "token_up": pl.Utf8,
    "token_down": pl.Utf8,
    "closed": pl.Boolean,
    "outcome": pl.Utf8,
}

PRICES_SCHEMA = {
    "window_start_ns": pl.Int64,
    "outcome_token": pl.Utf8,
    "ts_ns": pl.Int64,
    "price": pl.Float64,
}

class Catalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.done: set[int] = set()
        if self.manifest_path.exists():
            self.done = set(json.loads(self.manifest_path.read_text())["windows"])

    def write_window(self, meta: MarketMeta,
                     price_rows: list[tuple[str, int, float]]) -> None:
        date = utc_date(meta.window_start)
        self._append("markets", date, pl.DataFrame([{
            "window_start_ns": meta.window_start * NS,
            "window_end_ns": meta.window_end * NS,
            "slug": meta.slug,
            "condition_id": meta.condition_id,
            "token_up": meta.token_up,
            "token_down": meta.token_down,
            "closed": meta.closed,
            "outcome": meta.outcome,
        }], schema=MARKETS_SCHEMA))
        if price_rows:
            self._append("prices", date, pl.DataFrame([{
                "window_start_ns": meta.window_start * NS,
                "outcome_token": o,
                "ts_ns": t * NS,
                "price": p
            } for o, t, p in price_rows ],schema=PRICES_SCHEMA))
        self.done.add(meta.window_start)

    def _append(self, dataset: str, date: str, df:pl.DataFrame) -> None:
        d = self.root / dataset / f"data={date}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "part.parquet"
        if f.exists():
            df = pl.concat([pl.read_parquet(f), df]).unique(keep="last")
        df.write_parquet(f)

    def flush_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps({"windows": sorted(self.done)})
        )

