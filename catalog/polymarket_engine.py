import math
from collections import deque, defaultdict
from pathlib import Path

from pydantic import BaseModel

from contracts import (
    BarEvent, Event, InMemoryBus, InstrumentDefined, MarketResolved,
    MessageBus, OrderFilled, OrderSide, OrderType, SignalEvent, Strategy,
    SubmitOrder, TargetWeights, TimeEvent,
)
from engine import Ledger, MarketCache, MetricsEngine, RiskConfig, RiskEngine, SimClock

import polars as pl

NS = 1_000_000_000


class SeriesAllocatorConfig(BaseModel):
    weight_scale: float = 0.1
    norm_window: int = 500
    norm_min_obs: int = 50
    entry_cutoff_ns: int = 60 * NS

class SeriesAllocator:
    def __init__(self, cfg: SeriesAllocatorConfig, clock: SimClock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._scores: dict[tuple[str, str], float] = {}
        self._hist: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=cfg.norm_window)
        )
        self._expiry: dict[str, int] = {}
        self._dirty = False
        self._bus: MessageBus | None = None

    def register(self, bus: MessageBus) -> None:
        self._bus = bus
        bus.subscribe("signal.*", self._on_signal)
        bus.subscribe("instrument.defined.*", self._on_defined)
        bus.subscribe("market.resolved.*", self._on_resolved)
        bus.subscribe("system.tick_end", self._on_tick_end)

    def _on_defined(self, topic: str, e: Event) -> None:
        assert isinstance(e, InstrumentDefined)
        self._expiry[e.symbol] = e.expiry_ns

    def _on_resolved(self, topic: str, e: Event) -> None:
        assert isinstance(e, MarketResolved)
        self._expiry.pop(e.symbol, None)
        for key in [k for k in self._scores if k[1] == e.symbol]:
            del self._scores[key]
        self._dirty = True

    def _on_signal(self, topic: str, e: Event) -> None:
        assert isinstance(e, SignalEvent)
        history_key = (e.strategy_id, self._series_of(e.symbol))
        history = self._hist[history_key]
        history.append(e.score)
        if len(history) < self.cfg.norm_min_obs:
            z = e.score
        else:
            mean = sum(history) / len(history)
            var = sum((x - mean) ** 2 for x in history) / len(history)
            std = var ** 0.5
            z = 0.0 if std < 1e-12 else math.tanh((e.score - mean) / std / 2.0)
        self._scores[(e.strategy_id, e.symbol)] = z
        self._dirty = True

    def _on_tick_end(self, topic: str, e: Event) -> None:
        if not self._dirty:
            return
        if not self._bus:
            raise ModuleNotFoundError("MemoryBus not defined")
        self._dirty = False
        now = self.clock.now_ns()
        by_instrument: dict[str, list[float]] = defaultdict(list)
        for (_, instrument), s in self._scores.items():
            by_instrument[instrument].append(s)
        weights = {}
        for instrument, ss in by_instrument.items():
            expiry = self._expiry.get(instrument)
            if expiry is None or expiry - now < self.cfg.entry_cutoff_ns:
                continue # too close to resolution, no new instructions
            weights[instrument] = (sum(ss) / len(ss)) * self.cfg.weight_scale
            self._bus.publish(
                "portfolio.target.raw",
                TargetWeights(ts_init=now, ts_event=now, weights=weights)
            )

    def _series_of(self, instrument: str) -> str:
        return instrument.split(":", 1)[0]

class PolymarketExecutionConfig(BaseModel):
    spread_bps_of_price: float = 100.0
    fee_bps: float = 0.0
    min_trade_net_asset_value: float = 0.065

class PolymarketExecution:
    def __init__(self, cfg: PolymarketExecutionConfig, ledger: Ledger, clock: SimClock) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.clock = clock
        self._pending: dict[str, SubmitOrder] = {} # token_symbol -> order
        self._old = 0
        self._bus: MessageBus | None = None

    def register(self, bus: MessageBus) -> None:
        self._bus = bus
        bus.subscribe("data.bar.*", self._on_bar)
        bus.subscribe("portfolio.target", self._on_target)
        bus.subscribe("market.resolved.*", self._on_resolved)

    def _on_bar(self, topic: str, e: Event) -> None:
        assert isinstance(e, BarEvent)
        if not self._bus:
            raise ModuleNotFoundError("MemoryBus not defined")
        up_price, down_price = e.close, 1.0 - e.close
        for outcome, price in (("Up", up_price), ("Down", down_price)):
            token = f"{e.symbol}:{outcome}"
            self.ledger.mark(token, price)
            order = self._pending.pop(token, None)
            if order is None:
                continue
            slip = price * self.cfg.spread_bps_of_price / 1e4
            fill_price = price + slip if order.side == OrderSide.BUY else price - slip
            fill_price = min(0.999, max(0.001, fill_price))
            fee = abs(order.quantity) * fill_price * self.cfg.fee_bps / 1e4
            now = self.clock.now_ns()
            fill = OrderFilled(ts_event=now,
                               ts_init=now,
                               order_id=order.order_id,
                               symbol=token,
                               side=order.side,
                               quantity=order.quantity,
                               price=fill_price,
                               fee=fee,
                               origin=order.origin)
            self.ledger.apply_fill(fill)
            self._bus.publish(f"order.event.{token}", fill)

    def _on_target(self, topic: str, e: Event) -> None:
        assert isinstance(e, TargetWeights)
        nav = self.ledger.net_asset_value()
        if nav <= 0:
            return
        touched = set(e.weights)
        touched |= {s.rsplit(":", 1)[0] for s, q in self.ledger.positions.items() if q}
        for instrument in sorted(touched):
            w = e.weights.get(instrument, 0.0)
            targets = {"Up": max(w, 0.0) * nav, "Down": max(-w, 0.0) * nav}
            for outcome, target_dollars in targets.items():
                token = f"{instrument}:{outcome}"
                price = self.ledger.last_price.get(token)
                if not price or not (0.001 <= price <= 0.999):
                    continue
                current = self.ledger.positions.get(token, 0.0) * price
                delta = target_dollars - current
                if abs(delta) < self.cfg.min_trade_net_asset_value * nav:
                    continue
                qty = delta / price
                self._old += 1
                now = self.clock.now_ns()
                self._pending[token] = SubmitOrder(
                    ts_event=now, ts_init=now, order_id=f"p{self._old}",
                    symbol=token, venue="POLY",
                    side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET, quantity=abs(qty),
                    origin=e.source
                )

    def _on_resolved(self, topic: str, e: Event) -> None:
        assert isinstance(e, MarketResolved)
        for outcome, settle_price in e.settlement.items():
            token = f"{e.symbol}:{outcome}"
            self._pending.pop(token, None)
            qty = self.ledger.positions.get(token, 0.0)
            if qty:
                self.ledger.cash += qty * settle_price
                self.ledger.positions[token] = 0.0
            self.ledger.mark(token, settle_price)

def load_catalog_events(catalog_root: str | Path, series: str = "BTC5m",
                        venue: str = "POLY") -> list[tuple[int, int, str, Event]]:
    """Returns sorted [(ts_ns, priority, topic, event)]
    Priority within a timestamp: resolutions(0) -> definitions(1) -> bars(2).
    A window's resolution and the next window's definition share a timestamp;
    settlement must land before new instruments open.
    """

    root = Path(catalog_root)
    market = pl.read_parquet(root / "markets" / "date=*" / "part.parquet")
    price = pl.read_parquet(root / "prices" / "date=*" / "part.parquet")
    market = market.filter(pl.col("closed") & pl.col("outcome").is_not_null())

    events: list[tuple[int, int, str, Event]] = []
    known = set()
    for r in market.iter_rows(named=True):
        w0, w1 = r["window_start_ns"], r["window_end_ns"]
        symbol = f"{series}:{w0 // NS}"
        known.add(r["window_start_ns"])
        events.append((w0, 1, f"instrument.defined.{series}",InstrumentDefined(
                          ts_event=w0, ts_init=w0, symbol=symbol,
                          series=series, expiry_ns=w1,
                          meta={"slug": r["slug"],
                                "condition_id": r["condition_id"]}
        )))
        up_won = r["outcome"] == "Up"
        events.append((w1, 0, f"market.resolved.{series}", MarketResolved(
            ts_event=w1, ts_init=w1, symbol=symbol, series=series,
            outcome=r["outcome"], settlement={"Up": 1.0 if up_won else 0.0,
                                              "Down": 0.0 if up_won else 1.0}
        )))

    up = price.filter(pl.col("outcome_token") == "Up")
    for r in up.iter_rows(named=True):
        if r["window_start_ns"] not in known:
            continue
        symbol = f"{series}:{r['window_start_ns'] // NS}"
        p, t = r["price"], r["ts_ns"]
        events.append((t, 2, f"data.bar.{venue}.{symbol}", BarEvent(
            ts_event=t, ts_init=t, symbol=symbol, venue=venue,
            open=p, high=p, low=p, close=p, volume=0.0, bar_spec="1m"
        )))
    events.sort(key=lambda x: (x[0], x[1], x[2]))
    return events

class PolymarketBacktestConfig(BaseModel):
    initial_cash: float = 100_000.0
    allocator: SeriesAllocatorConfig = SeriesAllocatorConfig()
    risk: RiskConfig = RiskConfig(max_gross=0.5, max_per_symbol=0.1, drawdown_kill=0.3)
    execution: PolymarketExecutionConfig = PolymarketExecutionConfig()
    periods_per_year: float = 365 * 24 * 60

class PolymarketBacktestEngine:
    def __init__(self, cfg: PolymarketBacktestConfig, strategies: list[Strategy]) -> None:
        self.cfg = cfg
        self.bus = InMemoryBus()
        self.clock = SimClock()
        self.ledger = Ledger(cfg.initial_cash)
        self.cache = MarketCache(self.ledger)
        self.execution = PolymarketExecution(cfg.execution, self.ledger, self.clock)
        self.allocator = SeriesAllocator(cfg.allocator, self.clock)
        self.risk = RiskEngine(cfg.risk, self.ledger, self.clock)
        self.metrics = MetricsEngine(self.ledger, self.clock, cfg.periods_per_year)

        # Writing order is the contract, do not reorder.
        self.execution.register(self.bus)
        for s in strategies:
            s.register(self.bus, self.clock, self.cache)
        self.allocator.register(self.bus)
        self.risk.register(self.bus)
        self.metrics.register(self.bus)

    def run(self, events: list[tuple[int, int, str, Event]]) -> dict:
        i, n = 0, len(events)
        while i < n:
            ts = events[i][0]
            self.clock.set(ts)
            while i < n and events[i][0] == ts:
                _, _, topic, ev = events[i]
                if isinstance(ev, BarEvent):
                    self.cache.add(ev)
                self.bus.publish(topic, ev)
                i += 1
            self.bus.publish("system.tick_end",
                             TimeEvent(ts_event=ts, ts_init=ts, name="tick_end"))
        return self.metrics.summary()










