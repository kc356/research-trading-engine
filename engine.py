import math
from collections import defaultdict, deque

from pexpect.pxssh import pxssh
from pydantic import BaseModel

from contracts import BarEvent, OrderFilled, OrderSide, MessageBus, Event, SignalEvent, TargetWeights, SubmitOrder, \
    OrderType, Strategy, InMemoryBus, TimeEvent


class SimClock:
    def __init__(self) -> None:
        self._now = 0

    def now_ns(self) -> int:
        return self._now

    def set(self, ts: int) -> None:
        assert ts >= self._now, "clock cannot go backwards"
        self._now = ts

    def set_timer(self, name: str, interval_ns: int) -> None:
        raise NotImplementedError("timers: engine-merged, not yet needed")


class Ledger:
    def __init__(self, initial_cash: float) -> None:
        self.cash = initial_cash
        self.positions: dict[str, float] = defaultdict(float)
        self.last_price: dict[str, float] = {}
        self.fees_paid = 0.0

    def apply_fill(self, f: OrderFilled) -> None:
        signed = f.quantity if f.side == OrderSide.BUY else -f.quantity
        self.cash -= signed * f.price + f.fee
        self.fees_paid += f.fee
        self.positions[f.symbol] += signed

    def mark(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = price

    def net_asset_value(self) -> float:
        return self.cash + sum(
            q * self.last_price.get(s, 0.0) for s, q in self.positions.items()
        )

    def weight(self, symbol: str) -> float:
        nav = self.net_asset_value()
        if nav <= 0:
            return 0.0
        return self.positions[symbol] * self.last_price.get(symbol, 0.0) / nav


class MarketCache:
    def __init__(self, ledger: Ledger) -> None:
        self._bars: dict[str, list[BarEvent]] = defaultdict(list)
        self._ledger = ledger

    def add(self, bar: BarEvent) -> None:
        self._bars[bar.symbol].append(bar)

    def bars(self, symbol: str, n: int) -> list[BarEvent]:
        return self._bars[symbol][-n:]

    def position(self, symbol: str) -> float:
        return self._ledger.positions.get(symbol, 0.0)

    def net_asset_value(self) -> float:
        return self._ledger.net_asset_value()

class AllocatorConfig(BaseModel):
    weight_scale: float = 0.5
    norm_window: int = 60
    norm_min_obs: int = 20
    rebalance_every_n_ticks: int = 1

# Portfolio Manager
class Allocator:

    def __init__(self, cfg: AllocatorConfig, clock: SimClock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._scores: dict[tuple[str, str], float] = {}
        self._hist: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=cfg.norm_window)
        )
        self._bus: MessageBus | None = None
        self._tick = 0
        self._dirty = False

    def register(self, bus: MessageBus) -> None:
        self._bus = bus
        bus.subscribe("signal.*", self._on_signal)
        bus.subscribe("system.tick_end", self._on_tick_end)

    def _normalize(self, key: tuple[str, str], score: float) -> float:
        h = self._hist[key]
        h.append(score)
        if len(h) < self.cfg.norm_min_obs:
            return score
        mean = sum(h) / len(h)
        var = sum((x - mean) ** 2 for x in h) / len(h)
        std = var ** 0.5
        if std < 1e-12:
            return 0.0 if abs(score - mean) < 1e-12 else score
        return math.tanh((score - mean) / std / 2.0)

    def _on_signal(self, topic: str, e: Event) -> None: # why have topic this param
        assert isinstance(e, SignalEvent)
        key = (e.strategy_id, e.symbol)
        self._scores[key] = self._normalize(key, e.score)
        self._dirty = True

    def _on_tick_end(self, topic: str, e: Event) -> None:
        if not self._bus:
            return
        self._tick += 1
        if not self._dirty or self._tick % self.cfg.rebalance_every_n_ticks:
            return
        self._dirty = False
        by_sym: dict[str, list[float]] = defaultdict(list)
        for (_, sym), s in self._scores.items():
            by_sym[sym].append(s)
        weights = {
            sym: (sum(ss) / len(ss)) * self.cfg.weight_scale
            for sym, ss in by_sym.items()
        }
        now = self.clock.now_ns()
        self._bus.publish(
            "portfolio.target.raw",
            TargetWeights(ts_event=now, ts_init=now, weights=weights)
        )

class RiskConfig(BaseModel):
    max_gross: float = 1.0
    max_per_symbol: float = 0.5
    drawdown_kill: float = 0.25

class RiskEngine:
    def __init__(self, cfg: RiskConfig, ledger: Ledger, clock: SimClock) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.clock = clock
        self._peak_net_asset_value = ledger.net_asset_value()
        self.killed = False
        self._bus: MessageBus | None = None

    def register(self, bus: MessageBus) -> None:
        self._bus = bus
        bus.subscribe("portfolio.target.raw", self._on_raw_target)
        bus.subscribe("system.tick_end", self._on_tick_end)

    def _on_tick_end(self, topic: str, e: Event) -> None:
        if not self._bus:
            return
        nav = self.ledger.net_asset_value()
        self._peak_net_asset_value = max(self._peak_net_asset_value, nav)
        if not self.killed and nav < self._peak_net_asset_value * (1 - self.cfg.drawdown_kill):
            self.killed = True
            now = self.clock.now_ns()
            self._bus.publish(
                "portfolio.target",
                TargetWeights(ts_event=now, ts_init=now, weights={}, source="risk:kill"),
            )

    def _on_raw_target(self, topic: str, e: Event) -> None:
        if not self._bus:
            return
        assert isinstance(e, TargetWeights)
        if self.killed:
            return
        w = {
            s: max(-self.cfg.max_per_symbol, min(self.cfg.max_per_symbol, x))
            for s, x in e.weights.items()
        }
        gross = sum(abs(x) for x in w.values())
        if gross > self.cfg.max_gross:
            scale = self.cfg.max_gross / gross
            w = {s: x * scale for s, x in w.items()}
        self._bus.publish(
            "portfolio.target",
            TargetWeights(ts_event=e.ts_event, ts_init=e.ts_init, weights=w, source="risk:clamped"),
        )

class ExecutionConfig(BaseModel):
    slippage_bps: float = 5.0
    fee_bps: float = 2.0
    min_trade_weight: float = 0.01

class ExecutionEngine:
    def __init__(self, cfg: ExecutionConfig, ledger: Ledger, clock: SimClock) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.clock = clock
        self._pending: dict[str, SubmitOrder] = {}
        self._old = 0
        self._bus: MessageBus | None = None

    def register(self, bus: MessageBus) -> None:
        self._bus = bus
        bus.subscribe("data.bar.*", self._on_bar)
        bus.subscribe("portfolio.target", self._on_target)

    def _on_bar(self, topic: str, e: Event):
        if not self._bus:
            return
        assert isinstance(e, BarEvent)
        order = self._pending.pop(e.symbol, None)
        if order is not None:
            slip = self.cfg.slippage_bps / 1e4
            slipped_price = e.open * (1 + slip if order.side == OrderSide.BUY else 1 - slip)
            fee = abs(order.quantity) * slipped_price * self.cfg.fee_bps / 1e4
            now = self.clock.now_ns()
            fill = OrderFilled(ts_event=now, ts_init=now, order_id=order.order_id,
                               symbol=e.symbol, side=order.side,
                               quantity=order.quantity, price=slipped_price, fee=fee, origin=order.origin)
            self.ledger.apply_fill(fill)
            self._bus.publish(f"order.event.{e.symbol}", fill)
        self.ledger.mark(e.symbol, e.close)

    def _on_target(self, topic: str, e: Event) -> None:
        assert isinstance(e, TargetWeights)
        nav = self.ledger.net_asset_value()
        if nav <= 0:
            return
        symbols = set(e.weights) | {s for s, q in self.ledger.positions.items() if q}
        for sym in sorted(symbols):
            target_w = e.weights.get(sym, 0.0)
            delta_w = target_w - self.ledger.weight(sym)
            if abs(delta_w) < self.cfg.min_trade_weight:
                continue
            price = self.ledger.last_price.get(sym)
            if not price:
                continue
            qty = delta_w * nav / price
            self._old += 1
            now = self.clock.now_ns()
            self._pending[sym] = SubmitOrder(
                ts_event=now, ts_init=now, order_id=f"o{self._old}",
                symbol=sym, venue="SIM",
                side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=abs(qty),
                origin = e.source
            )

class MetricsEngine:
    def __init__(self, ledger: Ledger, clock: SimClock, periods_per_year: float):
        self.ledger = ledger
        self.clock = clock
        self.periods_per_year = periods_per_year
        self.equity: list[tuple[int, float]] = []

    def register(self, bus: MessageBus) -> None:
        bus.subscribe("system.tick_end", self._on_tick_end)

    def _on_tick_end(self, topic: str, e: Event) -> None:
        self.equity.append((self.clock.now_ns(), self.ledger.net_asset_value()))

    def summary(self) -> dict:
        navs = [n for _, n in self.equity]
        if len(navs) < 3:
            return {}
        returns = [navs[i] / navs[i-1] - 1 for i in range(1, len(navs))]
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
        std = var ** 0.5
        sharpe = (mean / std) * (self.periods_per_year ** 0.5) if std > 0 else 0.0
        peak, max_drawdown = navs[0], 0.0
        for n in navs:
            peak = max(peak, n)
            max_drawdown = max(max_drawdown, 1 - n / peak)
        return {
            "final_nav": navs[-1],
            "total_return": navs[-1] / navs[0] - 1,
            "ann_sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "fees_paid": self.ledger.fees_paid,
            "n_periods": len(navs),
        }

class BacktestConfig(BaseModel):
    initial_cash: float = 1_000_000.0
    periods_per_year: float = 365 * 24 # 1h bars default
    allocator: AllocatorConfig = AllocatorConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()

class BacktestEngine:
    def __init__(self, cfg: BacktestConfig, strategies: list[Strategy]) -> None:
        self.cfg = cfg
        self.bus = InMemoryBus()
        self.clock = SimClock()
        self.ledger = Ledger(cfg.initial_cash)
        self.cache = MarketCache(self.ledger)
        self.execution = ExecutionEngine(cfg.execution, self.ledger, self.clock)
        self.allocator = Allocator(cfg.allocator, self.clock)
        self.risk = RiskEngine(cfg.risk, self.ledger, self.clock)
        self.metrics = MetricsEngine(self.ledger, self.clock, cfg.periods_per_year)

        # do not reorder, the order is the contract
        self.execution.register(self.bus) # fills at open first
        for s in strategies:
            s.register(self.bus, self.clock, self.cache) # strategies on close
        self.allocator.register(self.bus) # combine on tick_end
        self.risk.register(self.bus) # clamp raw targets
        self.metrics.register(self.bus) # record equity

    def run(self, bars: list[BarEvent]) -> dict:
        bars = sorted(bars, key=lambda b: (b.ts_event, b.symbol))
        i, n = 0, len(bars)
        while i < n:
            ts = bars[i].ts_event
            self.clock.set(ts)
            while i < n and bars[i].ts_event == ts:
                bar = bars[i]
                self.cache.add(bar)
                self.bus.publish(f"data.bar.{bar.venue}.{bar.symbol}", bar)
                i += 1
            self.bus.publish(
                "system.tick_end", TimeEvent(ts_event=ts, ts_init=ts, name="tick_end")
            )
        return self.metrics.summary()



