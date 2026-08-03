import fnmatch
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Callable, TypeVar, Generic

from pydantic import BaseModel


class Clock(Protocol):
    def now_ns(self) -> int:
        ...

    def set_timer(self, name: str, interval_ns: int) -> None:
        ...


@dataclass(frozen=True, slots=True)
class Event:
    ts_event: int
    ts_init: int

@dataclass(frozen=True, slots=True)
class BarEvent(Event):
    symbol: str
    venue: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_spec: str = "1h"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass(frozen=True, slots=True)
class TradeTick(Event):
    symbol: str
    venue: str
    price: float
    size: float
    aggressor_side: OrderSide

@dataclass(frozen=True, slots=True)
class SignalEvent(Event):
    strategy_id: str
    symbol: str
    score: float #[-1.0, +1.0]
    horizon_ns: int = 0
    meta: dict = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TargetWeights(Event):
    weights: dict[str, float] #symbol -> target
    source: str = "allocator"

@dataclass(frozen=True, slots=True)
class PositionSnapshot(Event):
    positions: dict[str, float]
    nav: float
    cash: float

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

@dataclass(frozen=True, slots=True)
class SubmitOrder(Event):
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None = None
    origin: str = ""

@dataclass(frozen=True, slots=True)
class CancelOrder(Event):
    order_id: str

@dataclass(frozen=True, slots=True)
class OrderFilled(Event):
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    origin: str = ""

@dataclass(frozen=True, slots=True)
class TimeEvent(Event):
    name: str

Handler = Callable[[str, Event], None]

class MessageBus(Protocol):
    def subscribe(self, pattern: str, handler: Handler) -> None:
        ...

    def publish(self, topic: str, event: Event) -> None:
        ...

class InMemoryBus:
    def __init__(self) -> None:
        self._subs: list[tuple[str, Handler]] = []
        self._queue: list[tuple[str, Event]] = []
        self._draining = False

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    def publish(self, topic: str, event: Event) -> None:
        self._queue.append((topic, event))
        if self._draining:
            return
        try:
            while self._queue:
                t, e = self._queue.pop(0)
                for pattern, handler in self._subs:
                    if fnmatch.fnmatch(t, pattern):
                        handler(t, e)
        finally:
            self._draining = False

class Cache(Protocol):
    def bars(self, symbol: str, n: int) -> list[BarEvent]:
        ...

    def position(self, symbol: str) -> float:
        ...

    def nav(self) -> float:
        ...

class StrategyConfig(BaseModel):
    id: str
    symbols: list[str]
    bar_spec: str = "1h"

C = TypeVar("C", bound=StrategyConfig)

class Strategy(ABC, Generic[C]):
    def __init__(self, config: C) -> None:
        self.config = config
        self.id = config.id
        self._bus: MessageBus | None = None
        self.clock: Clock | None = None
        self.cache: Cache | None = None

    # writing (called by engine, not by user code)

    def _handle_bar(self, topic: str, event: Event) -> None:
        assert isinstance(event, BarEvent)
        if event.bar_spec == self.config.bar_spec:
            self.on_bar(event)

    def _handle_timer(self, topic: str, event: Event) -> None:
        assert isinstance(event, TimeEvent)
        self.on_timer(event)

    def register(self, bus: MessageBus, clock: Clock, cache: Cache) -> None:
        self._bus, self.clock, self.cache = bus, clock, cache
        for sym in self.config.symbols:
            bus.subscribe(f"data.bar.*.{sym}", self._handle_bar)

    def emit_signal(self, symbol: str, score: float, **meta) -> None:
        assert self._bus is not None and self.clock is not None
        score = max(-1.0, min(1.0, score))
        now = self.clock.now_ns()
        self._bus.publish(
            f"signal.{self.id}",
            SignalEvent(
                ts_event=now, ts_init=now,
                strategy_id=self.id, symbol=symbol,
                score=score, meta=meta,
            )
        )

    def on_bar(self, event: Event):
        ...

    def on_timer(self, event: TimeEvent):
        ...