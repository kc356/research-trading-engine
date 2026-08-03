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

    def register(self, bus: MessageBus, clock: Clock, cache: Cache) -> None:
        self._bus, self.clock, self.cache = bus, clock, cache
        for sym in self.config.symbols:
            bus.subscribe(f"data.bar.*.{sym}", self._handle_bar)