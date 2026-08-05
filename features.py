"""
features.py — FeatureEngine: the layer between raw alt data and strategies.

Raw collector output (data.alt.{source}.{scope}) is consumed here and
re-published as derived features (feature.{name}.{scope}). Strategies
subscribe to features, never to raw alt data.

Why a separate layer:
  - features are reusable across strategies and independently testable
  - identical code path computes features in backtest and live
  - point-in-time discipline is enforced in ONE place

The walk-forward rule for entity scoring (wallets, accounts, sources):
any score attached to an entity at time t may only use that entity's
history strictly before t. Labeling wallets with their full-sample
performance and then "following the smart money" in a backtest is
lookahead wearing a costume.
"""
from abc import ABC, abstractmethod

from contracts import MessageBus, Clock, DataEvent, Event, MarketResolved


class FeatureEngine(ABC):
    """Bus citizen: subscribes raw topics, publishes feature.{name}.{scope}."""

    name: str = "feature"

    def __init__(self) -> None:
        self._bus: MessageBus | None = None
        self.clock: Clock | None = None

    def register(self, bus: MessageBus, clock: Clock) -> None:
        self._bus, self.clock = bus, clock
        for topic in self.subscriptions():
            bus.subscribe(topic, self._handle)
        bus.subscribe("market.resolved.*", self._handle_resolved)

    def _handle(self, topic: str, event: Event) -> None:
        if isinstance(event, DataEvent):
            self.on_raw(topic, event)

    def _handle_resolved(self, topic: str, event: Event) -> None:
        assert isinstance(event, MarketResolved)
        self.on_resolution(event)

    def publish(self, scope: str, payload: dict) -> None:
        if not self.clock:
            raise ModuleNotFoundError("Clock is not injected")
        if not self._bus:
            raise ModuleNotFoundError("MessageBus is not injected")
        now = self.clock.now_ns()
        self._bus.publish(
            f"feature.{self.name}.{scope}",
            DataEvent(ts_event=now, ts_init=now, source=self.name,
                      scope=scope, payload=payload)
        )


    @abstractmethod
    def subscriptions(self) -> list[str]: ...
    @abstractmethod
    def on_raw(self, topic: str, event: DataEvent): ...
    def on_resolution(self, event: MarketResolved): ...
