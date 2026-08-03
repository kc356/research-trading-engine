"""Demo: momentum + mean-reversion on synthetic data, full event loop."""

import random

from pydantic import Field

from contracts import (
    BarEvent, StrategyConfig, Strategy, build_strategy, register_strategy,
)
from engine import BacktestConfig, BacktestEngine


class MeanRevConfig(StrategyConfig):
    window: int = Field(gt=2, default=20)
    z_entry: float = 1.5


@register_strategy("mean_reversion", MeanRevConfig)
class MeanRevStrategy(Strategy[MeanRevConfig]):
    def on_bar(self, bar: BarEvent) -> None:
        bars = self.cache.bars(bar.symbol, self.config.window)
        if len(bars) < self.config.window:
            return
        closes = [b.close for b in bars]
        mean = sum(closes) / len(closes)
        var = sum((c - mean) ** 2 for c in closes) / len(closes)
        std = var ** 0.5
        if std < 1e-9:
            return
        z = (bar.close - mean) / std
        if abs(z) < self.config.z_entry:
            self.emit_signal(bar.symbol, 0.0, z=z)
        else:
            self.emit_signal(bar.symbol, max(-1.0, min(1.0, -z / 3.0)), z=z)


def synthetic_bars(symbol: str, n: int, seed: int, drift: float, vol: float):
    rng = random.Random(seed)
    px, out = 100.0, []
    hour = 3_600_000_000_000
    for i in range(n):
        o = px
        px *= 1 + drift + vol * rng.gauss(0, 1)
        c = px
        ts = (i + 1) * hour
        out.append(BarEvent(ts_event=ts, ts_init=ts, symbol=symbol, venue="SIM",
                            open=o, high=max(o, c) * 1.001,
                            low=min(o, c) * 0.999, close=c, volume=1.0))
    return out

class MomentumConfig(StrategyConfig):
    lookback: int = Field(gt=1, default=90)
    threshold: float = Field(gt=0, default=1.0)

@register_strategy("momentum", MomentumConfig)
class MomentumStrategy(Strategy[MomentumConfig]):
    def on_bar(self, bar: BarEvent) -> None:
        if not self.cache:
            return
        bars = self.cache.bars(bar.symbol, self.config.lookback)
        if len(bars) < self.config.lookback:
            return # not enough history yet
        ret = bars[-1].close / bars[0].close - 1.0
        vol = _realized_vol(bars)
        if vol == 0:
            return
        z = ret / vol
        if abs(z) < self.config.threshold:
            self.emit_signal(bar.symbol, 0.0, z=z)
        else:
            self.emit_signal(bar.symbol, max(-1.0, min(1.0, z / 3.0)), z=z)

def _realized_vol(bars: list[BarEvent]) -> float:
    closes = [b.close for b in bars]
    rets = [closes[i] / closes[i-1] - 1.0 for i in range(1, len(closes))]
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return (var ** 0.5) / (len(rets) ** 0.5)

def main() -> None:
    run_cfg = {
        "strategies": [
            {"type": "momentum", "id": "momo", "symbols": ["AAA", "BBB"],
             "params": {"lookback": 90, "threshold": 0.8}},
            {"type": "mean_reversion", "id": "mr", "symbols": ["AAA", "BBB"],
             "params": {"window": 24, "z_entry": 1.5}},
        ],
    }
    strategies = [build_strategy(s) for s in run_cfg["strategies"]]

    bars = (synthetic_bars("AAA", 2000, seed=7, drift=2e-4, vol=8e-3)
            + synthetic_bars("BBB", 2000, seed=11, drift=-1e-4, vol=1.2e-2))

    engine = BacktestEngine(BacktestConfig(), strategies)
    summary = engine.run(bars)

    print("=== backtest summary ===")
    for k, v in summary.items():
        print(f"{k:>14}: {v:,.4f}" if isinstance(v, float) else f"{k:>14}: {v}")
    print(f"{'risk_killed':>14}: {engine.risk.killed}")
    print(f"{'positions':>14}: {dict(engine.ledger.positions)}")

    # determinism check: same inputs -> identical result
    engine2 = BacktestEngine(BacktestConfig(),
                             [build_strategy(s) for s in run_cfg["strategies"]])
    summary2 = engine2.run(list(bars))
    assert summary == summary2, "non-deterministic backtest!"
    print("determinism check: PASS")


if __name__ == "__main__":
    main()