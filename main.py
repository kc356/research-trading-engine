from pydantic import Field

from contracts import StrategyConfig, register_strategy, Strategy, Event, BarEvent


class MomentumConfig(StrategyConfig):
    lookback: int = Field(gt=1, default=90)
    threshold: float = Field(gt=0, default=1.0)

@register_strategy("momentum", MomentumConfig)
class MomentumStrategy(Strategy[MomentumConfig]):
    def on_bar(self, bar: BarEvent) -> None:
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
    return (var ** 0.5) & (len(rets) ** 0.5)