"""run_poly.py — First real backtest on the Polymarket BTC5M catalog.

Usage:  python run_poly.py --catalog ./catalog/polymarket_btc5m

The example strategy is a PIPELINE TEST, not alpha: late-window momentum
(if the market has drifted decisively by minute 3, follow it). Expected
result after spread: roughly zero minus costs. If it prints a big positive
Sharpe, suspect a bug before suspecting genius.
"""
from pydantic import Field

from contracts import StrategyConfig, register_strategy, Strategy, InstrumentDefined, MarketResolved, BarEvent
from polymarket_data import NS


class LateMomoConfig(StrategyConfig):
    min_elapsed_s: int = Field(gt=0, default=180) # act from minute 3
    band: float = 0.15 # |up_price - 0.5| must exceed this

@register_strategy("late_momentum", LateMomoConfig)
class LateMomentum(Strategy[LateMomoConfig]):
    """Follow decisive mid-window drift; stay flat in coin-flip windows."""

    def on_start(self) -> None:
        self._start_ns: dict[str, int] = {}

    def on_instrument(self, e: InstrumentDefined) -> None:
        self._start_ns[e.symbol] = e.ts_event

    def on_resolution(self, e: MarketResolved) -> None:
        self._start_ns.pop(e.symbol, None)

    def on_bar(self, bar: BarEvent) -> None:
        w0 = self._start_ns.get(bar.symbol)
        if w0 is None:
            return
        elapsed_s = (bar.ts_event - w0) // NS
        if elapsed_s < self.config.min_elapsed_s:
            return
        edge = bar.close - 0.5
        if abs(edge) < self.config.band:
            self.emit_signal(bar.symbol, 0.0, up_price=bar.close,
                             elapsed_s=elapsed_s)
        else:
            score = max(-1.0, min(1.0, edge / 0.4))
            self.emit_signal(bar.symbol, score, up_price=bar.close, elapsed_s=elapsed_s)
