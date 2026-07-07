from __future__ import annotations

import pandas as pd

from brain.paper_trading import PaperTradingConfig, run_paper_trading


def test_paper_trading_maintains_position_on_hold_and_marks_equity() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
            "predicted_action": ["BUY", "HOLD", "HOLD"],
            "confidence": [0.8, 0.5, 0.5],
            "metadata": [{"risk": {"position_size": 0.5}}, {}, {}],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
            "close": [100.0, 110.0, 121.0],
        }
    )

    result = run_paper_trading(
        predictions,
        prices,
        PaperTradingConfig(initial_capital=1000, fee_bps=0, slippage_bps=0),
    )

    assert result.timeline["position_state"].tolist() == ["LONG", "LONG", "LONG"]
    assert result.metrics["trade_count"] == 1
    assert result.metrics["open_position"] == "LONG"
    assert round(result.metrics["final_equity"], 2) == 1102.5


def test_paper_trading_rebalances_and_applies_costs() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "action": ["BUY", "SELL"],
            "confidence": [0.8, 0.9],
            "metadata": [{"risk": {"position_size": 0.5}}, {"risk": {"position_size": 0.25}}],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "close": [100.0, 120.0],
        }
    )

    result = run_paper_trading(
        predictions,
        prices,
        PaperTradingConfig(initial_capital=1000, fee_bps=10, slippage_bps=0),
    )

    assert result.timeline["exposure"].tolist() == [0.5, -0.25]
    assert result.metrics["trade_count"] == 2
    assert result.metrics["open_position"] == "SHORT"
    assert result.metrics["final_equity"] < 1100


def test_paper_trading_blocks_short_when_disabled() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
            "predicted_action": ["SELL"],
            "confidence": [0.9],
            "metadata": [{"risk": {"position_size": 0.5}}],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
            "close": [100.0],
        }
    )

    result = run_paper_trading(predictions, prices, PaperTradingConfig(allow_short=False))

    assert result.timeline.loc[0, "exposure"] == 0
    assert result.metrics["trade_count"] == 0
    assert result.metrics["open_position"] == "FLAT"


def test_paper_trading_empty_predictions_returns_flat_metrics() -> None:
    result = run_paper_trading(pd.DataFrame(), pd.DataFrame({"timestamp": [], "close": []}))

    assert result.metrics["final_equity"] == 10_000
    assert result.metrics["open_position"] == "FLAT"
    assert result.timeline.empty
