from __future__ import annotations

import pandas as pd

from brain.paper_trading import PaperTradingConfig, run_paper_trading
from brain.paper_trading_job import run_paper_trading_job


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


class FakePaperTradingRepository:
    def __init__(self, predictions: pd.DataFrame | None = None) -> None:
        self.predictions = predictions if predictions is not None else _prediction_frame()
        self.created_runs: list[dict] = []
        self.inserted_events: list[dict] = []

    def get_assets(self) -> list[dict]:
        return [
            {"id": "asset-btc", "ticker": "BTC-USD"},
            {"id": "asset-aapl", "ticker": "AAPL"},
        ]

    def get_prediction_feedback(self, **kwargs) -> pd.DataFrame:
        self.feedback_kwargs = kwargs
        return self.predictions

    def get_prices(self, asset_id: str, limit: int | None = None, ascending: bool = True) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
                "close": [100, 110, 121],
            }
        )
        return frame.sort_values("timestamp", ascending=ascending).reset_index(drop=True)

    def create_paper_trading_run(self, **kwargs) -> str:
        self.created_runs.append(kwargs)
        return f"paper-run-{len(self.created_runs)}"

    def insert_paper_trading_events(self, paper_trading_run_id: str, asset_id: str | None, timeline: pd.DataFrame) -> int:
        self.inserted_events.append(
            {"paper_trading_run_id": paper_trading_run_id, "asset_id": asset_id, "rows": len(timeline)}
        )
        return len(timeline)


def test_run_paper_trading_job_persists_selected_ticker() -> None:
    repository = FakePaperTradingRepository()

    report = run_paper_trading_job(
        repository,
        tickers=["BTC-USD"],
        limit=3,
        config=PaperTradingConfig(initial_capital=1000, fee_bps=0, slippage_bps=0),
    )

    assert report["attempted"] == 1
    assert report["succeeded"] == 1
    assert report["failed"] == 0
    assert repository.created_runs[0]["asset_id"] == "asset-btc"
    assert repository.created_runs[0]["model_run_id"] == "run-1"
    assert repository.created_runs[0]["metrics"]["trade_count"] == 1
    assert repository.inserted_events[0]["rows"] == 3


def test_run_paper_trading_job_skips_empty_predictions_by_default() -> None:
    repository = FakePaperTradingRepository(predictions=pd.DataFrame())

    report = run_paper_trading_job(repository, tickers=["AAPL"], limit=3)

    assert report["attempted"] == 1
    assert report["succeeded"] == 0
    assert report["skipped"] == [{"ticker": "AAPL", "asset_id": "asset-aapl", "reason": "no_predictions"}]
    assert repository.created_runs == []


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
            "predicted_action": ["BUY", "HOLD", "HOLD"],
            "confidence": [0.8, 0.5, 0.5],
            "model_name": ["extra_trees", "extra_trees", "extra_trees"],
            "model_version": ["v1", "v1", "v1"],
            "model_run_id": ["run-1", "run-1", "run-1"],
            "metadata": [{"risk": {"position_size": 0.5}}, {}, {}],
        }
    )
