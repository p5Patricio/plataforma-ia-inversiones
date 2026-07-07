from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient
from requests import RequestException

from app_config import AppConfig
from api.main import app, get_app_config, get_repository


class FakeRepository:
    def __init__(self, prediction: dict | None = None, prices: pd.DataFrame | None = None) -> None:
        self.prediction = prediction
        self.prices = prices if prices is not None else make_prices()
        self.feedback_kwargs: dict | None = None
        self.backtest_kwargs: dict | None = None

    def get_assets(self) -> list[dict]:
        return [{"id": "asset-1", "ticker": "AAPL", "name": "Apple Inc.", "asset_class": "stock"}]

    def get_asset_id(self, ticker: str) -> str:
        if ticker.upper() == "MISSING":
            raise ValueError("missing")
        return "asset-1"

    def get_prices(self, asset_id: str, limit: int | None = None, ascending: bool = True) -> pd.DataFrame:
        prices = self.prices.head(limit) if limit else self.prices
        return prices.sort_values("timestamp", ascending=ascending).reset_index(drop=True)

    def get_latest_prediction(
        self,
        asset_id: str,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> dict | None:
        return self.prediction

    def get_prediction_feedback(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
        asset_id: str | None = None,
        only_evaluated: bool = True,
        limit: int | None = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        self.feedback_kwargs = {
            "model_name": model_name,
            "model_version": model_version,
            "asset_id": asset_id,
            "only_evaluated": only_evaluated,
            "limit": limit,
            "ascending": ascending,
        }
        return pd.DataFrame(
            [
                {
                    "prediction_id": 1,
                    "timestamp": "2026-07-05T00:00:00+00:00",
                    "predicted_action": "HOLD",
                    "confidence": 0.45,
                    "probabilities": {"BUY": 0.3, "HOLD": 0.45, "SELL": 0.25},
                    "model_name": "extra_trees",
                    "model_version": "promoted",
                    "model_run_id": "run-1",
                    "feature_set": "technical_v2",
                    "label_method": "triple_barrier",
                    "horizon": 5,
                    "metadata": {"risk": {"position_size": 0, "blocked_reasons": ["confidence_below_trade_threshold"]}},
                    "actual_label": None,
                    "is_correct": None,
                    "outcome_return": None,
                    "prediction_created_at": "2026-07-05T00:01:00+00:00",
                }
            ]
        )

    def get_backtests(
        self,
        asset_id: str | None = None,
        model_run_id: str | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        self.backtest_kwargs = {
            "asset_id": asset_id,
            "model_run_id": model_run_id,
            "limit": limit,
            "ascending": ascending,
        }
        return pd.DataFrame(
            [
                {
                    "id": "backtest-1",
                    "name": "extra_trees:promoted:BTC-USD",
                    "model_run_id": "run-1",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "ended_at": "2026-07-01T00:00:00+00:00",
                    "created_at": "2026-07-06T00:00:00+00:00",
                    "metrics": {
                        "total_return": 0.25,
                        "max_drawdown": -0.04,
                        "profit_factor": 2.1,
                        "active_trade_count": 20,
                        "trade_count": 40,
                        "win_rate": 0.58,
                        "exposure": 0.5,
                        "final_equity": 12500,
                    },
                    "params": {"fee_bps": 5},
                    "model_runs": {
                        "model_name": "extra_trees",
                        "model_version": "promoted",
                        "feature_set": "technical_v2",
                        "label_method": "triple_barrier",
                        "horizon": 5,
                    },
                }
            ]
        )


class PredictionUnavailableRepository(FakeRepository):
    def get_latest_prediction(
        self,
        asset_id: str,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> dict | None:
        raise RequestException("prediction feedback unavailable")


def make_prices(rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = pd.Series(range(rows), dtype=float) + 100
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": 1000,
        }
    )


def override_repository(repository: FakeRepository) -> None:
    app.dependency_overrides[get_repository] = lambda: repository


def clear_overrides() -> None:
    app.dependency_overrides.clear()


def override_config(config: AppConfig) -> None:
    app.dependency_overrides[get_app_config] = lambda: config


def test_assets_endpoint_returns_repository_assets() -> None:
    override_repository(FakeRepository())
    client = TestClient(app)

    response = client.get("/api/assets")

    clear_overrides()
    assert response.status_code == 200
    assert response.json()[0]["ticker"] == "AAPL"


def test_analysis_endpoint_prefers_latest_prediction() -> None:
    override_repository(
        FakeRepository(
            prediction={
                "predicted_action": "BUY",
                "confidence": 0.72,
                "probabilities": {"BUY": 0.72, "HOLD": 0.2, "SELL": 0.08},
                "model_name": "baseline",
                "model_version": "v1",
                "model_run_id": "run-1",
                "feature_set": "technical_v1",
                "label_method": "triple_barrier",
                "horizon": 5,
                "metadata": {
                    "risk": {
                        "position_size": 0.05,
                        "stop_loss": 0.02,
                        "take_profit": 0.04,
                        "blocked_reasons": [],
                    }
                },
            }
        )
    )
    client = TestClient(app)

    response = client.get("/api/analysis/AAPL")

    clear_overrides()
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "prediction"
    assert payload["analysis"]["signal"] == "BUY"
    assert payload["analysis"]["model"]["version"] == "v1"
    assert payload["analysis"]["risk"]["position_size"] == 0.05


def test_analysis_endpoint_falls_back_to_indicator_logic() -> None:
    override_repository(FakeRepository(prediction=None))
    client = TestClient(app)

    response = client.get("/api/analysis/AAPL")

    clear_overrides()
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "fallback_indicators"
    assert payload["analysis"]["signal"] in {"BUY", "SELL", "HOLD"}


def test_analysis_endpoint_uses_real_prices_when_prediction_feedback_is_unavailable() -> None:
    override_repository(PredictionUnavailableRepository())
    client = TestClient(app)

    response = client.get("/api/analysis/AAPL")

    clear_overrides()
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "fallback_indicators"
    assert payload["analysis"]["signal"] in {"BUY", "SELL", "HOLD"}


def test_prediction_history_endpoint_returns_audit_rows() -> None:
    repository = FakeRepository()
    override_repository(repository)
    client = TestClient(app)

    response = client.get("/api/predictions/AAPL?limit=5&only_evaluated=false")

    clear_overrides()
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["action"] == "HOLD"
    assert payload[0]["model"]["name"] == "extra_trees"
    assert payload[0]["risk"]["blocked_reasons"] == ["confidence_below_trade_threshold"]
    assert repository.feedback_kwargs == {
        "model_name": None,
        "model_version": None,
        "asset_id": "asset-1",
        "only_evaluated": False,
        "limit": 5,
        "ascending": False,
    }


def test_prediction_history_endpoint_returns_empty_demo_history() -> None:
    app.dependency_overrides[get_repository] = lambda: None
    client = TestClient(app)

    response = client.get("/api/predictions/BTC-USD")

    clear_overrides()
    assert response.status_code == 200
    assert response.json() == []


def test_backtest_history_endpoint_returns_model_metrics() -> None:
    repository = FakeRepository()
    override_repository(repository)
    client = TestClient(app)

    response = client.get("/api/backtests/AAPL?limit=3")

    clear_overrides()
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["id"] == "backtest-1"
    assert payload[0]["model"]["name"] == "extra_trees"
    assert payload[0]["metrics"]["total_return"] == 0.25
    assert repository.backtest_kwargs == {
        "asset_id": "asset-1",
        "model_run_id": None,
        "limit": 3,
        "ascending": False,
    }


def test_backtest_history_endpoint_returns_empty_demo_history() -> None:
    app.dependency_overrides[get_repository] = lambda: None
    client = TestClient(app)

    response = client.get("/api/backtests/BTC-USD")

    clear_overrides()
    assert response.status_code == 200
    assert response.json() == []


def test_prices_endpoint_returns_descending_prices() -> None:
    override_repository(FakeRepository())
    client = TestClient(app)

    response = client.get("/api/prices/AAPL?limit=3")

    clear_overrides()
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 3
    assert rows[0]["timestamp"] > rows[-1]["timestamp"]


def test_demo_mode_identifies_demo_source_and_varies_price_shape() -> None:
    app.dependency_overrides[get_repository] = lambda: None
    client = TestClient(app)

    btc_prices = client.get("/api/prices/BTC-USD?limit=5")
    eth_prices = client.get("/api/prices/ETH-USD?limit=5")
    analysis = client.get("/api/analysis/BTC-USD")

    clear_overrides()
    assert btc_prices.status_code == 200
    assert eth_prices.status_code == 200
    assert analysis.status_code == 200
    assert analysis.json()["source"] == "demo_indicators"
    assert [row["close"] for row in btc_prices.json()] != [row["close"] for row in eth_prices.json()]


def test_assets_endpoint_returns_unavailable_when_demo_is_disabled() -> None:
    app.dependency_overrides[get_repository] = lambda: None
    override_config(AppConfig(environment="production", allow_demo_fallback=False))
    client = TestClient(app)

    response = client.get("/api/assets")

    clear_overrides()
    assert response.status_code == 503
    assert response.json()["detail"] == "Fuente de datos no disponible y modo demo desactivado"


def test_prices_endpoint_returns_unavailable_when_demo_is_disabled() -> None:
    app.dependency_overrides[get_repository] = lambda: None
    override_config(AppConfig(environment="production", allow_demo_fallback=False))
    client = TestClient(app)

    response = client.get("/api/prices/BTC-USD?limit=5")

    clear_overrides()
    assert response.status_code == 503
    assert response.json()["detail"] == "Fuente de datos no disponible y modo demo desactivado"
