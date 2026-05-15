from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, get_repository


class FakeRepository:
    def __init__(self, prediction: dict | None = None, prices: pd.DataFrame | None = None) -> None:
        self.prediction = prediction
        self.prices = prices if prices is not None else make_prices()

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
