from __future__ import annotations

import json

import pandas as pd

from collector.main import (
    AssetCollectionConfig,
    apply_date_overrides,
    load_asset_configs,
    run_collection,
)
from collector.providers import HistoricalPriceRequest


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[HistoricalPriceRequest] = []

    def fetch_prices(self, request: HistoricalPriceRequest) -> pd.DataFrame:
        self.requests.append(request)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
                "open": [10, 11],
                "high": [12, 13],
                "low": [9, 10],
                "close": [11, 12],
                "volume": [1000, 1100],
                "ticker": [request.ticker, request.ticker],
                "source": ["fake", "fake"],
            }
        )


class FakeRepository:
    def __init__(self) -> None:
        self.assets: list[dict] = []
        self.upserts: list[dict] = []

    def get_or_create_asset(self, ticker: str, name: str | None = None, asset_class: str | None = None) -> str:
        self.assets.append({"ticker": ticker, "name": name, "asset_class": asset_class})
        return f"asset-{ticker}"

    def upsert_prices(self, asset_id: str, prices: pd.DataFrame, batch_size: int = 500) -> int:
        self.upserts.append({"asset_id": asset_id, "prices": prices, "batch_size": batch_size})
        return len(prices)


def test_load_asset_configs_reads_json_array(tmp_path) -> None:
    assets_file = tmp_path / "assets.json"
    assets_file.write_text(
        json.dumps(
            [
                {
                    "provider": "stooq",
                    "ticker": "aapl.us",
                    "asset_ticker": "AAPL",
                    "name": "Apple Inc.",
                    "asset_class": "stock",
                    "start": "2020-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    assets = load_asset_configs(str(assets_file))

    assert assets == [
        AssetCollectionConfig(
            provider="stooq",
            ticker="aapl.us",
            asset_ticker="AAPL",
            name="Apple Inc.",
            asset_class="stock",
            start="2020-01-01",
        )
    ]


def test_apply_date_overrides_preserves_asset_metadata() -> None:
    assets = [
        AssetCollectionConfig(
            provider="binance",
            ticker="BTCUSDT",
            asset_ticker="BTCUSDT",
            name="Bitcoin / Tether",
            asset_class="crypto",
            start="2020-01-01",
        )
    ]

    overridden = apply_date_overrides(assets, start="2021-01-01", end="2021-12-31")

    assert overridden[0].ticker == "BTCUSDT"
    assert overridden[0].start == "2021-01-01"
    assert overridden[0].end == "2021-12-31"


def test_run_collection_fetches_and_upserts_each_asset() -> None:
    provider = FakeProvider()
    repository = FakeRepository()
    assets = [
        AssetCollectionConfig(
            provider="fake",
            ticker="AAPL",
            asset_ticker="AAPL",
            name="Apple Inc.",
            asset_class="stock",
            start="2020-01-01",
        )
    ]

    results = run_collection(
        assets,
        repository,  # type: ignore[arg-type]
        provider_factory=lambda _: provider,
        batch_size=100,
    )

    assert results[0].rows_loaded == 2
    assert provider.requests[0].ticker == "AAPL"
    assert repository.assets[0]["ticker"] == "AAPL"
    assert repository.upserts[0]["asset_id"] == "asset-AAPL"
    assert repository.upserts[0]["batch_size"] == 100
