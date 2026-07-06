from __future__ import annotations

import json

import pandas as pd

from collector.main import (
    AssetCollectionConfig,
    apply_date_overrides,
    load_asset_configs,
    run_collection,
)
from collector.market_data_job import filter_assets, run_market_data_job
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
        self.price_lookup: dict[str, pd.DataFrame] = {}
        self.features_loaded: list[dict] = []
        self.labels_loaded: list[dict] = []

    def get_or_create_asset(self, ticker: str, name: str | None = None, asset_class: str | None = None) -> str:
        self.assets.append({"ticker": ticker, "name": name, "asset_class": asset_class})
        return f"asset-{ticker}"

    def upsert_prices(self, asset_id: str, prices: pd.DataFrame, batch_size: int = 500) -> int:
        self.upserts.append({"asset_id": asset_id, "prices": prices, "batch_size": batch_size})
        self.price_lookup[asset_id] = prices
        return len(prices)

    def get_asset_id(self, ticker: str) -> str:
        return f"asset-{ticker}"

    def get_prices(self, asset_id: str, limit: int | None = None, ascending: bool = True) -> pd.DataFrame:
        prices = self.price_lookup.get(asset_id)
        if prices is None:
            prices = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC"),
                    "open": range(80),
                    "high": range(1, 81),
                    "low": range(80),
                    "close": range(1, 81),
                    "volume": 1000,
                }
            )
        prices = prices.sort_values("timestamp", ascending=ascending).reset_index(drop=True)
        return prices.head(limit) if limit else prices

    def upsert_features(self, asset_id: str, features: pd.DataFrame, feature_columns: list[str], feature_set: str, batch_size: int = 500) -> int:
        rows = len(features.dropna(subset=feature_columns))
        self.features_loaded.append({"asset_id": asset_id, "feature_set": feature_set, "rows": rows})
        return rows

    def upsert_labels(self, asset_id: str, labels: pd.DataFrame, label_method: str, horizon: int, batch_size: int = 500) -> int:
        rows = len(labels.dropna(subset=["label"]))
        self.labels_loaded.append({"asset_id": asset_id, "label_method": label_method, "horizon": horizon, "rows": rows})
        return rows


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


def test_filter_assets_matches_provider_or_stored_ticker() -> None:
    assets = [
        AssetCollectionConfig(provider="fake", ticker="BTCUSDT", asset_ticker="BTC-USD", name="Bitcoin", asset_class="crypto"),
        AssetCollectionConfig(provider="fake", ticker="AAPL", asset_ticker="AAPL", name="Apple", asset_class="stock"),
    ]

    assert [asset.asset_ticker for asset in filter_assets(assets, ["BTCUSDT"])] == ["BTC-USD"]
    assert [asset.asset_ticker for asset in filter_assets(assets, ["AAPL"])] == ["AAPL"]


def test_run_market_data_job_collects_and_materializes() -> None:
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

    result = run_market_data_job(
        repository=repository,  # type: ignore[arg-type]
        assets=assets,
        provider_factory=lambda _: provider,
        feature_sets=["technical_v1"],
        horizon=3,
        batch_size=100,
    )

    assert result["collection"]["succeeded"] == 1
    assert result["materialization"]["succeeded"] == 1
    assert result["failed"] == 0
    assert repository.features_loaded[0]["feature_set"] == "technical_v1"
    assert repository.labels_loaded[0]["horizon"] == 3


def test_run_market_data_job_records_materialization_errors() -> None:
    repository = FakeRepository()
    assets = [
        AssetCollectionConfig(
            provider="fake",
            ticker="AAPL",
            asset_ticker="AAPL",
            name="Apple Inc.",
            asset_class="stock",
        )
    ]

    result = run_market_data_job(
        repository=repository,  # type: ignore[arg-type]
        assets=assets,
        collect_prices=False,
        materialize=True,
        materialize_tickers=["MISSING"],
        feature_sets=["unknown_features"],
    )

    assert result["materialization"]["attempted"] == 1
    assert result["failed"] == 1
    assert result["errors"][0]["stage"] == "materialization"
