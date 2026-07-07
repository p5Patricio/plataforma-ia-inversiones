from __future__ import annotations

from collector.main import load_asset_configs


def test_core_assets_config_loads_with_collector_parser() -> None:
    assets = load_asset_configs("config/assets.core.json")

    tickers = {asset.asset_ticker for asset in assets}

    assert tickers == {"BTC-USD", "ETH-USD", "AAPL", "MSFT"}
    assert {asset.provider for asset in assets} == {"binance", "yfinance"}
    assert all(asset.interval == "1d" for asset in assets)
