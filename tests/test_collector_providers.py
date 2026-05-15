from __future__ import annotations

import pandas as pd

from collector.providers import HistoricalPriceRequest, get_provider, list_providers
from collector.providers.base import STANDARD_PRICE_COLUMNS, normalize_price_frame
from collector.providers.binance_provider import BinanceProvider
from collector.providers.stooq_provider import StooqProvider


class FakeResponse:
    def __init__(self, text: str = "", payload: list | None = None) -> None:
        self.text = text
        self._payload = payload or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def get(self, url: str, params: dict, timeout: int) -> FakeResponse:
        self.requests.append({"url": url, "params": params, "timeout": timeout})
        return self.responses.pop(0)


def test_normalize_price_frame_outputs_standard_columns() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": ["10", "11"],
            "high": ["12", "13"],
            "low": ["9", "10"],
            "close": ["11", "12"],
            "volume": ["1000", "1200"],
        }
    )

    result = normalize_price_frame(raw, ticker="aapl.us", source="test", timestamp_column="date")

    assert list(result.columns) == STANDARD_PRICE_COLUMNS
    assert result.loc[0, "ticker"] == "AAPL.US"
    assert result.loc[0, "source"] == "test"
    assert result["timestamp"].dt.tz is not None


def test_stooq_provider_parses_csv_response() -> None:
    csv = "Date,Open,High,Low,Close,Volume\n2024-01-01,10,12,9,11,1000\n"
    session = FakeSession([FakeResponse(text=csv)])
    provider = StooqProvider(session=session)

    result = provider.fetch_prices(
        HistoricalPriceRequest(ticker="aapl.us", interval="1d", start="2024-01-01", end="2024-01-31")
    )

    assert len(result) == 1
    assert list(result.columns) == STANDARD_PRICE_COLUMNS
    assert session.requests[0]["params"]["d1"] == "20240101"
    assert session.requests[0]["params"]["d2"] == "20240131"
    assert session.requests[0]["params"]["i"] == "d"


def test_binance_provider_parses_kline_response() -> None:
    payload = [
        [
            1704067200000,
            "42000.0",
            "43000.0",
            "41000.0",
            "42500.0",
            "123.45",
            1704153599999,
            "0",
            1,
            "0",
            "0",
            "0",
        ]
    ]
    session = FakeSession([FakeResponse(payload=payload)])
    provider = BinanceProvider(session=session)

    result = provider.fetch_prices(
        HistoricalPriceRequest(ticker="BTCUSDT", interval="1d", start="2024-01-01", end="2024-01-02")
    )

    assert len(result) == 1
    assert result.loc[0, "ticker"] == "BTCUSDT"
    assert result.loc[0, "source"] == "binance"
    assert result.loc[0, "close"] == 42500.0
    assert session.requests[0]["params"]["symbol"] == "BTCUSDT"
    assert session.requests[0]["params"]["interval"] == "1d"


def test_provider_registry_lists_initial_sources() -> None:
    assert {"binance", "stooq", "yfinance"}.issubset(set(list_providers()))
    assert get_provider("stooq").name == "stooq"
