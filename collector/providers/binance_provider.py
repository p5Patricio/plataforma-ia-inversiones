from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

from collector.providers.base import HistoricalPriceRequest, normalize_price_frame


BINANCE_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


def parse_date_ms(value: str | None) -> int | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, utc=True)
    return int(parsed.timestamp() * 1000)


@dataclass
class BinanceProvider:
    session: requests.Session | None = None

    name = "binance"
    base_url = "https://api.binance.com/api/v3/klines"

    def fetch_prices(self, request: HistoricalPriceRequest) -> pd.DataFrame:
        if request.interval not in BINANCE_INTERVALS:
            raise ValueError(f"Unsupported Binance interval: {request.interval}")

        rows = self._fetch_all_klines(request)
        if not rows:
            raise ValueError(f"{self.name} returned no prices for {request.ticker}")

        data = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        data["timestamp"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
        return normalize_price_frame(data, request.ticker, self.name)

    def _fetch_all_klines(self, request: HistoricalPriceRequest) -> list[list]:
        client = self.session or requests.Session()
        start_ms = parse_date_ms(request.start)
        end_ms = parse_date_ms(request.end)
        rows: list[list] = []

        while True:
            params: dict[str, str | int] = {
                "symbol": request.ticker.upper().replace("-", ""),
                "interval": request.interval,
                "limit": 1000,
            }
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms

            response = client.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            rows.extend(batch)
            last_open_time = int(batch[-1][0])
            next_start = last_open_time + 1
            if len(batch) < 1000 or start_ms == next_start:
                break
            if end_ms is not None and next_start >= end_ms:
                break
            start_ms = next_start

        return rows


def utc_now_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)
