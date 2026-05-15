from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from collector.providers.base import (
    HistoricalPriceRequest,
    ensure_non_empty,
    normalize_price_frame,
)


STOOQ_INTERVALS = {
    "1d": "d",
    "1wk": "w",
    "1mo": "m",
}


@dataclass
class StooqProvider:
    session: requests.Session | None = None

    name = "stooq"
    base_url = "https://stooq.com/q/d/l/"

    def fetch_prices(self, request: HistoricalPriceRequest) -> pd.DataFrame:
        interval = STOOQ_INTERVALS.get(request.interval)
        if not interval:
            raise ValueError(f"Unsupported Stooq interval: {request.interval}")

        params = {
            "s": request.ticker.lower(),
            "i": interval,
        }
        if request.start:
            params["d1"] = request.start.replace("-", "")
        if request.end:
            params["d2"] = request.end.replace("-", "")

        client = self.session or requests.Session()
        response = client.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        data = pd.read_csv(StringIO(response.text))
        data = ensure_non_empty(data, self.name, request.ticker)
        data = data.rename(
            columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return normalize_price_frame(data, request.ticker, self.name)
