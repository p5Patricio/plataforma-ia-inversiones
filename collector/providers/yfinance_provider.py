from __future__ import annotations

import pandas as pd
import yfinance as yf

from collector.providers.base import (
    HistoricalPriceRequest,
    ensure_non_empty,
    normalize_price_frame,
)


class YFinanceProvider:
    name = "yfinance"

    def fetch_prices(self, request: HistoricalPriceRequest) -> pd.DataFrame:
        kwargs: dict[str, str | bool] = {
            "interval": request.interval,
            "auto_adjust": False,
            "progress": False,
        }
        if request.start:
            kwargs["start"] = request.start
        if request.end:
            kwargs["end"] = request.end
        if not request.start and not request.end:
            kwargs["period"] = "max"

        data = yf.download(request.ticker, **kwargs)
        data = ensure_non_empty(data, self.name, request.ticker)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index().rename(
            columns={
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return normalize_price_frame(data, request.ticker, self.name)
