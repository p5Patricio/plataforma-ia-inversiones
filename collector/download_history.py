from __future__ import annotations

import argparse
from pathlib import Path

from collector.providers import HistoricalPriceRequest, get_provider, list_providers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical OHLCV data to a standard CSV")
    parser.add_argument("--provider", required=True, choices=list_providers())
    parser.add_argument("--ticker", required=True, help="Provider ticker, e.g. AAPL.US, SPY, BTCUSDT")
    parser.add_argument("--out", required=True, help="CSV destination path")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--start", help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", help="End date, e.g. 2026-07-05")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    request = HistoricalPriceRequest(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )
    prices = provider.fetch_prices(request)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out, index=False)
    print(f"Saved {len(prices)} rows to {out}")


if __name__ == "__main__":
    main()
