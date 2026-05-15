from __future__ import annotations

from collector.providers.binance_provider import BinanceProvider
from collector.providers.base import PriceProvider
from collector.providers.stooq_provider import StooqProvider
from collector.providers.yfinance_provider import YFinanceProvider


PROVIDERS = {
    "binance": BinanceProvider,
    "stooq": StooqProvider,
    "yfinance": YFinanceProvider,
}


def list_providers() -> list[str]:
    return sorted(PROVIDERS)


def get_provider(name: str) -> PriceProvider:
    normalized = name.lower()
    provider_class = PROVIDERS.get(normalized)
    if not provider_class:
        raise ValueError(f"Unknown provider '{name}'. Available: {', '.join(list_providers())}")
    return provider_class()
