from __future__ import annotations

from dataclasses import dataclass
import os


VALID_ENVIRONMENTS = {"development", "staging", "production", "test"}
TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AppConfig:
    environment: str = "development"
    allow_demo_fallback: bool = True
    cors_origins: tuple[str, ...] = ("*",)

    @classmethod
    def from_env(cls) -> "AppConfig":
        environment = normalize_environment(os.getenv("APP_ENV", "development"))
        return cls(
            environment=environment,
            allow_demo_fallback=parse_bool_env(
                "ALLOW_DEMO_FALLBACK",
                default=environment != "production",
            ),
            cors_origins=parse_csv_env("API_CORS_ORIGINS", default=("*",)),
        )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def normalize_environment(value: str) -> str:
    environment = value.strip().lower()
    if environment not in VALID_ENVIRONMENTS:
        allowed = ", ".join(sorted(VALID_ENVIRONMENTS))
        raise RuntimeError(f"APP_ENV must be one of: {allowed}")
    return environment


def parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    value = raw_value.strip().lower()
    if value in TRUTHY_VALUES:
        return True
    if value in FALSY_VALUES:
        return False

    raise RuntimeError(f"{name} must be true or false")


def parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    values = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    return values or default
