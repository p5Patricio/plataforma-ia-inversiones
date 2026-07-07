from __future__ import annotations

import pytest

from app_config import AppConfig


def test_development_defaults_keep_demo_fallback_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ALLOW_DEMO_FALLBACK", raising=False)
    monkeypatch.delenv("API_CORS_ORIGINS", raising=False)

    config = AppConfig.from_env()

    assert config.environment == "development"
    assert config.allow_demo_fallback is True
    assert config.cors_origins == ("*",)


def test_production_defaults_disable_demo_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("ALLOW_DEMO_FALLBACK", raising=False)

    config = AppConfig.from_env()

    assert config.is_production is True
    assert config.allow_demo_fallback is False


def test_explicit_demo_fallback_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ALLOW_DEMO_FALLBACK", "true")

    config = AppConfig.from_env()

    assert config.allow_demo_fallback is True


def test_cors_origins_are_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:5173, https://app.example.com")

    config = AppConfig.from_env()

    assert config.cors_origins == ("http://localhost:5173", "https://app.example.com")


def test_invalid_environment_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "sandbox")

    with pytest.raises(RuntimeError, match="APP_ENV"):
        AppConfig.from_env()
