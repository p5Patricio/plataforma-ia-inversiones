from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import cos, sin
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from requests import RequestException

from app_config import AppConfig
from brain.logic import generate_signals
from brain.risk import RiskPolicy, apply_risk_policy
from collector.supabase_repository import SupabaseConfig, SupabaseRepository


APP_CONFIG = AppConfig.from_env()

app = FastAPI(title="Plataforma IA Inversiones API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(APP_CONFIG.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskProfilePayload(BaseModel):
    name: str = Field(default="default", min_length=1, max_length=80)
    max_position_size: float = Field(default=0.10, ge=0, le=1)
    min_confidence_to_trade: float = Field(default=0.60, ge=0, le=1)
    max_expected_risk: float = Field(default=0.05, ge=0, le=1)
    stop_loss: float = Field(default=0.02, ge=0, le=1)
    take_profit: float = Field(default=0.04, ge=0, le=5)
    allow_short: bool = True

def get_repository() -> SupabaseRepository | None:
    try:
        return SupabaseRepository(SupabaseConfig.from_env())
    except RuntimeError:
        return None


def get_app_config() -> AppConfig:
    return APP_CONFIG


def get_access_token(authorization: Annotated[str | None, Header()] = None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization Bearer token requerido")
    return token


def get_optional_user_id(
    access_token: str | None = Depends(get_access_token),
    repository: SupabaseRepository | None = Depends(get_repository),
) -> str | None:
    if access_token is None:
        return None
    if repository is None:
        raise HTTPException(status_code=503, detail="Supabase no disponible para autenticar usuario")
    try:
        return repository.get_auth_user(access_token)["id"]
    except (RuntimeError, RequestException):
        raise HTTPException(status_code=401, detail="Token de usuario invalido") from None


@app.get("/")
def read_root():
    return {"message": "API de Plataforma IA Inversiones funcionando"}


@app.get("/api/assets")
def get_assets(
    repository: SupabaseRepository | None = Depends(get_repository),
    config: AppConfig = Depends(get_app_config),
):
    if repository is None:
        require_demo_fallback(config)
        return demo_assets()

    try:
        return repository.get_assets()
    except (RuntimeError, RequestException):
        require_demo_fallback(config)
        return demo_assets()


@app.get("/api/prices/{ticker}")
def get_prices(
    ticker: str,
    limit: int = 100,
    repository: SupabaseRepository | None = Depends(get_repository),
    config: AppConfig = Depends(get_app_config),
):
    if repository is None:
        require_demo_ticker(ticker, config)
        return demo_prices(ticker, limit=limit)

    try:
        asset_id = repository.get_asset_id(ticker)
        prices = repository.get_prices(asset_id, limit=limit, ascending=False)
        return prices.to_dict(orient="records")
    except ValueError:
        if not should_use_demo_ticker(ticker, config):
            raise HTTPException(status_code=404, detail="Activo no encontrado") from None
    except (RuntimeError, RequestException):
        require_demo_ticker(ticker, config)

    return demo_prices(ticker, limit=limit)


@app.get("/api/analysis/{ticker}")
def analyze_ticker(
    ticker: str,
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    user_id: str | None = Depends(get_optional_user_id),
    repository: SupabaseRepository | None = Depends(get_repository),
    config: AppConfig = Depends(get_app_config),
):
    if repository is None:
        require_demo_ticker(ticker, config)
        prices = demo_prices(ticker, limit=100)
        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "source": "demo_indicators",
            "analysis": generate_signals(prices),
        }

    try:
        asset_id = repository.get_asset_id(ticker)
        try:
            prediction = repository.get_latest_prediction(
                asset_id=asset_id,
                model_name=model_name,
                model_version=model_version,
            )
        except (RuntimeError, RequestException):
            prediction = None

        if prediction:
            profile = get_user_risk_profile(repository, user_id)
            if profile:
                prediction = apply_user_risk_profile_to_prediction(prediction, profile)
            return {
                "ticker": ticker.upper(),
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "source": "prediction",
                "analysis": format_prediction_analysis(prediction),
            }

        prices = repository.get_prices(asset_id, limit=100, ascending=False)
        if prices.empty:
            raise HTTPException(status_code=404, detail="No hay precios para analizar")

        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "source": "fallback_indicators",
            "analysis": generate_signals(prices.to_dict(orient="records")),
        }
    except ValueError:
        if not should_use_demo_ticker(ticker, config):
            raise HTTPException(status_code=404, detail="Activo no encontrado") from None
    except (RuntimeError, RequestException):
        require_demo_ticker(ticker, config)

    prices = demo_prices(ticker, limit=100)
    return {
        "ticker": ticker.upper(),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "source": "demo_indicators",
        "analysis": generate_signals(prices),
    }


@app.get("/api/predictions/{ticker}")
def get_prediction_history(
    ticker: str,
    limit: int = Query(default=20, ge=1, le=100),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    only_evaluated: bool = Query(default=False),
    repository: SupabaseRepository | None = Depends(get_repository),
    config: AppConfig = Depends(get_app_config),
):
    if repository is None:
        require_demo_ticker(ticker, config)
        return []

    try:
        asset_id = repository.get_asset_id(ticker)
        feedback = repository.get_prediction_feedback(
            asset_id=asset_id,
            model_name=model_name,
            model_version=model_version,
            only_evaluated=only_evaluated,
            limit=limit,
            ascending=False,
        )
        return [format_prediction_history_row(row) for row in feedback.to_dict(orient="records")]
    except ValueError:
        if not should_use_demo_ticker(ticker, config):
            raise HTTPException(status_code=404, detail="Activo no encontrado") from None
    except (RuntimeError, RequestException):
        require_demo_ticker(ticker, config)

    return []


@app.get("/api/backtests/{ticker}")
def get_backtest_history(
    ticker: str,
    limit: int = Query(default=10, ge=1, le=50),
    repository: SupabaseRepository | None = Depends(get_repository),
    config: AppConfig = Depends(get_app_config),
):
    if repository is None:
        require_demo_ticker(ticker, config)
        return []

    try:
        asset_id = repository.get_asset_id(ticker)
        backtests = repository.get_backtests(asset_id=asset_id, limit=limit, ascending=False)
        return [format_backtest_history_row(row) for row in backtests.to_dict(orient="records")]
    except ValueError:
        if not should_use_demo_ticker(ticker, config):
            raise HTTPException(status_code=404, detail="Activo no encontrado") from None
    except (RuntimeError, RequestException):
        require_demo_ticker(ticker, config)

    return []


@app.get("/api/risk-profile")
def get_risk_profile(
    user_id: str | None = Depends(get_optional_user_id),
    repository: SupabaseRepository | None = Depends(get_repository),
):
    if user_id is None or repository is None:
        return format_risk_profile(None, source="default")

    try:
        profile = repository.get_default_user_risk_profile(user_id)
    except (RuntimeError, RequestException):
        profile = None
    return format_risk_profile(profile, source="user" if profile else "default")


@app.put("/api/risk-profile")
def update_risk_profile(
    payload: RiskProfilePayload,
    user_id: str | None = Depends(get_optional_user_id),
    repository: SupabaseRepository | None = Depends(get_repository),
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Autenticacion requerida")
    if repository is None:
        raise HTTPException(status_code=503, detail="Supabase no disponible para guardar perfil de riesgo")

    try:
        profile = repository.upsert_default_user_risk_profile(user_id, payload.model_dump())
    except (RuntimeError, RequestException):
        raise HTTPException(status_code=503, detail="No se pudo guardar el perfil de riesgo") from None
    return format_risk_profile(profile, source="user")


def require_demo_fallback(config: AppConfig) -> None:
    if not config.allow_demo_fallback:
        raise HTTPException(
            status_code=503,
            detail="Fuente de datos no disponible y modo demo desactivado",
        )


def require_demo_ticker(ticker: str, config: AppConfig) -> None:
    require_demo_fallback(config)
    if not is_demo_ticker(ticker):
        raise HTTPException(status_code=404, detail="Activo no encontrado") from None


def should_use_demo_ticker(ticker: str, config: AppConfig) -> bool:
    return config.allow_demo_fallback and is_demo_ticker(ticker)


def format_risk_profile(profile: dict | None, source: str) -> dict:
    policy = risk_policy_from_profile(profile)
    return {
        "source": source,
        "profile": {
            "name": (profile or {}).get("name", "default"),
            "max_position_size": policy.max_position_size,
            "min_confidence_to_trade": policy.min_confidence_to_trade,
            "max_expected_risk": policy.max_expected_risk,
            "stop_loss": policy.stop_loss,
            "take_profit": policy.take_profit,
            "allow_short": policy.allow_short,
        },
    }


def risk_policy_from_profile(profile: dict | None) -> RiskPolicy:
    default_policy = RiskPolicy()
    if not profile:
        return default_policy
    return RiskPolicy(
        max_position_size=float(profile.get("max_position_size", default_policy.max_position_size)),
        min_confidence_to_trade=float(profile.get("min_confidence_to_trade", default_policy.min_confidence_to_trade)),
        max_expected_risk=float(profile.get("max_expected_risk", default_policy.max_expected_risk)),
        stop_loss=float(profile.get("stop_loss", default_policy.stop_loss)),
        take_profit=float(profile.get("take_profit", default_policy.take_profit)),
        allow_short=bool(profile.get("allow_short", default_policy.allow_short)),
    )


def get_user_risk_profile(repository: SupabaseRepository | None, user_id: str | None) -> dict | None:
    if repository is None or user_id is None:
        return None
    try:
        return repository.get_default_user_risk_profile(user_id)
    except (RuntimeError, RequestException):
        return None


def apply_user_risk_profile_to_prediction(prediction: dict, profile: dict) -> dict:
    metadata = dict(prediction.get("metadata") or {})
    current_risk = dict(metadata.get("risk") or {})
    base_action = current_risk.get("pre_risk_action") or prediction.get("predicted_action") or prediction.get("action") or "HOLD"
    prediction_frame = pd.DataFrame(
        [
            {
                "action": base_action,
                "confidence": prediction.get("confidence") or 0,
                "expected_risk": prediction.get("expected_risk"),
                "metadata": metadata,
            }
        ]
    )
    adjusted = apply_risk_policy(prediction_frame, risk_policy_from_profile(profile)).iloc[0]
    adjusted_prediction = dict(prediction)
    adjusted_prediction["predicted_action"] = adjusted["action"]
    adjusted_prediction["metadata"] = adjusted["metadata"]
    adjusted_prediction["metadata"]["risk"]["profile_source"] = "user"
    adjusted_prediction["metadata"]["risk"]["profile_name"] = profile.get("name", "default")
    return adjusted_prediction


def format_prediction_analysis(prediction: dict) -> dict:
    metadata = prediction.get("metadata") or {}
    risk = metadata.get("risk") or {}
    probabilities = prediction.get("probabilities") or {}
    action = prediction.get("predicted_action") or prediction.get("action") or "HOLD"
    confidence = prediction.get("confidence") or 0

    reasons = [
        f"Modelo {prediction.get('model_name')}:{prediction.get('model_version')}",
        f"Feature set {prediction.get('feature_set')}",
    ]
    blocked = risk.get("blocked_reasons") or []
    if blocked:
        reasons.append(f"Bloqueada por riesgo: {', '.join(blocked)}")

    return {
        "signal": action,
        "confidence": float(confidence),
        "reason": "Prediccion versionada del modelo",
        "reasons": reasons,
        "probabilities": probabilities,
        "model": {
            "name": prediction.get("model_name"),
            "version": prediction.get("model_version"),
            "run_id": prediction.get("model_run_id"),
            "feature_set": prediction.get("feature_set"),
            "label_method": prediction.get("label_method"),
            "horizon": prediction.get("horizon"),
        },
        "risk": {
            "position_size": risk.get("position_size", 0),
            "stop_loss": risk.get("stop_loss"),
            "take_profit": risk.get("take_profit"),
            "blocked_reasons": blocked,
            "profile_source": risk.get("profile_source"),
            "profile_name": risk.get("profile_name"),
        },
        "feedback": {
            "actual_label": prediction.get("actual_label"),
            "is_correct": prediction.get("is_correct"),
            "outcome_return": prediction.get("outcome_return"),
        },
        "indicators": {
            "rsi": None,
            "close": None,
            "sma_20": None,
        },
        "prediction_timestamp": prediction.get("timestamp"),
        "expected_return": prediction.get("expected_return"),
        "expected_risk": prediction.get("expected_risk"),
    }


def format_backtest_history_row(backtest: dict) -> dict:
    metrics = backtest.get("metrics") or {}
    params = backtest.get("params") or {}
    model = backtest.get("model_runs") or {}
    return {
        "id": backtest.get("id"),
        "name": backtest.get("name"),
        "started_at": backtest.get("started_at"),
        "ended_at": backtest.get("ended_at"),
        "created_at": backtest.get("created_at"),
        "metrics": {
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "profit_factor": metrics.get("profit_factor"),
            "active_trade_count": metrics.get("active_trade_count"),
            "trade_count": metrics.get("trade_count"),
            "win_rate": metrics.get("win_rate"),
            "exposure": metrics.get("exposure"),
            "final_equity": metrics.get("final_equity"),
        },
        "params": params,
        "model": {
            "name": model.get("model_name"),
            "version": model.get("model_version"),
            "run_id": backtest.get("model_run_id"),
            "feature_set": model.get("feature_set"),
            "label_method": model.get("label_method"),
            "horizon": model.get("horizon"),
        },
    }


def format_prediction_history_row(prediction: dict) -> dict:
    metadata = prediction.get("metadata") or {}
    risk = metadata.get("risk") or {}
    blocked = risk.get("blocked_reasons") or []
    return {
        "prediction_id": prediction.get("prediction_id"),
        "ticker": prediction.get("ticker"),
        "timestamp": prediction.get("timestamp"),
        "created_at": prediction.get("prediction_created_at"),
        "action": prediction.get("predicted_action") or prediction.get("action") or "HOLD",
        "confidence": prediction.get("confidence"),
        "probabilities": prediction.get("probabilities") or {},
        "expected_return": prediction.get("expected_return"),
        "expected_risk": prediction.get("expected_risk"),
        "model": {
            "name": prediction.get("model_name"),
            "version": prediction.get("model_version"),
            "run_id": prediction.get("model_run_id"),
            "feature_set": prediction.get("feature_set"),
            "label_method": prediction.get("label_method"),
            "horizon": prediction.get("horizon"),
        },
        "risk": {
            "position_size": risk.get("position_size", 0),
            "stop_loss": risk.get("stop_loss"),
            "take_profit": risk.get("take_profit"),
            "blocked_reasons": blocked,
        },
        "feedback": {
            "actual_label": prediction.get("actual_label"),
            "is_correct": prediction.get("is_correct"),
            "outcome_return": prediction.get("outcome_return"),
        },
    }


def demo_assets() -> list[dict]:
    return [
        {"id": "demo-btc-usd", "ticker": "BTC-USD", "name": "Bitcoin", "asset_class": "crypto"},
        {"id": "demo-eth-usd", "ticker": "ETH-USD", "name": "Ethereum", "asset_class": "crypto"},
        {"id": "demo-aapl", "ticker": "AAPL", "name": "Apple Inc.", "asset_class": "stock"},
        {"id": "demo-msft", "ticker": "MSFT", "name": "Microsoft Corp.", "asset_class": "stock"},
    ]


def is_demo_ticker(ticker: str) -> bool:
    return ticker.upper() in {asset["ticker"] for asset in demo_assets()}


def demo_prices(ticker: str, limit: int = 100) -> list[dict]:
    normalized = ticker.upper()
    profiles = {
        "BTC-USD": {"base": 106000.0, "trend": 0.0018, "cycle": 0.035, "period": 6.0, "phase": 0.0, "noise": 0.014},
        "ETH-USD": {"base": 5200.0, "trend": 0.0012, "cycle": 0.052, "period": 4.2, "phase": 1.4, "noise": 0.019},
        "AAPL": {"base": 225.0, "trend": 0.0007, "cycle": 0.018, "period": 9.0, "phase": 2.2, "noise": 0.006},
        "MSFT": {"base": 495.0, "trend": 0.0010, "cycle": 0.024, "period": 7.4, "phase": 3.1, "noise": 0.008},
    }
    profile = profiles.get(
        normalized,
        {"base": 100.0, "trend": 0.001, "cycle": 0.02, "period": 8.0, "phase": 0.8, "noise": 0.007},
    )
    rows = []
    today = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    for index in range(max(limit, 30)):
        age = max(limit, 30) - index
        trend = 1 + (index * profile["trend"])
        cycle = sin((index / profile["period"]) + profile["phase"]) * profile["cycle"]
        noise = cos((index * 1.7) + profile["phase"]) * profile["noise"]
        close = profile["base"] * trend * (1 + cycle + noise)
        rows.append(
            {
                "timestamp": (today - timedelta(days=age)).isoformat(),
                "open": round(close * 0.992, 2),
                "high": round(close * 1.018, 2),
                "low": round(close * 0.982, 2),
                "close": round(close, 2),
                "volume": int(1_000_000 + index * 7_500),
            }
        )

    return list(reversed(rows[-limit:]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
