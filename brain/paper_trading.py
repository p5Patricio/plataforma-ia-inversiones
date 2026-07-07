from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


TRADE_ACTIONS = {"BUY", "SELL"}


@dataclass(frozen=True)
class PaperTradingConfig:
    initial_capital: float = 10_000.0
    default_position_size: float = 0.10
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    allow_short: bool = True

    @property
    def one_way_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class PaperTradingResult:
    metrics: dict[str, Any]
    timeline: pd.DataFrame


def run_paper_trading(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    config: PaperTradingConfig | None = None,
) -> PaperTradingResult:
    """Simulate a paper account from chronological predictions and observed prices."""
    config = config or PaperTradingConfig()
    if predictions.empty:
        return PaperTradingResult(metrics=_empty_metrics(config), timeline=pd.DataFrame())

    required_predictions = {"timestamp", "confidence"}
    if not ({"predicted_action", "action"} & set(predictions.columns)):
        required_predictions.add("predicted_action")
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(f"predictions missing columns: {sorted(missing_predictions)}")

    required_prices = {"timestamp", "close"}
    missing_prices = required_prices - set(prices.columns)
    if missing_prices:
        raise ValueError(f"prices missing columns: {sorted(missing_prices)}")

    signals = _prepare_signals(predictions)
    price_frame = _prepare_prices(prices)
    if price_frame.empty:
        return PaperTradingResult(metrics=_empty_metrics(config), timeline=pd.DataFrame())

    signals = pd.merge_asof(
        signals.sort_values("timestamp"),
        price_frame,
        on="timestamp",
        direction="forward",
    ).dropna(subset=["close"])

    equity = float(config.initial_capital)
    exposure = 0.0
    last_price: float | None = None
    rows = []

    for _, row in signals.iterrows():
        price = float(row["close"])
        mark_return = 0.0
        if last_price is not None and exposure:
            mark_return = (price / last_price) - 1
            equity *= 1 + (exposure * mark_return)

        target_exposure = _target_exposure(str(row["action"]), exposure, _position_size(row, config), config)
        exposure_delta = target_exposure - exposure
        cost = abs(exposure_delta) * config.one_way_cost
        if cost:
            equity *= 1 - cost

        exposure = target_exposure
        last_price = price
        rows.append(
            {
                "timestamp": row["timestamp"],
                "action": str(row["action"]),
                "confidence": float(row["confidence"]),
                "price": price,
                "mark_return": mark_return,
                "exposure": exposure,
                "exposure_delta": exposure_delta,
                "cost": cost,
                "equity": equity,
                "position_state": _position_state(exposure),
                "metadata": row.get("metadata") or {},
            }
        )

    timeline = pd.DataFrame(rows)
    return PaperTradingResult(metrics=_metrics_from_timeline(timeline, config), timeline=timeline)


def _prepare_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    signals = predictions.copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    signals["action"] = (
        signals["predicted_action"] if "predicted_action" in signals.columns else signals["action"]
    ).fillna("HOLD")
    signals["confidence"] = pd.to_numeric(signals["confidence"], errors="coerce").fillna(0.0)
    if "metadata" not in signals.columns:
        signals["metadata"] = [{} for _ in range(len(signals))]
    return signals.sort_values("timestamp").reset_index(drop=True)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    price_frame = prices[["timestamp", "close"]].copy()
    price_frame["timestamp"] = pd.to_datetime(price_frame["timestamp"], utc=True)
    price_frame["close"] = pd.to_numeric(price_frame["close"], errors="coerce")
    return price_frame.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)


def _target_exposure(action: str, current_exposure: float, position_size: float, config: PaperTradingConfig) -> float:
    if action == "BUY":
        return position_size
    if action == "SELL":
        return -position_size if config.allow_short else 0.0
    return current_exposure


def _position_size(row: pd.Series, config: PaperTradingConfig) -> float:
    metadata = row.get("metadata") or {}
    risk = metadata.get("risk") if isinstance(metadata, dict) else None
    if isinstance(risk, dict):
        size = risk.get("position_size")
        if size is not None and not pd.isna(size):
            return max(0.0, min(1.0, float(size)))
    return config.default_position_size


def _position_state(exposure: float) -> str:
    if exposure > 0:
        return "LONG"
    if exposure < 0:
        return "SHORT"
    return "FLAT"


def _metrics_from_timeline(timeline: pd.DataFrame, config: PaperTradingConfig) -> dict[str, Any]:
    if timeline.empty:
        return _empty_metrics(config)

    equity = timeline["equity"]
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1
    equity_returns = equity.pct_change().fillna(equity.iloc[0] / config.initial_capital - 1)
    gains = equity_returns[equity_returns > 0].sum()
    losses = equity_returns[equity_returns < 0].sum()
    trades = timeline[timeline["exposure_delta"].abs() > 0]

    return {
        "initial_capital": config.initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / config.initial_capital - 1),
        "max_drawdown": float(drawdowns.min()),
        "signal_count": int(len(timeline)),
        "trade_count": int(len(trades)),
        "active_signal_count": int(timeline["action"].isin(TRADE_ACTIONS).sum()),
        "average_abs_exposure": float(timeline["exposure"].abs().mean()),
        "open_exposure": float(timeline["exposure"].iloc[-1]),
        "open_position": _position_state(float(timeline["exposure"].iloc[-1])),
        "last_price": float(timeline["price"].iloc[-1]),
        "profit_factor": None if losses == 0 else float(gains / abs(losses)),
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "allow_short": config.allow_short,
    }


def _empty_metrics(config: PaperTradingConfig) -> dict[str, Any]:
    return {
        "initial_capital": config.initial_capital,
        "final_equity": config.initial_capital,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "signal_count": 0,
        "trade_count": 0,
        "active_signal_count": 0,
        "average_abs_exposure": 0.0,
        "open_exposure": 0.0,
        "open_position": "FLAT",
        "last_price": None,
        "profit_factor": None,
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "allow_short": config.allow_short,
    }
