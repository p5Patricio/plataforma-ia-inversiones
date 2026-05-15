from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADE_ACTIONS = {"BUY", "SELL"}


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    position_size: float = 1.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    allow_short: bool = True

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict
    trades: pd.DataFrame


def run_prediction_backtest(feedback: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """Backtest evaluated predictions using label outcome returns as trade outcomes."""
    config = config or BacktestConfig()
    if feedback.empty:
        return BacktestResult(metrics=_empty_metrics(config), trades=pd.DataFrame())

    required = {"timestamp", "predicted_action", "confidence", "outcome_return"}
    missing = required - set(feedback.columns)
    if missing:
        raise ValueError(f"feedback missing columns: {sorted(missing)}")

    evaluated = feedback.dropna(subset=["outcome_return"]).copy()
    evaluated["timestamp"] = pd.to_datetime(evaluated["timestamp"], utc=True)
    evaluated["confidence"] = pd.to_numeric(evaluated["confidence"], errors="coerce")
    evaluated["outcome_return"] = pd.to_numeric(evaluated["outcome_return"], errors="coerce")
    evaluated = evaluated.sort_values("timestamp")

    rows = []
    equity = config.initial_capital
    for _, row in evaluated.iterrows():
        action = str(row["predicted_action"])
        gross_return = _gross_return_for_action(action, row["outcome_return"], config)
        cost = config.round_trip_cost if action in TRADE_ACTIONS else 0.0
        net_return = (gross_return * config.position_size) - cost
        equity *= 1 + net_return
        rows.append(
            {
                "timestamp": row["timestamp"],
                "action": action,
                "confidence": row["confidence"],
                "gross_return": gross_return,
                "net_return": net_return,
                "cost": cost,
                "equity": equity,
                "metadata": {
                    "actual_label": row.get("actual_label"),
                    "model_name": row.get("model_name"),
                    "model_version": row.get("model_version"),
                },
            }
        )

    trades = pd.DataFrame(rows)
    return BacktestResult(metrics=_metrics_from_trades(trades, config), trades=trades)


def _gross_return_for_action(action: str, outcome_return: float, config: BacktestConfig) -> float:
    if action == "BUY":
        return float(outcome_return)
    if action == "SELL" and config.allow_short:
        return float(-outcome_return)
    return 0.0


def _metrics_from_trades(trades: pd.DataFrame, config: BacktestConfig) -> dict:
    if trades.empty:
        return _empty_metrics(config)

    returns = trades["net_return"]
    equity = trades["equity"]
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    traded = trades[trades["action"].isin(TRADE_ACTIONS)]

    return {
        "initial_capital": config.initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / config.initial_capital - 1),
        "max_drawdown": float(drawdowns.min()),
        "trade_count": int(len(trades)),
        "active_trade_count": int(len(traded)),
        "exposure": float(len(traded) / len(trades)),
        "win_rate": _nullable_float((returns > 0).mean()),
        "average_net_return": _nullable_float(returns.mean()),
        "profit_factor": None if losses == 0 else float(gains / abs(losses)),
        "sharpe_like": _sharpe_like(returns),
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "position_size": config.position_size,
        "allow_short": config.allow_short,
    }


def _empty_metrics(config: BacktestConfig) -> dict:
    return {
        "initial_capital": config.initial_capital,
        "final_equity": config.initial_capital,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
        "active_trade_count": 0,
        "exposure": 0.0,
        "win_rate": None,
        "average_net_return": None,
        "profit_factor": None,
        "sharpe_like": None,
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "position_size": config.position_size,
        "allow_short": config.allow_short,
    }


def _sharpe_like(returns: pd.Series) -> float | None:
    std = returns.std()
    if pd.isna(std) or std == 0:
        return None
    return float((returns.mean() / std) * np.sqrt(252))


def _nullable_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
