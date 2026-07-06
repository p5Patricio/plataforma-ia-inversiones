from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Any


@dataclass(frozen=True)
class PromotionCriteria:
    min_total_return: float = 0.0
    min_profit_factor: float = 1.0
    max_drawdown_floor: float = -0.25
    min_active_trades: int = 20
    require_positive_edge_vs_no_trade: bool = True


def build_candidate_row(
    summary: dict[str, Any],
    criteria: PromotionCriteria | None = None,
    drawdown_penalty: float = 1.0,
) -> dict[str, Any]:
    """Create a comparable candidate row from a backtest summary."""
    metrics = summary["model"]
    promotion = evaluate_promotion(summary, criteria or PromotionCriteria())
    objective_score = score_candidate(summary, drawdown_penalty=drawdown_penalty)

    return {
        "candidate_id": summary.get("candidate_id"),
        "scope": summary.get("scope"),
        "target_ticker": summary.get("target_ticker"),
        "model_name": summary.get("model_name"),
        "min_confidence": summary.get("min_confidence"),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "profit_factor": metrics.get("profit_factor"),
        "active_trade_count": metrics.get("active_trade_count"),
        "win_rate": metrics.get("win_rate"),
        "exposure": metrics.get("exposure"),
        "participating_asset_count": summary.get("participating_asset_count"),
        "objective_score": objective_score,
        "return_to_drawdown": return_to_drawdown(metrics),
        "promotion": promotion,
    }


def rank_candidate_summaries(
    summaries: list[dict[str, Any]],
    criteria: PromotionCriteria | None = None,
    drawdown_penalty: float = 1.0,
) -> list[dict[str, Any]]:
    rows = [
        build_candidate_row(summary, criteria=criteria, drawdown_penalty=drawdown_penalty)
        for summary in summaries
    ]
    return sorted(
        rows,
        key=lambda row: (row["promotion"]["status"] == "pass", row["objective_score"]),
        reverse=True,
    )


def score_candidate(summary: dict[str, Any], drawdown_penalty: float = 1.0) -> float:
    """Score candidates by return after penalizing drawdown and rewarding enough trades."""
    metrics = summary["model"]
    total_return = _float(metrics.get("total_return"))
    max_drawdown = abs(_float(metrics.get("max_drawdown")))
    active_trades = int(metrics.get("active_trade_count") or 0)
    profit_factor = _profit_factor_for_selection(metrics)

    trade_sample_bonus = min(log1p(active_trades) / 100, 0.05)
    profit_factor_bonus = min(max(profit_factor - 1.0, 0.0) * 0.02, 0.04)
    return total_return - (drawdown_penalty * max_drawdown) + trade_sample_bonus + profit_factor_bonus


def evaluate_promotion(summary: dict[str, Any], criteria: PromotionCriteria) -> dict[str, Any]:
    metrics = summary["model"]
    baselines = summary.get("baselines", {})
    no_trade_return = _float(baselines.get("no_trade", {}).get("total_return"))
    total_return = _float(metrics.get("total_return"))
    max_drawdown = _float(metrics.get("max_drawdown"))
    profit_factor = _profit_factor_for_selection(metrics)
    active_trades = int(metrics.get("active_trade_count") or 0)

    passed = []
    failed = []

    if total_return >= criteria.min_total_return:
        passed.append(f"total_return>={criteria.min_total_return}")
    else:
        failed.append(f"total_return_below_{criteria.min_total_return}")

    if profit_factor >= criteria.min_profit_factor:
        passed.append(f"profit_factor>={criteria.min_profit_factor}")
    else:
        failed.append(f"profit_factor_below_{criteria.min_profit_factor}")

    if max_drawdown >= criteria.max_drawdown_floor:
        passed.append(f"max_drawdown>={criteria.max_drawdown_floor}")
    else:
        failed.append(f"max_drawdown_below_{criteria.max_drawdown_floor}")

    if active_trades >= criteria.min_active_trades:
        passed.append(f"active_trades>={criteria.min_active_trades}")
    else:
        failed.append(f"active_trades_below_{criteria.min_active_trades}")

    if not criteria.require_positive_edge_vs_no_trade or total_return > no_trade_return:
        passed.append("positive_edge_vs_no_trade")
    else:
        failed.append("no_positive_edge_vs_no_trade")

    return {
        "status": "pass" if not failed else "fail",
        "passed": passed,
        "failed": failed,
        "criteria": {
            "min_total_return": criteria.min_total_return,
            "min_profit_factor": criteria.min_profit_factor,
            "max_drawdown_floor": criteria.max_drawdown_floor,
            "min_active_trades": criteria.min_active_trades,
            "require_positive_edge_vs_no_trade": criteria.require_positive_edge_vs_no_trade,
        },
    }


def return_to_drawdown(metrics: dict[str, Any]) -> float | None:
    total_return = _float(metrics.get("total_return"))
    max_drawdown = abs(_float(metrics.get("max_drawdown")))
    if max_drawdown == 0:
        return None
    return total_return / max_drawdown


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _profit_factor_for_selection(metrics: dict[str, Any]) -> float:
    profit_factor = metrics.get("profit_factor")
    if profit_factor is not None:
        return float(profit_factor)
    total_return = _float(metrics.get("total_return"))
    active_trades = int(metrics.get("active_trade_count") or 0)
    if total_return > 0 and active_trades > 0:
        return float("inf")
    return 0.0
