from __future__ import annotations

import pandas as pd

from brain.features import prepare_price_frame


BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


def fixed_horizon_labels(
    prices: list[dict] | pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.015,
    sell_threshold: float = -0.015,
) -> pd.DataFrame:
    """
    Label each row by future return at a fixed horizon.

    Labels are for supervised training only. They must never be joined back into
    features before the decision timestamp.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    df = prepare_price_frame(prices)
    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = df["future_close"] / df["close"] - 1
    df["label"] = HOLD
    df.loc[df["future_return"] > buy_threshold, "label"] = BUY
    df.loc[df["future_return"] < sell_threshold, "label"] = SELL
    df.loc[df["future_return"].isna(), "label"] = pd.NA

    return df[["timestamp", "future_return", "label"]]


def triple_barrier_labels(
    prices: list[dict] | pd.DataFrame,
    horizon: int = 5,
    profit_take: float = 0.03,
    stop_loss: float = 0.015,
) -> pd.DataFrame:
    """
    Label rows by the first barrier touched: take-profit, stop-loss, or timeout.

    This first implementation models a long candidate. A stop-loss touch is
    labeled SELL because the model should avoid or exit that opportunity.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if profit_take <= 0 or stop_loss <= 0:
        raise ValueError("profit_take and stop_loss must be positive")

    df = prepare_price_frame(prices)
    rows = []

    for index, row in df.iterrows():
        entry_price = row["close"]
        upper = entry_price * (1 + profit_take)
        lower = entry_price * (1 - stop_loss)
        window = df.iloc[index + 1 : index + horizon + 1]

        if len(window) < horizon:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "label": pd.NA,
                    "outcome_return": pd.NA,
                    "label_exit_timestamp": pd.NaT,
                }
            )
            continue

        label = HOLD
        exit_timestamp = pd.NaT
        outcome_return = pd.NA

        for _, future in window.iterrows():
            if future["high"] >= upper:
                label = BUY
                exit_timestamp = future["timestamp"]
                outcome_return = profit_take
                break
            if future["low"] <= lower:
                label = SELL
                exit_timestamp = future["timestamp"]
                outcome_return = -stop_loss
                break

        if pd.isna(exit_timestamp):
            final = window.iloc[-1]
            exit_timestamp = final["timestamp"]
            outcome_return = final["close"] / entry_price - 1

        rows.append(
            {
                "timestamp": row["timestamp"],
                "label": label,
                "outcome_return": outcome_return,
                "label_exit_timestamp": exit_timestamp,
            }
        )

    return pd.DataFrame(rows)
