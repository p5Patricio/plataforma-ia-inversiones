from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "log_return_1d",
    "volatility_10d",
    "volatility_20d",
    "sma_10_ratio",
    "sma_20_ratio",
    "ema_10_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "volume_zscore_20",
    "atr_14",
    "drawdown_20",
]


def prepare_price_frame(prices: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Normalize raw OHLCV rows into a chronological DataFrame."""
    df = pd.DataFrame(prices).copy()
    missing = BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def build_features(prices: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """
    Build point-in-time features using only data available at each timestamp.

    The target labels are intentionally created elsewhere to keep leakage checks
    simple: this module never looks forward.
    """
    df = prepare_price_frame(prices)
    close = df["close"]
    volume = df["volume"]

    df["return_1d"] = close.pct_change(1)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["log_return_1d"] = np.log(close / close.shift(1))
    df["volatility_10d"] = df["log_return_1d"].rolling(10).std()
    df["volatility_20d"] = df["log_return_1d"].rolling(20).std()

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    df["sma_10_ratio"] = close / sma_10 - 1
    df["sma_20_ratio"] = close / sma_20 - 1
    df["ema_10_ratio"] = close / ema_10 - 1

    exp_fast = close.ewm(span=12, adjust=False).mean()
    exp_slow = close.ewm(span=26, adjust=False).mean()
    df["macd"] = exp_fast - exp_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi_14"] = calculate_rsi(close)

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    df["volume_zscore_20"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    df["atr_14"] = calculate_atr(df)

    rolling_high = close.rolling(20).max()
    df["drawdown_20"] = close / rolling_high - 1

    return df.replace([np.inf, -np.inf], np.nan)
