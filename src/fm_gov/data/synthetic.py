from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_market(n: int = 2500, seed: int = 7) -> pd.DataFrame:
    """
    Generate a reproducible synthetic financial time series with regimes:
      - low volatility
      - high volatility
      - crash
    Includes engineered features and a next-period volatility target.
    """
    rng = np.random.default_rng(seed)

    # Regime definition
    regimes = np.zeros(n, dtype=int)
    regimes[int(0.35 * n): int(0.7 * n)] = 1      # high volatility
    regimes[int(0.7 * n): int(0.78 * n)] = 2     # crash
    regimes[int(0.78 * n):] = 0

    mu = np.where(regimes == 2, -0.0015, 0.0003)
    sigma = np.where(regimes == 0, 0.008, np.where(regimes == 1, 0.018, 0.04))

    returns = mu + sigma * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"close": close})
    df["ret"] = np.log(df["close"]).diff().fillna(0.0)

    # Feature engineering
    df["vol_10"] = df["ret"].rolling(10).std().fillna(0.0)
    df["vol_30"] = df["ret"].rolling(30).std().fillna(0.0)
    df["mom_5"] = df["ret"].rolling(5).sum().fillna(0.0)
    df["mom_20"] = df["ret"].rolling(20).sum().fillna(0.0)
    df["dd_60"] = (df["close"] / df["close"].rolling(60).max() - 1.0).fillna(0.0)

    df["regime"] = regimes

    # Target: next-window realized volatility proxy
    df["target_next_vol"] = (
        df["ret"].rolling(5).std().shift(-1).fillna(method="ffill").fillna(0.0)
    )

    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.7):
    """
    Time-ordered train/test split (no shuffling).
    """
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
