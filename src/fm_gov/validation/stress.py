import numpy as np
import pandas as pd


def stress_noise(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    stressed = df.copy()
    for col in ["vol_10", "vol_30", "mom_5", "mom_20", "dd_60"]:
        noise = np.random.normal(0, scale, size=len(stressed))
        stressed[col] = stressed[col] + noise
    return stressed


def stress_volatility_regime(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    stressed = df.copy()
    stressed["vol_10"] *= multiplier
    stressed["vol_30"] *= multiplier
    return stressed
