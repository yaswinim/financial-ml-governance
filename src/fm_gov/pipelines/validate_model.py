import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from fm_gov.data.synthetic import make_synthetic_market, time_split
from fm_gov.validation.stress import stress_noise, stress_volatility_regime
from fm_gov.config import PATHS
from fm_gov.logging import get_logger

logger = get_logger("validate")


def validate(model_id: str = "rf_vol_model_v1"):
    model_dir = PATHS.registry / model_id
    model = joblib.load(model_dir / "model.joblib")

    df = make_synthetic_market()
    _, test = time_split(df)

    features = ["vol_10", "vol_30", "mom_5", "mom_20", "dd_60"]
    y_true = test["target_next_vol"]

    results = {}

    # Baseline
    base_pred = model.predict(test[features])
    results["baseline_mae"] = float(mean_absolute_error(y_true, base_pred))

    # Noise stress
    noise_df = stress_noise(test, scale=0.01)
    noise_pred = model.predict(noise_df[features])
    results["noise_mae"] = float(mean_absolute_error(y_true, noise_pred))

    # Volatility spike
    vol_df = stress_volatility_regime(test, multiplier=2.5)
    vol_pred = model.predict(vol_df[features])
    results["vol_spike_mae"] = float(mean_absolute_error(y_true, vol_pred))

    # Save
    out = model_dir / "stress_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Validation + stress testing complete")
    logger.info(results)

    return results


if __name__ == "__main__":
    validate()
