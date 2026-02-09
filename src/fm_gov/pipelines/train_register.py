from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from fm_gov.data.synthetic import make_synthetic_market, time_split
from fm_gov.config import PATHS
from fm_gov.logging import get_logger

logger = get_logger("train")


def train_and_register(random_state: int = 42):
    PATHS.registry.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = make_synthetic_market()
    train, test = time_split(df)

    features = [
        "vol_10",
        "vol_30",
        "mom_5",
        "mom_20",
        "dd_60",
    ]
    target = "target_next_vol"

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    # 2) Train model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 3) Evaluate
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": features,
    }

    # 4) Register model
    model_id = f"rf_vol_model_v1"
    model_dir = PATHS.registry / model_id
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model trained and registered")
    logger.info(metrics)

    return model_id, metrics


if __name__ == "__main__":
    train_and_register()
