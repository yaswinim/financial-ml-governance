import json
from pathlib import Path

from fm_gov.config import PATHS


def score_model(model_id: str = "rf_vol_model_v1"):
    model_dir = PATHS.registry / model_id

    with open(model_dir / "metrics.json") as f:
        metrics = json.load(f)

    with open(model_dir / "stress_results.json") as f:
        stress = json.load(f)

    baseline = stress["baseline_mae"]
    noise_ratio = stress["noise_mae"] / baseline
    vol_ratio = stress["vol_spike_mae"] / baseline

    risk_score = 0
    if noise_ratio > 1.3:
        risk_score += 1
    if vol_ratio > 2.0:
        risk_score += 2
    if metrics["r2"] < 0.3:
        risk_score += 1

    if risk_score <= 1:
        decision = "APPROVE"
    elif risk_score == 2:
        decision = "APPROVE WITH CAUTION"
    else:
        decision = "REJECT"

    result = {
        "baseline_mae": baseline,
        "noise_degradation": noise_ratio,
        "volatility_degradation": vol_ratio,
        "r2": metrics["r2"],
        "risk_score": risk_score,
        "decision": decision,
    }

    with open(model_dir / "risk_assessment.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    print(score_model())
