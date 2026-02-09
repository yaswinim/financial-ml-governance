import json
from pathlib import Path

from fm_gov.config import PATHS


def generate_model_card(model_id: str = "rf_vol_model_v1"):
    model_dir = PATHS.registry / model_id

    with open(model_dir / "metrics.json") as f:
        metrics = json.load(f)

    with open(model_dir / "stress_results.json") as f:
        stress = json.load(f)

    with open(model_dir / "risk_assessment.json") as f:
        risk = json.load(f)

    card = f"""
# Model Card — {model_id}

## Model Overview
- **Model Type:** RandomForestRegressor
- **Objective:** Predict next-period realized volatility
- **Training Samples:** {metrics['n_train']}
- **Test Samples:** {metrics['n_test']}

## Performance
- **MAE:** {metrics['mae']:.6f}
- **R²:** {metrics['r2']:.3f}

## Stress Testing Results
| Scenario | MAE | Degradation |
|--------|-----|-------------|
| Baseline | {stress['baseline_mae']:.6f} | 1.00× |
| Noise Injection | {stress['noise_mae']:.6f} | {stress['noise_mae']/stress['baseline_mae']:.2f}× |
| Volatility Spike | {stress['vol_spike_mae']:.6f} | {stress['vol_spike_mae']/stress['baseline_mae']:.2f}× |

## Risk Assessment
- **Risk Score:** {risk['risk_score']}
- **Decision:** **{risk['decision']}**

## Interpretation
The model demonstrates reasonable predictive power under normal conditions, but exhibits
significant degradation under volatility regime shifts. This behavior indicates sensitivity
to structural breaks, making the model unsuitable for production deployment without further
robustness enhancements.

## Limitations
- Trained on synthetic data
- Single-model architecture
- No retraining or adaptive mechanisms

## Governance Status
🚫 **Not approved for production**
"""

    out = model_dir / "MODEL_CARD.md"
    out.write_text(card.strip())

    return out


if __name__ == "__main__":
    print(generate_model_card())
