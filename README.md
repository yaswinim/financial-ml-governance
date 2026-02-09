
# Financial ML Model Governance and Stress Testing

This project implements an end-to-end model governance pipeline for financial machine learning systems.  
Rather than focusing only on predictive accuracy, the system evaluates robustness, failure behavior, and deployment risk under stressed market conditions.

The pipeline trains a volatility forecasting model, applies controlled stress scenarios, quantifies performance degradation, assigns a risk score, and generates a formal Model Card documenting the final governance decision.

---

## Project overview

The system follows a realistic financial ML lifecycle:

1. Generate market data with multiple regimes (normal, high volatility, crash)
2. Train a time-aware machine learning model
3. Validate baseline performance
4. Stress test the model under adverse conditions
5. Quantify degradation and assign a risk score
6. Produce a deployment decision and model documentation

The objective is to determine whether a model should be deployed, not just whether it can be trained.

---

## Key components

### Model training
- Predicts next-period realized volatility
- Uses time-ordered train/test splits
- Persists trained models and metrics as versioned artifacts

### Stress testing
The model is evaluated under:
- Noise injection (simulating unstable or noisy inputs)
- Volatility regime shocks (simulating market crashes)

Performance degradation is measured relative to baseline conditions.

### Risk scoring
Stress-test results are converted into:
- Degradation ratios
- A composite risk score
- A governance decision:
  - APPROVE
  - APPROVE WITH CAUTION
  - REJECT

Rejecting unsafe models is an expected and correct outcome.

### Model card
A structured Model Card is automatically generated containing:
- Model purpose and scope
- Performance metrics
- Stress test behavior
- Risk assessment
- Deployment decision
- Limitations and assumptions

---

## Running the project

### Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn scipy joblib pyyaml matplotlib streamlit pytest ruff
