# AI-Integrated Cybersecurity Decision Intelligence

Research prototype integrating **AI + Cybersecurity + Decision Intelligence** for explainable, risk-aware and human-governed decision support.

## Pipeline

```text
CYBERSECURITY DATA
        ↓
DATA PREPROCESSING
        ↓
AI / MACHINE LEARNING
        ↓
PREDICTION
        ↓
EXPLAINABLE AI
        ↓
CYBER RISK & UNCERTAINTY
        ↓
DECISION INTELLIGENCE
        ↓
HUMAN-INTERPRETABLE ACTION
```

The code does not hard-code research results. Tables, figures and metrics are produced when the experiment is executed on actual input data.

## Dataset setup

Place one or more CSV files from a legitimate public cybersecurity dataset in `data/raw/`. Good choices include UNSW-NB15 and CIC-IDS2017. The loader combines CSV files and looks for a target column named `label`, `class`, `target`, `attack`, or `attack_cat`.

For a first local test, use a manageable subset. The default configuration samples at most 150,000 rows.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Test

```bash
pytest -q
```

## Run

```bash
python run_experiment.py
```

Outputs are generated under `reports/figures`, `reports/tables`, and `reports/results`.

## Models

The experiment compares Logistic Regression, Random Forest and XGBoost using precision, recall, F1, ROC-AUC and PR-AUC. Explainability uses SHAP when the selected model supports tree explanations.

## Decision Intelligence

The prototype combines attack probability, an experimental severity proxy, prediction uncertainty and asset criticality into a risk-aware decision score. It maps the score to LOW, MODERATE, HIGH or CRITICAL and generates a human-readable priority and recommended action.

The default weights and thresholds are research prototype assumptions, not validated industry standards. Human review remains part of the decision process.

## Security

Do not commit secrets, credentials, `.env` files, private logs or confidential enterprise datasets.
