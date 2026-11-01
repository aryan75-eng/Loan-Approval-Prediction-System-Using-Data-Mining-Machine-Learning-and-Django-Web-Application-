# Loan Approval Prediction System

> Academic mini-project — financial analytics / supervised binary classification  
> **For educational purposes only. Not a real lending decision system.**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Pipeline | scikit-learn 1.4, pandas 2.2, matplotlib / seaborn |
| Web Framework | Django 4.2 |
| Frontend | Bootstrap 5 + Chart.js (CDN) |
| Database | SQLite (dev) |
| Dataset | [Kaggle Loan Prediction Problem](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) — 614 rows × 13 cols |

---

## Quick Start

### 1 — Prerequisites

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2 — Environment

```bash
copy .env.example .env
# Edit .env: set SECRET_KEY and DEBUG
```

### 3 — Place Dataset

Download `loan_train.csv` from Kaggle and place it at:

```
data/raw/loan_train.csv
```

### 4 — Train ML Model (offline)

```bash
python src/model_training.py
```

Produces:
- `models/best_model.joblib`
- `models/feature_columns.json`
- `models/model_metadata.json`
- `visualizations/*.png`
- `reports/model_comparison.csv`

### 5 — Set Up Django

```bash
cd django_app
python manage.py migrate
python manage.py createsuperuser
python manage.py load_model_metrics   # import reports/model_comparison.csv → DB
python manage.py runserver
```

Open http://127.0.0.1:8000/

---

## Project Structure

```
loan_prediction_system/
├── data/            # Raw + processed data
├── src/             # Offline ML pipeline
├── models/          # Trained model artifacts
├── visualizations/  # EDA & evaluation plots
├── reports/         # Metrics CSV & text reports
├── tests/           # Test suite
└── django_app/      # Django web application
```

---

## Running Tests

```bash
cd django_app
pytest ../tests/ -v
```

---

## License

MIT — see `LICENSE`
