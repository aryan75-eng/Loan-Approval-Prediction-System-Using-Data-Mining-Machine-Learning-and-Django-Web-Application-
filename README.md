# 🏦 Loan Approval Prediction System

> A full-stack Machine Learning web application that predicts loan approval decisions using Data Mining techniques, trained scikit-learn classifiers, and a Django web interface.
>
> **⚠️ For educational purposes only. Not a real lending decision system.**

---

## 🌐 Live Demo

**🚀 [https://loan-approval-prediction-system-using-vpq8.onrender.com/](https://loan-approval-prediction-system-using-vpq8.onrender.com/)**

> *Hosted on Render (free tier). The app may take ~30 seconds to wake up on first visit.*

---

## 📸 Features

- 🔐 **User Authentication** — Register, login, logout
- 📊 **Interactive Dashboard** — Overview stats and charts
- 📁 **Dataset Upload** — Upload CSV/XLSX datasets for analysis
- 🔍 **EDA & Analytics** — Preprocessing overview and visualizations
- 🤖 **Loan Prediction** — Real-time prediction form with ML model
- 📈 **Model Performance** — Compare Logistic Regression, Decision Tree, Random Forest
- 📜 **Prediction History** — Track all past predictions

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Pipeline** | scikit-learn 1.4, pandas 2.2, numpy 1.26, matplotlib, seaborn |
| **Web Framework** | Django 4.2 |
| **Frontend** | Bootstrap 5 + Chart.js (CDN) |
| **Database** | SQLite |
| **Static Files** | WhiteNoise |
| **Production Server** | Gunicorn |
| **Deployment** | Render.com |
| **Python Version** | 3.11.9 |

---

## 📂 Dataset

### Kaggle — Loan Prediction Problem Dataset

| Property | Value |
|----------|-------|
| **Source** | Kaggle |
| **URL** | 🔗 [https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) |
| **File** | `loan_train.csv` |
| **Rows** | 614 applicants |
| **Columns** | 13 features + 1 target |
| **Task** | Binary Classification (Approved / Rejected) |

### Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | ID | Unique loan identifier |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Categorical | 0, 1, 2, 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numeric | Monthly income of applicant (₹) |
| `CoapplicantIncome` | Numeric | Monthly income of co-applicant (₹) |
| `LoanAmount` | Numeric | Loan amount requested (in thousands) |
| `Loan_Amount_Term` | Numeric | Term of loan in months |
| `Credit_History` | Numeric | 1 = Good credit history, 0 = Bad |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Loan_Status` | **Target** | **Y = Approved, N = Rejected** |

---

## 🤖 ML Pipeline

The offline pipeline (`src/`) follows these stages:

```
Raw CSV → Clean → EDA → Outlier Capping → Split → Preprocess → Train → Evaluate → Save
```

### Models Trained & Compared

| Model | Cross-Val Strategy | Metric |
|-------|--------------------|--------|
| Logistic Regression | 5-Fold Stratified CV | F1 Score |
| Decision Tree (max_depth=6) | 5-Fold Stratified CV | F1 Score |
| Random Forest (200 estimators, max_depth=8) | 5-Fold Stratified CV | F1 Score |

> **Best model** is selected by highest test F1 (tie-break: recall) and saved to `models/best_model.joblib`

### Preprocessing (sklearn Pipeline)

- **Numeric features**: `SimpleImputer(median)` → `StandardScaler`
- **Categorical features**: `SimpleImputer(most_frequent)` → `OneHotEncoder`
- **Outlier capping**: IQR Winsorization on `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`

---

## 📁 Project Structure

```
loan_prediction_system/
│
├── data/
│   ├── raw/                  # Place loan_train.csv here
│   └── processed/            # Auto-generated train/test splits
│
├── src/                      # Offline ML pipeline modules
│   ├── config.py             # Paths, constants, feature lists
│   ├── data_loader.py        # Load & validate raw CSV
│   ├── data_cleaning.py      # Missing value handling
│   ├── eda.py                # Exploratory data analysis plots
│   ├── outlier_detection.py  # IQR capping + boxplots
│   ├── preprocessing.py      # sklearn ColumnTransformer
│   ├── feature_selection.py  # Correlation heatmap, feature importance
│   ├── model_training.py     # Train + compare all models (main entry)
│   ├── model_evaluation.py   # Metrics, confusion matrix, CSV export
│   ├── prediction.py         # Inference helper
│   └── utils.py              # Logger, seed, directory helpers
│
├── models/                   # Saved model artifacts (auto-generated)
│   ├── best_model.joblib
│   ├── feature_columns.json
│   └── model_metadata.json
│
├── visualizations/           # EDA & evaluation plots (auto-generated)
├── reports/                  # Metrics CSV & classification reports
│
├── django_app/               # Django Web Application
│   ├── accounts/             # User registration & login
│   ├── analytics/            # EDA views & preprocessing overview
│   ├── dashboard/            # Home dashboard with stats
│   ├── datasets/             # Dataset upload & preview
│   ├── predictions/          # Prediction form, result, history
│   ├── config/               # Django settings, URLs, WSGI/ASGI
│   ├── static/               # CSS & JavaScript
│   ├── templates/            # Base HTML, partials, error pages
│   ├── build.sh              # Render build script
│   └── runtime.txt           # Python 3.11.9 pin for Render
│
├── tests/                    # Test suite (pytest)
│   ├── test_ml_pipeline.py
│   ├── test_ml_service.py
│   ├── test_forms.py
│   └── test_views.py
│
├── .env.example              # Environment variable template
├── .gitignore
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment blueprint
└── LICENSE                   # MIT License
```

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.11+
- Git

### 1 — Clone the Repository

```bash
git clone https://github.com/aryan75-eng/Loan-Approval-Prediction-System-Using-Data-Mining-Machine-Learning-and-Django-Web-Application-.git
cd Loan-Approval-Prediction-System-Using-Data-Mining-Machine-Learning-and-Django-Web-Application-
```

### 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Configure Environment

```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS/Linux
```

Edit `.env` and set:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
```

### 5 — Download the Dataset

1. Download **`loan_train.csv`** from Kaggle:
   👉 [https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

2. Place it at:
   ```
   data/raw/loan_train.csv
   ```

### 6 — Train the ML Model (offline)

```bash
python src/model_training.py
```

This generates:
- `models/best_model.joblib` — Best trained model
- `models/feature_columns.json` — Feature column names
- `models/model_metadata.json` — Training metadata & metrics
- `visualizations/*.png` — EDA & evaluation plots
- `reports/model_comparison.csv` — All model metrics

### 7 — Set Up Django

```bash
cd django_app
python manage.py migrate
python manage.py createsuperuser
python manage.py load_model_metrics   # Load model metrics into DB
python manage.py runserver
```

Open → **http://127.0.0.1:8000/**

---

## 🧪 Running Tests

```bash
cd django_app
pytest ../tests/ -v
```

---

## 🚀 Deployment (Render)

The app is configured for [Render.com](https://render.com) deployment.

| Field | Value |
|-------|-------|
| **Root Directory** | `django_app` |
| **Build Command** | `./build.sh` |
| **Start Command** | `gunicorn config.wsgi:application --bind 0.0.0.0:$PORT` |
| **Runtime** | Python 3.11 |

**Required Environment Variables on Render:**

| Key | Value |
|-----|-------|
| `SECRET_KEY` | Generate a secure random key |
| `DEBUG` | `False` |
| `ALLOWED_HOSTS` | `.onrender.com` |
| `DJANGO_SETTINGS_MODULE` | `config.settings` |

---

## 🔑 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | — | Django secret key (required in production) |
| `DEBUG` | `True` | Set to `False` in production |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated allowed hostnames |
| `MAX_UPLOAD_SIZE` | `10485760` | Max file upload size in bytes (10 MB) |

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE)

---

## 👤 Author

**Aryan** — [@aryan75-eng](https://github.com/aryan75-eng)

---

## 🙏 Acknowledgements

- [Kaggle — Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- [scikit-learn](https://scikit-learn.org/)
- [Django](https://www.djangoproject.com/)
- [Render](https://render.com)
