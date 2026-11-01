# Dataset Provenance & Schema

## Source

**Kaggle Loan Prediction Problem Dataset**  
URL: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset  
License: Community Data License Agreement

## Download

1. Go to the Kaggle URL above
2. Download `train.csv`
3. Rename it to `loan_train.csv`
4. Place it at `data/raw/loan_train.csv`

## Schema (614 rows × 13 columns)

| Column | Type | Description |
|--------|------|-------------|
| Loan_ID | object | Unique loan identifier (dropped before training) |
| Gender | object | Male / Female |
| Married | object | Yes / No |
| Dependents | object | 0 / 1 / 2 / 3+ |
| Education | object | Graduate / Not Graduate |
| Self_Employed | object | Yes / No |
| ApplicantIncome | int64 | Monthly applicant income |
| CoapplicantIncome | float64 | Monthly co-applicant income |
| LoanAmount | float64 | Loan amount (in thousands) |
| Loan_Amount_Term | float64 | Term of loan in months |
| Credit_History | float64 | 1.0 = meets guidelines, 0.0 = does not |
| Property_Area | object | Urban / Semiurban / Rural |
| Loan_Status | object | **Target** — Y (Approved) / N (Rejected) |

## Missing Values (approximate)

| Column | Missing |
|--------|---------|
| Gender | 13 |
| Married | 3 |
| Dependents | 15 |
| Self_Employed | 32 |
| LoanAmount | 22 |
| Loan_Amount_Term | 14 |
| Credit_History | 50 |

## Notes

- `Dependents` contains `3+` — treated as string category
- `Credit_History` is binary float — treated as numeric
- Class distribution: ~69% Approved (Y), ~31% Rejected (N) — mild imbalance
