# Mental Health in Tech — Prediction Model

## Business Problem
Tech companies struggle to identify employees who need mental health support.
This model predicts which workers are likely to seek treatment so HR can
provide proactive support before crisis hits.

## Dataset
- Source: [Kaggle — Mental Health in Tech Survey]((https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey))
- 1259 tech workers surveyed
- 24 raw features + 5 engineered features
- Target: treatment (Yes=1 / No=0)
- 58.6% of workers sought treatment

## Project Steps
1. Load and explore data
2. EDA — found key patterns
3. Clean data (age, gender, missing values)
4. Feature Engineering (5 new features)
5. Train/test split (stratified)
6. ML Pipeline (impute + encode + scale)
7. Compared 4 models + XGBoost
8. Overfitting check
9. Tuned XGBoost with GridSearchCV
10. Final evaluation

## Feature Engineering
| Feature | Description |
|---|---|
| Company_Support_Score | Combines 5 company support columns |
| Openness_Score | Combines 5 workplace openness columns |
| Risk_Level | family_history + work_interfere combined |
| Is_Small_Company | Company size simplified (0 or 1) |
| Age_Group | Young / Adult / Senior / Elder |

## Model Comparison
| Model | Accuracy | F1 | CV Mean |
|---|---|---|---|
| Logistic Regression | 78.1% | 78.1% | 73.3% |
| Decision Tree | 72.9% | 73.8% | 71.9% |
| Random Forest | 78.5% | 78.6% | 74.7% |
| **XGBoost** | **81.3%** | **81.1%** | **74.5%** |

## Overfitting Check
| Model | Train | Test | Gap | Status |
|---|---|---|---|---|
| Random Forest | 100.0% | 78.5% | 21.5% | Overfitting! |
| XGBoost | 82.3% | 81.3% | 1.0% | Good fit |

## Final Results
- Best Model: Tuned XGBoost
- Accuracy: 81.3%
- F1 Score: 81.1%
- CV Score: 74.8%

## Key Findings
- work_interfere is the strongest predictor
- family_history doubles treatment likelihood
- Company support score significantly reduces risk
- Random Forest overfit badly (100% train accuracy!)
- XGBoost best model with only 1% train/test gap
