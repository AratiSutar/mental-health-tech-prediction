# Mental Health in Tech — Prediction Model

## Business Problem
Tech companies struggle to identify employees
who need mental health support. This model
predicts which workers are likely to seek
treatment so HR can take proactive action.

## Dataset
- Source: Kaggle — Mental Health in Tech Survey
- 1259 tech workers
- 24 raw features + 5 engineered
- Target: treatment (Yes/No)
- 58.6% sought treatment

## Steps Followed
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
11. Predicted new employee profiles

## Feature Engineering
| Feature | Description |
|---|---|
| Company_Support_Score | 5 support columns combined |
| Openness_Score | 5 openness columns combined |
| Risk_Level | family_history + work_interfere |
| Is_Small_Company | Company size (0 or 1) |
| Age_Group | Young/Adult/Senior/Elder |

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
- work_interfere = strongest predictor
- family_history doubles treatment likelihood
- Company support reduces risk significantly
- Random Forest overfit to 100% train accuracy!
- XGBoost: only 1% train/test gap

## Predictions
| Employee | Result | Confidence |
|---|---|---|
| Junior Dev (supportive) | No Treatment | 87.4% |
| Senior Dev (no support) | Needs Treatment | 85.3% |
| Freelancer (high risk) | Needs Treatment | 77.6% |

## Tools
Python · Pandas · Scikit-learn · XGBoost ·
Matplotlib · Seaborn · Google Colab

Arati Sutar
[LinkedIn]((https://www.linkedin.com/in/aratisutar/))
