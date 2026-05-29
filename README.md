# PulseML — Cardiovascular Risk Prediction System

End-to-end machine learning project on **~70,000 real patient records** from the public [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). Includes **hyperparameter tuning** (RandomizedSearchCV), 5 classifiers, cross-validation, saved models, and a Streamlit dashboard.



## Dataset

| Property | Value |
|----------|--------|
| Source | Sulanova cardiovascular disease dataset (public) |
| Raw size | ~70,000 patients |
| After cleaning | ~65,000+ (invalid BP/height/weight removed) |
| Target | `cardio` — presence of cardiovascular disease |
| Features | Age, gender, height, weight, BMI, blood pressure, cholesterol, glucose, lifestyle |

---



---

## Models (all hyperparameter-tuned)

| Model | Notes |
|-------|--------|
| Logistic Regression | Linear baseline |
| Random Forest | Bagging ensemble |
| Gradient Boosting | sklearn GBM |
| XGBoost | Tuned `scale_pos_weight` for imbalance |
| Hist Gradient Boosting | Fast on large data (replaces SVM) |

Best model auto-selected by composite ROC-AUC + F1.

### Latest tuned results (68,404 patients after cleaning)

| Model | Accuracy | F1 | ROC-AUC | 5-fold CV AUC |
|-------|----------|-----|---------|---------------|
| **XGBoost (best)** | 73.3% | 0.72 | **0.801** | 0.802 |
| Gradient Boosting | 73.4% | 0.72 | 0.800 | 0.801 |
| Random Forest | 73.1% | 0.71 | 0.798 | 0.800 |
| Hist Gradient Boosting | 73.2% | 0.72 | 0.798 | 0.800 |
| Logistic Regression | 72.5% | 0.71 | 0.791 | 0.792 |

Balanced classes (~50% disease rate) · Hyperparameter tuning: 18 iterations × 3-fold CV on 30k sample, refit on full train set.

---

## Project structure

```
aiml_project/
├── data/cardiovascular_disease.csv   # Downloaded on first run
├── models/*.joblib                   # Tuned pipelines
├── models/best_hyperparameters.json  # Tuning results per model
├── figures/metrics.json              # Final metrics
├── src/train.py                      # Tuning + training
├── run_pipeline.py
└── app/streamlit_app.py
```


