# Credit Card Fraud Detection

A machine learning system that detects fraudulent credit card transactions, built end-to-end: data preprocessing, class imbalance handling, model comparison, threshold optimization, and a live deployed web app.

**Live demo:**  https://credit-card-fraud-detection-1-4tl3.onrender.com

---

## Problem

Credit card fraud detection is a classic imbalanced classification problem. In this dataset, only **0.17% of transactions are fraudulent** (492 out of 284,807) — meaning a model that predicts "legit" for everything would be 99.83% accurate while catching zero fraud. The real challenge is building a model that reliably catches fraud without overwhelming the system with false alarms.

## Dataset

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European cardholder transactions. Features `V1`-`V28` are PCA-transformed for confidentiality; `Amount` and `Time` are the only raw features.

## Approach

1. **Preprocessing** — removed 1,081 duplicate rows, scaled `Amount` and `Time` with separate `RobustScaler` instances (robust to the extreme outliers common in transaction amounts), and performed a stratified 80/20 train/test split to preserve the fraud ratio in both sets.

2. **Class imbalance** — applied **SMOTE** (Synthetic Minority Oversampling) to the training set only, generating synthetic fraud examples rather than duplicating existing ones. SMOTE was never applied to test data, to avoid leaking synthetic signal into evaluation.

3. **Model comparison** — trained and compared three models: Logistic Regression, Random Forest, and XGBoost. Evaluated using **PR-AUC** (Precision-Recall AUC) rather than accuracy or ROC-AUC, since PR-AUC is the more honest metric under this level of class imbalance.

   | Model | PR-AUC |
   |---|---|
   | Logistic Regression | 0.67 |
   | Random Forest | ~0.81 |
   | **XGBoost (selected)** | ~0.81 |

   Random Forest and XGBoost performed very closely — within 1-2% of each other, with the exact ranking varying slightly by run due to the small absolute number of fraud cases (~490 total). XGBoost was selected as the final model as the more widely used industry standard for tabular fraud detection.

4. **Threshold optimization** — rather than using the default 0.5 cutoff, calculated precision, recall, and F1-score across all possible thresholds and selected the F1-optimal point: **0.9249**. Also tested a lower threshold (0.3) to check the precision/recall tradeoff — it only caught 1 additional fraud case at the cost of 10x more false positives, confirming the F1-optimal threshold was the better choice.

5. **Deployment** — saved the final model and scalers, built a Gradio web interface, and deployed it live on Render.

## Results (final deployed model)

At threshold 0.9249:

```
                Predicted Legit   Predicted Fraud
Actual Legit         56,647              4
Actual Fraud            21              74
```

- **Precision: 95%** — when the model flags a transaction as fraud, it's correct 95% of the time
- **Recall: 78%** — the model catches 78% of all actual fraud
- **False positive rate: 0.007%** — only 4 legitimate transactions wrongly flagged out of 56,651

See `graphs/` for the full visual comparison (PR curves, ROC curves, confusion matrices).

## Project structure

```
├── src/
│   ├── preprocessing.py     # Load, clean, scale, split data
│   ├── train.py              # SMOTE + train/compare 3 models
│   ├── threshold.py          # Find F1-optimal decision threshold
│   ├── predict.py            # Clean prediction interface (FraudPredictor class)
│   └── generate_graphs.py    # Generate comparison graphs
├── tests/
│   └── test_predict.py       # Automated tests for the prediction pipeline
├── graphs/                   # Saved comparison charts (PR-AUC, ROC, confusion matrices)
├── fraud_detection.ipynb     # Full exploratory notebook (EDA through deployment)
├── app.py                    # Gradio web app (deployed on Render)
├── fraud_model.pkl           # Trained XGBoost model
├── scaler_amount.pkl         # Fitted scaler for transaction amount
├── scaler_time.pkl           # Fitted scaler for transaction time
├── threshold.json            # Optimized decision threshold
└── requirements.txt
```

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Retrain from scratch
python src/train.py

# Run the tests
pytest tests/test_predict.py -v

# Regenerate comparison graphs
python src/generate_graphs.py

# Run the web app
python app.py
```

## Known limitations

- **Features are not human-interpretable.** `V1`-`V28` are PCA-transformed for privacy, so the deployed app demonstrates predictions using real held-out test transactions rather than free-form manual input — there's no way to construct a meaningful new `V1`-`V28` value without access to the original bank's raw data pipeline.
- **No production monitoring or retraining pipeline yet.** The current deployment is a validated model behind a demo UI, not a full MLOps system with drift detection or automated retraining — a natural next step if this were scaled to production.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, Gradio, matplotlib/seaborn, pytest
