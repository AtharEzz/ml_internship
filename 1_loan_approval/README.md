# Loan Approval Prediction

Predicting loan default risk on an imbalanced banking dataset, as part of the Elevvo Pathways Machine Learning Internship.

## Problem

Banking datasets for loan approval are typically heavily imbalanced (far more approved than rejected applications), which makes standard classification models biased toward the majority class and unreliable at catching genuinely risky loans.

## Approach

Several approaches were compared to handle the class imbalance, including ensemble methods (Random Forest, XGBoost) and synthetic oversampling (SMOTE). The final model selected was **Logistic Regression with a tuned decision threshold (0.45 instead of the default 0.5)**, chosen over the alternatives for three reasons:

1. **Performance**: outperformed ensemble baselines (Random Forest F1: 0.71–0.72, XGBoost F1: 0.68–0.74)
2. **Interpretability**: fully interpretable coefficients, important for banking compliance and explainability requirements
3. **No synthetic data risk**: avoided the data-quality risks that come with synthetic oversampling (e.g., SMOTE)

## Key Engineering Steps

- Log-transformed skewed features (`Income`, `LoanAmount`)
- Standardized all numerical features
- Tuned the decision threshold to 0.45 to optimize F1-score for the minority (rejected) class

## Results

| Metric | Score |
|---|---|
| F1-score (Rejected class) | 0.77 |
| Recall (Rejected class) | 0.71 — catches 71% of risky loans |
| Precision (Rejected class) | 0.84 — only ~16% false rejections |

By adjusting the decision threshold from 0.5 to 0.45, F1-score for the minority class improved by 6 percentage points, outperforming ensemble methods while maintaining the interpretability banking stakeholders typically require.

## Folder Structure

- `data/` — dataset files
- `notebooks/` — analysis and modeling notebook(s)
- `models/` — saved model artifacts
- `reports/` — supporting output/reports
