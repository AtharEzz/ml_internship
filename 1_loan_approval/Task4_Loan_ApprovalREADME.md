## Final Model: Logistic Regression with Threshold Tuning

**Performance:**
- F1-Score (Rejected): 0.77
- Recall (Rejected): 0.71 (catches 71% of risky loans)
- Precision (Rejected): 0.84 (only 16% false rejections)

**Why This Model?**
1. Outperformed ensemble methods (RF F1=0.71-0.72, XGBoost F1=0.68-0.74)
2. Fully interpretable — critical for banking compliance
3. No synthetic data risk (unlike SMOTE)
4. Lightweight and fast to deploy

**Key Engineering Steps:**
1. Log-transformed skewed features (`Income`, `LoanAmount`)
2. Standardized all numerical features
3. Tuned decision threshold to 0.45 to optimize F1 for minority class

For imbalanced classification, by adjusting the decision threshold from 0.5 to 0.45, I improved F1-score for the minority class by 6 percentage points — outperforming ensemble methods — while maintaining high interpretability for business stakeholders.