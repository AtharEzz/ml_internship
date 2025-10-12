# Walmart Weekly Sales Forecasting  
**Internship Task 7 — Elevvo MLOps Track**  
*Predict future sales using historical data with time-aware modeling*

---

##  Objective
Build a robust forecasting model to predict weekly sales for Walmart stores using historical sales, store metadata, and macroeconomic indicators. The solution respects temporal dependencies and avoids data leakage.

---

##  Dataset
- **Source**: [Walmart Recruiting - Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- **Files Used**:
  - `train.csv`: Historical weekly sales (2010–2012)
  - `stores.csv`: Store metadata (type, size)
  - `features.csv`: External features (temperature, fuel price, CPI, unemployment, holidays)

---

##  Methodology

### 1. **Data Preprocessing**
- Merged `train`, `stores`, and `features` on `Store` and `Date`
- Resolved duplicate `IsHoliday` columns (prioritized `features.csv`)
- Sorted by `['Store', 'Dept', 'Date']` to preserve temporal order per group

### 2. **Feature Engineering**
- **Time-based**: `Year`, `Month`, `WeekOfYear`
- **Lag features**: `Weekly_Sales_Lag1`, `Lag2`, `Lag3`
- **Rolling statistics**: 3-week, 4-week, and 8-week moving averages + standard deviation
- **Categorical encoding**: `LabelEncoder` for `Store` and `Dept`

### 3. **Validation Strategy**
- **Chronological split**:
  - Train: `2010-02-19` → `2011-12-30`
  - Validation: `2012-01-06` → `2012-10-26`
- **No random shuffling** — prevents future data leakage

### 4. **Modeling**
- **Algorithm**: XGBoost Regressor
- **Key hyperparameters**:
  ```python
  {
      'n_estimators': 1000,
      'max_depth': 10,
      'learning_rate': 0.05,
      'early_stopping_rounds': 20
  }
  ```

---

##  Project Structure

```
walmart-sales-forecasting/
├── data/                              # Raw datasets
│   ├── train.csv
│   ├── stores.csv
│   └── features.csv
├── src/
│   └── walmart_forecasting.ipynb     # Full pipeline
├── reports/                           # Model artifacts & outputs
│   ├── walmart_sales_xgb_model.json
│   ├── walmart_sales_xgb_metadata.json
│   ├── feature_names.json
│   ├── le_store.pkl
│   ├── le_dept.pkl
│   └── actual_vs_predicted.png
└── README.md
```

---

##  How to Reproduce

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### 2. Load & Validate Model
```python
import xgboost as xgb
import joblib
import json

# Load model
model = xgb.XGBRegressor()
model.load_model('reports/walmart_sales_xgb_model.json')

# Load encoders & features
le_store = joblib.load('reports/le_store.pkl')
le_dept = joblib.load('reports/le_dept.pkl')

with open('reports/feature_names.json') as f:
    features = json.load(f)

# Reproduce MAE: $1,408.34 (see notebook for full inference pipeline)
```

---

##  Results

### Performance Metrics
- **Final MAE**: $1,408.34 (13.2% improvement over baseline)
- **Final RMSE**: $3,015.89
- **Model**: XGBoost + Rolling Features + Early Stopping

### Model Comparison
The iterative development process showed consistent improvement:

![MAE vs RMSE Performance Tracking](../reports/MAE%20vs%20RMSE%20performance%20tracking.png)

![MAE Improvement vs Baseline](../reports/MAE%20Improvement%20vs%20Baseline.png)

| Model | MAE ($) | RMSE ($) | Improvement vs Baseline |
|-------|---------|----------|------------------------|
| Random Forest (Baseline) | 1,624.73 | 3,597.12 | 0.0% |
| XGBoost | 1,476.93 | 3,270.45 | 9.1% |
| XGBoost + Rolling Features | 1,444.46 | 3,136.78 | 11.1% |
| **XGBoost + Rolling + Early Stopping** | **1,408.34** | **3,015.89** | **13.2%** |

![Actual vs Predicted Sales](../reports/actual_vs_predicted.png)

### Prediction Quality
The model successfully captures:
- **Seasonal trends**: Strong performance during holiday spikes
- **Weekly patterns**: Consistent tracking of regular sales cycles
- **Store-department dynamics**: Accurate predictions across different store-department combinations

---

## Key Learnings
- **Time-series integrity is non-negotiable**: Lag features must be group-wise and never break temporal order
- **XGBoost + early stopping outperforms Random Forest** for tabular forecasting
- **Rolling features add meaningful signal** beyond simple lags
- **Model persistence requires saving both model and preprocessing artifacts** (encoders, feature names, metadata)

---

##  Notes
- All preprocessing steps maintain temporal order within each Store-Department group
- Feature engineering is designed to avoid look-ahead bias
- Model artifacts are serialized for reproducible deployment

---

