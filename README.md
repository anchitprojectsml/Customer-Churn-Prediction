# Customer-Churn-Prediction
# Customer Churn Prediction (Production Ready)

## Problem Statement
Predict telecom customer churn using Machine Learning to reduce business revenue loss.

---

##  Model Performance

| Model | Test Accuracy | CV Score | Improvement |
|-------|--------------|----------|------------|
| Raw XGBoost | 77.26% | - | - |
| Engineered XGBoost | **79.32%** | **80.11%** | +2.06% |

Train Accuracy: 83.70%  
Test Accuracy: 79.32%  
 Balanced Model (No Overfitting)

---

##  Feature Engineering
- tenure_monthly_ratio (Business logic based feature)
- Polynomial features (tenure²)
- Risk flags (high_value_short_tenure)

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

---

##  Business Impact
Estimated potential annual savings: ₹60Cr+

---

##  Files Included
- churn_model.pkl
- scaler.pkl
- features.pkl

  

---

## 📈 Future Improvements
- Deployment as web app
- Real-time prediction API

