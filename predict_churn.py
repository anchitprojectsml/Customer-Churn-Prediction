
import joblib
import pandas as pd
import numpy as np

def predict_churn(customer_data):
    model = joblib.load('/content/churn_model.pkl')
    scaler_loaded = joblib.load('/content/scaler.pkl')
    features = joblib.load('/content/features.pkl')
    
    # Complete customer data (missing = 0)
    full_data = {f: 0 for f in features}
    full_data.update(customer_data)
    
    df = pd.DataFrame([full_data])
    
    # Scale numeric
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler_loaded.transform(df[numeric_cols])
    
    prob = model.predict_proba(df)[0][1]
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    
    return {
        'churn_probability': f"{prob:.1%}",
        'risk_level': risk,
        'recommendation': 'Retention call!' if risk == 'HIGH' else 'Monitor'
    }


sample_customer = {
    'tenure': 6,           # Short tenure
    'MonthlyCharges': 95,  # High bill  
    'TotalCharges': 500,   # Low total (new customer)
    'PaperlessBilling': 1, # Risky!
    'Contract': 0          # Month-to-month!
}


print(predict_churn(sample_customer))
