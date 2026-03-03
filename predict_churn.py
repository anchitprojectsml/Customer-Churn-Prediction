
X_final = X_engineered
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

final_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.01, random_state=42)
final_model.fit(X_train_final, y_train_final)


joblib.dump(final_model, '/content/churn_model_final.pkl')
print("✅ Final engineered model saved!")


def predict_churn_engineered(customer_data):
    model = joblib.load('/content/churn_model_final.pkl')
    scaler_loaded = joblib.load('/content/scaler.pkl')
    features = X_final.columns.tolist()
    
    full_data = {f: 0 for f in features}
    full_data.update(customer_data)
    full_data['tenure_monthly_ratio'] = full_data['tenure'] / (full_data['MonthlyCharges'] + 1)
    full_data['high_value_short_tenure'] = 1 if full_data['MonthlyCharges'] > 85 and full_data['tenure'] < 12 else 0
    full_data['risky_contract'] = 1 if full_data['PaperlessBilling'] == 1 and full_data['Contract'] == 0 else 0
    full_data['tenure_group'] = 0 if full_data['tenure'] <= 12 else 1 if full_data['tenure'] <= 36 else 2
    
    df = pd.DataFrame([full_data], columns=features)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler_loaded.transform(df[numeric_cols])
    
    prob = model.predict_proba(df)[0][1]
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    
    return {
        'churn_probability': f"{prob:.1%}",
        'risk_level': risk,
        'recommendation': 'Retention call NOW!' if risk == 'HIGH' else 'Monitor'
    }


sample_customer = {
    'tenure': 6, 
    'MonthlyCharges': 95, 
    'TotalCharges': 500,
    'PaperlessBilling': 1,
    'Contract': 0
}


print("(Engineered Features):")
print(predict_churn_engineered(sample_customer))
