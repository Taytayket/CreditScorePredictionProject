import numpy as np
import pandas as pd
import json
import joblib
from data_pipeline import ppl
from data_process import encode_categoricals, engineer_features
import sys

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)

def load_model(json_path="models/logistic_model.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    model = LogisticRegression()
    model.weights = np.array(data["weights"])
    model.bias = data["bias"]
    return model

def load_scaler(scaler_path="models/scaler.pkl"):
    return joblib.load(scaler_path)

def preprocess_input(input_dict):
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Reuse preprocessing steps
    df = engineer_features(df)
    df = encode_categoricals(df)

    # Drop any extra columns (like loan_status if present)
    df = df.drop(columns=["loan_status"], errors='ignore')

    return df

def map_risk_level(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

def calculate_credit_score(prob):
    return int(round((1 - prob) * 100))

def predict_credit_score(input_dict):
    model = load_model()
    scaler = load_scaler()

    # Preprocess
    df = preprocess_input(input_dict)
    X_scaled = scaler.transform(df)

    # Predict
    proba = model.predict_proba(X_scaled)[0]
    score = calculate_credit_score(proba)
    risk = map_risk_level(proba)

    # Feature contributions
    contributions = (X_scaled[0] * model.weights).tolist()
    feature_names = df.columns.tolist()
    explanation = sorted(
        zip(feature_names, contributions), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:8]  # Top 8 features


    return {
        "predicted_probability": float(round(proba, 4)),
        "credit_score": int(score),
        "risk_level": str(risk),
        "explanation": [
            {"feature": str(f), "contribution": float(c)} for f, c in explanation
        ]

    }

# Example usage

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        df = ppl("data/accepted_2007_to_2018Q4.csv")  
        results = []

        for i, row in df.iterrows():
            result = predict_credit_score(row.to_dict())
            result["row_id"] = i + 1
            results.append(result)

        df_result = pd.DataFrame(results)
        df_result.to_csv("data/prediction_result.csv", index=False)
        print("Batch prediction completed and saved to data/prediction_result.csv")
    else:
        sample_input = {
            "loan_amnt": 1000000,
            "term": "36 months",
            "int_rate": 13.56,
            "installment": 340.5,
            "grade": "C",
            "sub_grade": "C3",
            "emp_length": "3 years",
            "home_ownership": "RENT",
            "annual_inc": 55000,
            "purpose": "credit_card",
            "dti": 18.5,
            "open_acc": 9,
            "total_acc": 25,
            "earliest_cr_line": "1985-07-01"
        }

        result = predict_credit_score(sample_input)
        print("Prediction Result:")
        print(result)