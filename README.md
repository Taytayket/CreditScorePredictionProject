# CreditScorePredictionProject
A full-stack credit risk scoring application using a custom-built Logistic Regression model (NumPy), FastAPI backend, and Streamlit frontend. Inspired by LendingClub's real-world loan data.

---

## Features

- Custom Logistic Regression (built from scratch using NumPy)
- Full ETL pipeline: cleaning, encoding, feature engineering
- RESTful API using FastAPI
- Interactive frontend using Streamlit
- Visual explanation of score contributions
- Batch inference for large datasets (optional)

---

## Credit Scoring Logic

The model predicts the **default probability** of a loan, and maps that to:

- A **credit score** (0–100)
- A **risk level**: Low / Medium / High
- An **explanation chart** (top 8 features contributing to the score)

---

## Project Structure
credit-score-prediction-pro/
├── app.py                 # Streamlit frontend
├── api.py                 # FastAPI backend
├── predict.py             # Scoring & explainability logic
├── train_model.py         # Custom logistic regression training script
├── data_process.py        # Data cleaning, encoding, feature engineering
├── data_pipeline.py       # ETL wrapper
├── models/
│   ├── logistic_model.json
│   └── scaler.pkl
├── data/
│   └── accepted_2007_to_2018Q4.csv   # (Not uploaded - Download at: https://www.kaggle.com/datasets/wordsforthewise/lending-club/data)
