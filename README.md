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
```
credit-score-prediction-pro/
├── app.py                # Streamlit frontend
├── api.py                # FastAPI backend
├── predict.py            # Scoring & explainability logic
├── train_model.py        # Custom logistic regression training script
├── data_process.py       # Data cleaning, encoding, feature engineering
├── data_pipeline.py      # ETL wrapper
├── models/
│   ├── logistic_model.json
│   └── scaler.pkl
├── data/
│   └── accepted_2007_to_2018Q4.csv   # (Not uploaded - Download below)
```

## Installation
git clone https://github.com/Taytayket/CreditScorePredictionProject.git

cd CreditScorePredictionProject

pip install:
fastapi, uvicorn, streamlit, pandas, numpy, scikit-learn, joblib, matplotlib


## Usage
#Start FastAPI backend: 

uvicorn api:app --reload

#Open Streamlit frontend: 

streamlit run app.py

You can now access:

	•	API docs: http://127.0.0.1:8000/docs
 
	•	Streamlit UI: http://localhost:8501
 

## Dataset
This project uses the Lending Club Loan Data available on Kaggle. It contains detailed records of loan applications and repayment outcomes from 2007 to 2018, published by LendingClub.

The dataset includes over 200 features, covering:
	•	Loan details: loan_amnt, term, int_rate, installment, etc.
	•	Borrower attributes: annual_inc, emp_length, home_ownership, purpose, etc.
	•	Credit history: earliest_cr_line, open_acc, total_acc, dti, etc.
	•	Loan outcome: loan_status (e.g., Fully Paid, Charged Off, Default)

Processing:
	•	The target variable loan_status is converted into a binary classification:
	•	Fully Paid → 0 (Non-default)
	•	Charged Off or Default → 1 (Default)
	•	Only a subset of important features (around 15–20) is selected for modeling
	•	Preprocessing includes missing value handling, feature engineering, label encoding, and log transformations

⚠️Note:
Due to the large size of the original dataset (~1.68 GB uncompressed), it is not included in this GitHub repo. However, all data processing logic is provided in the codebase.


You can download the dataset directly from Kaggle:
🔗 https://www.kaggle.com/datasets/wordsforthewise/lending-club/data

Once downloaded, place the file accepted_2007_to_2018Q4.csv inside the data/ folder.
