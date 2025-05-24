from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
from data_pipeline import ppl
from predict import predict_credit_score


def convert_to_json_safe(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_safe(v) for v in obj]  
    elif isinstance(obj, (np.generic, np.number)):
        return obj.item()  
    else:
        return obj

app = FastAPI(title="Credit Score API", description="Logistic Regression-based Credit Risk Scoring")

class LoanApplication(BaseModel):
    loan_amnt: float
    term: Literal[
                    "12 months", "24 months", "36 months", "48 months", "60 months", "72 months"]
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    purpose: str
    dti: float
    open_acc: int
    total_acc: int
    earliest_cr_line: str 

@app.post("/predict")
def predict(application: LoanApplication):
    input_dict = application.dict()
    result = predict_credit_score(input_dict)


    print("Raw result to return:")
    print(result)
    print("Types in explanation:")
    for f, c in result["explanation"]:
        print("  →", type(f), type(c))
    return convert_to_json_safe(result)