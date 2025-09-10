from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/ewa_risk_pipeline.joblib")

class Employee(BaseModel):
    employee_id: str
    salary_monthly: float
    tenure_days: int
    num_withdrawals_last_30d: int
    num_withdrawals_last_90d: int
    avg_withdraw_amount: float
    avg_withdraw_pct_of_salary: float
    last_withdraw_days_ago: int
    savings_balance: float
    other_loans: int
    department: str
    job_level: str

@app.post("/predict")
def predict(emp: Employee):
    row = pd.DataFrame([emp.dict()])
    prob = model.predict_proba(row)[:,1][0]
    label = "high" if prob >= 0.6 else ("medium" if prob>=0.3 else "low")
    return {"employee_id": emp.employee_id, "risk_score": float(prob), "risk_label": label}
