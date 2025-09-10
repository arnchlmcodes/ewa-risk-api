from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_smoke():
    payload = {
        "employee_id": "E00001",
        "salary_monthly": 30000,
        "tenure_days": 240,
        "num_withdrawals_last_30d": 2,
        "num_withdrawals_last_90d": 3,
        "avg_withdraw_amount": 5000,
        "avg_withdraw_pct_of_salary": 0.1667,
        "last_withdraw_days_ago": 4,
        "savings_balance": 1200,
        "other_loans": 1,
        "department": "support",
        "job_level": "junior"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "risk_score" in r.json()
    assert 0.0 <= r.json()['risk_score'] <= 1.0
