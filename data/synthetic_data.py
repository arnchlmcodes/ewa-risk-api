import numpy as np, pandas as pd
np.random.seed(42)

N = 20000
departments = ['sales','ops','engineering','hr','support']
job_levels = ['junior','mid','senior']

def gen_row(i):
    salary = np.random.choice([15000,20000,25000,35000,50000], p=[0.2,0.25,0.25,0.2,0.1])
    tenure = np.random.exponential(scale=300)
    avg_withdraw_pct = np.clip(np.random.beta(1.5,6)*0.6 + (0.1 if np.random.rand()<0.05 else 0), 0, 1)
    avg_withdraw_amount = avg_withdraw_pct * salary
    num_30 = np.random.poisson(0.3 + 2*avg_withdraw_pct)
    num_90 = num_30 + np.random.poisson(0.5)
    last_withdraw = np.random.randint(1,60) if num_30>0 else np.random.randint(61,180)
    savings = np.random.exponential(scale=5000)
    other_loans = np.random.binomial(1, 0.1 if salary>30000 else 0.25)
    dept = np.random.choice(departments)
    level = np.random.choice(job_levels, p=[0.5,0.35,0.15])
    p = 0.05 + 0.5*avg_withdraw_pct + (0.2 if last_withdraw<7 else 0) - 0.00002*savings + 0.15*other_loans
    label = np.random.binomial(1, np.clip(p, 0, 0.95))
    return {
        'employee_id': f'E{i:05d}','department':dept,'job_level':level,
        'salary_monthly':salary,'tenure_days':int(tenure),
        'num_withdrawals_last_30d':int(num_30),'num_withdrawals_last_90d':int(num_90),
        'avg_withdraw_amount':float(avg_withdraw_amount),'avg_withdraw_pct_of_salary':float(avg_withdraw_pct),
        'last_withdraw_days_ago':int(last_withdraw),'savings_balance':float(savings),'other_loans':int(other_loans),
        'label_request_next_cycle':int(label)
    }

df = pd.DataFrame([gen_row(i) for i in range(N)])
df.to_csv('synthetic_ewa.csv', index=False)
