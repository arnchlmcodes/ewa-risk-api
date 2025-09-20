
import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)

N = 2000  # Number of employees
DAYS = 90  # Number of days to simulate
departments = ['sales', 'ops', 'engineering', 'hr', 'support']
job_levels = ['junior', 'mid', 'senior']

# Generate employee base info
employees = pd.DataFrame({
    'employee_id': [f'E{i:05d}' for i in range(N)],
    'department': np.random.choice(departments, N),
    'job_level': np.random.choice(job_levels, N, p=[0.5, 0.35, 0.15]),
    'salary_monthly': np.random.choice([15000, 20000, 25000, 35000, 50000], N, p=[0.2, 0.25, 0.25, 0.2, 0.1]),
    'tenure_days': np.random.exponential(scale=300, size=N).astype(int),
    'savings_balance': np.random.exponential(scale=5000, size=N),
    'other_loans': np.random.binomial(1, 0.15, N)
})

# Simulate daily transactions for each employee
records = []
for _, row in employees.iterrows():
    emp_id = row['employee_id']
    salary = row['salary_monthly']
    balance = row['savings_balance'] + salary * np.random.uniform(0.5, 1.5)
    payday = np.random.choice([1, 15])  # 1st or 15th of month
    prev_ewa_dates = []
    for day in range(DAYS):
        date = pd.Timestamp('2025-06-01') + timedelta(days=day)
        # Simulate income on payday
        deposit = salary / 2 if date.day == payday else 0
        # Simulate spending
        spend = np.random.normal(500, 200)
        spend = max(0, spend)
        # Categorize transaction
        category = np.random.choice(['necessity', 'discretionary'], p=[0.7, 0.3])
        # Update balance
        balance += deposit - spend
        # Simulate EWA usage
        ewa_used = np.random.binomial(1, 0.04 if balance > 500 else 0.15)
        ewa_amount = np.random.uniform(100, 500) if ewa_used else 0
        if ewa_used:
            prev_ewa_dates.append(date)
        balance += ewa_amount
        # Repayment (simulate next payday)
        repayment = ewa_amount if date.day == payday and ewa_used else 0
        balance -= repayment
        records.append({
            'employee_id': emp_id,
            'date': date,
            'deposit': deposit,
            'spend': spend,
            'category': category,
            'balance': balance,
            'ewa_used': ewa_used,
            'ewa_amount': ewa_amount,
            'repayment': repayment
        })

transactions = pd.DataFrame(records)

# Feature engineering
features = []
for emp_id, group in transactions.groupby('employee_id'):
    group = group.sort_values('date').reset_index(drop=True)
    # Rolling averages of spending
    group['spend_3d_avg'] = group['spend'].rolling(3).mean()
    group['spend_7d_avg'] = group['spend'].rolling(7).mean()
    group['spend_30d_avg'] = group['spend'].rolling(30).mean()
    # Velocity (spending acceleration)
    group['spend_velocity'] = group['spend'].diff().rolling(7).mean()
    # Seasonality: days to payday, month-end
    group['days_to_payday'] = group['date'].apply(lambda d: min((d.day - 1) % 15, (15 - (d.day - 1) % 15)))
    group['is_month_end'] = group['date'].dt.is_month_end.astype(int)
    # Income-to-expense ratio and volatility
    total_income = group['deposit'].sum()
    total_spend = group['spend'].sum()
    income_expense_ratio = total_income / (total_spend + 1e-3)
    spend_volatility = group['spend'].std()
    # Transaction categorization
    necessity_spend = group[group['category'] == 'necessity']['spend'].sum()
    discretionary_spend = group[group['category'] == 'discretionary']['spend'].sum()
    # Account balance trend
    balance_trend = group['balance'].iloc[-30:].mean() - group['balance'].iloc[:30].mean()
    # EWA usage patterns
    ewa_count = group['ewa_used'].sum()
    ewa_total = group['ewa_amount'].sum()
    repayment_rate = group['repayment'].sum() / (ewa_total + 1e-3)
    # Aggregate features for the last day
    last = group.iloc[-1]
    features.append({
        'employee_id': emp_id,
        'department': employees.loc[employees['employee_id'] == emp_id, 'department'].values[0],
        'job_level': employees.loc[employees['employee_id'] == emp_id, 'job_level'].values[0],
        'salary_monthly': employees.loc[employees['employee_id'] == emp_id, 'salary_monthly'].values[0],
        'tenure_days': employees.loc[employees['employee_id'] == emp_id, 'tenure_days'].values[0],
        'savings_balance': employees.loc[employees['employee_id'] == emp_id, 'savings_balance'].values[0],
        'other_loans': employees.loc[employees['employee_id'] == emp_id, 'other_loans'].values[0],
        'spend_3d_avg': last['spend_3d_avg'],
        'spend_7d_avg': last['spend_7d_avg'],
        'spend_30d_avg': last['spend_30d_avg'],
        'spend_velocity': last['spend_velocity'],
        'days_to_payday': last['days_to_payday'],
        'is_month_end': last['is_month_end'],
        'income_expense_ratio': income_expense_ratio,
        'spend_volatility': spend_volatility,
        'necessity_spend': necessity_spend,
        'discretionary_spend': discretionary_spend,
        'balance_trend': balance_trend,
        'ewa_count': ewa_count,
        'ewa_total': ewa_total,
        'repayment_rate': repayment_rate,
        'final_balance': last['balance'],
        # Target: will use EWA in next 15 days?
        'ewa_next15': int(group.iloc[-15:]['ewa_used'].sum() > 0)
    })

df_features = pd.DataFrame(features)
df_features.to_csv('synthetic_ewa.csv', index=False)
print("Synthetic data with advanced features saved to synthetic_ewa.csv")
