import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

df = pd.read_csv("data/synthetic_ewa.csv")
y = df.pop('label_request_next_cycle')
X = df.drop(['employee_id'], axis=1)

num_cols = ['salary_monthly','tenure_days','avg_withdraw_amount','avg_withdraw_pct_of_salary','last_withdraw_days_ago','savings_balance']
cat_cols = ['department','job_level']

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])

preproc = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

pipe = Pipeline([('preproc', preproc), ('clf', XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric='logloss'))])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
preds = pipe.predict_proba(X_test)[:,1]
print("ROC AUC:", roc_auc_score(y_test, preds))
joblib.dump(pipe, "model/ewa_risk_pipeline.joblib")
