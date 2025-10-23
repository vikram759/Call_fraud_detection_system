# src/models/supervised.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import numpy as np

def train_lightgbm(feature_csv="data/engineered_features.csv", label_col='label'):
    df = pd.read_csv(feature_csv)
    # Simplify labels to binary: fraud vs normal (for baseline)
    df['is_fraud'] = df[label_col].apply(lambda x: 0 if x=='normal' else 1)
    X = df.drop(columns=['msisdn','window','label','is_fraud'])
    y = df['is_fraud']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val)
    params = {
        'objective':'binary',
        'metric':'auc',
        'learning_rate':0.05,
        'num_leaves':31,
        'verbosity': -1,
        'seed':42
    }
    bst = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=200, callbacks=[lgb.early_stopping(stopping_rounds=20)])
    preds = bst.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    print("Validation AUC:", auc)
    joblib.dump(bst, "outputs/models/lgbm_baseline.pkl")
    return bst, X_val, y_val, preds

if __name__ == "__main__":
    bst, X_val, y_val, preds = train_lightgbm()
