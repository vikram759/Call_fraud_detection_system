import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve

def train_isolation_forest(feature_csv="data/engineered_features.csv"): 
    df = pd.read_csv(feature_csv)
    X = df.drop(columns=['msisdn','window','label'])
    # Train Isolation Forest
    iso_forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso_forest.fit(X)
    # Get anomaly scores
    scores = -iso_forest.decision_function(X)  # higher scores indicate more anomalous
    
    df['iso_score']=scores
    return iso_forest, df


if __name__=="__main__":
    iso_forest, df = train_isolation_forest()
    print(df[["msisdn","label","iso_score"]].head())
    