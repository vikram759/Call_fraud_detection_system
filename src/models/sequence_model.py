# src/models/sequence_models.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def build_sequence_dataset(win_feats_csv="data/engineered_features.csv", seq_len=4):
    df = pd.read_csv(win_feats_csv)
    # we need sequences per msisdn ordered by window
    seqs = []
    labels = []
    group = df.groupby('msisdn')
    for ms, g in group:
        g = g.sort_values('window')
        features = g[['call_count','sms_count','data_count','unique_callees','avg_duration']].fillna(0).values
        labs = (g['label'] != 'normal').astype(int).values
        # sliding windows over sequence
        for i in range(len(features)-seq_len+1):
            seqs.append(features[i:i+seq_len])
            labels.append(1 if labs[i+seq_len-1]==1 else 0)
    X = np.array(seqs)
    y = np.array(labels)
    return X,y

def train_lstm(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    model = models.Sequential([
        layers.Input(shape=X_train.shape[1:]),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=10, batch_size=128)
    return model, X_val, y_val

if __name__ == "__main__":
    X,y = build_sequence_dataset()
    model, X_val, y_val = train_lstm(X,y)
    model.save("outputs/models/lstm_seq.h5")
