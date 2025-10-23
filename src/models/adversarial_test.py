import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

CSV_PATH = "data/engineered_features.csv"
MODEL_PATH = "outputs/models/lstm_seq.h5"
SEQ_LEN = 6
EPSILON = 0.05  # perturbation strength

def build_sequence_dataset(df, seq_len=4):
    seqs, labels = [], []
    group = df.groupby("msisdn")
    for ms, g in group:
        g = g.sort_values("window")
        features = g[
            ['call_count','sms_count','data_count','unique_callees','avg_duration']
        ].fillna(0).values
        labs = (g['label'] != 'normal').astype(int).values
        for i in range(len(features) - seq_len + 1):
            seqs.append(features[i:i+seq_len])
            labels.append(1 if labs[i+seq_len-1]==1 else 0)
    return np.array(seqs), np.array(labels)

# === Step 1: Load data and model ===
df = pd.read_csv(CSV_PATH)
X, y = build_sequence_dataset(df)
model = load_model(MODEL_PATH)

print(f"Original samples: {X.shape}, labels: {y.shape}")

# === Step 2: Create adversarial examples ===
# Add small perturbations within normalized range [-1, 1]
noise = np.random.uniform(-EPSILON, EPSILON, X.shape)
X_adv = np.clip(X + noise, -1, 1)  # keep within normalized limits

# === Step 3: Evaluate on original and adversarial data ===
orig_preds = model.predict(X)
adv_preds = model.predict(X_adv)

orig_acc = np.mean((orig_preds > 0.5).astype(int).flatten() == y)
adv_acc = np.mean((adv_preds > 0.5).astype(int).flatten() == y)

print(f"\n🔹 Original Accuracy: {orig_acc:.4f}")
print(f"🔸 Adversarial Accuracy: {adv_acc:.4f}")
print(f"Accuracy Drop: {(orig_acc - adv_acc)*100:.2f}%")

# === Step 4: Save sample adversarial data ===v 
with open("outputs/adversarial_results.txt", "w", encoding='utf-8') as f:
    f.write(f"\n🔹 Original Accuracy: {orig_acc:.4f}\n")
    f.write(f"🔸 Adversarial Accuracy: {adv_acc:.4f}\n")
    f.write(f"Accuracy Drop: {(orig_acc - adv_acc)*100:.2f}%\n")

adv_flat = X_adv.reshape(-1, X.shape[-1])
adv_df = pd.DataFrame(adv_flat, columns=['call_count','sms_count','data_count','unique_callees','avg_duration'])
adv_df.to_csv("outputs/adversarial_features.csv", index=False)
print("\n✅ Adversarial data saved to outputs/adversarial_features.csv")
