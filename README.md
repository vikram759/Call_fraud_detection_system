# 📞 Call Fraud Detection

**Call Fraud Detection** is a full end-to-end machine learning project designed to identify fraudulent call activity using a combination of **synthetic data generation**, **feature engineering**, **supervised learning (LightGBM)**, and **sequential modeling (LSTM)**.
The project also includes **adversarial testing** to evaluate the robustness of the trained models.

---

## 🧠 Overview

Fraudulent calls can cause major financial and network security issues in telecom systems.
This project simulates real-world scenarios by generating synthetic call data and building intelligent models that can identify abnormal or suspicious patterns.

The pipeline consists of:
1. Generating synthetic call data.
2. Creating temporal and graph-based features.
3. Training supervised and sequential ML models.
4. Evaluating adversarial robustness.

---

## 🗂️ Project Structure
```
├── data/
│ ├── raw_data.csv # Synthetic raw call data
│ ├── engineered_features.csv # Feature engineered dataset
│
├── outputs/
│ ├── models/ # Saved trained models
│ ├── adversarial_features.csv # Generated adversarial dataset
│ └── adversarial_results.txt # Accuracy and logs of adversarial tests
│
├── src/
│ ├── models/
│ │ ├── adversarial_test.py # Adversarial test execution
│ │ ├── sequence_model.py # LSTM sequence model
│ │ ├── supervised.py # LightGBM supervised model
│ │ ├── unsupervised.py # (Optional) clustering/autoencoder models
│ │ └── features.py # Feature engineering logic
│ └── synth_data.py # Synthetic data generation script
│
├── venv/ # Virtual environment
├── requirements.txt # Dependencies
└── .gitignore

```

## ⚙️ Project Workflow

### **1. Synthetic Data Generation**
- Script: `src/synth_data.py`
- Generates realistic **telecom call logs** for multiple MSISDNs (mobile subscribers).
- Includes timestamps, call durations, call frequencies, and partner connections.
- Output: `raw_data.csv`

---

### **2. Feature Engineering**
- Created **window-based temporal features** for each MSISDN.
- Extracted statistics like:
  - Number of calls per time window
  - Unique numbers called
  - Duration distributions
  - Connection graph metrics
- Generated visual graphs showing multiple user connections.
- Output: `engineered_features.csv`

---

### **3. Supervised Model (LightGBM)**
- Script: `src/models/supervised.py`
- Trains a **LightGBM classifier** on the engineered features.
- Model learns static behavioral patterns of fraudulent users.
---

### **4. Sequential Model (LSTM)**
- Script: `src/models/sequence_model.py`
- Uses **LSTM (Long Short-Term Memory)** architecture to capture **temporal dependencies** in call behavior.
- Trained on `engineered_features.csv` with timestamp-ordered windows.
- Achieved **94.2% accuracy**, demonstrating strong sequence learning ability.

---

### **5. Adversarial Testing**
- Script: `src/models/adversarial_test.py`
- Generates **adversarial feature perturbations** to test model robustness.
- Evaluated the LSTM model trained in `sequence_model.py`.
- The model maintained **93% accuracy** even on adversarially perturbed data.
- Outputs stored in:
  - `outputs/adversarial_features.csv`
  - `outputs/adversarial_results.txt`

---

## 🧩 Key Highlights

✅ **Synthetic data generation pipeline** simulating realistic telecom usage.  
✅ **Feature engineering** for both static and sequential data.  
✅ **Two-stage model training** — LightGBM (supervised) + LSTM (sequential).  
✅ **Adversarial robustness testing** to evaluate real-world resilience.  
✅ **Modular, reproducible codebase** following clean ML pipeline design.

---

## 📊 Results Summary

| Model Type          | Dataset Used               | Accuracy |
|----------------------|-----------------------------|-----------|
| LSTM (Sequential)     | Timestamp-based Features    | **94.2%** |
| LSTM (Adversarial)    | Adversarially Perturbed Data| **93.0%** |

---

## 🚀 How to Run

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/call-fraud-detection.git
cd call-fraud-detection
```
### **2. Create Virtual Environment**
```bash
python -m venv venv
```
### **3. Activate Environment**
#### Windows
```bash
venv\Scripts\activate

```
#### Linux/Mac

```bash
source venv/bin/activate
```
### **Install Dependencies**
```bash
pip install -r requirements.txt

```
### **5. Run the Pipeline**
#### Step 1 - Generate Synthetic data
```bash
python -m venv venv
```
#### Step 2 — Train Supervised Model (LightGBM)
```bash
python src/models/supervised.py

```
#### Step 3 — Train LSTM Sequence Model
```bash
python src/models/sequence_model.py

```
#### Step 4 — Perform Adversarial Testing
```bash
python src/models/adversarial_test.py

```

## Future Improvements
-Integrate Graph Neural Networks (GNNs) for relational fraud detection.

-Extend synthetic dataset with real-world call traces for transfer learning.

-Develop a live dashboard for fraud monitoring and alerting.

-Implement unsupervised anomaly detection for zero-day frauds.

---

## 🧑‍💻 Author
### **Vikramjit Singh**
#### 📧 ustat0803@gmail.com
#### 🌐 https://www.linkedin.com/in/vikramjit-singh-70b920296/
