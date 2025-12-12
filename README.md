# 🚗 Vehicle Fault Diagnosis – Machine Learning Model  
### 🔧 AI-Powered Automotive Problem Detection for Smart Driver Assistance Apps

![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview
This project contains a **Machine Learning model** designed for predicting **vehicle mechanical problems** based on user-typed symptom descriptions.  
It powers features such as:

- 🚗 Vehicle issue diagnosis  
- 🛠 Suggested fixes  
- ⚠ Severity analysis  
- 🤖 AI Chatbot integration  
- 📱 Mobile app backend (Android / Flutter / Web)

The model is trained on a **real automotive diagnostic dataset**, containing thousands of problem descriptions, mechanical diagnoses, ECU codes, and recommended fixes.

---

## 🔥 Key Features
- **TF-IDF + SVM / Logistic Regression Pipeline**
- **Text classification for mechanical issue prediction**
- **99+ unique vehicle problem categories**
- **Natural language symptom understanding**
- **Easy to integrate with apps, APIs & chatbots**
- Supports **custom dataset retraining**
- Exported using **joblib**

## 🧠 Model Architecture

📁 Vehicle-Fault-Diagnosis-Model

📄 README.md → This file

📄 vehicle_fault_model_pipeline.joblib

📄 model_metadata.joblib

📄 vehicle_fault_dataset_cleaned.csv

📓 notebook.ipynb → Complete Google Colab training notebook

📄 class_labels.csv → All predicted categories


---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/Tharushax1/Vehicle-Fault-Diagnosis-ML-Model.git
cd vehicle-fault-ml-model

pip install -r requirements.txt

📥 Using the Model

Load the trained model:

import joblib

model = joblib.load("vehicle_fault_model_pipeline.joblib")
meta = joblib.load("model_metadata.joblib")

🔍 Predict Fault
def predict_fault(text):
    pred = model.predict([text])[0]
    return pred

print(predict_fault("Steering wheel shakes above 80 km/h"))

🧾 Example Output
{
  "predicted_fault": "Wheel Imbalance",
  "recommended_fix": "Wheel balancing & alignment",
  "confidence": 0.93
}

📊 Dataset

This model was trained on a dataset containing:
10,000+ real vehicle issues
Symptom descriptions
Diagnosed mechanical faults
Severity levels
Repair recommendations
ECU error codes

Dataset columns include:

  Column Name	                  Description
Problem Description	          User symptom text
Diagnosis	                    Actual mechanical issue
Severity	                    Risk level
Recommended Fix	              Suggested repair
Classification	              Category of problem
ECU Codes                    On-board diagnostics

🧪 Training Notebook
A complete Google Colab notebook is included:

Data cleaning
TF-IDF vectorization
Model training & tuning
Evaluation metrics
Exporting the trained pipeline

You can retrain using your own dataset.

📈 Model Performance
Metric	  Score
Accuracy	90–94%
F1 Score	0.89–0.92
Classes	  90+ categories

Performance may vary depending on retraining dataset.

🧑‍💻 Author
Tharusha Pehesara Bandara
🚀 BSc(Hons) IT – Horizon Campus
