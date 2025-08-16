# MSIN0144 – Multimodel Threshold-Optimised Dashboard

This repository implements a machine learning pipeline and interactive dashboard for predicting student failure risk using the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset).  
The project includes data preprocessing, model training, unified prediction, and a Dash-based dashboard for educator-facing insights.

---

## 📂 Project Structure

```
repo/
├─ src/                    # Source code
│  ├─ trainers/            # Training scripts for individual models
│  ├─ prediction/          # Unified prediction pipeline
│  └─ dashboard/           # Dash app
├─ notebooks/              # Exploratory data analysis and experiments
├─ data/                   # OULAD CSV files (not included in repo)
│  └─ README.md            # Instructions for downloading dataset
├─ artifacts/              # Trained models and outputs (generated locally)
├─ main.py                 # One-click pipeline entry point
├─ requirements.txt        # Dependencies
└─ README.md               # This file
```

---

## 📊 Models Implemented
- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

Each model is trained on weekly-level student-course records (demographics + VLE activity), targeting the probability of failing the **next scheduled assessment**.

---

## 📥 Data Setup
The following OULAD CSV files are required in the `data/raw/anonymisedData/` directory:

- `assessments.csv`
- `courses.csv`
- `studentAssessment.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `vle.csv`

Download from the official OULAD website:  
👉 [https://analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset)

> ⚠️ Do not upload dataset files to GitHub. Only place them locally under `data/`.

---

## 🚀 How to Run

There are **two ways** to run the project:

### **Option 1: Full Pipeline (One-Click)**
From the project root:
```bash
python -m main
```

This will:
1. Train all four models sequentially  
2. Save model artifacts (`.pkl`) under `artifacts/`  
3. Generate unified predictions under `data/prediction/`  
4. Launch the dashboard at **http://localhost:8050/**  

⚠️ **Note**: This requires significant memory and compute resources, as all models are trained in a single run.

---

### **Option 2: Step-by-Step Execution**

#### 1. Train Models Individually
Run each of the following in sequence:
```bash
python -m src.trainers.train_logreg
python -m src.trainers.train_rf_SMOTE
python -m src.trainers.train_xgb
python -m src.trainers.train_catboost
```
Trained model artifacts (`.pkl`) will appear under `artifacts/`.

#### 2. Generate Predictions
```bash
python -m src.prediction.unified_model_predictor
```
Predictions will be stored as CSVs under `data/prediction/`.

#### 3. Launch Dashboard
```bash
python -m src.dashboard.dashboard
```
View at: [http://localhost:8050/](http://localhost:8050/)

---

## ⚙️ Requirements
Install dependencies via:
```bash
pip install -r requirements.txt
```

Recommended environment:  
- Python 3.9+  
- ≥16GB RAM (for full pipeline)  

---

