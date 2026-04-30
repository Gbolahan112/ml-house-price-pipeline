# 🏠 House Price Prediction (ML Pipeline Project)

## 📌 Overview
This project builds a production-style machine learning pipeline to predict house prices using the California Housing dataset.

It demonstrates end-to-end ML workflow including:
- Data loading
- Feature engineering
- Model training
- Model evaluation
- Hyperparameter tuning
- Cross-validation
- Model comparison

---

## ⚙️ Technologies Used
- Python
- scikit-learn
- pandas
- numpy
- matplotlib

---

## 🧠 Models Used

### 1. Ridge Regression
- Linear baseline model
- R² Score: ~0.64

### 2. Random Forest (Optimized)
- Non-linear ensemble model
- R² Score: ~0.80
- RMSE: ~0.50

---

## 🔍 Key Features

- Feature engineering (ratios like rooms per household)
- Model comparison (linear vs ensemble)
- Hyperparameter tuning using GridSearchCV
- Cross-validation for model reliability
- Feature importance analysis

---

## 📊 Results

| Model | R² Score | RMSE |
|------|--------|------|
| Ridge | ~0.64 | ~0.68 |
| Random Forest | ~0.80 | ~0.50 |

Cross-validation R² ≈ 0.80 → model is stable and generalizes well.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py