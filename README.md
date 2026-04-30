# 🏠 House Price Prediction - ML Pipeline

##  Project Overview
This project builds an end-to-end machine learning pipeline to predict house prices using the California Housing dataset.

The goal is to demonstrate a structured ML workflow including data processing, feature engineering, model training, evaluation, and optimization.

---

##  Tech Stack
- Python
- scikit-learn
- pandas
- numpy

---

##  Models Used

###  Ridge Regression (Baseline)
- R² Score: 0.64  
- RMSE: 0.69  

###  Random Forest (Optimized)
- Tuned using GridSearchCV  
- R² Score: 0.80  
- RMSE: 0.51  

---

##  Model Validation

Cross-validation (5-fold):

- Mean R² Score: 0.80  
- Stable performance across folds  
- Indicates good generalization (low overfitting)

---

##  Feature Engineering

Custom features created:
- Rooms per household  
- Bedrooms per household  

These features improved model performance by capturing more meaningful relationships.

---

##  Feature Importance (Random Forest)

Top predictors:
- Median Income (MedInc)  
- Average Occupancy (AveOccup)  
- Latitude & Longitude (location)  
- House Age  

---

##  Pipeline Overview

1. Load dataset  
2. Create engineered features  
3. Split data (train/test)  
4. Train models (Ridge, Random Forest)  
5. Tune Random Forest with GridSearchCV  
6. Evaluate using R² and RMSE  
7. Validate with cross-validation  
8. Save model for inference  

---

##  Project Structure

ml_project/
│
├── src/
│ ├── data.py
│ ├── features.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore



## Example Prediction
from src.predict import load_model, make_prediction
model = load_model("models/rf_model.pkl")
prediction = make_prediction(model, sample_data)

print(prediction)



