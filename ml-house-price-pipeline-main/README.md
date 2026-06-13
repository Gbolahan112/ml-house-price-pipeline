# 🏠 House Price Prediction - ML Pipeline

## 📌 Project Overview
This project builds an end-to-end machine learning pipeline to predict house prices using the California Housing dataset.

It demonstrates a complete ML workflow including:
- Data processing
- Feature engineering
- Model training
- Evaluation
- Deployment (API + Web App)

---

## ⚙️ Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- FastAPI
- Streamlit
- joblib

---

## 🧠 Models Used

### 🔹 Ridge Regression (Baseline)
- R² Score: ~0.64  
- RMSE: ~0.68  

### 🔹 Random Forest
- R² Score: ~0.74  
- RMSE: ~0.58  

### 🔹 Gradient Boosting (Final Model)
- R² Score: ~0.79  
- RMSE: ~0.52  

👉 Gradient Boosting was selected as the final model based on performance.

---

## 📊 Model Validation

- 5-fold cross-validation  
- Mean R² Score: ~0.75  
- Stable performance across folds  
- Indicates good generalization  

---

## 🔍 Feature Engineering

Custom features created:

- Rooms per household  
- Bedrooms per household  
- Income per person  

These features improved model performance by capturing more meaningful relationships in the data.

---

## 📈 Feature Importance (Random Forest)

Top predictors:

- Median Income (MedInc)  
- Average Occupancy (AveOccup)  
- Latitude & Longitude (location)  
- House Age  

---

## 🧠 Pipeline Overview

1. Load dataset  
2. Apply feature engineering  
3. Split data into train/test  
4. Train multiple models  
5. Evaluate using R² and RMSE  
6. Perform cross-validation  
7. Select best model  
8. Save model for inference  

---

## 🚀 Project Structure

ml_project/
│
├── src/
│ ├── data.py
│ ├── features.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│
├── api/
│ └── app.py
│
├── models/
│
├── app_streamlit.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore


---

## 🌐 Deployment

This project includes:

- ✅ FastAPI backend for predictions  
- ✅ Streamlit web app for user interaction  

👉 The app allows users to input house features and get real-time predictions.

---

## ▶️ How to Run Locally

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python main.py

# Run Streamlit app
streamlit run app_streamlit.py

Example Prediction

from src.predict import load_model, make_prediction

model = load_model("models/gb_model.pkl")
prediction = make_prediction(model, sample_data)

print(prediction)

Key Learnings
Building structured ML pipelines
Feature engineering impact on performance
Model comparison (linear vs ensemble)
Cross-validation for reliability
Trade-off between model size and performance
Deploying ML models with FastAPI and Streamlit
🚀 Future Improvements
Add more advanced models (XGBoost, LightGBM)
Deploy API to cloud (AWS / Render)
Improve UI/UX of Streamlit app
Use real-world production datasets
👨‍💻 Author

Built by Abubakr Agbolahan
Machine Learning & Data Analytics Project
