# src/predict.py

import joblib


def load_model(model_path="models/model.pkl"):
    """
    Load a trained model from disk
    """
    model = joblib.load(model_path)
    return model


def make_prediction(model, new_data):
    """
    Make predictions using the loaded model
    """
    prediction = model.predict(new_data)
    return prediction