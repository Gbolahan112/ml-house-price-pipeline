# main.py
from src.data import load_data
from src.features import add_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import load_model, make_prediction
import pandas as pd


def main():
    df = load_data()
    df = add_features(df)

    ridge_model, rf_model, gb_model, X_test, y_test = train_model(df)

    # 🔹 Evaluate Ridge
    r2_ridge, rmse_ridge = evaluate_model(ridge_model, X_test, y_test)
    print("Ridge Model:")
    print(f"R2 Score: {r2_ridge}")
    print(f"RMSE: {rmse_ridge}")

    # 🔹 Evaluate Random Forest
    r2_rf, rmse_rf = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Model:")
    print(f"R2 Score: {r2_rf}")
    print(f"RMSE: {rmse_rf}")

    # 🔹 Evaluate Gradient Boosting
    r2_gb, rmse_gb = evaluate_model(gb_model, X_test, y_test)
    print("\nGradient Boosting Model:")
    print(f"R2 Score: {r2_gb}")
    print(f"RMSE: {rmse_gb}")

    # 🔹 Feature Importance (Safe)
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        feature_names = X_test.columns

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nFeature Importance (Random Forest):")
        print(feature_importance_df.head(10))

    # 🔹 Sample Prediction
    loaded_model = load_model("models/rf_model.pkl")
    sample = X_test.iloc[:1]
    prediction = make_prediction(loaded_model, sample)

    print("\nSample Prediction (RF):", prediction)


if __name__ == "__main__":
    main()