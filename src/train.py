# src/train.py

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def train_model(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔹 Ridge Model
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    ridge_pipeline.fit(X_train, y_train)

    # 🔥 NEW: Grid Search for Random Forest
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
        

    # 🔥 Get best model
    rf_model = grid.best_estimator_

    # 🔥 ADD THIS BLOCK HERE
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(
        rf_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("\nCross-validation R2 scores:", cv_scores)
    print("Mean CV R2:", cv_scores.mean())

    # 🔥 NEW: Best model selected automatically
    rf_model = grid.best_estimator_

    # 🔥 Save both models
    joblib.dump(ridge_pipeline, "models/ridge_model.pkl")
    joblib.dump(rf_model, "models/rf_model.pkl")

    return ridge_pipeline, rf_model, X_test, y_test