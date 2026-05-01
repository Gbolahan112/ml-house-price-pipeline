# src/train.py

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

    # 🔹 Random Forest (OPTIMIZED FOR SIZE + PERFORMANCE)
    rf_model = RandomForestRegressor(
        n_estimators=50,       # 🔥 reduced from 200
        max_depth=8,           # 🔥 reduced depth
        min_samples_split=10,  # 🔥 prevents over-complex trees
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    # 🔹 Cross-validation (for RF)
    cv_scores = cross_val_score(
        rf_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("\nCross-validation R2 scores:", cv_scores)
    print("Mean CV R2:", cv_scores.mean())

    # 🔹 Gradient Boosting Model
    gb_model = GradientBoostingRegressor(
        n_estimators=150,   # slightly reduced
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    # 🔥 Save models
    joblib.dump(ridge_pipeline, "models/ridge_model.pkl")
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(gb_model, "models/gb_model.pkl")

    return ridge_pipeline, rf_model, gb_model, X_test, y_test