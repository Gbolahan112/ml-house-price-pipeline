import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    return {
        "Gradient Boosting (Best)": joblib.load("models/gb_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "Ridge Regression": joblib.load("models/ridge_model.pkl")
    }

models = load_models()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit 🚀")

# -------------------------------
# UI - Header
# -------------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Compare models and predict house prices")

st.markdown("---")

# -------------------------------
# Layout (Columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Features")

    MedInc = st.number_input("Median Income", min_value=0.0, value=5.0)
    HouseAge = st.number_input("House Age", min_value=0, value=20)
    AveRooms = st.number_input("Average Rooms", min_value=0.1, value=5.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.1, value=1.0)
    Population = st.number_input("Population", min_value=1, value=300)
    AveOccup = st.number_input("Average Occupancy", min_value=0.1, value=3.0)
    Latitude = st.number_input("Latitude", value=34.0)
    Longitude = st.number_input("Longitude", value=-118.0)

with col2:
    st.subheader("📊 Prediction Output")

    if st.button("Predict Price"):

        data = pd.DataFrame([{
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        }])

        # Feature engineering
        data["Rooms_per_Household"] = data["AveRooms"] / data["AveOccup"]
        data["Bedrooms_per_Household"] = data["AveBedrms"] / data["AveOccup"]
        data["Income_per_Person"] = data["MedInc"] / data["Population"]

        data = data[
            [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude",
                "Rooms_per_Household", "Bedrooms_per_Household",
                "Income_per_Person"
            ]
        ]

        prediction = model.predict(data)
        price = prediction[0] * 100000

        st.success(f"💰 Estimated Price: ${price:,.0f}")
        st.write(f"Model used: **{selected_model_name}**")

# -------------------------------
# Feature Importance (RF only)
# -------------------------------
if selected_model_name == "Random Forest":

    st.markdown("---")
    st.subheader("📈 Feature Importance (Random Forest)")

    feature_names = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
        "Rooms_per_Household", "Bedrooms_per_Household",
        "Income_per_Person"
    ]

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Gbolahan | ML Pipeline Project 🚀")