import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        "models",
        "gb_model.pkl"
    )
    return joblib.load(model_path)


model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    "This app predicts house prices using a trained Gradient Boosting model."
)

# -------------------------------
# UI - Header
# -------------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")

st.markdown("---")

# -------------------------------
# Layout (Columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Features")

    MedInc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=5.0
    )

    HouseAge = st.number_input(
        "House Age",
        min_value=0,
        value=20
    )

    AveRooms = st.number_input(
        "Average Rooms",
        min_value=0.1,
        value=5.0
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.1,
        value=1.0
    )

    Population = st.number_input(
        "Population",
        min_value=1,
        value=300
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        min_value=0.1,
        value=3.0
    )

    Latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    Longitude = st.number_input(
        "Longitude",
        value=-118.0
    )

with col2:
    st.subheader("📊 Prediction Output")

    if st.button("Predict Price"):

        try:
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

            # Feature Engineering
            data["Rooms_per_Household"] = (
                data["AveRooms"] /
                data["AveOccup"]
            )

            data["Bedrooms_per_Household"] = (
                data["AveBedrms"] /
                data["AveOccup"]
            )

            data["Income_per_Person"] = (
                data["MedInc"] /
                data["Population"]
            )

            # Arrange Features
            data = data[
                [
                    "MedInc",
                    "HouseAge",
                    "AveRooms",
                    "AveBedrms",
                    "Population",
                    "AveOccup",
                    "Latitude",
                    "Longitude",
                    "Rooms_per_Household",
                    "Bedrooms_per_Household",
                    "Income_per_Person"
                ]
            ]

            # Prediction
            prediction = model.predict(data)
            price = prediction[0] * 100000

            st.success(
                f"💰 Estimated Price: ${price:,.0f}"
            )

            st.write(
                "Model used: Gradient Boosting"
            )

        except Exception as e:
            st.error(
                f"Error: {str(e)}"
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Built by Gbolahan | ML Pipeline Project 🚀"
)