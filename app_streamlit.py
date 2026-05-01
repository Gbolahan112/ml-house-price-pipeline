import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    # 🔥 Load BEST model (Gradient Boosting)
    return joblib.load("models/gb_model.pkl")

model = load_model()

# -------------------------------
# UI - Header
# -------------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")
st.write("Fill in the details below and click **Predict Price**")

st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("About")
st.sidebar.write(
    "This app predicts house prices using a Gradient Boosting model. "
    "Feature engineering is applied automatically during prediction."
)

# -------------------------------
# Inputs
# -------------------------------
MedInc = st.number_input("Median Income", min_value=0.0, value=5.0)
HouseAge = st.number_input("House Age", min_value=0, value=20)
AveRooms = st.number_input("Average Rooms", min_value=0.1, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.1, value=1.0)
Population = st.number_input("Population", min_value=1, value=300)
AveOccup = st.number_input("Average Occupancy", min_value=0.1, value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    try:
        # Create DataFrame
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

        # 🔥 Feature Engineering
        data["Rooms_per_Household"] = data["AveRooms"] / data["AveOccup"]
        data["Bedrooms_per_Household"] = data["AveBedrms"] / data["AveOccup"]
        data["Income_per_Person"] = data["MedInc"] / data["Population"]

        # Ensure correct column order
        data = data[
            [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude",
                "Rooms_per_Household", "Bedrooms_per_Household",
                "Income_per_Person"
            ]
        ]

        # Predict
        prediction = model.predict(data)

        # Format output (clean currency)
        price = prediction[0] * 100000

        st.success(f"💰 Estimated House Price: ${price:,.0f}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit | ML Pipeline Project by You 🚀")