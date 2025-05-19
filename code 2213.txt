import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Assume best_model is trained as in your code; load it if saved
# best_model = joblib.load("best_model.pkl")  # Uncomment if you saved your model

# Define prediction function
def predict_house_value(longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                        population, households, median_income, ocean_proximity, model):

    # Derived features
    rooms_per_household = np.log1p(total_rooms) / households
    bedrooms_per_room = total_bedrooms / np.log1p(total_rooms)
    population_per_household = np.log1p(population) / households

    # Construct input DataFrame
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [np.log1p(total_rooms)],
        'total_bedrooms': [total_bedrooms],
        'population': [np.log1p(population)],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity],
        'rooms_per_household': [rooms_per_household],
        'bedrooms_per_room': [bedrooms_per_room],
        'population_per_household': [population_per_household]
    })

    prediction = model.predict(input_data)
    usd_value = np.expm1(prediction[0])  # Reverse log1p
    return usd_value

# Streamlit UI
st.set_page_config(page_title="California House Price Predictor", layout="centered")

st.title("🏡 California House Price Predictor")
st.markdown("Enter house features below to predict **median house value (in USD)**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.slider("Longitude", -124.0, -114.0, -119.0, 0.1)
        latitude = st.slider("Latitude", 32.0, 42.0, 36.5, 0.1)
        housing_median_age = st.slider("Housing Median Age", 1, 52, 25)
        total_rooms = st.slider("Total Rooms", 100, 40000, 2000, 100)
        total_bedrooms = st.slider("Total Bedrooms", 1, 7000, 400, 10)

    with col2:
        population = st.slider("Population", 50, 50000, 1500, 100)
        households = st.slider("Households", 50, 7000, 500, 10)
        median_income = st.slider("Median Income", 0.5, 15.0, 3.0, 0.1)
        ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])

    submitted = st.form_submit_button("Predict")

# Load or reference your trained model here (assume available in memory)
if "best_model" not in st.session_state:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    # Dummy pipeline if not actually loaded from training
    st.warning("⚠️ `best_model` not found. Load your trained model manually.")
    # Or place: st.session_state.best_model = joblib.load("best_model.pkl")

if submitted:
    if "best_model" in st.session_state:
        price = predict_house_value(longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                                    population, households, median_income, ocean_proximity,
                                    st.session_state.best_model)
        st.success(f"🏠 Predicted House Value: **${price:,.2f} USD**")
    else:
        st.error("🚫 Model not loaded. Please ensure `best_model` is available in memory or saved.")
