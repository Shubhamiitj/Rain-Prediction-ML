import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model_dt.pkl")
scaler = joblib.load("scaler.pkl")

city_mapping = {
    "Austin": 0,
    "Charlotte": 1,
    "Chicago": 2,
    "Columbus": 3,
    "Dallas": 4,
    "Denver": 5,
    "Fort Worth": 6,
    "Houston": 7,
    "Indianapolis": 8,
    "Jacksonville": 9,
    "Los Angeles": 10,
    "New York": 11,
    "Philadelphia": 12,
    "Phoenix": 13,
    "San Antonio": 14,
    "San Diego": 15,
    "San Francisco": 16,
    "San Jose": 17,
    "Seattle": 18,
    "Washington D.C.": 19
}

st.title("🌧 Rain Prediction")

city = st.selectbox("Location", list(city_mapping.keys()))

temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=60.0)
wind_speed = st.number_input("Wind Speed", value=10.0)
precipitation = st.number_input("Precipitation", value=0.0)
cloud_cover = st.number_input("Cloud Cover (%)", value=50.0)
pressure = st.number_input("Pressure (hPa)", value=1013.0)

if st.button("Predict"):

    location = city_mapping[city]

    # User enters REAL values here
    numeric_df = pd.DataFrame(
        [[temperature, humidity, wind_speed,
          precipitation, cloud_cover, pressure]],
        columns=[
            "Temperature",
            "Humidity",
            "Wind Speed",
            "Precipitation",
            "Cloud Cover",
            "Pressure"
        ]
    )

    # Scaling happens automatically in background
    scaled = scaler.transform(numeric_df)

    final_df = pd.DataFrame(
        [[
            location,
            scaled[0][0],
            scaled[0][1],
            scaled[0][2],
            scaled[0][3],
            scaled[0][4],
            scaled[0][5]
        ]],
        columns=[
            "Location",
            "Temperature",
            "Humidity",
            "Wind Speed",
            "Precipitation",
            "Cloud Cover",
            "Pressure"
        ]
    )

    prediction = model.predict(final_df)

    if prediction[0] == 1:
        st.success("🌧 Rain Tomorrow")
    else:
        st.success("☀ No Rain Tomorrow")