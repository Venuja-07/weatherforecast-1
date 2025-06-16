
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Forecast for Farmers", layout="centered")

st.title("🌾 Weather Forecast for Farmers in Australia")

# Load weather data and forecast
df = pd.read_csv("farmer_weather_data.csv", parse_dates=["Date"])
prophet = pd.read_csv("prophet_forecast.csv", parse_dates=["ds"])
lstm = pd.read_csv("lstm_forecast.csv", parse_dates=["ds"])

# Section 1: Recent 7-day advice
st.subheader("🗓️ 7-Day Weather Summary & Advice")
st.dataframe(df.tail(7)[["Date", "Day Temp (°C)", "Rainfall (mm)", "Max Wind Speed (km/h)", "Humidity (%)", "Advice"]])

# Section 2: Forecast visualization
st.subheader("📈 30-Day Forecasts")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df["Day Temp (°C)"], label="Actual Temp", color="black")
ax.plot(prophet["ds"], prophet["yhat"], label="Prophet Forecast", linestyle="--")
ax.plot(lstm["ds"], lstm["yhat"], label="LSTM Forecast", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

st.markdown("ℹ️ This dashboard uses real data from Open-Meteo and machine learning models (Prophet, LSTM) to provide weather insights.")
