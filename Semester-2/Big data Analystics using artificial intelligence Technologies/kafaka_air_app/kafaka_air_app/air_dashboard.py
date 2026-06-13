import pandas as pd
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('air_pollution_data.csv')

# Display Data
st.title("🌍 Real-Time Air Pollution Dashboard")
data = load_data()

st.subheader("Latest 10 Air Pollution Updates")
st.write(data.tail(10))

st.subheader("🌫️ Air Quality Index (AQI) Trends")
st.line_chart(data.set_index('timestamp')['aqi'])

st.subheader("🧪 Pollutant Concentrations")
if not data.empty:
    latest_components = data.iloc[-1]['components']
    st.json(latest_components)
else:
    st.write("No data available.")