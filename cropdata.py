
#####NEW NEW CODE 


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ---- SESSION STATE LOGIN INIT ----
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "login_attempted" not in st.session_state:
    st.session_state["login_attempted"] = False

# ---- LOGIN SCREEN ----
def login():
    st.title("🔐 Crop Yield Dashboard Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state["login_attempted"] = True
        if username == "Bruce" and password == "1234":
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    if st.session_state["login_attempted"] and not st.session_state["authenticated"]:
        st.error("Incorrect username or password.")
    
    st.stop()  # Prevent the rest of the app from running

# ---- SHOW LOGIN IF NOT AUTHENTICATED ----
if not st.session_state["authenticated"]:
    login()

# ---- DASHBOARD CONTENT BELOW ----



# Load data
df = pd.read_excel("crop_data1.xlsx")

# Title
st.title("🌾 Crop Yield Optimization Dashboard")

# Sidebar inputs
st.sidebar.header("🔍 Enter Environmental Conditions")
rain = st.sidebar.slider("Rainfall (mm)", 300, 1300, 800)
fert = st.sidebar.slider("Fertilizer", 40, 100, 70)
temp = st.sidebar.slider("Temperature (°C)", 20, 40, 30)
n = st.sidebar.slider("Nitrogen (N)", 60, 90, 75)
p = st.sidebar.slider("Phosphorus (P)", 15, 30, 20)
k = st.sidebar.slider("Potassium (K)", 15, 30, 20)

# --- Section 1: Data Overview ---
st.header("📊 Data Overview")
st.write("Sample of data:")
st.dataframe(df.head())

# st.subheader("Correlation Heatmap")
# fig, ax = plt.subplots()
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
# st.pyplot(fig)

# --- Section 2: Yield Predictor ---
st.header("🤖 Yield Predictor")

# Prepare features and model
X = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
y = df["Yield (Q/acre)"]
model = LinearRegression()
model.fit(X, y)

# Predict based on input
input_data = np.array([[rain, fert, temp, n, p, k]])
predicted_yield = model.predict(input_data)[0]
st.success(f"Estimated Yield: **{predicted_yield:.2f} Q/acre**")

# --- Section 3: Recommendations ---
st.header("📌 Recommendation")
if predicted_yield < 9:
    st.warning("⚠️ Yield is below average. Consider increasing Nitrogen or checking rainfall patterns.")
elif predicted_yield > 11:
    st.info("✅ Conditions are favorable for high yield.")
else:
    st.write("🟡 Moderate yield expected. You may fine-tune fertilizer or irrigation levels.")

# ---- LOGOUT BUTTON ----
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.session_state["login_attempted"] = False
    st.experimental_rerun()


#####Good codes ends here#####

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta

# ---- SESSION STATE LOGIN INIT ----
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "login_attempted" not in st.session_state:
    st.session_state["login_attempted"] = False

# ---- LOGIN SCREEN ----
def login():
    st.title("🔐 Crop Yield Dashboard Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state["login_attempted"] = True
        if username == "Bruce" and password == "1234":
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    if st.session_state["login_attempted"] and not st.session_state["authenticated"]:
        st.error("Incorrect username or password.")
    
    st.stop()  # Prevent the rest of the app from running

# ---- SHOW LOGIN IF NOT AUTHENTICATED ----
if not st.session_state["authenticated"]:
    login()

# ---- DASHBOARD CONTENT BELOW ----

# Crop planting requirements
CROP_REQUIREMENTS = {
    "Wheat": {"min_temp": 10, "max_temp": 25, "min_rainfall": 50, "max_rainfall": 100},
    "Maize": {"min_temp": 15, "max_temp": 30, "min_rainfall": 60, "max_rainfall": 150},
    "Rice": {"min_temp": 20, "max_temp": 35, "min_rainfall": 100, "max_rainfall": 200}
}

# Function to fetch weather forecast
def get_weather_forecast(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to check if weather is suitable for planting
def is_suitable_for_planting(forecast, crop):
    requirements = CROP_REQUIREMENTS[crop]
    temp = forecast["main"]["temp"]
    rainfall = forecast.get("rain", {}).get("3h", 0)  # Rainfall in last 3 hours (mm)
    return (requirements["min_temp"] <= temp <= requirements["max_temp"] and
            requirements["min_rainfall"] <= rainfall * 8 <= requirements["max_rainfall"])  # Scale to daily

# Load data
df = pd.read_excel("crop_data1.xlsx")

# Title
st.title("🌾 Crop Yield Optimization Dashboard")

# Sidebar inputs
st.sidebar.header("🔍 Enter Environmental Conditions")
rain = st.sidebar.slider("Rainfall (mm)", 300, 1300, 800)
fert = st.sidebar.slider("Fertilizer", 40, 100, 70)
temp = st.sidebar.slider("Temperature (°C)", 20, 40, 30)
n = st.sidebar.slider("Nitrogen (N)", 60, 90, 75)
p = st.sidebar.slider("Phosphorus (P)", 15, 30, 20)
k = st.sidebar.slider("Potassium (K)", 15, 30, 20)

# Planting Scheduler inputs
st.sidebar.header("🌱 Planting Scheduler")
city = st.sidebar.text_input("City (e.g., Nairobi, Iowa)", "Nairobi")
crop = st.sidebar.selectbox("Select Crop", list(CROP_REQUIREMENTS.keys()))
api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

# --- Section 1: Data Overview ---
st.header("📊 Data Overview")
st.write("Sample of data:")
st.dataframe(df.head())

# st.subheader("Correlation Heatmap")
# fig, ax = plt.subplots()
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
# st.pyplot(fig)

# --- Section 2: Yield Predictor ---
st.header("🤖 Yield Predictor")

# Prepare features and model
X = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
y = df["Yield (Q/acre)"]
model = LinearRegression()
model.fit(X, y)

# Predict based on input
input_data = np.array([[rain, fert, temp, n, p, k]])
predicted_yield = model.predict(input_data)[0]
st.success(f"Estimated Yield: **{predicted_yield:.2f} Q/acre**")

# --- Section 3: Recommendations ---
st.header("📌 Recommendation")
if predicted_yield < 9:
    st.warning("⚠️ Yield is below average. Consider increasing Nitrogen or checking rainfall patterns.")
elif predicted_yield > 11:
    st.info("✅ Conditions are favorable for high yield.")
else:
    st.write("🟡 Moderate yield expected. You may fine-tune fertilizer or irrigation levels.")

# --- Section 4: Planting Scheduler ---
st.header("🗓️ Planting Scheduler")
if st.button("Get Planting Schedule"):
    if not api_key:
        st.error("Please provide a valid OpenWeatherMap API key.")
    else:
        weather_data = get_weather_forecast(city, api_key)
        if weather_data:
            forecast_list = weather_data["list"]
            suitable_dates = []
            
            # Analyze 7-day forecast
            current_date = datetime.now()
            for forecast in forecast_list:
                forecast_time = datetime.fromtimestamp(forecast["dt"])
                if is_suitable_for_planting(forecast, crop):
                    suitable_dates.append(forecast_time.date())
            
            # Deduplicate and sort dates
            suitable_dates = sorted(list(set(suitable_dates)))
            
            if suitable_dates:
                st.success(f"Recommended planting dates for {crop} in {city}:")
                date_range = f"{min(suitable_dates).strftime('%B %d')} - {max(suitable_dates).strftime('%B %d')}"
                st.write(f"Best planting period: {date_range}")
                
                # Create a timeline visualization
                df_dates = pd.DataFrame({
                    "Date": suitable_dates,
                    "Suitability": ["Suitable" for _ in suitable_dates]
                })
                fig = px.scatter(df_dates, x="Date", y="Suitability", title=f"Suitable Planting Dates for {crop}",
                                 labels={"Suitability": ""}, height=200)
                fig.update_yaxes(showticklabels=False)
                fig.update_traces(marker=dict(size=12, color="#36A2EB"))
                st.plotly_chart(fig)
            else:
                st.warning(f"No suitable planting dates found for {crop} in the next 7 days.")
        else:
            st.error("Failed to fetch weather data. Check city name or API key.")

# ---- LOGOUT BUTTON ----
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.session_state["login_attempted"] = False
    st.experimental_rerun()
