import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="🚒 London Fire Brigade Dashboard", layout="wide")

# Data source
DATA_URL = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%20from%202021%20-%202024.csv"

# Load and clean the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(DATA_URL)

        # Debugging: Check if data is actually loaded
        if data.empty or data.isnull().all().all():
            st.error("❌ Data is empty! Check the dataset source or file format.")
            return pd.DataFrame()

        # Convert datetime columns
        datetime_cols = ['DateAndTimeMobilised', 'DateAndTimeMobile', 'DateAndTimeArrived', 
                         'DateAndTimeLeft', 'DateAndTimeReturned']
        for col in datetime_cols:
            data[col] = pd.to_datetime(data[col], errors='coerce')

        # Convert numeric columns
        numeric_cols = ['HourOfCall', 'TurnoutTimeSeconds', 'TravelTimeSeconds', 'AttendanceTimeSeconds', 'PumpOrder']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Convert Latitude & Longitude
        data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
        data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        # Extract features
        data['DayOfWeek'] = data['DateAndTimeMobilised'].dt.dayofweek
        data['Month'] = data['DateAndTimeMobilised'].dt.month

        # One-hot encode BoroughName
        data = pd.get_dummies(data, columns=['BoroughName'], drop_first=True)

        return data

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# Train predictive model
@st.cache_data
def train_model(data):
    features = ['DayOfWeek', 'Month', 'HourOfCall'] + [col for col in data.columns if 'BoroughName_' in col]
    target = 'AttendanceTimeSeconds'

    if data.empty or data[target].isnull().all():
        st.error("⚠️ No valid data for training. Ensure dataset contains AttendanceTimeSeconds.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae

# Load data
data = load_data()

# Train model if data is valid
if not data.empty:
    model, mae = train_model(data)
else:
    model, mae = None, None

# Streamlit UI
st.title("🚒 London Fire Brigade Dashboard")

# Sidebar filters
st.sidebar.header("User Inputs")
if not data.empty:
    selected_year = st.sidebar.selectbox("Select Year", sorted(data['DateAndTimeMobilised'].dt.year.unique()))
    selected_hour = st.sidebar.slider("Select Hour", 0, 23, 12)

    # Filter data
    filtered_data = data[(data['DateAndTimeMobilised'].dt.year == selected_year) &
                         (data['HourOfCall'] == selected_hour)]

    st.header("📊 Filtered Data")
    st.write(filtered_data)

    # Geospatial Visualization
    st.header("📌 Incident Map")
    london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

    if not filtered_data.empty:
        from folium.plugins import HeatMap
        HeatMap(filtered_data[['Latitude', 'Longitude']].values.tolist(), radius=10).add_to(london_map)

    folium_static(london_map)

    # Predictive Model
    st.header("🔮 Predict Response Time")
    if model:
        st.write(f"Mean Absolute Error: {mae:.2f} sec")
        day_of_week = st.slider("Day of Week", 0, 6, 1)
        month = st.slider("Month", 1, 12, 6)
        hour = st.slider("Hour", 0, 23, 12)

        input_data = {
            'DayOfWeek': day_of_week,
            'Month': month,
            'HourOfCall': hour,
        }
        input_df = pd.DataFrame([input_data])

        predicted_time = model.predict(input_df)[0]
        st.write(f"Predicted Response Time: {predicted_time:.2f} sec")

else:
    st.error("❌ No valid data available.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
