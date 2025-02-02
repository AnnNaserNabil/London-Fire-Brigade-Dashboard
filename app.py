import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Streamlit page configuration
st.set_page_config(page_title="🚒 London Fire Brigade Dashboard", layout="wide")

# Data source
DATA_URL = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%20from%202021%20-%202024.csv"

# Load and clean the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(DATA_URL, encoding='utf-8', low_memory=False)
        data.columns = data.columns.str.strip()
        
        # Convert datetime columns
        datetime_cols = ['DateAndTimeMobilised', 'DateAndTimeMobile', 'DateAndTimeArrived', 
                         'DateAndTimeLeft', 'DateAndTimeReturned']
        for col in datetime_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')

        # Convert numeric columns
        numeric_cols = ['HourOfCall', 'TurnoutTimeSeconds', 'TravelTimeSeconds', 'AttendanceTimeSeconds', 'PumpOrder']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Fix Latitude & Longitude issue
        if 'Latitude' in data.columns and 'Longitude' in data.columns:
            data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
            data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
            data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        # Extract features
        if 'DateAndTimeMobilised' in data.columns:
            data['DayOfWeek'] = data['DateAndTimeMobilised'].dt.dayofweek
            data['Month'] = data['DateAndTimeMobilised'].dt.month

        # One-hot encode BoroughName (if present)
        if 'BoroughName' in data.columns:
            data = pd.get_dummies(data, columns=['BoroughName'], drop_first=True)

        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# Train ML model
@st.cache_data
def train_model(data):
    features = ['HourOfCall', 'TurnoutTimeSeconds', 'TravelTimeSeconds', 'AttendanceTimeSeconds', 'DayOfWeek', 'Month']
    target = 'PumpOrder'
    if target not in data.columns:
        return None, None

    data = data.dropna(subset=features + [target])
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return model, mae

# Load and preprocess data
data = load_data()

if not data.empty:
    st.success("✅ Data loaded successfully!")
    st.write(data.head())
    model, mae = train_model(data)
    if model:
        st.success(f"✅ Model trained! Mean Absolute Error: {mae:.2f}")
else:
    st.error("❌ No valid data available.")
