import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Set page title and layout
st.set_page_config(page_title="🚒 London Fire Brigade Dashboard", layout="wide")

# Data source URL
DATA_URL = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%20from%202021%20-%202024.csv"

# Load and clean the dataset
@st.cache_data
def load_data():
    try:
        # Load dataset
        data = pd.read_csv(DATA_URL)

        # Debugging: Show raw data structure
        st.write("✅ Data loaded successfully! Preview:", data.head())

        # Ensure necessary columns exist
        required_columns = {'BoroughName', 'DateAndTimeMobilised', 'HourOfCall', 'AttendanceTimeSeconds'}
        if not required_columns.issubset(data.columns):
            st.error(f"⚠️ Missing required columns! Expected {required_columns}, but got {data.columns}.")
            return pd.DataFrame()

        # Convert to datetime & handle errors
        data['DateAndTimeMobilised'] = pd.to_datetime(data['DateAndTimeMobilised'], errors='coerce')
        data.dropna(subset=['DateAndTimeMobilised'], inplace=True)

        # Convert HourOfCall to numeric
        data['HourOfCall'] = pd.to_numeric(data['HourOfCall'], errors='coerce')
        data.dropna(subset=['HourOfCall'], inplace=True)

        # Extract additional time features
        data['DayOfWeek'] = data['DateAndTimeMobilised'].dt.dayofweek
        data['Month'] = data['DateAndTimeMobilised'].dt.month

        # Ensure AttendanceTimeSeconds is valid
        data.dropna(subset=['AttendanceTimeSeconds'], inplace=True)

        # Add latitude and longitude (Example mapping)
        borough_coords = {
            "Camden": (51.5290, -0.1255),
            "Westminster": (51.4975, -0.1357),
            "Lambeth": (51.4963, -0.1115),
        }
        data['Latitude'] = data['BoroughName'].map(lambda x: borough_coords.get(x, (None, None))[0])
        data['Longitude'] = data['BoroughName'].map(lambda x: borough_coords.get(x, (None, None))[1])
        data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        # One-hot encode BoroughName
        data = pd.get_dummies(data, columns=['BoroughName'], drop_first=True)

        return data

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# Train the predictive model
@st.cache_data
def train_model(data):
    features = ['DayOfWeek', 'Month', 'HourOfCall'] + [col for col in data.columns if 'BoroughName_' in col]
    target = 'AttendanceTimeSeconds'

    # Ensure we have enough data
    if data.empty or len(data) < 10:
        st.error("⚠️ Not enough data for training. Check data loading and filtering.")
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

# Streamlit App UI
st.title("🚒 London Fire Brigade Dashboard")

# Sidebar filters
st.sidebar.header("User Inputs")
if not data.empty:
    selected_year = st.sidebar.selectbox("Select Year", sorted(data['DateAndTimeMobilised'].dt.year.unique()))
    selected_borough = st.sidebar.selectbox("Select Borough", list(borough_coords.keys()))

    # Filter data
    filtered_data = data[(data['DateAndTimeMobilised'].dt.year == selected_year) & 
                         (data.get(f'BoroughName_{selected_borough}', pd.Series([0]*len(data))) == 1)]

    # Show filtered data
    st.header("📊 Filtered Data")
    st.write(filtered_data)

    # Geospatial Visualization
    st.header("📌 Incident Map")
    london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
    
    if not filtered_data.empty:
        from folium.plugins import HeatMap
        HeatMap(filtered_data[['Latitude', 'Longitude']].values.tolist(), radius=15).add_to(london_map)

    folium_static(london_map)

    # Trends Visualization
    st.header("📈 Trends Over Time")
    if not filtered_data.empty:
        hourly_counts = filtered_data['HourOfCall'].value_counts().sort_index()
        st.bar_chart(hourly_counts)

    # Predictive Modeling Section
    st.header("🔮 Predictive Modeling")

    # Model Performance
    if model:
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f} seconds")

        # Predict Response Time
        st.subheader("Predict Response Time")
        day_of_week = st.slider("Day of Week", 0, 6, 1)
        month = st.slider("Month", 1, 12, 6)
        hour_of_call = st.slider("Hour of Call", 0, 23, 12)

        # Prepare input for prediction
        input_features = {
            'DayOfWeek': day_of_week,
            'Month': month,
            'HourOfCall': hour_of_call,
        }
        for col in data.columns:
            if 'BoroughName_' in col:
                input_features[col] = 1 if col == f'BoroughName_{selected_borough}' else 0

        input_df = pd.DataFrame([input_features])

        # Predict response time
        predicted_time = model.predict(input_df)[0]
        st.write(f"🕒 **Predicted Response Time:** {predicted_time:.2f} seconds")
    else:
        st.warning("⚠️ Model not trained due to insufficient data.")

else:
    st.error("❌ No data available. Please check your dataset.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
