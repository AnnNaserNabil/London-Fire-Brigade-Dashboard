import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Set page title and layout
st.set_page_config(page_title="London Fire Brigade Dashboard", layout="wide")

# Load the dataset from URL
@st.cache_data
def load_data():
    url = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%20from%202021%20-%202024.csv"
    data = pd.read_csv(url)
    
    # Example: Add latitude and longitude columns (replace with actual geocoding)
    borough_coords = {
        "Camden": (51.5290, -0.1255),
        "Westminster": (51.4975, -0.1357),
        "Lambeth": (51.4963, -0.1115),
        # Add more boroughs as needed
    }
    data['Latitude'] = data['BoroughName'].map(lambda x: borough_coords.get(x, (None, None))[0])
    data['Longitude'] = data['BoroughName'].map(lambda x: borough_coords.get(x, (None, None))[1])
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Feature engineering for predictive modeling
    data['DateAndTimeMobilised'] = pd.to_datetime(data['DateAndTimeMobilised'])
    data['DayOfWeek'] = data['DateAndTimeMobilised'].dt.dayofweek
    data['Month'] = data['DateAndTimeMobilised'].dt.month
    data['HourOfCall'] = data['HourOfCall'].astype(int)
    data = pd.get_dummies(data, columns=['BoroughName'], drop_first=True)
    
    return data

# Train a predictive model
@st.cache_data
def train_model(data):
    features = ['DayOfWeek', 'Month', 'HourOfCall'] + [col for col in data.columns if 'BoroughName_' in col]
    target = 'AttendanceTimeSeconds'
    
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, mae

# Load data
data = load_data()

# Train model
model, mae = train_model(data)

# Streamlit app
st.title("🚒 London Fire Brigade Dashboard")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
selected_year = st.sidebar.selectbox("Select Year", data['CalYear'].unique())
selected_borough = st.sidebar.selectbox("Select Borough", data['BoroughName'].unique())

# Filter data based on user inputs
filtered_data = data[(data['CalYear'] == selected_year) & (data['BoroughName'] == selected_borough)]

# Display filtered data
st.header("Filtered Data")
st.write(filtered_data)

# Geospatial Analysis: Map incidents
st.header("📌 Incident Map")
london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
HeatMap(filtered_data[['Latitude', 'Longitude']].values.tolist(), radius=15).add_to(london_map)
folium_static(london_map)

# Trends over time
st.header("📈 Trends Over Time")
hourly_counts = filtered_data['HourOfCall'].value_counts().sort_index()
st.bar_chart(hourly_counts)

# Predictive Modeling Section
st.header("🔮 Predictive Modeling")

# Display model performance
st.subheader("Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f} seconds")

# Predict response time for user inputs
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

# Predict
predicted_time = model.predict(input_df)[0]
st.write(f"Predicted Response Time: {predicted_time:.2f} seconds")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
