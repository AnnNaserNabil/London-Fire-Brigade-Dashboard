import streamlit as st
import pandas as pd
import numpy as np
import folium
import logging
import plotly.express as px
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import shap
import mlflow

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="London Fire Brigade Dashboard", layout="wide")

# Data source URL
DATA_URL = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%20from%202021%20-%202024.csv"

# Function to load and preprocess the data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load dataset
        data = pd.read_csv(DATA_URL)
        
        # Convert date columns to datetime
        data['DateAndTimeMobilised'] = pd.to_datetime(data['DateAndTimeMobilised'], errors='coerce')
        
        # Drop rows with missing critical values
        data.dropna(subset=['AttendanceTimeSeconds', 'BoroughName', 'HourOfCall'], inplace=True)
        
        # Feature engineering
        data['DayOfWeek'] = data['DateAndTimeMobilised'].dt.dayofweek
        data['Month'] = data['DateAndTimeMobilised'].dt.month
        data['HourOfCall'] = data['HourOfCall'].astype(int)
        
        # One-hot encoding for BoroughName
        data = pd.get_dummies(data, columns=['BoroughName'], drop_first=True)
        
        # Convert Latitude and Longitude to numeric and drop invalid entries
        if 'Latitude' in data.columns and 'Longitude' in data.columns:
            data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
            data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
            data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        return data
    except Exception as e:
        st.error(f"Error loading and processing data: {e}")
        logger.error(f"Data loading error: {e}")
        return pd.DataFrame()

# Load the data
data = load_and_preprocess_data()

# Ensure data is available
if not data.empty:
    st.success("Data loaded successfully!")

    # Display dataset preview
    st.header("Data Overview")
    st.write(data.head())

    # Sidebar for user input
    st.sidebar.header("Filters")
    selected_year = st.sidebar.selectbox("Select Year", data['CalYear'].unique())
    selected_borough = st.sidebar.selectbox("Select Borough", [col.replace("BoroughName_", "") for col in data.columns if "BoroughName_" in col])

    # Filter data based on user input
    filtered_data = data[(data['CalYear'] == selected_year) & (data[f'BoroughName_{selected_borough}'] == 1)]

    # Display filtered data
    st.header("Filtered Data")
    st.write(filtered_data)

    # Geospatial Visualization - Heatmap
    st.header("Incident Map")
    if not filtered_data[['Latitude', 'Longitude']].isnull().values.any():
        london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
        HeatMap(filtered_data[['Latitude', 'Longitude']].values.tolist(), radius=15).add_to(london_map)
        folium_static(london_map)
    else:
        st.warning("Some incidents have missing geospatial data and will not be displayed on the map.")

    # Trend Analysis - Hour of Call
    st.header("Trends Over Time")
    st.bar_chart(filtered_data['HourOfCall'].value_counts().sort_index())

    # Model Training and Hyperparameter Tuning
    @st.cache_data
    def train_predictive_model(data):
        features = ['DayOfWeek', 'Month', 'HourOfCall'] + [col for col in data.columns if 'BoroughName_' in col]
        target = 'AttendanceTimeSeconds'
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
        
        # Hyperparameter tuning using RandomizedSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize and train the RandomForestRegressor model
        model = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_absolute_error', cv=3, random_state=42)
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        return best_model, mae, X_train

    # Train the model and evaluate performance
    model, mae, X_train = train_predictive_model(data)
    st.success(f"Model trained successfully! Mean Absolute Error: {mae:.2f} seconds")

    # SHAP Feature Importance Visualization
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    st.header("Feature Importance")
    feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    fig = px.bar(feature_importance, x='Feature', y='Importance')
    st.plotly_chart(fig)

    # User Input for Response Time Prediction
    st.subheader("Predict Response Time")
    day_of_week = st.slider("Day of Week", 0, 6, 1)
    month = st.slider("Month", 1, 12, 6)
    hour_of_call = st.slider("Hour of Call", 0, 23, 12)

    input_features = {'DayOfWeek': day_of_week, 'Month': month, 'HourOfCall': hour_of_call}
    for col in data.columns:
        if 'BoroughName_' in col:
            input_features[col] = 1 if col == f'BoroughName_{selected_borough}' else 0

    input_df = pd.DataFrame([input_features])
    predicted_time = model.predict(input_df)[0]
    st.write(f"Predicted Response Time: {predicted_time:.2f} seconds")

    # Log model performance using MLflow
    mlflow.log_metric("MAE", mae)
    st.success("Model performance logged with MLflow.")

else:
    st.error("Failed to load or process the data.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by the London Fire Brigade Data Team | Powered by Streamlit")
