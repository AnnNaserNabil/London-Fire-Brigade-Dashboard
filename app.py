import streamlit as st
import pandas as pd
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Load data
data = load_data()

if not data.empty:
    st.success("✅ Data loaded successfully!")
    st.write(data.head())

    # Sidebar filters
    st.sidebar.header("User Inputs")
    selected_year = st.sidebar.selectbox("Select Year", data['CalYear'].unique())
    selected_borough = st.sidebar.selectbox("Select Borough", [col.replace("BoroughName_", "") for col in data.columns if "BoroughName_" in col])

    # Filter data
    filtered_data = data[(data['CalYear'] == selected_year) & (data[f'BoroughName_{selected_borough}'] == 1)]

    # Display filtered data
    st.header("Filtered Data")
    st.write(filtered_data)

# Geospatial Analysis
st.header("📌 Incident Map")
if 'Latitude' in filtered_data.columns and 'Longitude' in filtered_data.columns:
    # Check if there are missing values in Latitude or Longitude
    if not filtered_data[['Latitude', 'Longitude']].isnull().values.any():
        london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
        HeatMap(filtered_data[['Latitude', 'Longitude']].values.tolist(), radius=15).add_to(london_map)
        folium_static(london_map)
    else:
        st.warning("⚠️ Some incidents have missing geospatial data and will not be displayed on the map.")
else:
    st.error("❌ Latitude and Longitude columns are missing from the data.")


    # Trends over time
    st.header("📈 Trends Over Time")
    st.bar_chart(filtered_data['HourOfCall'].value_counts().sort_index())

    # Train predictive model with hyperparameter tuning
    @st.cache_data
    def train_model(data):
        features = ['DayOfWeek', 'Month', 'HourOfCall'] + [col for col in data.columns if 'BoroughName_' in col]
        target = 'AttendanceTimeSeconds'
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        model = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_absolute_error', cv=3, random_state=42)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        return best_model, mae, X_train

    # Train the model
    model, mae, X_train = train_model(data)
    st.success(f"✅ Model trained! Mean Absolute Error: {mae:.2f} seconds")

    # Explain model predictions
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    st.header("🔍 Feature Importance")
    fig = px.bar(pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False), x='Feature', y='Importance')
    st.plotly_chart(fig)

    # Predict response time
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

    # Logging model performance
    mlflow.log_metric("MAE", mae)
    st.success("📊 Model performance logged with MLflow!")

else:
    st.error("❌ No valid data available.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
