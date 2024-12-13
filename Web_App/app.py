import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
import calendar
import certifi
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from bs4 import BeautifulSoup
import json
from itertools import product
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX

mongo_uri = st.secrets["MONGO_URI"]
if not mongo_uri:
    st.error("MongoDB URI is not set!")
    st.stop()
else:
    # Connect to MongoDB with SSL certificate validation
    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client["AgriPredict"]
    collection = db["WhiteSesame"]
    best_params_collection = db["BestParams"]
    impExp = db["impExp"]
    users_collection = db["user"]

state_market_dict = {
    "Karnataka": [
        "Kalburgi",
        "Basava Kalayana",
        "Lingasugur",
        "Kustagi",
        "Bangalore",
        "Bagalakot",
        "Hubli (Amaragol)"
    ],
    "Gujarat": [
        "Siddhpur",
        "Jasdan",
        "Gondal",
        "Morbi",
        "Botad",
        "Visavadar",
        "Dahod",
        "Rajkot",
        "Junagadh",
        "Savarkundla",
        "Bhavnagar",
        "Rajula",
        "Dhoraji",
        "Amreli",
        "Mahuva(Station Road)",
        "Mansa",
        "Porbandar",
        "Dasada Patadi",
        "Halvad",
        "Chotila",
        "Bhanvad",
        "Dhansura",
        "Babra",
        "Upleta",
        "Palitana",
        "Jetpur(Dist.Rajkot)",
        "S.Mandvi",
        "Mandvi",
        "Khambha",
        "Kadi",
        "Taleja",
        "Himatnagar",
        "Lakhani",
        "Rapar",
        "Una",
        "Dhari",
        "Bagasara",
        "Jam Jodhpur",
        "Veraval",
        "Dhragradhra",
        "Deesa"
    ],
    "Uttar Pradesh": [
        "Bangarmau",
        "Sultanpur",
        "Maudaha",
        "Mauranipur",
        "Lalitpur",
        "Konch",
        "Muskara",
        "Raath",
        "Varipaal",
        "Auraiya",
        "Orai",
        "Banda",
        "Kishunpur",
        "Ait",
        "Jhansi",
        "Kurara",
        "Chirgaon",
        "Charkhari",
        "Moth",
        "Jalaun",
        "Sirsaganj",
        "Shikohabad"
    ],
    "Madhya Pradesh": [
        "Naugaon",
        "Mehar",
        "Kailaras",
        "Datia",
        "LavKush Nagar(Laundi)",
        "Ajaygarh",
        "Rajnagar",
        "Sevda",
        "Neemuch",
        "Sheopurkalan",
        "Lashkar",
        "Alampur",
        "Niwadi",
        "Dabra",
        "Ujjain",
        "Bijawar",
        "Sidhi",
        "Barad",
        "Pohari",
        "Shahagarh",
        "Lateri",
        "Banapura",
        "Panna",
        "Garhakota",
        "Katni",
        "Chhatarpur",
        "Beohari",
        "Satna",
        "Sabalgarh",
        "Hanumana",
        "Bhander",
        "Banmorkalan",
        "Jaora",
        "Bagli",
        "Singroli"
    ],
    "Telangana": [
        "Warangal"
    ]
}

import pandas as pd
import numpy as np

def create_forecasting_features(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Reported Date')
        df.index = pd.to_datetime(df.index)

    # Create a mapping of target values for lag features
    target_map = df['Modal Price (Rs./Quintal)'].to_dict()

    # Time-based features
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    # Lag features
    df['lag'] = (df.index - pd.Timedelta(days=14)).map(target_map)
    df['lag28'] = (df.index - pd.Timedelta(days=28)).map(target_map)
    df['lag56'] = (df.index - pd.Timedelta(days=56)).map(target_map)
    df['lag_3months'] = (df.index - pd.DateOffset(months=3)).map(target_map)
    df['lag_6months'] = (df.index - pd.DateOffset(months=6)).map(target_map)

    # Rolling features
    for window in [28, 42, 56]:  # Monthly, 6-week, and 8-week windows
        df[f'rolling_mean_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).std()

    # Exponential moving averages for smoothing recent trends
    df['ema28'] = df['Modal Price (Rs./Quintal)'].ewm(span=28, adjust=False).mean()
    df['ema42'] = df['Modal Price (Rs./Quintal)'].ewm(span=42, adjust=False).mean()
    df['ema56'] = df['Modal Price (Rs./Quintal)'].ewm(span=56, adjust=False).mean()

    # Seasonal averages (historical values based on seasonality)
    df['monthly_avg'] = df.groupby('month')['Modal Price (Rs./Quintal)'].transform('mean')
    df['weekly_avg'] = df.groupby('weekofyear')['Modal Price (Rs./Quintal)'].transform('mean')
    df['dayofweek_avg'] = df.groupby('dayofweek')['Modal Price (Rs./Quintal)'].transform('mean')

    # Fourier terms for periodicity
    df['fourier_sin_365'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['fourier_cos_365'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    df['fourier_sin_14'] = np.sin(2 * np.pi * df.index.dayofyear / 14)
    df['fourier_cos_14'] = np.cos(2 * np.pi * df.index.dayofyear / 14)

    # Statistical history features
    df['recent_min_14'] = (df.index - pd.Timedelta(days=14)).map(target_map).min()
    df['recent_max_14'] = (df.index - pd.Timedelta(days=14)).map(target_map).max()
    df['recent_range_14'] = df['recent_max_14'] - df['recent_min_14']

    # Long-term trends
    df['yearly_avg'] = df.groupby('year')['Modal Price (Rs./Quintal)'].transform('mean')
    df['cumulative_mean'] = df['Modal Price (Rs./Quintal)'].expanding().mean()

    # Calculate correlations and print
    correlation = df.corr()['Modal Price (Rs./Quintal)'].sort_values()
    print("Correlation with Modal Price:\n", correlation)
    to_remove = correlation[correlation.abs() < 0.2].index.to_list()
    df.drop(columns=to_remove, inplace=True)
    return df.reset_index()

def preprocess_data(df):
    # Retain only 'Reported Date' and 'Modal Price (Rs./Quintal)' columns
    df = df[['Reported Date', 'Modal Price (Rs./Quintal)']]

    # Ensure 'Reported Date' is in datetime format
    df['Reported Date'] = pd.to_datetime(df['Reported Date'])

    # Group by 'Reported Date' and calculate mean of 'Modal Price (Rs./Quintal)'
    df = df.groupby('Reported Date', as_index=False).mean()

    # Generate a full date range from the minimum to the maximum date
    full_date_range = pd.date_range(df['Reported Date'].min(), df['Reported Date'].max())
    df = df.set_index('Reported Date').reindex(full_date_range).rename_axis('Reported Date').reset_index()

    # Detect and remove outliers for every 30 days
    df['Modal Price (Rs./Quintal)'] = (
        df['Modal Price (Rs./Quintal)'].fillna(method='ffill').fillna(method='bfill')
    )
    return df
    
def train_and_evaluate(df):
    import streamlit as st

    # Add progress bar for hyperparameter tuning
    progress_bar = st.progress(0)

    # Helper function to update progress during hyperparameter tuning
    def update_tuning_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress)

    df = create_forecasting_features(df)

    # Split the data into training and testing sets
    train_df = df[df['Reported Date'] < '2024-01-01']
    test_df = df[df['Reported Date'] >= '2024-01-01']

    X_train = train_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'])
    y_train = train_df['Modal Price (Rs./Quintal)']
    X_test = test_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'])
    y_test = test_df['Modal Price (Rs./Quintal)']

    # Hyperparameter tuning
    st.write("Performing hyperparameter tuning...")
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150],
        'booster': ['gbtree', 'dart']
    }

    model = XGBRegressor()
    param_combinations = len(param_grid['learning_rate']) * len(param_grid['max_depth']) * \
                         len(param_grid['n_estimators']) * len(param_grid['booster'])

    current_combination = 0  # Counter for combinations

    def custom_grid_search():
        nonlocal current_combination
        best_score = float('-inf')
        best_params = None
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                for n_estimators in param_grid['n_estimators']:
                    for booster in param_grid['booster']:
                        model.set_params(
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            booster=booster
                        )
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'n_estimators': n_estimators,
                                'booster': booster
                            }
                        # Update progress bar
                        current_combination += 1
                        update_tuning_progress(current_combination, param_combinations)
        return best_params

    best_params = custom_grid_search()

    # Train the best model with the identified parameters
    st.write("Training the best model and making predictions...")
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")

    # Prepare data for plotting
    train_plot_df = train_df[['Reported Date', 'Modal Price (Rs./Quintal)']].copy()
    train_plot_df['Type'] = 'Train'

    test_plot_df = test_df[['Reported Date', 'Modal Price (Rs./Quintal)']].copy()
    test_plot_df['Type'] = 'Test'

    predicted_plot_df = test_df[['Reported Date']].copy()
    predicted_plot_df['Modal Price (Rs./Quintal)'] = y_pred
    predicted_plot_df['Type'] = 'Predicted'

    plot_df = pd.concat([train_plot_df, test_plot_df, predicted_plot_df])

    fig = go.Figure()

    for plot_type, color, dash in [('Train', 'blue', None), ('Test', 'orange', None),
                                   ('Predicted', 'green', 'dot')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig.add_trace(go.Scatter(
            x=data['Reported Date'],
            y=data['Modal Price (Rs./Quintal)'],
            mode='lines',
            name=f"{plot_type} Data",
            line=dict(color=color, dash=dash)
        ))

    fig.update_layout(
        title="Train, Test, and Predicted Data",
        xaxis_title="Date",
        yaxis_title="Modal Price (Rs./Quintal)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Return best parameters
    return best_params

import pandas as pd

def optimize_data_types(df):
    # Optimize numerical data by downcasting
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    
    int_cols = df.select_dtypes(include=['int']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    
    # Convert dates if they're not already in datetime format
    if df['Reported Date'].dtype == 'object':
        df['Reported Date'] = pd.to_datetime(df['Reported Date'])
    
    return df

def forecast_next_14_days(df, _best_params):
    st.write("Optimizing data types...")
    df = optimize_data_types(df)  # Assuming function 'optimize_data_types' is defined elsewhere

    st.write("Determining the last reported date...")
    last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
    future_df = pd.DataFrame({'Reported Date': future_dates})

    st.write("Combining old and future data frames for feature creation...")
    full_df = pd.concat([df, future_df], ignore_index=True)

    st.write("Creating forecasting features...")
    full_df = create_forecasting_features(full_df)  # Assuming function 'create_forecasting_features' is defined

    # Split data back into original and future sets
    original_df = full_df[full_df['Reported Date'] <= last_date]
    future_df = full_df[full_df['Reported Date'] > last_date]

    st.write("Preparing training data...")
    X_train = original_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')
    y_train = original_df['Modal Price (Rs./Quintal)']

    st.write("Preparing data for future forecasting...")
    X_future = future_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')

    st.write("Training the model...")
    _best_params['tree_method'] = 'hist'
    model = XGBRegressor(**_best_params)
    model.fit(X_train, y_train)

    st.write("Making predictions for the next 14 days...")
    future_predictions = model.predict(X_future)
    future_df['Modal Price (Rs./Quintal)'] = future_predictions

    # Get last 14 actual values for comparison
    actual_last_14_df = original_df[original_df['Reported Date'] > (last_date - pd.Timedelta(days=14))]

    # Predicted data (using the last 14 actual values)
    predicted_plot_df = actual_last_14_df[['Reported Date']].copy()
    predicted_plot_df['Modal Price (Rs./Quintal)'] = model.predict(
        actual_last_14_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore'))
    predicted_plot_df['Type'] = 'Actual'

    # Forecasted future data
    future_plot_df = future_df[['Reported Date', 'Modal Price (Rs./Quintal)']].copy()
    future_plot_df['Type'] = 'Forecasted'

    # Add the last actual point to the forecasted data for continuity
    last_actual_point = predicted_plot_df.iloc[[-1]].copy()
    last_actual_point['Type'] = 'Forecasted'
    future_plot_df = pd.concat([last_actual_point, future_plot_df])

    # Concatenate all relevant data for plotting
    plot_df = pd.concat([predicted_plot_df, future_plot_df])

    st.write("Plotting the results...")
    fig = go.Figure()

    for plot_type, color, dash in [('Actual', 'blue', 'solid'), ('Forecasted', 'red', 'dash')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig.add_trace(go.Scatter(
            x=data['Reported Date'],
            y=data['Modal Price (Rs./Quintal)'],
            mode='lines',
            name=f"{plot_type} Data",
            line=dict(color=color, dash=dash)
        ))

    fig.update_layout(
        title="Actual vs Forecasted Modal Price (Rs./Quintal)",
        xaxis_title="Date",
        yaxis_title="Modal Price (Rs./Quintal)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("Forecasting for the next 14 days successfully completed!")


def fetch_and_process_data(query_filter):
    try:
        cursor = collection.find(query_filter)
        data = list(cursor)
        if data:
            df = pd.DataFrame(data)
            st.write("Preprocessing data...")
            df = preprocess_data(df)
            return df
        else:
            st.warning("⚠️ No data found for the selected filter.")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

# Function to save best_params to MongoDB
def save_best_params(filter_key, best_params):
    best_params["filter_key"] = filter_key
    best_params["last_updated"] = datetime.now().isoformat()
    best_params_collection.replace_one({"filter_key": filter_key}, best_params, upsert=True)

# Function to retrieve best_params from MongoDB
def get_best_params(filter_key):
    record = best_params_collection.find_one({"filter_key": filter_key})
    return record

# Function to handle training and forecasting
def train_and_forecast(df, filter_key):
    if df is not None:
        # Train the model and save parameters to MongoDB
        best_params = train_and_evaluate(df)
        save_best_params(filter_key, best_params)
        forecast_next_14_days(df, best_params)

# Function to forecast using stored best_params
def forecast(df, filter_key):
    record = get_best_params(filter_key)
    if record:
        st.info(f"ℹ️ The model was trained on {record['last_updated']}.")
        forecast_next_14_days(df, record)
    else:
        st.warning("⚠️ Model is not trained yet. Please train the model first.")

# Section for forecasting imp-exp
def create_forecasting_features_impExp(df, target_column):
    df = df.copy()

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Reported Date')
        df.index = pd.to_datetime(df.index)

    # Create a mapping of target values for lag features
    target_map = df[target_column].to_dict()

    # Time-based features
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    # Lag features
    df['lag14'] = (df.index - pd.Timedelta(days=14)).map(target_map)
    df['lag28'] = (df.index - pd.Timedelta(days=28)).map(target_map)
    df['lag56'] = (df.index - pd.Timedelta(days=56)).map(target_map)
    df['lag_3months'] = (df.index - pd.DateOffset(months=3)).map(target_map)
    df['lag_6months'] = (df.index - pd.DateOffset(months=6)).map(target_map)
    for window in [7, 14, 28]:  # Weekly, bi-weekly, and monthly windows
        df[f'rolling_mean_{window}'] = df[target_column].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_column].rolling(window=window, min_periods=1).std()

    # Exponential moving averages for smoothing recent trends
    df['ema7'] = df[target_column].ewm(span=7, adjust=False).mean()
    df['ema14'] = df[target_column].ewm(span=14, adjust=False).mean()

    # Seasonal averages (historical values based on seasonality)
    df['monthly_avg'] = df.groupby('month')[target_column].transform('mean')
    df['weekly_avg'] = df.groupby('weekofyear')[target_column].transform('mean')
    df['dayofweek_avg'] = df.groupby('dayofweek')[target_column].transform('mean')

    # Fourier terms for periodicity (optional for strong seasonality)
    df['fourier_sin_365'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['fourier_cos_365'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    df['fourier_sin_14'] = np.sin(2 * np.pi * df.index.dayofyear / 14)
    df['fourier_cos_14'] = np.cos(2 * np.pi * df.index.dayofyear / 14)

    # Statistical history features
    df['recent_min_14'] = (df.index - pd.Timedelta(days=14)).map(target_map).min()
    df['recent_max_14'] = (df.index - pd.Timedelta(days=14)).map(target_map).max()
    df['recent_range_14'] = df['recent_max_14'] - df['recent_min_14']

    # Long-term trends
    df['yearly_avg'] = df.groupby('year')[target_column].transform('mean')

    # Trend Feature
    df['cumulative_mean'] = df[target_column].expanding().mean()

    # Reset the index to include 'Reported Date' in the output
    return df.reset_index()


def train_and_evaluate_impExp(df, target_column):
    # MongoDB connection setup
    impExpParams = db["impExpParams"]

    # Add progress bar for hyperparameter tuning
    progress_bar = st.progress(0)

    # Helper function to update progress during hyperparameter tuning
    def update_tuning_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress)

    # Create forecasting features (ensure this function is defined elsewhere)
    df = create_forecasting_features_impExp(df, target_column)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.25, shuffle=False, stratify=None)

    # Separate features and target
    X_train = train_df.drop(columns=[target_column, 'Reported Date'])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column, 'Reported Date'])
    y_test = test_df[target_column]

    # Hyperparameter tuning
    st.write("Performing hyperparameter tuning...")
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150],
        'booster': ['gbtree', 'dart']
    }

    model = XGBRegressor()
    param_combinations = len(param_grid['learning_rate']) * len(param_grid['max_depth']) * \
                         len(param_grid['n_estimators']) * len(param_grid['booster'])

    current_combination = 0  # Counter for combinations

    def custom_grid_search():
        nonlocal current_combination
        best_score = float('-inf')
        best_params = None
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                for n_estimators in param_grid['n_estimators']:
                    for booster in param_grid['booster']:
                        model.set_params(
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            booster=booster
                        )
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'n_estimators': n_estimators,
                                'booster': booster
                            }
                        # Update progress bar
                        current_combination += 1
                        update_tuning_progress(current_combination, param_combinations)
        return best_params

    best_params = custom_grid_search()

    # Save the best parameters to MongoDB
    st.write("Saving the best parameters to MongoDB...")
    impExpParams.insert_one({
        "target_column": target_column,
        "best_params": best_params
    })

    # Train the best model with the identified parameters
    st.write("Training the best model and making predictions...")
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")

    # Prepare data for plotting
    train_plot_df = train_df[['Reported Date', target_column]].copy()
    train_plot_df['Type'] = 'Train'

    test_plot_df = test_df[['Reported Date', target_column]].copy()
    test_plot_df['Type'] = 'Test'

    predicted_plot_df = test_df[['Reported Date']].copy()
    predicted_plot_df[target_column] = y_pred
    predicted_plot_df['Type'] = 'Predicted'

    plot_df = pd.concat([train_plot_df, test_plot_df, predicted_plot_df])

    # Plotly Visualization
    fig = go.Figure()

    for plot_type, color, dash in [('Train', 'blue', None), ('Test', 'orange', None),
                                   ('Predicted', 'green', 'dot')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig.add_trace(go.Scatter(
            x=data['Reported Date'],
            y=data[target_column],
            mode='lines',
            name=f"{plot_type} Data",
            line=dict(color=color, dash=dash)
        ))

    fig.update_layout(
        title="Train, Test, and Predicted Data",
        xaxis_title="Date",
        yaxis_title=target_column,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Return best parameters
    return best_params
def forecast_next_14_days_impExp(df, best_params, target_column):
    # Step 1: Create the future dataframe for the next 14 days
    last_date = last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
    future_df = pd.DataFrame({'Reported Date': future_dates})

    # Concatenate future_df with the original dataframe
    full_df = pd.concat([df, future_df], ignore_index=True)

    # Create forecasting features (ensure this function is defined elsewhere)
    full_df = create_forecasting_features_impExp(full_df, target_column)

    # Step 3: Split data back into original and future sets
    original_df = full_df[full_df['Reported Date'] <= last_date]
    future_df = full_df[full_df['Reported Date'] > last_date]

    # Prepare the training dataset
    X_train = original_df.drop(columns=[target_column, 'Reported Date'], errors='ignore')
    y_train = original_df[target_column]

    # Prepare the dataset for forecasting
    X_future = future_df.drop(columns=[target_column, 'Reported Date'], errors='ignore')

    # Step 4: Train the model with the best parameters on the full dataset
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Step 5: Forecast for the next 14 days
    future_predictions = model.predict(X_future)
    future_df[target_column] = future_predictions

    # Get the test data for plotting
    test_df = original_df[original_df['Reported Date'] >= '2024-01-01']

    # Get max date from predicted data
    max_date = test_df['Reported Date'].max()

    # Filter test data to plot the last 14 days of predictions
    test_last_14_df = test_df[test_df['Reported Date'] > (max_date - pd.Timedelta(days=14))]

    # Predicted data
    predicted_plot_df = test_last_14_df[['Reported Date']].copy()
    predicted_plot_df[target_column] = model.predict(
        test_last_14_df.drop(columns=[target_column, 'Reported Date'], errors='ignore'))
    predicted_plot_df['Type'] = 'Predicted'

    # Forecasted future data
    future_plot_df = future_df[['Reported Date', target_column]].copy()
    future_plot_df['Type'] = 'Forecasted'

    # Concatenate all relevant data
    plot_df = pd.concat([predicted_plot_df, future_plot_df])

    # Plotly Visualization
    fig = go.Figure()

    # Plot predicted data (last 14 days)
    for plot_type, color, dash in [('Predicted', 'green', 'dot'), ('Forecasted', 'red', 'dash')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig.add_trace(go.Scatter(
            x=data['Reported Date'],
            y=data[target_column],
            mode='lines',
            name=f"{plot_type} Data",
            line=dict(color=color, dash=dash)
        ))

    # Update layout
    fig.update_layout(
        title="Last 14 Days of Predictions and Forecasted Next 14 Days",
        xaxis_title="Date",
        yaxis_title=target_column,
        template="plotly_white"
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a success message in Streamlit
    st.success("Forecasting for the next 14 days successfully completed!")

def forecast_modal_price(df, db):
    """
    Forecasts the Modal Price using exogenous variables and SARIMAX.

    Parameters:
    - df: DataFrame containing the data.
    - db: MongoDB database connection to check for existing parameters.
    """
    # MongoDB collections
    impExpParams = db["impExpParams"]
    impExpForecasts = db["impExpForecasts"]

    # Columns to forecast and use as exogenous variables
    columns_to_check = ["QUANTITY_IMPORT", "QUANTITY_EXPORT", "VALUE_IMPORT", "VALUE_EXPORT"]
    missing_columns = []
    params_dict = {}

    # Check if parameters exist in the database
    for column in columns_to_check:
        param_doc = impExpParams.find_one({"target_column": column})
        if param_doc:
            params_dict[column] = param_doc["best_params"]
        else:
            missing_columns.append(column)

    # Train and forecast for missing columns
    for column in missing_columns:
        st.write(f"Training and forecasting for missing column: {column}")
        best_params = train_and_evaluate_impExp(df, column)
        forecast_next_14_days_impExp(df, best_params, column)  # Updates df with predictions
        params_dict[column] = best_params

    # Split data into 75% train and 25% test
    split_index = int(len(df) * 0.75)
    train_endog = df["Modal Price (Rs./Quintal)"][:split_index]
    test_endog = df["Modal Price (Rs./Quintal)"][split_index:]
    train_exog = df[[f"{col}" for col in columns_to_check]][:split_index]
    test_exog = df[[f"{col}" for col in columns_to_check]][split_index:]

    # Extract end date for forecasting
    end_date = df["Reported Date"].max()
    forecast_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=14)

    # SARIMAX grid search for Modal Price
    st.write("Performing SARIMAX grid search for Modal Price...")
    p = d = q = range(0, 3)
    P = D = Q = range(0, 3)
    m = [7]
    param_grid = list(product(p, d, q, P, D, Q, m))

    results_list = []
    progress_bar = st.progress(0)
    total_combinations = len(param_grid)

    for i, params in enumerate(tqdm(param_grid, desc="Grid Search Progress")):
        progress_bar.progress((i + 1) / total_combinations)
        try:
            (p, d, q, P, D, Q, m) = params
            model = SARIMAX(
                train_endog,
                exog=train_exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
            )
            results = model.fit(disp=False)
            predictions = results.get_prediction(start=len(train_endog), end=len(train_endog) + len(test_endog) - 1,
                                                 exog=test_exog).predicted_mean
            rmse = mean_squared_error(test_endog, predictions, squared=False)
            mae = mean_absolute_error(test_endog, predictions)
            results_list.append({'Params': params, 'RMSE': rmse, 'MAE': mae})
        except Exception as e:
            continue

    # Select best parameters based on RMSE
    best_result = sorted(results_list, key=lambda x: x['RMSE'])[0]
    best_params = best_result['Params']
    st.write(f"Best Parameters for SARIMAX: {best_params}")
    st.write(f"Best RMSE: {best_result['RMSE']}")
    st.write(f"Best MAE: {best_result['MAE']}")

    # Save best parameters to MongoDB
    impExpParams.update_one(
        {"target_column": "Modal Price (Rs./Quintal)"},
        {"$set": {"best_params": best_params}},
        upsert=True
    )

    # Train the best SARIMAX model
    (p, d, q, P, D, Q, m) = best_params
    best_model = SARIMAX(
        train_endog,
        exog=train_exog,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
    )
    best_results = best_model.fit(disp=False)

    # Forecast for the next 14 days
    best_forecast = best_results.get_forecast(steps=14, exog=test_exog.iloc[-14:])
    best_forecast_mean = best_forecast.predicted_mean
    best_forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Modal Price': best_forecast_mean.values})

    # Combine data for plotting
    train_plot_df = pd.DataFrame({'Date': df["Reported Date"][:split_index], 'Value': train_endog, 'Type': 'Train'})
    test_plot_df = pd.DataFrame({'Date': df["Reported Date"][split_index:], 'Value': test_endog, 'Type': 'Test'})
    pred_plot_df = pd.DataFrame({'Date': df["Reported Date"][split_index:], 'Value': predictions, 'Type': 'Predicted'})
    forecast_plot_df = pd.DataFrame({'Date': forecast_dates, 'Value': best_forecast_mean, 'Type': 'Forecasted'})

    plot_df = pd.concat([train_plot_df, test_plot_df, pred_plot_df, forecast_plot_df])

    # Prepare data for MongoDB storage
    timestamp = datetime.now().isoformat()
    last_14_days_test = test_endog.tail(14).reset_index()
    forecasted_data = best_forecast_df.reset_index()

    # Structure the document
    forecast_document = {
        "timestamp": timestamp,
        "last_14_days": last_14_days_test[["Reported Date", "Modal Price (Rs./Quintal)"]].to_dict(orient="records"),
        "forecasted": forecasted_data[["Date", "Forecasted Modal Price"]].to_dict(orient="records")
    }

    # Insert the document into MongoDB
    impExpForecasts.insert_one(forecast_document)
    st.success("Last 14 Days and Forecasted Data successfully stored in MongoDB.")

    # Plot 1: Full data (Train, Test, Predicted, Forecasted)
    fig1 = go.Figure()
    for plot_type, color, dash in [('Train', 'blue', None), ('Test', 'orange', None),
                                   ('Predicted', 'green', 'dot'), ('Forecasted', 'purple', 'dash')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig1.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Value'],
            mode='lines',
            name=plot_type,
            line=dict(color=color, dash=dash)
        ))

    fig1.update_layout(
        title="Train, Test, Predicted, and Forecasted Data",
        xaxis_title="Date",
        yaxis_title="Modal Price",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Last 14 days of test, predictions, and next 14 days forecast
    fig2 = go.Figure()
    recent_plot_df = pd.concat([test_plot_df[-14:], pred_plot_df[-14:], forecast_plot_df])

    for plot_type, color, dash in [('Test', 'orange', None), ('Predicted', 'green', 'dot'), ('Forecasted', 'purple', 'dash')]:
        data = recent_plot_df[recent_plot_df['Type'] == plot_type]
        fig2.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Value'],
            mode='lines',
            name=plot_type,
            line=dict(color=color, dash=dash)
        ))

    fig2.update_layout(
        title="Last 14 Days and Forecasted Next 14 Days",
        xaxis_title="Date",
        yaxis_title="Modal Price",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Return the forecasted DataFrame
    return best_forecast_df


def forecast_modal_price_xgboost(df, db):
    """
    Forecasts the Modal Price using exogenous variables and XGBoost.

    Parameters:
    - df: DataFrame containing the data.
    - db: MongoDB database connection to check for existing parameters.
    """
    # MongoDB collections
    impExpParams = db["impExpParams"]
    impExpForecasts = db["impExpForecasts"]

    # Columns to forecast and use as exogenous variables
    columns_to_check = ["QUANTITY_IMPORT", "QUANTITY_EXPORT", "VALUE_IMPORT", "VALUE_EXPORT"]

    # Ensure forecasting features are created
    df = create_forecasting_features(df)

    # Split data into train (75%) and test (25%)
    split_index = int(len(df) * 0.75)
    train_df = df[:split_index]
    test_df = df[split_index:]

    # Prepare the train and test datasets
    X_train = train_df.drop(columns=["Modal Price (Rs./Quintal)", "Reported Date"])
    y_train = train_df["Modal Price (Rs./Quintal)"]
    X_test = test_df.drop(columns=["Modal Price (Rs./Quintal)", "Reported Date"])
    y_test = test_df["Modal Price (Rs./Quintal)"]

    # Hyperparameter tuning with progress bar
    st.write("Performing hyperparameter tuning...")
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.075, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150],
        'booster': ['gbtree', 'dart']
    }
    total_combinations = len(param_grid['learning_rate']) * len(param_grid['max_depth']) * len(param_grid['n_estimators']) * len(param_grid['booster'])
    progress_bar = st.progress(0)
    step = 0

    model = XGBRegressor()
    best_params = None
    best_rmse = float("inf")
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for n_estimators in param_grid['n_estimators']:
                for booster in param_grid['booster']:
                    # Update progress bar
                    step += 1
                    progress_bar.progress(step / total_combinations)

                    # Set parameters
                    params = {
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'booster': booster
                    }
                    model.set_params(**params)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    rmse = mean_squared_error(y_test, predictions, squared=False)

                    # Track the best parameters
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = params

    st.write(f"Best Parameters for XGBoost: {best_params}")
    st.write(f"Best RMSE: {best_rmse}")

    # Save best parameters to MongoDB
    impExpParams.update_one(
        {"target_column": "Modal Price (Rs./Quintal)"},
        {"$set": {"best_params": best_params}},
        upsert=True
    )

    # Train the model with best parameters
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Predictions on the test set
    test_predictions = best_model.predict(X_test)
    test_df["Modal Price (Rs./Quintal)"] = test_predictions  # Overwrite the test values for plotting
    test_df["Type"] = "Test"

    # Prepare future data for forecasting
    last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
    future_df = pd.DataFrame({'Reported Date': future_dates})
    df = pd.concat([df, future_df], ignore_index=True)

    # Create forecasting features
    df = create_forecasting_features(df)

    # Prepare the data for forecasting
    forecast_df = df[df['Reported Date'] > last_date]
    X_forecast = forecast_df.drop(columns=["Modal Price (Rs./Quintal)", "Reported Date"], errors="ignore")

    # Forecast for the next 14 days
    forecast_predictions = best_model.predict(X_forecast)
    forecast_df["Modal Price (Rs./Quintal)"] = forecast_predictions
    forecast_df["Type"] = "Forecasted"

    # Combine data for plotting
    train_df["Type"] = "Train"
    combined_plot_df = pd.concat([train_df, test_df])

    # Plot 1: Train, Test, and Predicted Data
    fig1 = go.Figure()

    for plot_type, color, dash in [('Train', 'blue', None), ('Test', 'orange', None)]:
        data = combined_plot_df[combined_plot_df['Type'] == plot_type]
        fig1.add_trace(go.Scatter(
            x=data['Reported Date'],
            y=data["Modal Price (Rs./Quintal)"],
            mode='lines',
            name=plot_type,
            line=dict(color=color, dash=dash)
        ))

    fig1.update_layout(
        title="Train, Test, and Predicted Data",
        xaxis_title="Date",
        yaxis_title="Modal Price (Rs./Quintal)",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Last 14 Days of Test and Forecasted Data
    last_14_test_df = test_df.tail(14)
    fig2 = go.Figure()

    # Connect last test point with forecast
    connection_df = pd.DataFrame({
        'Reported Date': [last_14_test_df.iloc[-1]['Reported Date'], forecast_df.iloc[0]['Reported Date']],
        'Modal Price (Rs./Quintal)': [last_14_test_df.iloc[-1]["Modal Price (Rs./Quintal)"], forecast_df.iloc[0]["Modal Price (Rs./Quintal)"]],
        'Type': 'Connection'
    })

    # Plot last 14 days of test data
    fig2.add_trace(go.Scatter(
        x=last_14_test_df['Reported Date'],
        y=last_14_test_df["Modal Price (Rs./Quintal)"],
        mode='lines+markers',
        name='Last 14 Days Test',
        line=dict(color='orange')
    ))

    # Plot connection between test and forecasted
    fig2.add_trace(go.Scatter(
        x=connection_df['Reported Date'],
        y=connection_df['Modal Price (Rs./Quintal)'],
        mode='lines',
        name='Connection',
        line=dict(color='orange')
    ))

    # Plot forecasted data
    fig2.add_trace(go.Scatter(
        x=forecast_df['Reported Date'],
        y=forecast_df["Modal Price (Rs./Quintal)"],
        mode='lines+markers',
        name='Forecasted',
        line=dict(color='purple', dash='dash')
    ))

    fig2.update_layout(
        title="Last 14 Days of Test and Forecasted Next 14 Days",
        xaxis_title="Date",
        yaxis_title="Modal Price (Rs./Quintal)",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Save last 14 days and forecasted data to MongoDB
    timestamp = datetime.now().isoformat()
    last_14_days_test = last_14_test_df[["Reported Date", "Modal Price (Rs./Quintal)"]].to_dict(orient="records")
    forecasted_values = forecast_df[["Reported Date", "Modal Price (Rs./Quintal)"]].to_dict(orient="records")

    forecast_document = {
        "timestamp": timestamp,
        "last_14_days": last_14_days_test,
        "forecasted": forecasted_values
    }

    impExpForecasts.insert_one(forecast_document)
    st.success("Last 14 Days and Forecasted Data successfully stored in MongoDB.")

    # Return the forecasted DataFrame
    return forecast_df


def collection_to_dataframe(collection, drop_id=True):
    """
    Converts a MongoDB collection to a pandas DataFrame.

    Args:
        collection: MongoDB collection object.
        drop_id (bool): Whether to drop the '_id' column. Default is True.

    Returns:
        pd.DataFrame: DataFrame containing the collection data.
    """
    # Fetch all documents from the collection
    documents = list(collection.find())

    # Convert to a pandas DataFrame
    df = pd.DataFrame(documents)

    # Drop the MongoDB "_id" column if specified
    if drop_id and '_id' in df.columns:
        df = df.drop(columns=['_id'])

    return df


def display_statistics(df):
    st.title("📊 National Market Statistics Dashboard")
    st.markdown("""
        <style>
            h1 {
                color: #2e7d32;
                font-size: 36px;
                font-weight: bold;
            }
            h3 {
                color: #388e3c;
                font-size: 28px;
                font-weight: 600;
            }
            p {
                font-size: 16px;
                line-height: 1.6;
            }
            .highlight {
                background-color: #f1f8e9;
                padding: 10px;
                border-radius: 8px;
                font-size: 16px;
                color: #2e7d32;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)

    # Ensure 'Reported Date' is in datetime format
    df['Reported Date'] = pd.to_datetime(df['Reported Date'])
    national_data = df.groupby('Reported Date').agg({
        'Modal Price (Rs./Quintal)': 'mean',
        'Arrivals (Tonnes)': 'sum'
    }).reset_index()

    st.subheader("🗓️ Key Statistics")
    latest_date = national_data['Reported Date'].max()
    latest_price = national_data[national_data['Reported Date'] == latest_date]['Modal Price (Rs./Quintal)'].mean()
    latest_arrivals = national_data[national_data['Reported Date'] == latest_date]['Arrivals (Tonnes)'].sum()

    st.markdown("<p class='highlight'>This section provides the most recent statistics for the market. It includes the latest available date, the average price of commodities, and the total quantity of goods arriving at the market. These metrics offer an up-to-date snapshot of market conditions.</p>", unsafe_allow_html=True)
    st.write(f"**Latest Date**: {latest_date.strftime('%Y-%m-%d')}")
    st.write(f"**Latest Modal Price**: {latest_price:.2f} Rs./Quintal")
    st.write(f"**Latest Arrivals**: {latest_arrivals:.2f} Tonnes")

    st.subheader("📆 This Day in Previous Years")
    st.markdown("<p class='highlight'>This table shows the modal price and total arrivals for this exact day across previous years. It provides a historical perspective to compare against current market conditions. This section examines historical data for the same day in previous years. By analyzing trends for this specific day, you can identify seasonal patterns, supply-demand changes, or any deviations that might warrant closer attention.</p>", unsafe_allow_html=True)
    today = latest_date
    previous_years_data = national_data[national_data['Reported Date'].dt.dayofyear == today.dayofyear]

    if not previous_years_data.empty:
        previous_years_data['Year'] = previous_years_data['Reported Date'].dt.year.astype(str)
        display_data = (previous_years_data[['Year', 'Modal Price (Rs./Quintal)', 'Arrivals (Tonnes)']]
                        .sort_values(by='Year', ascending=False)
                        .reset_index(drop=True))
        st.table(display_data)
    else:
        st.write("No historical data available for this day in previous years.")

    st.subheader("📅 Monthly Averages Over Years")
    st.markdown("<p class='highlight'>This section displays the average modal prices and arrivals for each month across all years. It helps identify seasonal trends and peak activity months, which can be crucial for inventory planning and market predictions.</p>", unsafe_allow_html=True)
    national_data['Month'] = national_data['Reported Date'].dt.month
    monthly_avg_price = national_data.groupby('Month')['Modal Price (Rs./Quintal)'].mean().reset_index()
    monthly_avg_arrivals = national_data.groupby('Month')['Arrivals (Tonnes)'].mean().reset_index()
    monthly_avg = pd.merge(monthly_avg_price, monthly_avg_arrivals, on='Month')
    monthly_avg['Month'] = monthly_avg['Month'].apply(lambda x: calendar.month_name[x])
    monthly_avg.columns = ['Month', 'Average Modal Price (Rs./Quintal)', 'Average Arrivals (Tonnes)']
    st.write(monthly_avg)

    st.subheader("📆 Yearly Averages")
    st.markdown("<p class='highlight'>Yearly averages provide insights into long-term trends in pricing and arrivals. By examining these values, you can detect overall growth, stability, or volatility in the market.</p>", unsafe_allow_html=True)
    national_data['Year'] = national_data['Reported Date'].dt.year
    yearly_avg_price = national_data.groupby('Year')['Modal Price (Rs./Quintal)'].mean().reset_index()
    yearly_avg_arrivals = national_data.groupby('Year')['Arrivals (Tonnes)'].mean().reset_index()
    yearly_avg = pd.merge(yearly_avg_price, yearly_avg_arrivals, on='Year')
    yearly_avg['Year'] = yearly_avg['Year'].apply(lambda x: f"{int(x)}")
    yearly_avg.columns = ['Year', 'Average Modal Price (Rs./Quintal)', 'Average Arrivals (Tonnes)']
    st.write(yearly_avg)

    st.subheader("📈 Largest Daily Price Changes (Past Year)")
    st.markdown("<p class='highlight'>This analysis identifies the most significant daily price changes in the past year. These fluctuations can highlight periods of market volatility, potentially caused by external factors like weather, policy changes, or supply chain disruptions.</p>", unsafe_allow_html=True)
    one_year_ago = latest_date - pd.DateOffset(years=1)
    recent_data = national_data[national_data['Reported Date'] >= one_year_ago]
    recent_data['Daily Change (%)'] = recent_data['Modal Price (Rs./Quintal)'].pct_change() * 100
    largest_changes = recent_data[['Reported Date', 'Modal Price (Rs./Quintal)', 'Daily Change (%)']].nlargest(5, 'Daily Change (%)')
    largest_changes['Reported Date'] = largest_changes['Reported Date'].dt.date
    largest_changes = largest_changes.reset_index(drop=True)
    st.write(largest_changes)

    st.subheader("🏆 Top 5 Highest and Lowest Prices (Past Year)")
    st.markdown("<p class='highlight'>This section highlights the highest and lowest prices over the past year. These values reflect the extremes of market dynamics, helping to understand price ceilings and floors in the recent period.</p>", unsafe_allow_html=True)
    highest_prices = recent_data.nlargest(5, 'Modal Price (Rs./Quintal)')[['Reported Date', 'Modal Price (Rs./Quintal)']]
    lowest_prices = recent_data.nsmallest(5, 'Modal Price (Rs./Quintal)')[['Reported Date', 'Modal Price (Rs./Quintal)']]
    highest_prices['Reported Date'] = highest_prices['Reported Date'].dt.date
    lowest_prices['Reported Date'] = lowest_prices['Reported Date'].dt.date
    highest_prices = highest_prices.reset_index(drop=True)
    lowest_prices = lowest_prices.reset_index(drop=True)
    st.write("**Top 5 Highest Prices**")
    st.write(highest_prices)
    st.write("**Top 5 Lowest Prices**")
    st.write(lowest_prices)

    st.subheader("🗂️ Data Snapshot")
    st.markdown("<p class='highlight'>This snapshot provides a concise overview of the latest data, including rolling averages and lagged values. These metrics help identify short-term trends and lagged effects in pricing.</p>", unsafe_allow_html=True)
    national_data['Rolling Mean (14 Days)'] = national_data['Modal Price (Rs./Quintal)'].rolling(window=14).mean()
    national_data['Lag (14 Days)'] = national_data['Modal Price (Rs./Quintal)'].shift(14)
    national_data['Reported Date'] = national_data['Reported Date'].dt.date
    national_data = national_data.sort_values(by='Reported Date', ascending=False)
    st.dataframe(national_data.head(14).reset_index(drop=True), use_container_width=True, height=525)




def fetch_and_store_data():
    latest_doc = collection.find_one(sort=[("Reported Date", -1)])
    if latest_doc and "Reported Date" in latest_doc:
        latest_date = latest_doc["Reported Date"]
    else:
        latest_date = None

    if latest_date:
        from_date = (latest_date + timedelta(days=1)).strftime('%d %b %Y')
    else:
        # If no latest date, set a default from_date
        from_date = "01 Jan 2000"

    to_date = (datetime.now() - timedelta(days=1)).strftime('%d %b %Y')

    # Build the URL to be requested
    base_url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    params = {
        "Tx_Commodity": "11",
        "Tx_State": "0",
        "Tx_District": "0",
        "Tx_Market": "0",
        "DateFrom": from_date,
        "DateTo": to_date,
        "Fr_Date": from_date,
        "To_Date": to_date,
        "Tx_Trend": "2",
        "Tx_CommodityHead": "Sesamum(Sesame,Gingelly,Til)",
        "Tx_StateHead": "--Select--",
        "Tx_DistrictHead": "--Select--",
        "Tx_MarketHead": "--Select--"
    }
    
    full_url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    api_url = "https://api.scraperapi.com"
    api_key = "bbbbde6b56c0fde1e2a61c914eb22d14"
    scraperapi_params = {
        'api_key': api_key,
        'url': full_url
    }

    response = requests.get(api_url, params=scraperapi_params)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find("table", {"class": "tableagmark_new"})

        if table:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []

            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells:
                    rows.append(cells)

            df = pd.DataFrame(rows, columns=headers)
            df = df[df['Variety']=="White"]
            df["Reported Date"] = pd.to_datetime(df["Reported Date"], format='%d %b %Y', errors='coerce')
            df.dropna(subset=["Reported Date"], inplace=True)
            df.sort_values(by="Reported Date", inplace=True)
            df.rename(columns={"State Name": "state"}, inplace=True)

            # Type casting for the columns
            df["Modal Price (Rs./Quintal)"] = pd.to_numeric(df["Modal Price (Rs./Quintal)"], errors='coerce').astype("int64")
            df["Arrivals (Tonnes)"] = pd.to_numeric(df["Arrivals (Tonnes)"], errors='coerce').astype("float64")
            df["state"] = df["state"].astype("string")
            df["Market Name"] = df["Market Name"].astype("string")

            for index, row in df.iterrows():
                document = row.to_dict()
                collection.insert_one(document)

            return df

    else:
        print(f"Failed to fetch data with status code: {response.status_code}")
        return None


    
def get_dataframe_from_collection(collection):
    # Fetch all documents from the collection
    data = list(collection.find())

    # Convert the list of documents into a DataFrame
    df = pd.DataFrame(data)

    # Drop the MongoDB-specific '_id' column (optional, if not needed)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    return df

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and check_password_hash(user['password'], password):
        return True
    return False

# CSS for responsive and professional design
st.markdown("""
    <style>
        /* Main layout adjustments */
        .main { max-width: 1200px; margin: 0 auto; }

        /* Header style */
        h1 { 
            color: #4CAF50; 
            font-family: 'Arial Black', sans-serif; 
        }

        /* Button Styling */
        .stButton>button {
            background-color: #4CAF50; 
            color: white; 
            font-size: 16px;
            border-radius: 12px; 
            padding: 12px 20px; 
            margin: 10px auto;
            border: none;
            cursor: pointer;
            transition: background-color 0.4s ease, transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }

        /* Hover Effects for Button */
        .stButton>button:hover { 
            background-color: #2196F3; /* Change color on hover */
            color: #ffffff; /* Ensure text is readable */
            transform: scale(1.1) rotate(-2deg); /* Slight zoom and tilt */
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.3); /* Enhance shadow effect */
        }

        /* Animation Effect */
        .stButton>button:after {
            content: ''; 
            position: absolute; 
            top: 0; 
            left: 0; 
            right: 0; 
            bottom: 0;
            border-radius: 12px;
            background: linear-gradient(45deg, #4CAF50, #2196F3, #FFC107, #FF5722);
            z-index: -1; /* Ensure gradient stays behind the button */
            opacity: 0;
            transition: opacity 0.5s ease;
        }

        /* Glow Effect on Hover */
        .stButton>button:hover:after {
            opacity: 1;
            animation: glowing 2s infinite alternate;
        }

        /* Keyframes for Glow Animation */
        @keyframes glowing {
            0% { box-shadow: 0 0 5px #4CAF50, 0 0 10px #4CAF50; }
            100% { box-shadow: 0 0 20px #2196F3, 0 0 30px #2196F3; }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .stButton>button { 
                width: 100%; 
                font-size: 14px; 
            }
            h1 { 
                font-size: 24px; 
            }
        }
    </style>
""", unsafe_allow_html=True)
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    st.title("🌾 AgriPredict Dashboard")
    if st.button("Get Live Data Feed"):
        fetch_and_store_data()
    # Top-level radio buttons for switching views
    view_mode = st.radio("", ["Statistics", "Plots", "Predictions"], horizontal=True)

    if view_mode == "Plots":
        st.sidebar.header("Filters")
        selected_period = st.sidebar.selectbox(
            "Select Time Period",
            ["2 Weeks", "1 Month", "3 Months", "1 Year", "5 Years"],
            index=1
        )
        period_mapping = {
            "2 Weeks": 14,
            "1 Month": 30,
            "3 Months": 90,
            "1 Year": 365,
            "5 Years": 1825
        }
        st.session_state.selected_period = period_mapping[selected_period]
        selected_state = st.sidebar.selectbox("Select State", list(state_market_dict.keys()))
        market_wise = st.sidebar.checkbox("Market Wise Analysis")
        if market_wise:
            markets = state_market_dict.get(selected_state, [])
            selected_market = st.sidebar.selectbox("Select Market", markets)
            query_filter = {"state": selected_state, "Market Name": selected_market}
        else:
            query_filter = {"state": selected_state}

        # Dropdown for data type
        data_type = st.sidebar.radio(
            "Select Data Type",
            ["Price", "Volume", "Both"]
        )

        # Add date filtering based on selected period
        query_filter["Reported Date"] = {
            "$gte": datetime.now() - timedelta(days=st.session_state.selected_period)
        }

        # Submit button to trigger the query and plot
        if st.sidebar.button("✨ Let's go!"):
            # Fetch data from MongoDB
            try:
                cursor = collection.find(query_filter)
                data = list(cursor)

                if data:
                    # Convert MongoDB data to a DataFrame
                    df = pd.DataFrame(data)
                    df['Reported Date'] = pd.to_datetime(df['Reported Date'])

                    # Group by Reported Date
                    df_grouped = (
                        df.groupby('Reported Date', as_index=False)
                        .agg({
                            'Arrivals (Tonnes)': 'sum',
                            'Modal Price (Rs./Quintal)': 'mean'
                        })
                    )

                    # Create a complete date range
                    date_range = pd.date_range(
                        start=df_grouped['Reported Date'].min(),
                        end=df_grouped['Reported Date'].max()
                    )
                    df_grouped = df_grouped.set_index('Reported Date').reindex(date_range).rename_axis('Reported Date').reset_index()

                    # Fill missing values
                    df_grouped['Arrivals (Tonnes)'] = df_grouped['Arrivals (Tonnes)'].fillna(method='ffill').fillna(method='bfill')
                    df_grouped['Modal Price (Rs./Quintal)'] = df_grouped['Modal Price (Rs./Quintal)'].fillna(method='ffill').fillna(method='bfill')

                    st.subheader(f"📈 Trends for {selected_state} ({'Market: ' + selected_market if market_wise else 'State'})")

                    if data_type == "Both":
                        # Min-Max Scaling
                        scaler = MinMaxScaler()
                        df_grouped[['Scaled Price', 'Scaled Arrivals']] = scaler.fit_transform(
                            df_grouped[['Modal Price (Rs./Quintal)', 'Arrivals (Tonnes)']]
                        )

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df_grouped['Reported Date'],
                            y=df_grouped['Scaled Price'],
                            mode='lines',
                            name='Scaled Price',
                            line=dict(width=1, color='green'),
                            text=df_grouped['Modal Price (Rs./Quintal)'],
                            hovertemplate='Date: %{x}<br>Scaled Price: %{y:.2f}<br>Actual Price: %{text:.2f}<extra></extra>'
                        ))

                        fig.add_trace(go.Scatter(
                            x=df_grouped['Reported Date'],
                            y=df_grouped['Scaled Arrivals'],
                            mode='lines',
                            name='Scaled Arrivals',
                            line=dict(width=1, color='blue'),
                            text=df_grouped['Arrivals (Tonnes)'],
                            hovertemplate='Date: %{x}<br>Scaled Arrivals: %{y:.2f}<br>Actual Arrivals: %{text:.2f}<extra></extra>'
                        ))

                        fig.update_layout(
                            title="Price and Arrivals Trend",
                            xaxis_title='Date',
                            yaxis_title='Scaled Values',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif data_type == "Price":
                        # Plot Modal Price
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_grouped['Reported Date'],
                            y=df_grouped['Modal Price (Rs./Quintal)'],
                            mode='lines',
                            name='Modal Price',
                            line=dict(width=1, color='green')
                        ))
                        fig.update_layout(title="Modal Price Trend", xaxis_title='Date', yaxis_title='Price', template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

                    elif data_type == "Volume":
                        # Plot Arrivals (Tonnes)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_grouped['Reported Date'],
                            y=df_grouped['Arrivals (Tonnes)'],
                            mode='lines',
                            name='Arrivals',
                            line=dict(width=1, color='blue')
                        ))
                        fig.update_layout(title="Arrivals Trend", xaxis_title='Date', yaxis_title='Volume', template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("⚠️ No data found for the selected filters.")

            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")

    elif view_mode == "Predictions":
        st.subheader("📊 Model Analysis")
        sub_option = st.radio("Select one of the following", ["India", "States", "Market"], horizontal=True)

        if sub_option == "States":
            states = ["Karnataka", "Madhya Pradesh", "Gujarat", "Uttar Pradesh", "Telangana"]
            selected_state = st.selectbox("Select State for Model Training", states)
            filter_key = f"state_{selected_state}"  # Unique key for each state

            if st.button("Forecast"):
                query_filter = {"state": selected_state}
                df = fetch_and_process_data(query_filter)
                forecast(df, filter_key)

        elif sub_option == "Market":
            kharif_rabi_option = st.radio("Select one of the following", ["Kharif", "Rabi"], horizontal=True)

            # Define market options based on Kharif/Rabi selection
            market_options = {
                "Kharif": ["Rajkot", "Neemuch", "Kalburgi"],
                "Rabi": ["Rajkot", "Warangal"]
            }

            selected_market = st.selectbox("Select Market for Model Training", market_options[kharif_rabi_option])
            filter_key = f"market_{selected_market}"  # Unique key for each market
            if st.button("Forecast"):
                query_filter = {"Market Name": selected_market}
                df = fetch_and_process_data(query_filter)
                forecast(df, filter_key)
        elif sub_option == "India":
            df = collection_to_dataframe(impExp)
            if True:
                if st.button("Forecast"):
                    query_filter = {}
                    df = fetch_and_process_data(query_filter)
                    forecast(df, "India")

    elif view_mode=="Statistics":
        document = collection.find_one()
        print(document)
        df = get_dataframe_from_collection(collection)
        print(df)
        display_statistics(df)
else:
    with st.form("login_form"):
        st.subheader("Please log in")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if authenticate_user(username, password):
                st.session_state.authenticated = True  # Set the authentication state to True
                st.session_state['username'] = username  # Store username in session state
                st.write("Login successful!")
                st.rerun()  # Page will automatically rerun to show the protected content
            else:
                st.error("Invalid username or password")
