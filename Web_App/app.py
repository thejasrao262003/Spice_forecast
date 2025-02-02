import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import calendar
import certifi
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from bs4 import BeautifulSoup
import json
from itertools import product
from tqdm import tqdm
import io
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
    best_params_collection_1m = db["BestParams_1m"]
    best_params_collection_3m = db["BestParams_3m"]
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
def create_forecasting_features(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Reported Date')
        df.index = pd.to_datetime(df.index)

    target_map = df['Modal Price (Rs./Quintal)'].to_dict()

    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    df['lag14'] = (df.index - pd.Timedelta(days=14)).map(target_map)
    df['lag28'] = (df.index - pd.Timedelta(days=28)).map(target_map)
    df['lag56'] = (df.index - pd.Timedelta(days=56)).map(target_map)
    df['lag_3months'] = (df.index - pd.DateOffset(months=3)).map(target_map)
    df['lag_6months'] = (df.index - pd.DateOffset(months=6)).map(target_map)
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).std()

    df['ema7'] = df['Modal Price (Rs./Quintal)'].ewm(span=7, adjust=False).mean()
    df['ema14'] = df['Modal Price (Rs./Quintal)'].ewm(span=14, adjust=False).mean()
    df['monthly_avg'] = df.groupby('month')['Modal Price (Rs./Quintal)'].transform('mean')
    df['weekly_avg'] = df.groupby('weekofyear')['Modal Price (Rs./Quintal)'].transform('mean')
    df['dayofweek_avg'] = df.groupby('dayofweek')['Modal Price (Rs./Quintal)'].transform('mean')

    df['fourier_sin_365'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['fourier_cos_365'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df['fourier_sin_14'] = np.sin(2 * np.pi * df.index.dayofyear / 14)
    df['fourier_cos_14'] = np.cos(2 * np.pi * df.index.dayofyear / 14)

    df['recent_min_14'] = (df.index - pd.Timedelta(days=14)).map(target_map).min()
    df['recent_max_14'] = (df.index - pd.Timedelta(days=14)).map(target_map).max()
    df['recent_range_14'] = df['recent_max_14'] - df['recent_min_14']

    df['yearly_avg'] = df.groupby('year')['Modal Price (Rs./Quintal)'].transform('mean')
    df['cumulative_mean'] = df['Modal Price (Rs./Quintal)'].expanding().mean()

    return df.reset_index()

def create_forecasting_features_1m(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Reported Date')
        df.index = pd.to_datetime(df.index)

    target_map = df['Modal Price (Rs./Quintal)'].to_dict()

    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    df['lag_30'] = (df.index - pd.Timedelta(days=30)).map(target_map)
    df['lag_60'] = (df.index - pd.Timedelta(days=60)).map(target_map)
    df['lag_90'] = (df.index - pd.Timedelta(days=90)).map(target_map)
    df['lag_6months'] = (df.index - pd.DateOffset(months=6)).map(target_map)
    df['lag_12months'] = (df.index - pd.DateOffset(months=12)).map(target_map)

    for window in [30, 60, 90]:
        df[f'rolling_mean_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).std()

    df['ema_30'] = df['Modal Price (Rs./Quintal)'].ewm(span=30, adjust=False).mean()
    df['ema_60'] = df['Modal Price (Rs./Quintal)'].ewm(span=60, adjust=False).mean()

    df['monthly_avg'] = df.groupby('month')['Modal Price (Rs./Quintal)'].transform('mean')
    df['weekly_avg'] = df.groupby('weekofyear')['Modal Price (Rs./Quintal)'].transform('mean')
    df['dayofweek_avg'] = df.groupby('dayofweek')['Modal Price (Rs./Quintal)'].transform('mean')

    df['fourier_sin_365'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['fourier_cos_365'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df['fourier_sin_30'] = np.sin(2 * np.pi * df.index.dayofyear / 30)
    df['fourier_cos_30'] = np.cos(2 * np.pi * df.index.dayofyear / 30)

    df['recent_min_30'] = (df.index - pd.Timedelta(days=30)).map(target_map).min()
    df['recent_max_30'] = (df.index - pd.Timedelta(days=30)).map(target_map).max()
    df['recent_range_30'] = df['recent_max_30'] - df['recent_min_30']

    df['yearly_avg'] = df.groupby('year')['Modal Price (Rs./Quintal)'].transform('mean')
    df['cumulative_mean'] = df['Modal Price (Rs./Quintal)'].expanding().mean()

    return df.reset_index()

def create_forecasting_features_3m(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Reported Date')
        df.index = pd.to_datetime(df.index)

    target_map = df['Modal Price (Rs./Quintal)'].to_dict()

    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    df['lag_3months'] = (df.index - pd.DateOffset(months=3)).map(target_map)
    df['lag_6months'] = (df.index - pd.DateOffset(months=6)).map(target_map)
    df['lag_9months'] = (df.index - pd.DateOffset(months=9)).map(target_map)
    df['lag_12months'] = (df.index - pd.DateOffset(months=12)).map(target_map)

    for window in [90, 180, 270, 365]:
        df[f'rolling_mean_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).std()

    df['ema90'] = df['Modal Price (Rs./Quintal)'].ewm(span=90, adjust=False).mean()
    df['ema180'] = df['Modal Price (Rs./Quintal)'].ewm(span=180, adjust=False).mean()
    df['monthly_avg'] = df.groupby('month')['Modal Price (Rs./Quintal)'].transform('mean')
    df['weekly_avg'] = df.groupby('weekofyear')['Modal Price (Rs./Quintal)'].transform('mean')
    df['dayofweek_avg'] = df.groupby('dayofweek')['Modal Price (Rs./Quintal)'].transform('mean')

    df['fourier_sin_90'] = np.sin(2 * np.pi * df.index.dayofyear / 90)
    df['fourier_cos_90'] = np.cos(2 * np.pi * df.index.dayofyear / 90)
    df['fourier_sin_30'] = np.sin(2 * np.pi * df.index.dayofyear / 30)
    df['fourier_cos_30'] = np.cos(2 * np.pi * df.index.dayofyear / 30)

    df['recent_min_90'] = (df.index - pd.Timedelta(days=90)).map(target_map).min()
    df['recent_max_90'] = (df.index - pd.Timedelta(days=90)).map(target_map).max()
    df['recent_range_90'] = df['recent_max_90'] - df['recent_min_90']

    df['yearly_avg'] = df.groupby('year')['Modal Price (Rs./Quintal)'].transform('mean')
    df['cumulative_mean'] = df['Modal Price (Rs./Quintal)'].expanding().mean()

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

def train_and_evaluate_1m(df):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Add progress bar for hyperparameter tuning
    progress_bar = st.progress(0)

    # Helper function to update progress during hyperparameter tuning
    def update_tuning_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress)
    
    df = create_forecasting_features_1m(df)
    
    # Define train-test split for a 1-month horizon
    split_date = pd.to_datetime("2024-01-01")
    test_horizon = pd.DateOffset(days=30)  # 1-month horizon
    
    train_df = df[df['Reported Date'] < split_date]
    test_df = df[(df['Reported Date'] >= split_date) & (df['Reported Date'] < split_date + test_horizon)]

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

def train_and_evaluate_3m(df):
    import streamlit as st

    # Add progress bar for hyperparameter tuning
    progress_bar = st.progress(0)

    # Helper function to update progress during hyperparameter tuning
    def update_tuning_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress)

    df = create_forecasting_features_3m(df)
    train_df = df[df['Reported Date'] < '2023-10-01']
    test_df = df[df['Reported Date'] >= '2023-10-01']

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

def forecast_next_14_days(df, _best_params, key):
    last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
    future_df = pd.DataFrame({'Reported Date': future_dates})
    
    # Assuming 'create_forecasting_features' function is defined elsewhere
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = create_forecasting_features(full_df)

    original_df = full_df[full_df['Reported Date'] <= last_date]
    future_df = full_df[full_df['Reported Date'] > last_date]

    X_train = original_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')
    y_train = original_df['Modal Price (Rs./Quintal)']
    X_future = future_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')

    model = XGBRegressor(**_best_params)
    model.fit(X_train, y_train)

    future_predictions = model.predict(X_future)
    future_df['Modal Price (Rs./Quintal)'] = future_predictions

    # Pass model to plot_data
    plot_data(original_df, future_df, last_date, model, 14)
    download_button(future_df, key)
    
def forecast_next_30_days(df, _best_params, key):
    last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({'Reported Date': future_dates})
    
    # Assuming 'create_forecasting_features' function is defined elsewhere
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = create_forecasting_features_1m(full_df)

    original_df = full_df[full_df['Reported Date'] <= last_date]
    future_df = full_df[full_df['Reported Date'] > last_date]

    X_train = original_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')
    y_train = original_df['Modal Price (Rs./Quintal)']
    X_future = future_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')

    model = XGBRegressor(**_best_params)
    model.fit(X_train, y_train)

    future_predictions = model.predict(X_future)
    future_df['Modal Price (Rs./Quintal)'] = future_predictions

    # Pass model to plot_data
    plot_data(original_df, future_df, last_date, model, 30)
    download_button(future_df, key)

def forecast_next_90_days(df, _best_params, key):
    last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90)
    future_df = pd.DataFrame({'Reported Date': future_dates})
    
    # Assuming 'create_forecasting_features' function is defined elsewhere
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = create_forecasting_features_3m(full_df)

    original_df = full_df[full_df['Reported Date'] <= last_date]
    future_df = full_df[full_df['Reported Date'] > last_date]

    X_train = original_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')
    y_train = original_df['Modal Price (Rs./Quintal)']
    X_future = future_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')

    model = XGBRegressor(**_best_params)
    model.fit(X_train, y_train)

    future_predictions = model.predict(X_future)
    future_df['Modal Price (Rs./Quintal)'] = future_predictions

    # Pass model to plot_data
    plot_data(original_df, future_df, last_date, model, 90)
    download_button(future_df, key)

def plot_data(original_df, future_df, last_date, model, days):
    actual_last_df = original_df[original_df['Reported Date'] > (last_date - pd.Timedelta(days=days))]
    predicted_plot_df = actual_last_df[['Reported Date']].copy()
    predicted_plot_df['Modal Price (Rs./Quintal)'] = model.predict(
        actual_last_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore'))
    predicted_plot_df['Type'] = 'Actual'

    future_plot_df = future_df[['Reported Date', 'Modal Price (Rs./Quintal)']].copy()
    future_plot_df['Type'] = 'Forecasted'
    last_actual_point = predicted_plot_df.iloc[[-1]].copy()
    last_actual_point['Type'] = 'Forecasted'
    future_plot_df = pd.concat([last_actual_point, future_plot_df])
    plot_df = pd.concat([predicted_plot_df, future_plot_df])

    fig = go.Figure()
    for plot_type, color, dash in [('Actual', 'blue', 'solid'), ('Forecasted', 'red', 'dash')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig.add_trace(go.Scatter(x=data['Reported Date'], y=data['Modal Price (Rs./Quintal)'], mode='lines', name=f"{plot_type} Data", line=dict(color=color, dash=dash)))
    fig.update_layout(title="Actual vs Forecasted Modal Price (Rs./Quintal)", xaxis_title="Date", yaxis_title="Modal Price (Rs./Quintal)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def download_button(future_df, key):
    # Create a new DataFrame with only 'Reported Date' and 'Modal Price (Rs./Quintal)'
    download_df = future_df[['Reported Date', 'Modal Price (Rs./Quintal)']].copy()

    # Format 'Reported Date' to display only the date in YYYY-MM-DD format
    download_df['Reported Date'] = download_df['Reported Date'].dt.strftime('%Y-%m-%d')

    # Write to Excel without the index
    towrite = io.BytesIO()
    download_df.to_excel(towrite, index=False, engine='xlsxwriter')  # Using 'xlsxwriter' for the Excel engine
    towrite.seek(0)

    # Create a download button for the Excel file
    st.download_button(label="Download Forecasted Values",
                       data=towrite,
                       file_name=f'forecasted_prices_{key}.xlsx',
                       mime='application/vnd.ms-excel')



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

def save_best_params(collection, filter_key, best_params):
    best_params["filter_key"] = filter_key
    best_params["last_updated"] = datetime.now().isoformat()
    
    existing_entry = collection.find_one({"filter_key": filter_key})
    if existing_entry:
        collection.replace_one({"filter_key": filter_key}, best_params)
    else:
        collection.insert_one(best_params)

# Function to retrieve best_params from MongoDB
def get_best_params(filter_key, collection):
    record = collection.find_one({"filter_key": filter_key})
    return record
# Function to handle training and forecasting
def train_and_forecast(df, filter_key, days):
    if df is not None:
        # Train the model and save parameters to MongoDB
        if days==14:
            best_params = train_and_evaluate(df)
            save_best_params(filter_key, best_params, best_params_collection)
            forecast_next_14_days(df, best_params, filter_key)
        elif days==30:
            best_params = train_and_evaluate_1m(df)
            save_best_params(filter_key, best_params, best_params_collection_1m)
            forecast_next_30_days(df, best_params, filter_key)
        elif days==90:
            best_params = train_and_evaluate_3m(df)
            save_best_params(filter_key, best_params, best_params_collection_3m)
            forecast_next_90_days(df, best_params, filter_key)

def forecast(df, filter_key, days):
    if days==14:
        record = get_best_params(filter_key, best_params_collection)
        if record:
            st.info(f"ℹ️ The model was trained on {record['last_updated']}.")
            forecast_next_14_days(df, record, filter_key)
        else:
            st.warning("⚠️ Model is not trained yet. Please train the model first.")
    if days==30:
        record = get_best_params(filter_key, best_params_collection_1m)
        if record:
            st.info(f"ℹ️ The model was trained on {record['last_updated']}.")
            forecast_next_30_days(df, record, filter_key)
        else:
            st.warning("⚠️ Model is not trained yet. Please train the model first.")
    if days==90:
        record = get_best_params(filter_key, best_params_collection_3m)
        if record:
            st.info(f"ℹ️ The model was trained on {record['last_updated']}.")
            forecast_next_90_days(df, record, filter_key)
        else:
            st.warning("⚠️ Model is not trained yet. Please train the model first.")
            
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



def editable_spreadsheet():
    st.title("Sowing Report Prediction Model")

    # Excel file uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

    # Check if an Excel file is uploaded
    if uploaded_file is not None:
        # Read the Excel file
        df_excel = pd.read_excel(uploaded_file)
        
        # Display the DataFrame from the Excel file
        st.write("Excel data loaded:", df_excel)

        # Form for inputting filtering options and area for calculation
        with st.form("input_form"):
            input_region = st.text_input("Enter Region to Filter By", placeholder="Region Name")
            input_season = st.text_input("Enter Season to Filter By", placeholder="Season (e.g., Winter)")
            input_area = st.number_input("Enter Area (in hectares) for Production Calculation", min_value=0.0, format="%.2f")
            submit_button = st.form_submit_button("Calculate Production")

        if submit_button:
            if input_region and input_season and input_area > 0:
                # Filter data by the region and season specified
                filtered_df = df_excel[
                    (df_excel['Region'].str.lower() == input_region.lower()) &
                    (df_excel['Season'].str.lower() == input_season.lower())
                ]

                if not filtered_df.empty:
                    process_dataframe(filtered_df, input_area)
                else:
                    st.error("No data found for the specified region and season.")
            else:
                st.error("Please enter valid region, season, and area to proceed.")

def process_dataframe(df, area):
    if 'Yield' in df.columns:
        average_yield = df['Yield'].mean()
        predicted_production = average_yield * area
        st.success(f"The predicted Production Volume for the specified region and season is: {predicted_production:.2f} units")
    else:
        st.error("The DataFrame does not contain a necessary 'Yield' column for calculation.")



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
    yearly_sum_arrivals = national_data.groupby('Year')['Arrivals (Tonnes)'].sum().reset_index()
    yearly_avg = pd.merge(yearly_avg_price, yearly_sum_arrivals, on='Year')
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

    editable_spreadsheet()



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
    view_mode = st.radio("", ["Statistics", "Plots", "Predictions", "Exim"], horizontal=True)

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
            "2 Years": 730,
            "5 Years": 1825
        }
        st.session_state.selected_period = period_mapping[selected_period]
        
        # Add 'India' option to the list of states
        state_options = list(state_market_dict.keys()) + ['India']
        selected_state = st.sidebar.selectbox("Select", state_options)
        
        market_wise = False
        if selected_state != 'India':
            market_wise = st.sidebar.checkbox("Market Wise Analysis")
            if market_wise:
                markets = state_market_dict.get(selected_state, [])
                selected_market = st.sidebar.selectbox("Select Market", markets)
                query_filter = {"state": selected_state, "Market Name": selected_market}
            else:
                query_filter = {"state": selected_state}
        else:
            query_filter = {}  # For India, no specific state filter
        
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
        
                    if selected_state == 'India':
                        # Aggregate data for all of India
                        df_grouped = df.groupby('Reported Date', as_index=False).agg({
                            'Arrivals (Tonnes)': 'sum',
                            'Modal Price (Rs./Quintal)': 'mean'
                        })
                    else:
                        # Regular grouping by Reported Date
                        df_grouped = df.groupby('Reported Date', as_index=False).agg({
                            'Arrivals (Tonnes)': 'sum',
                            'Modal Price (Rs./Quintal)': 'mean'
                        })
        
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
                        fig.update_layout(title="Modal Price Trend", xaxis_title='Date', yaxis_title='Price (/Quintall)', template='plotly_white')
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
                        fig.update_layout(title="Arrivals Trend", xaxis_title='Date', yaxis_title='Volume (in Tonnes)', template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
        
                else:
                    st.warning("⚠️ No data found for the selected filters.")
        
            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")
    elif view_mode == "Predictions":
        st.subheader("📊 Model Analysis")
        sub_option = st.radio("Select one of the following", ["India", "States", "Market"], horizontal=True)
        sub_timeline = st.radio("Select one of the following horizons", ["14 days", "1 month", "3 month"], horizontal=True)
        if sub_option == "States":
            states = ["Karnataka", "Madhya Pradesh", "Gujarat", "Uttar Pradesh", "Telangana"]
            selected_state = st.selectbox("Select State for Model Training", states)
            filter_key = f"state_{selected_state}"  # Unique key for each state

            if st.button("Forecast"):
                query_filter = {"state": selected_state}
                df = fetch_and_process_data(query_filter)
                if sub_timeline == "14 days":
                    forecast(df, filter_key, 14)
                elif sub_timeline == "1 month":
                    forecast(df, filter_key, 30)
                else:
                    forecast(df, filter_key, 90)
        elif sub_option == "Market":
            market_options = ["Rajkot", "Neemuch", "Kalburgi", "Warangal"]
            selected_market = st.selectbox("Select Market for Model Training", market_options)
            filter_key = f"market_{selected_market}"  # Unique key for each market
            if st.button("Forecast"):
                query_filter = {"Market Name": selected_market}
                df = fetch_and_process_data(query_filter)
                if sub_timeline == "14 days":
                    forecast(df, filter_key, 14)
                elif sub_timeline == "1 month":
                    forecast(df, filter_key, 30)
                else:
                    forecast(df, filter_key, 90)
        
        elif sub_option == "India":
            df = collection_to_dataframe(impExp)
            if True:
                if st.button("Forecast"):
                    query_filter = {}
                    df = fetch_and_process_data(query_filter)
                    if sub_timeline == "14 days":
                        forecast(df, "India", 14)
                    elif sub_timeline == "1 month":
                        forecast(df, "India", 30)
                    else:
                        forecast(df, "India", 90)

    elif view_mode=="Statistics":
        document = collection.find_one()
        print(document)
        df = get_dataframe_from_collection(collection)
        print(df)
        display_statistics(df)
    elif view_mode == "Exim":
        df = collection_to_dataframe(impExp)
    
        # Add radio buttons for user selection
        plot_option = st.radio(
            "Select the data to visualize:",
            ["Import Price", "Import Quantity", "Export Price", "Export Quantity"],
            horizontal=True
        )
    
        # Dropdown for time period selection
        time_period = st.selectbox(
            "Select time period:",
            ["1 Month", "6 Months", "1 Year", "2 Years"]
        )
    
        # Convert Reported Date to datetime
        df["Reported Date"] = pd.to_datetime(df["Reported Date"], format="%Y-%m-%d")
    
        # Filter data based on the time period
        if time_period == "1 Month":
            start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
        elif time_period == "6 Months":
            start_date = pd.Timestamp.now() - pd.DateOffset(months=6)
        elif time_period == "1 Year":
            start_date = pd.Timestamp.now() - pd.DateOffset(years=1)
        elif time_period == "2 Years":
            start_date = pd.Timestamp.now() - pd.DateOffset(years=2)
    
        filtered_df = df[df["Reported Date"] >= start_date]
    
        # Process data based on the selected option
        if plot_option == "Import Price":
            grouped_df = (
                filtered_df.groupby("Reported Date", as_index=False)["VALUE_IMPORT"]
                .mean()
                .rename(columns={"VALUE_IMPORT": "Average Import Price"})
            )
            y_axis_label = "Average Import Price (Rs.)"
        elif plot_option == "Import Quantity":
            grouped_df = (
                filtered_df.groupby("Reported Date", as_index=False)["QUANTITY_IMPORT"]
                .sum()
                .rename(columns={"QUANTITY_IMPORT": "Total Import Quantity"})
            )
            y_axis_label = "Total Import Quantity (Tonnes)"
        elif plot_option == "Export Price":
            grouped_df = (
                filtered_df.groupby("Reported Date", as_index=False)["VALUE_EXPORT"]
                .mean()
                .rename(columns={"VALUE_EXPORT": "Average Export Price"})
            )
            y_axis_label = "Average Export Price (Rs.)"
        elif plot_option == "Export Quantity":
            grouped_df = (
                filtered_df.groupby("Reported Date", as_index=False)["QUANTITY_IMPORT"]
                .sum()
                .rename(columns={"QUANTITY_IMPORT": "Total Export Quantity"})
            )
            y_axis_label = "Total Export Quantity (Tonnes)"
    
        # Plot using Plotly
        fig = px.line(
            grouped_df,
            x="Reported Date",
            y=grouped_df.columns[1],  # Dynamic y-axis column name
            title=f"{plot_option} Over Time",
            labels={"Reported Date": "Date", grouped_df.columns[1]: y_axis_label},
        )
        st.plotly_chart(fig)
        
        
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
