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
import certifi
import os
import time
import json
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Load environment variables

mongo_uri = st.secrets["MONGO_URI"]
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

    # Ensure the index is a DatetimeIndex
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
    df['lag14'] = (df.index - pd.Timedelta(days=14)).map(target_map)
    df['lag28'] = (df.index - pd.Timedelta(days=28)).map(target_map)
    df['lag56'] = (df.index - pd.Timedelta(days=56)).map(target_map)
    df['lag_3months'] = (df.index - pd.DateOffset(months=3)).map(target_map)
    df['lag_6months'] = (df.index - pd.DateOffset(months=6)).map(target_map)
    for window in [7, 14, 28]:  # Weekly, bi-weekly, and monthly windows
        df[f'rolling_mean_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['Modal Price (Rs./Quintal)'].rolling(window=window, min_periods=1).std()

    # Exponential moving averages for smoothing recent trends
    df['ema7'] = df['Modal Price (Rs./Quintal)'].ewm(span=7, adjust=False).mean()
    df['ema14'] = df['Modal Price (Rs./Quintal)'].ewm(span=14, adjust=False).mean()
    # Seasonal averages (historical values based on seasonality)
    df['monthly_avg'] = df.groupby('month')['Modal Price (Rs./Quintal)'].transform('mean')
    df['weekly_avg'] = df.groupby('weekofyear')['Modal Price (Rs./Quintal)'].transform('mean')
    df['dayofweek_avg'] = df.groupby('dayofweek')['Modal Price (Rs./Quintal)'].transform('mean')

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
    df['yearly_avg'] = df.groupby('year')['Modal Price (Rs./Quintal)'].transform('mean')

    # Trend Feature
    df['cumulative_mean'] = df['Modal Price (Rs./Quintal)'].expanding().mean()

    # Reset the index to include 'Reported Date' in the output
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
    def remove_outliers(group):
        q1 = group['Modal Price (Rs./Quintal)'].quantile(0.25)
        q3 = group['Modal Price (Rs./Quintal)'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Replace outliers with NaN
        group['Modal Price (Rs./Quintal)'] = group['Modal Price (Rs./Quintal)'].apply(
            lambda x: x if lower_bound <= x <= upper_bound else None
        )
        return group

    # Apply outlier detection in rolling 30-day windows
    df['Month'] = (df.index // 30)  # Group by every 30 days
    df = df.groupby('Month').apply(remove_outliers).reset_index(drop=True)

    # Impute missing values using the mean of forward and backward fill
    df['Modal Price (Rs./Quintal)'] = (
        df['Modal Price (Rs./Quintal)'].fillna(method='ffill').fillna(method='bfill')
    )

    # Drop the temporary 'Month' column
    df.drop(columns=['Month'], inplace=True)

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



def forecast_next_14_days(df, best_params):
    import streamlit as st

    # Step 1: Create the future dataframe for the next 14 days
    last_date = df['Reported Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
    future_df = pd.DataFrame({'Reported Date': future_dates})

    # Concatenate future_df with the original dataframe
    full_df = pd.concat([df, future_df], ignore_index=True)

    full_df = create_forecasting_features(full_df)

    # Step 3: Split data back into original and future sets
    original_df = full_df[full_df['Reported Date'] <= last_date]
    future_df = full_df[full_df['Reported Date'] > last_date]

    # Prepare the training dataset
    X_train = original_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')
    y_train = original_df['Modal Price (Rs./Quintal)']

    # Prepare the dataset for forecasting
    X_future = future_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore')

    # Step 4: Train the model with the best parameters on the full dataset
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Step 5: Forecast for the next 14 days
    future_predictions = model.predict(X_future)
    future_df['Modal Price (Rs./Quintal)'] = future_predictions

    test_df = original_df[original_df['Reported Date'] >= '2024-01-01']

    # Get max date from predicted data
    max_date = test_df['Reported Date'].max()

    # Filter test data to plot the last 14 days of predictions
    test_last_14_df = test_df[test_df['Reported Date'] > (max_date - pd.Timedelta(days=14))]

    # Predicted data
    predicted_plot_df = test_last_14_df[['Reported Date']].copy()
    predicted_plot_df['Modal Price (Rs./Quintal)'] = model.predict(
        test_last_14_df.drop(columns=['Modal Price (Rs./Quintal)', 'Reported Date'], errors='ignore'))
    predicted_plot_df['Type'] = 'Predicted'

    # Forecasted future data
    future_plot_df = future_df[['Reported Date', 'Modal Price (Rs./Quintal)']].copy()
    future_plot_df['Type'] = 'Forecasted'

    # Concatenate all relevant data
    plot_df = pd.concat([predicted_plot_df, future_plot_df])

    fig = go.Figure()

    for plot_type, color, dash in [('Predicted', 'green', 'dot'), ('Forecasted', 'red', 'dash')]:
        data = plot_df[plot_df['Type'] == plot_type]
        fig.add_trace(go.Scatter(
            x=data['Reported Date'],
            y=data['Modal Price (Rs./Quintal)'],
            mode='lines',
            name=f"{plot_type} Data",
            line=dict(color=color, dash=dash)
        ))

    fig.update_layout(
        title="Last 14 Days of Predictions and Forecasted Next 14 Days",
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

if not mongo_uri:
    st.error("MongoDB URI is not set!")
    st.stop()
else:
    # Connect to MongoDB with SSL certificate validation
    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client["AgriPredict"]
    collection = db["WhiteSesame"]
    best_params_collection = db["BestParams"]

# CSS for responsive and professional design
st.markdown("""
    <style>
        .main { max-width: 1200px; margin: 0 auto; }
        h1 { color: #4CAF50; font-family: 'Arial Black', sans-serif; }
        .stButton>button {
            background-color: #4CAF50; color: white; font-size: 16px;
            border-radius: 8px; padding: 12px; margin: 5px;
        }
        .stButton>button:hover { background-color: #45a049; }
        .stSelectbox>div { font-size: 14px; margin-top: 5px; }
        .plotly-graph-div {
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        @media (max-width: 768px) {
            .stButton>button { width: 100%; font-size: 14px; }
            .stSelectbox>div { font-size: 12px; }
            h1 { font-size: 24px; }
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 AgriPredict Dashboard")

# Top-level radio buttons for switching views
view_mode = st.radio("Select View", ["Plots", "Predictions"], horizontal=True)

if view_mode == "Plots":
    st.sidebar.header("Filters")
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        ["2 Weeks", "1 Month", "3 Months", "1 Year", "5 Years"],
        index=1
    )

    # Mapping selected period to days
    period_mapping = {
        "2 Weeks": 14,
        "1 Month": 30,
        "3 Months": 90,
        "1 Year": 365,
        "5 Years": 1825
    }
    st.session_state.selected_period = period_mapping[selected_period]

    # Dropdown for state selection
    selected_state = st.sidebar.selectbox("Select State", list(state_market_dict.keys()))

    # Dropdown for market analysis
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

        if st.button("Train and Forecast"):
            query_filter = {"state": selected_state}
            df = fetch_and_process_data(query_filter)
            train_and_forecast(df, filter_key)

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

        if st.button("Train and Forecast"):
            query_filter = {"Market Name": selected_market}
            df = fetch_and_process_data(query_filter)
            train_and_forecast(df, filter_key)

        if st.button("Forecast"):
            query_filter = {"Market Name": selected_market}
            df = fetch_and_process_data(query_filter)
            forecast(df, filter_key)
