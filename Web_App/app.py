import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import certifi
import json
import os

mongo_uri = st.secrets["MONGO_URI"]

if not mongo_uri:
    st.error("MongoDB URI is not set!")
    st.stop()
else:
    # Connect to MongoDB with SSL certificate validation
    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client["AgriPredict"]
    collection = db["WhiteSesame"]

# CSS to increase the width of the container
st.markdown("""
    <style>
        /* Adjust the width of the main container */
        .main {
            max-width: 1200px;  /* Increase the width */
            margin: 0 auto;  /* Center the container */
        }

        /* Main background */
        body {
            background-color: #f9f9f9;
        }

        /* Title styling */
        h1 {
            color: #4CAF50;
            font-family: 'Arial Black', sans-serif;
        }

        /* Buttons */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 14px;
            border-radius: 8px;
            padding: 10px 20px;
            margin: 5px;
            white-space: nowrap;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Selectbox styling */
        .stSelectbox>div {
            padding: 10px;
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
        }

        /* Checkbox styling */
        .stCheckbox>label {
            font-size: 14px;
            color: #555;
        }

        /* Containers */
        .stContainer {
            border-radius: 12px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Chart area */
        .plotly-graph-div {
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Footer */
        footer {
            font-size: 12px;
            text-align: center;
            color: #888;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 AgriPredict Dashboard")

# Load the state-market dictionary from the JSON file
with open('all_state_market_dict.json', 'r') as file:
    state_market_dict = json.load(file)

# UI for Dashboard
with st.container():
    with st.expander("AgriPredict Dashboard", expanded=True):
        # Adjust the columns to fit more elements within the container
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 3, 3])

        # Buttons for periods
        with col1:
            if st.button('2W', key='2_weeks'):
                st.session_state.selected_period = 14

        with col2:
            if st.button('1M', key='1_month'):
                st.session_state.selected_period = 30

        with col3:
            if st.button('3M', key='3_months'):
                st.session_state.selected_period = 90

        with col4:
            if st.button('1Y', key='1_year'):
                st.session_state.selected_period = 365

        with col5:
            if st.button('5Y', key='5_year'):
                st.session_state.selected_period = 1825

        # Dropdown for states
        with col6:
            states = list(state_market_dict.keys())
            selected_state = st.selectbox(
                "Choose a state",
                states,
                key="state_selectbox",
                index=0
            )

        # Dropdown for selecting between Price, Volume, or Both
        with col7:
            data_type = st.selectbox(
                "Select Data Type",
                ["Price", "Volume", "Both"]
            )

        # Checkbox for market-wise analysis
        st.write("")
        with st.container():
            market_wise = st.checkbox("Market wise", key="market_checkbox")

        if market_wise:
            # Get markets for the selected state
            markets = state_market_dict.get(selected_state, [])
            selected_market = st.selectbox(
                "Choose a market",
                markets,
                key="market_selectbox",
                index=0
            )
            query_filter = {"state": selected_state, "Market Name": selected_market}
        else:
            query_filter = {"state": selected_state}

        # Add date filtering based on selected period
        if 'selected_period' in st.session_state:
            days_period = st.session_state.selected_period
            query_filter["Reported Date"] = {
                "$gte": datetime.now() - timedelta(days=days_period)
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
                df_grouped['Arrivals (Tonnes)'] = df_grouped['Arrivals (Tonnes)'].fillna(
                    method='ffill').fillna(method='bfill')
                df_grouped['Modal Price (Rs./Quintal)'] = df_grouped['Modal Price (Rs./Quintal)'].fillna(
                    method='ffill').fillna(method='bfill')

                st.subheader(f"📈 Trend Graph for {selected_state} ({'Market: ' + selected_market if market_wise else 'State'})")

                if data_type == "Both":
                    # Min-Max Scaling
                    scaler = MinMaxScaler()
                    df_grouped[['Scaled Price', 'Scaled Arrivals']] = scaler.fit_transform(
                        df_grouped[['Modal Price (Rs./Quintal)', 'Arrivals (Tonnes)']]
                    )

                    fig = go.Figure()

                    # Plot Scaled Price with actual values on hover
                    fig.add_trace(go.Scatter(
                        x=df_grouped['Reported Date'],
                        y=df_grouped['Scaled Price'],
                        mode='lines',
                        name='Scaled Price',
                        line=dict(width=1, color='green'),
                        text=df_grouped['Modal Price (Rs./Quintal)'],  # Actual Modal Price values
                        hovertemplate='Date: %{x}<br>Scaled Price: %{y:.2f}<br>Actual Price: %{text:.2f}<extra></extra>'
                    ))

                    # Plot Scaled Arrivals with actual values on hover
                    fig.add_trace(go.Scatter(
                        x=df_grouped['Reported Date'],
                        y=df_grouped['Scaled Arrivals'],
                        mode='lines',
                        name='Scaled Arrivals',
                        line=dict(width=1, color='blue'),
                        text=df_grouped['Arrivals (Tonnes)'],  # Actual Arrivals values
                        hovertemplate='Date: %{x}<br>Scaled Arrivals: %{y:.2f}<br>Actual Arrivals: %{text:.2f}<extra></extra>'
                    ))

                    fig.update_layout(
                        title="Price and Arrivals Trend",
                        xaxis_title='Date',
                        yaxis_title='Scaled Values',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig)

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
                    st.plotly_chart(fig)

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
                    st.plotly_chart(fig)

                else:
                    st.warning("⚠️ No relevant data found for the selected options.")
            else:
                st.warning("⚠️ No data found for the selected filters.")

        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
