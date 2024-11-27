import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.title("AgriPredict")
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            font-size: 14px;
            margin: 4px;
            border: none;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox>div {
            padding: 10px;
            background-color: #f4f4f9;
            border-radius: 8px;
            font-size: 14px;
        }
        .stTitle {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .stSubheader {
            font-size: 18px;
            font-weight: 500;
            color: #333;
        }
        .stContainer {
            margin-bottom: 20px;
        }
        .stBox {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }
        .css-1s2u09g {
            width: 300px;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    with st.expander("AgriPredict Dashboard", expanded=True):
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 2, 2])

        # Buttons for periods
        with col1:
            if st.button('2W', key='2_weeks'):
                st.session_state.selected_period = '2 Weeks'

        with col2:
            if st.button('1M', key='1_month'):
                st.session_state.selected_period = '1 Month'

        with col3:
            if st.button('3M', key='3_months'):
                st.session_state.selected_period = '3 Months'

        with col4:
            if st.button('1Y', key='1_year'):
                st.session_state.selected_period = '1 Year'

        with col5:
            if st.button('5Y', key='5_year'):
                st.session_state.selected_period = '5 Year'

        # State dropdown in the 6th column
        with col6:
            states = ["Karnataka", "Uttar Pradesh", "Madhya Pradesh", "Telangana", "Gujarat"]
            selected_state = st.selectbox("Choose a state", states, key="state_selectbox", index=0, help="Select a state for trend analysis.")

        # Dropdown for selecting between Price, Volume, or Both
        with col7:
            data_type = st.selectbox("Select Data Type", ["Price", "Volume", "Both"], help="Choose the data type you want to analyze.")

        st.subheader("Trend Graph")

        if 'selected_period' in st.session_state:
            period = st.session_state.selected_period
            st.write(f"Displaying trend for {selected_state} over {period}.")
            url = f"http://localhost:5000/get_data?state={selected_state}&period={period}"

            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                df = pd.DataFrame(data)

                # Ensure the 'Reported Date' column is in datetime format and extract the date part only
                df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce').dt.date

                if data_type == "Price" and 'Modal Price (Rs./Quintal)' in df.columns:
                    # Plot Modal Price
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df['Reported Date'],
                        y=df['Modal Price (Rs./Quintal)'],
                        mode='lines',
                        name=f'{selected_state} - {period} Modal Price',
                        text=df['Modal Price (Rs./Quintal)'],
                        hoverinfo='text+x+y',
                        line=dict(width=1)
                    ))

                    fig.update_layout(
                        title=f"{selected_state} - {period} Modal Price Trend",
                        xaxis_title='Date',
                        yaxis_title='Modal Price',
                        hovermode='closest'
                    )

                    st.plotly_chart(fig)

                elif data_type == "Volume" and 'Arrivals (Tonnes)' in df.columns:
                    # Plot Arrivals (Tonnes)
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df['Reported Date'],
                        y=df['Arrivals (Tonnes)'],
                        mode='lines',
                        name=f'{selected_state} - {period} Arrivals (Tonnes)',
                        text=df['Arrivals (Tonnes)'],
                        hoverinfo='text+x+y',
                        line=dict(width=1)
                    ))

                    fig.update_layout(
                        title=f"{selected_state} - {period} Arrivals (Tonnes) Trend",
                        xaxis_title='Date',
                        yaxis_title='Arrivals (Tonnes)',
                        hovermode='closest'
                    )

                    st.plotly_chart(fig)

                elif data_type == "Both":
                    if 'Modal Price (Rs./Quintal)' in df.columns and 'Arrivals (Tonnes)' in df.columns:
                        # Min-Max Scaling for both columns
                        scaler = MinMaxScaler()
                        scaled_values = scaler.fit_transform(df[['Modal Price (Rs./Quintal)', 'Arrivals (Tonnes)']])
                        df['Scaled Price'] = scaled_values[:, 0]
                        df['Scaled Arrivals'] = scaled_values[:, 1]

                        # Plot both scaled values on the same graph
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df['Reported Date'],
                            y=df['Scaled Price'],
                            mode='lines',
                            name=f'{selected_state} - {period} Scaled Modal Price',
                            text=df['Modal Price (Rs./Quintal)'],
                            hoverinfo='text+x+y',
                            line=dict(width=1, color='blue')
                        ))

                        fig.add_trace(go.Scatter(
                            x=df['Reported Date'],
                            y=df['Scaled Arrivals'],
                            mode='lines',
                            name=f'{selected_state} - {period} Scaled Arrivals',
                            text=df['Arrivals (Tonnes)'],
                            hoverinfo='text+x+y',
                            line=dict(width=1, color='green')
                        ))

                        fig.update_layout(
                            title=f"{selected_state} - {period} Scaled Modal Price and Arrivals Trend",
                            xaxis_title='Date',
                            yaxis_title='Scaled Values (0 to 1)',
                            hovermode='closest',
                            legend=dict(x=0, y=1, traceorder='normal'),
                        )

                        st.plotly_chart(fig)

                else:
                    st.write("Selected column(s) not found in the data.")

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data from Flask: {e}")
