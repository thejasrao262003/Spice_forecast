from flask import Flask, request, jsonify
import pandas as pd
from datetime import timedelta

app = Flask(__name__)

def get_date_range(period):
    if period == '2 Weeks':
        return timedelta(weeks=2)
    elif period == '1 Month':
        return timedelta(days=30)
    elif period == '3 Months':
        return timedelta(days=90)
    elif period == '1 Year':
        return timedelta(days=365)
    elif period == '5 Year':
        return timedelta(days=365*5)
    else:
        return timedelta(days=0)

@app.route('/get_data', methods=['GET'])
def get_data():
    state = request.args.get('state')
    period = request.args.get('period')
    market = request.args.get('market')  # Market parameter (optional)
    print(state, period, market)

    if not state or not period:
        return jsonify({"error": "State or period not provided"}), 400

    try:
        file_path = f"Data/{state}_processed.csv"
        df = pd.read_csv(file_path)

        # Ensure Reported Date is in datetime format
        df['Reported Date'] = pd.to_datetime(df['Reported Date']).dt.date

        # Calculate the date range
        date_range = get_date_range(period)
        last_date = df['Reported Date'].max()
        start_date = last_date - date_range

        # Filter rows within the date range
        filtered_df = df[df['Reported Date'] >= start_date]

        # Handle market filter if provided
        if market:
            filtered_df = filtered_df[filtered_df['Market Name'] == market]

        # Group data by Reported Date, summing 'Arrivals (Tonnes)' and averaging 'Modal Price (Rs./Quintal)'
        grouped_df = filtered_df.groupby('Reported Date').agg(
            Arrivals_sum=('Arrivals (Tonnes)', 'sum'),
            Modal_Price_avg=('Modal Price (Rs./Quintal)', 'mean')
        ).reset_index()

        # Rename columns to the original names
        grouped_df = grouped_df.rename(columns={
            'Arrivals_sum': 'Arrivals (Tonnes)',
            'Modal_Price_avg': 'Modal Price (Rs./Quintal)'
        })

        # Ensure 'Reported Date' is in datetime64 format before merging
        grouped_df['Reported Date'] = pd.to_datetime(grouped_df['Reported Date'])

        # Create a range of all dates between start_date and last_date
        all_dates = pd.date_range(start=start_date, end=last_date, freq='D')
        all_dates_df = pd.DataFrame(all_dates, columns=['Reported Date'])

        # Ensure 'Reported Date' in all_dates_df is of datetime64 type
        all_dates_df['Reported Date'] = pd.to_datetime(all_dates_df['Reported Date'])

        # Merge with the grouped data
        result_df = pd.merge(all_dates_df, grouped_df, on='Reported Date', how='left')

        # Impute missing values with forward fill and backward fill
        result_df['Arrivals (Tonnes)'] = result_df['Arrivals (Tonnes)'].fillna(method='ffill').fillna(method='bfill')
        result_df['Modal Price (Rs./Quintal)'] = result_df['Modal Price (Rs./Quintal)'].fillna(method='ffill').fillna(method='bfill')

        # Return only the necessary columns
        result_df = result_df[['Reported Date', 'Arrivals (Tonnes)', 'Modal Price (Rs./Quintal)']]

        # Convert the dataframe to JSON and return it
        data = result_df.to_dict(orient='records')
        return jsonify(data)

    except FileNotFoundError:
        return jsonify({"error": f"File for state {state} not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
