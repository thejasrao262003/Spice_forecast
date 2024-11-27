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
    if not state or not period:
        return jsonify({"error": "State or period not provided"}), 400
    try:
        file_path = f"Data/{state}_processed_imputed.csv"
        df = pd.read_csv(file_path)

        # Ensure Reported Date is in datetime format
        df['Reported Date'] = pd.to_datetime(df['Reported Date']).dt.date

        # Calculate the date range
        date_range = get_date_range(period)
        last_date = df['Reported Date'].max()
        start_date = last_date - date_range

        # Filter rows within the date range
        filtered_df = df[df['Reported Date'] >= start_date]
        data = filtered_df.to_dict(orient='records')
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": f"File for state {state} not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
