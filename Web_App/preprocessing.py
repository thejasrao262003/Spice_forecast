import pandas as pd

def process_state_data(state):
    import pandas as pd  # Ensure you import pandas

    # Read the CSV file for the given state
    df = pd.read_csv(f'Data/{state}.csv')

    # Filter rows where 'Variety' is 'White'
    df = df[df['Variety'] == 'White']

    # Convert the 'Reported Date' column to datetime format
    df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d %b %Y', errors='coerce')

    # Select relevant columns and ensure they are in the right data type
    df = df[['Reported Date', 'Market Name', 'Arrivals (Tonnes)', 'Modal Price (Rs./Quintal)', 'Variety']]
    df['Arrivals (Tonnes)'] = pd.to_numeric(df['Arrivals (Tonnes)'], errors='coerce')
    df['Modal Price (Rs./Quintal)'] = pd.to_numeric(df['Modal Price (Rs./Quintal)'], errors='coerce')
    df = df.sort_values(by='Reported Date')

    # Save the DataFrame before grouping to a CSV file
    df.to_csv(f'Data/{state}_processed.csv', index=False)  # Updated filename

    # Group by 'Reported Date' and aggregate
    grouped = df.groupby('Reported Date').agg(
        {'Arrivals (Tonnes)': 'sum',
         'Modal Price (Rs./Quintal)': 'mean'}).reset_index()

    return grouped


def check_and_impute(state):
    # Read the processed CSV for the state
    df = pd.read_csv(f'Data/{state}_processed.csv')

    # Convert 'Reported Date' to DateTime format (in case of issues with formatting)
    df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')

    # Calculate the start and end date for the data
    start_date = df['Reported Date'].min()
    end_date = df['Reported Date'].max()

    # Create a complete date range from start to end date
    complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')  # daily frequency

    # Merge the complete date range with the existing data
    df_complete = pd.DataFrame({'Reported Date': complete_dates})

    # Merge the existing data with the complete date range to fill in missing dates
    df = pd.merge(df_complete, df, on='Reported Date', how='left')

    # Impute missing values for 'Arrivals (Tonnes)' and 'Modal Price (Rs./Quintal)'
    df['Arrivals (Tonnes)'] = df['Arrivals (Tonnes)'].fillna(method='bfill').fillna(method='ffill')
    df['Modal Price (Rs./Quintal)'] = df['Modal Price (Rs./Quintal)'].fillna(method='bfill').fillna(method='ffill')

    # Check for missing values in the data columns
    missing_values = df.isnull().sum()

    # Log the results in a text file
    with open('missing_values_report.txt', 'a') as f:
        f.write(f"State: {state}\n")
        f.write(f"Start Date: {start_date}\n")
        f.write(f"End Date: {end_date}\n")
        f.write(f"Total Days: {len(complete_dates)}\n")
        f.write(f"Missing Dates: {missing_values['Reported Date']}\n")
        f.write(f"Missing Values (Arrivals): {missing_values['Arrivals (Tonnes)']}\n")
        f.write(f"Missing Values (Modal Price): {missing_values['Modal Price (Rs./Quintal)']}\n")
        f.write("\n")

    # Save the imputed dataframe to a new file
    df.to_csv(f'Data/{state}_processed_imputed.csv', index=False)

# List of states
states = ["Karnataka", "Uttar Pradesh", "Madhya Pradesh", "Telangana", "Gujarat"]

# Apply the function to each state
for state in states:
    process_state_data(state)  # Process and filter rows with Variety as 'White'
    check_and_impute(state)   # Impute missing data
