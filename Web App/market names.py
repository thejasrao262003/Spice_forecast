import os
import pandas as pd
import json

# List of states to process
states = ["Karnataka", "Gujarat", "Uttar Pradesh", "Madhya Pradesh", "Telangana"]

# Directory containing the CSV files
data_directory = 'Data/'

# Dictionary to hold the final output for all states
all_state_market_dict = {}

# Iterate through each state in the list
for state_input in states:
    # File name for the specific state
    file_name = f"{state_input}_processed.csv"
    file_path = os.path.join(data_directory, file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Ensure the "Market Name" column exists in the DataFrame
        if 'Market Name' in df.columns:
            # Get the unique list of values in the "Market Name" column
            unique_markets = df['Market Name'].unique().tolist()

            # Add the state's market list to the dictionary
            all_state_market_dict[state_input] = unique_markets
        else:
            print(f"'Market Name' column not found in {file_name}.")
    else:
        print(f"File for {state_input} not found in the directory.")

# Write the final dictionary to a JSON file
with open('all_state_market_dict.json', 'w') as json_file:
    json.dump(all_state_market_dict, json_file, indent=4)

print("JSON file has been created with the state-market dictionary for all states.")
