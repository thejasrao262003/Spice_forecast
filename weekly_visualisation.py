import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Load the cleaned CSV file
df_cleaned = pd.read_csv('Data/df_simple.csv')

# Ensure 'Price Date' is in datetime format
df_cleaned['Price Date'] = pd.to_datetime(df_cleaned['Price Date'])

# Set 'Price Date' as the index
df_cleaned.set_index('Price Date', inplace=True)

# Create the directory if it does not exist
output_directory = 'Weekly_Graph_Variety_Wise'
os.makedirs(output_directory, exist_ok=True)

# Get unique varieties
varieties = df_cleaned['Variety'].unique()

# Function to sanitize filenames
def sanitize_filename(name):
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# Loop through each variety and plot its modal prices
for variety in varieties:
    # Filter the DataFrame for the current variety
    variety_data = df_cleaned[df_cleaned['Variety'] == variety]

    # Further aggregate by Grade if necessary (choosing the first Grade for simplicity)
    if not variety_data.empty:
        # Resample the data weekly and calculate the mean modal price for each week
        weekly_data = variety_data.resample('W').mean(numeric_only=True)

        # Create a new figure for each variety
        plt.figure(figsize=(12, 6))

        # Plot the data if there are any valid entries
        if not weekly_data.empty:
            plt.plot(weekly_data.index, weekly_data['Modal Price (Rs./Quintal)'], marker='o', linestyle='-', label=variety)

            # Adding title and labels
            plt.title(f'Weekly Modal Prices for {variety}')
            plt.xlabel('Weeks')
            plt.ylabel('Modal Price (Rs./Quintal)')
            plt.xticks(rotation=45)
            plt.legend(title='Variety')
            plt.grid()
            plt.tight_layout()

            # Sanitize the variety name for the filename
            safe_variety_name = sanitize_filename(variety)

            # Save the plot in the specified directory
            plt.savefig(os.path.join(output_directory, f'weekly_modal_price_{safe_variety_name}.png'))
            plt.close()  # Close the figure to free memory
        else:
            print(f"No data available for {variety}.")

print(f"Plots saved in directory: {output_directory}")
