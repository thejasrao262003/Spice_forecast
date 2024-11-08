import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the CSV file and select relevant columns
df = pd.read_csv('Data/White_Variety.csv')
df = df[["Price Date", "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)", "Modal Price (Rs./Quintal)"]]

# Convert 'Price Date' to datetime
df['Price Date'] = pd.to_datetime(df['Price Date'])

# Group by 'Price Date' and calculate the mean prices for each date
grouped_data = df.groupby("Price Date")
average_prices = grouped_data.mean()

# 'Price Date' is now the index, so we don’t need to set it again
# Create the output directory if it does not exist
output_directory = 'Daily_Graphs'
os.makedirs(output_directory, exist_ok=True)

# Define the columns we want to plot
price_columns = {
    'Max Price (Rs./Quintal)': 'Daily Max Price',
    'Min Price (Rs./Quintal)': 'Daily Min Price',
    'Modal Price (Rs./Quintal)': 'Daily Modal Price'
}

# Plot each price column as a separate plot
for column, title in price_columns.items():
    plt.figure(figsize=(12, 6))
    plt.plot(average_prices.index, average_prices[column], marker='o', linestyle='-', color='b')

    # Add title and labels
    plt.title(f'{title} Over Time')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()

    # Save the plot in the specified directory
    plt.savefig(os.path.join(output_directory, f'{title.replace(" ", "_").lower()}.png'))
    plt.close()  # Close the figure to free memory

print(f"Daily time plots saved in directory: {output_directory}")
