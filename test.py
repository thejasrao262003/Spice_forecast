import pandas as pd

# Read the file
df = pd.read_csv("Data/White_Variety.csv")
print("Initial Data:\n", df.head())

# Remove unnecessary columns
df = df.drop(columns=['Variety', 'Grade'])

# Reorder columns so that 'Price Date' comes first
cols = ['Price Date'] + [col for col in df.columns if col != 'Price Date']
df = df[cols]

# Convert 'Price Date' to datetime format
df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')

# Step 1: Detect and remove outliers using the IQR method
q1 = df['Modal Price (Rs./Quintal)'].quantile(0.25)
q3 = df['Modal Price (Rs./Quintal)'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Filter out rows with outliers in 'Modal Price (Rs./Quintal)'
df = df[(df['Modal Price (Rs./Quintal)'] >= lower_bound) & (df['Modal Price (Rs./Quintal)'] <= upper_bound)]
print("Data after outlier removal:\n", df.head())

# Step 2: Aggregate by 'Price Date' to ensure unique dates
df = df.groupby('Price Date').mean().reset_index()

# Step 3: Reindex to fill in missing dates
df = df.set_index('Price Date')  # Set 'Price Date' as index
date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')  # Full date range
df = df.reindex(date_range)  # Reindex the DataFrame to include all dates
df.index.name = 'Price Date'  # Rename the index to 'Price Date'

# Impute missing values for 'Modal Price (Rs./Quintal)' with the previous day's value
df['Modal Price (Rs./Quintal)'] = df['Modal Price (Rs./Quintal)'].ffill()

# Reset the index after handling missing dates
df.reset_index(inplace=True)

# Display the data after handling missing dates
print("Data after filling missing dates:\n", df.head())

# Optionally, save the final data to a new file
df.to_csv("Data/aggregated_spice_price_data_2019_2024.csv", index=False)
