import pandas as pd
import time
df = pd.read_csv("Data/combined_spice_price_data.csv")
df = df.drop(columns=["Unnamed: 0", "Sl no.", "Commodity"])
df['Price Date'] = pd.to_datetime(df['Price Date'], format='%d %b %Y')
df_sorted = df.sort_values(by='Price Date')
df_sorted.reset_index(drop=True, inplace=True)
start_date = df_sorted['Price Date'].min()
end_date = df_sorted['Price Date'].max()
all_dates = pd.date_range(start=start_date, end=end_date)
present_dates = df_sorted['Price Date'].unique()
missing_dates = [date for date in all_dates if date not in present_dates]
print("Missing Dates:")
print(missing_dates)
print(f"Number of Missing Dates: {len(missing_dates)}")
print(df_sorted)
df_sorted.to_csv("Data/semi_cleaned.csv", index=False)