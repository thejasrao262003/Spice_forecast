import pandas as pd
import os
data_directory = "Data"
combined_csv_file = "Data/combined_spice_price_data.csv"
dataframes = []
for filename in os.listdir(data_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_directory, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv(combined_csv_file, index=False)
print(f"All CSV files combined into {combined_csv_file}")

