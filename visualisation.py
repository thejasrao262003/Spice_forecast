import pandas as pd

df = pd.read_csv('Data/semi_cleaned.csv')
# count_min_price_zero = df[df['Min Price (Rs./Quintal)'] == 0].shape[0]
# df_cleaned = df[df['Min Price (Rs./Quintal)'] != 0]
# new_count = df_cleaned.shape[0]
# df = df_cleaned.copy()
#
# average_modal_price = df.groupby('Variety')['Modal Price (Rs./Quintal)'].mean().reset_index()
# average_modal_price = average_modal_price.sort_values(by='Modal Price (Rs./Quintal)', ascending=True)
#
# # Print the sorted average modal price
# print("Average Modal Price per Variety (Ascending Order):")
# print(average_modal_price.to_string(index=False))
#
# # Calculate the price changes
# price_changes = df.groupby('Variety').agg(
#     Min_Price=('Min Price (Rs./Quintal)', 'min'),
#     Max_Price=('Max Price (Rs./Quintal)', 'max'),
#     Modal_Price=('Modal Price (Rs./Quintal)', 'mean')
# ).reset_index()
#
# # Calculate absolute gain/loss and percentage gain/loss
# price_changes['Absolute_Change'] = price_changes['Max_Price'] - price_changes['Min_Price']
# price_changes['Percentage_Change'] = (price_changes['Absolute_Change'] / price_changes['Min_Price']) * 100
#
# # Find highest gainer and biggest loser in percentage
# highest_gainer_percentage = price_changes.loc[price_changes['Percentage_Change'].idxmax()]
# biggest_loser_percentage = price_changes.loc[price_changes['Percentage_Change'].idxmin()]
#
# # Find highest gainer and biggest loser in absolute terms
# highest_gainer_absolute = price_changes.loc[price_changes['Absolute_Change'].idxmax()]
# biggest_loser_absolute = price_changes.loc[price_changes['Absolute_Change'].idxmin()]
#
# # Display results
# print("\nHighest Gainer (Percentage):")
# print(highest_gainer_percentage)
#
# print("\nBiggest Loser (Percentage):")
# print(biggest_loser_percentage)
#
# print("\nHighest Gainer (Absolute):")
# print(highest_gainer_absolute)
#
# print("\nBiggest Loser (Absolute):")
# print(biggest_loser_absolute)
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned CSV file (if not already loaded)
df_cleaned = pd.read_csv('Data/df_simple.csv')
df_cleaned['Price Date'] = pd.to_datetime(df_cleaned['Price Date'])
df_cleaned = df_cleaned[["Price Date", "Min Price (Rs./Quintal)","Max Price (Rs./Quintal)","Modal Price (Rs./Quintal)"]]
# Set 'Price Date' as the index
df_cleaned.set_index('Price Date', inplace=True)

# Resample the data weekly and calculate the mean modal price for each week
weekly_data = df_cleaned.resample('W').mean()

# Create a plot
plt.figure(figsize=(12, 6))

# Plot the modal prices for all years
plt.plot(weekly_data.index, weekly_data['Modal Price (Rs./Quintal)'], marker='o', linestyle='-', color='blue')

# Adding title and labels
plt.title('Weekly Modal Prices for Sesame')
plt.xlabel('Weeks')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()