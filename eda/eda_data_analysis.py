import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load data sets

data=pd.read_csv('sales_data.csv',parse_dates=['date'])

#basic info

print("dataset info:")
print(data.info())
print("\nsummary statistics:")
print(data.describe())

#price vs units sold scatter plot

# plt.figure()
# # Use 'base_price' column; fallback to 'price' if present
# if 'price' in data.columns:
#     x = data['price']
#     xlabel = 'price'
# else:
#     x = data['base_price']
#     xlabel = 'base_price'
# plt.scatter(x, data["units_sold"])
# plt.xlabel(xlabel)
# plt.ylabel("units sold")
# plt.title("Price vs Units Sold")
# plt.show()

# #Demand over time
# plt.figure()
# plt.plot(data['date'] ,data['units_sold'])
# plt.xlabel("Date")
# plt.ylabel("Units_Sold")
# plt.title("Units sold over time")
# plt.show()

# #average demand by day of week
# avg_demand_dow=data.groupby('day_of_week')['units_sold'].mean()
# plt.figure()
# avg_demand_dow.plot(kind="bar")
# plt.xlabel("Day of Week (0=Mon, 6=Sun)")
# plt.ylabel("Average Units Sold")
# plt.title("Average Demand by Day of Week")
# plt.show()

# #Average demand by month
# # -------------------------------
# avg_by_month = data.groupby("month")["units_sold"].mean()

# plt.figure()
# avg_by_month.plot(kind="bar")
# plt.xlabel("Month")
# plt.ylabel("Average Units Sold")
# plt.title("Average Demand by Month")
# plt.show()


# Determine which price column exists (supporting 'price' or 'base_price')
possible_price_cols = ['price', 'Price', 'base_price', 'basePrice', 'Base_Price']
price_col = None
for c in possible_price_cols:
    if c in data.columns:
        price_col = c
        break
if price_col is None:
    raise KeyError(f"No price column found in data. Expected one of: {possible_price_cols}")

# Ensure units_sold exists
if 'units_sold' not in data.columns:
    raise KeyError("No 'units_sold' column found in data")

# Convert to numeric (coerce errors) and drop NaNs for plotting / regression
data[price_col] = pd.to_numeric(data[price_col], errors='coerce')
data['units_sold'] = pd.to_numeric(data['units_sold'], errors='coerce')
plot_data = data[[price_col, 'units_sold']].dropna()

# Fit linear regression (price -> units_sold)
z = np.polyfit(plot_data[price_col], plot_data['units_sold'], 1)
p = np.poly1d(z)

plt.figure()
plt.scatter(plot_data[price_col], plot_data['units_sold'], alpha=0.3)
# Sort x for a clean regression line
sorted_idx = np.argsort(plot_data[price_col])
plt.plot(plot_data[price_col].iloc[sorted_idx], p(plot_data[price_col].iloc[sorted_idx]), color='red')
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.title(f"Price vs Demand with Trend Line ({price_col})")
plt.show()