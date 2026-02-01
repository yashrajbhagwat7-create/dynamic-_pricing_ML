import numpy as np
import pandas as pd

np.random.seed(42)

# Dates
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
n = len(dates)

# Base price around which elasticity works
base_price = 100

# Generate prices (reasonable range)
price = np.random.normal(loc=100, scale=7, size=n)
price = np.clip(price, 85, 115)

# Base demand
base_demand = 120

# 🔥 PRICE ELASTICITY (KEY FIX)
elasticity_strength = 0.035
price_effect = np.exp(-elasticity_strength * (price - base_price))

# Seasonality
month = dates.month
day_of_week = dates.dayofweek

monthly_effect = 1 + 0.25 * np.sin(2 * np.pi * month / 12)
weekday_effect = np.where(day_of_week < 5, 1.0, 0.85)

# Noise
noise = np.random.normal(1.0, 0.08, n)

# Final demand
units_sold = (
    base_demand
    * price_effect
    * monthly_effect
    * weekday_effect
    * noise
)

units_sold = np.maximum(units_sold.astype(int), 1)

# Dataset
df = pd.DataFrame({
    "date": dates,
    "price": price,
    "month": month,
    "day_of_week": day_of_week,
    "units_sold": units_sold
})

df.head()


# -------------------------------
# Save
# -------------------------------
df.to_csv("sales_data_v2.csv", index=False)

print("✅ Improved synthetic sales data generated")
