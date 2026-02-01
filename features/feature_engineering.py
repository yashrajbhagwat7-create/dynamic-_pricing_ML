import pandas as pd
import numpy as np


def prepare_features(csv_path:str):

    #load data
    data=pd.read_csv(csv_path,parse_dates=['date'])

    #time basaed features

    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding (VERY IMPORTANT)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    # Handle price column (support 'price' or 'base_price')
    possible_price_cols = ['price', 'Price', 'base_price', 'basePrice', 'Base_Price']
    price_col = next((c for c in possible_price_cols if c in data.columns), None)
    if price_col is None:
        raise KeyError(f"No price column found in data. Expected one of: {possible_price_cols}")
    # Create canonical 'price' column as numeric
    data['price'] = pd.to_numeric(data[price_col], errors='coerce')

    # Ensure promotion column exists
    if 'promotion' not in data.columns:
        data['promotion'] = 0

    # Price-based features
    data['price_squared'] = data['price'] ** 2
    data['price_log'] = np.log(data['price'].replace(0, np.nan))
    REFERENCE_PRICE = 100
    data["price_diff"] = data["price"] - REFERENCE_PRICE

    # Drop rows with missing critical values
    data = data.dropna(subset=['price', 'units_sold'])

    # -------------------------------
    # Feature selection
    # -------------------------------
    feature_cols = [
        "price",
        "price_squared",
        "price_log",
        "promotion",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos"
    ]

    X = data[feature_cols]
    y = data["units_sold"]

    return X, y

