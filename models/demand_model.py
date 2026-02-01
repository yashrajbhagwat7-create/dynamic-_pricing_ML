import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import sys

# Robust import: support package, direct script execution, or local-file fallback
try:
    # If run as a package (e.g., python -m models.demand_model)
    from ..features.feature_engineering import prepare_features
except Exception:
    try:
        # Try absolute import (works if project root is in sys.path and no shadowing package)
        from ..features.feature_engineering import prepare_features
    except Exception:
        # As a last resort, load the local file explicitly to avoid shadowing by an installed package
        import importlib.util
        project_root = os.path.dirname(os.path.dirname(__file__))
        fe_path = os.path.join(project_root, 'features', 'feature_engineering.py')
        if not os.path.exists(fe_path):
            raise ImportError(f"Could not locate local feature_engineering.py at {fe_path}")
        spec = importlib.util.spec_from_file_location('local_feature_engineering', fe_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prepare_features = mod.prepare_features


def train_models(csv_path, return_metrics=False):
    """Train and return LinearRegression and RandomForestRegressor models.

    Args:
        csv_path (str): Path to CSV to load features from.
        return_metrics (bool): If True, also return a dict of metrics.

    Returns:
        (Lr_model, Rf_model) or (Lr_model, Rf_model, metrics)
    """
    # Load features
    X, y = prepare_features(csv_path)

    # Time-based split (IMPORTANT)
    split_index = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # model 1 :  linear regression
    Lr_model = LinearRegression()
    Lr_model.fit(X_train, y_train)
    y_pred = Lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, y_pred)
    lr_mse = mean_squared_error(y_test, y_pred)

    # model 2: random forest
    Rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    Rf_model.fit(X_train, y_train)
    y_pred_rf = Rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_mse = mean_squared_error(y_test, y_pred_rf)

    metrics = {
        'lr_mae': lr_mae,
        'lr_mse': lr_mse,
        'rf_mae': rf_mae,
        'rf_mse': rf_mse,
    }

    if return_metrics:
        return Lr_model, Rf_model, metrics
    return Lr_model, Rf_model


def main():
    # Choose CSV path sensibly
    project_root = os.path.dirname(os.path.dirname(__file__))
    possible_paths = [
        os.path.join(project_root, 'data', 'sales_data_v2.csv'),
        os.path.join(project_root, 'data', 'sales_data.csv'),
        os.path.join(project_root, 'sales_data.csv')
    ]
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break
    else:
        raise FileNotFoundError('No sales_data CSV found. Checked: ' + ', '.join(possible_paths))

    Lr_model, Rf_model, metrics = train_models(csv_path, return_metrics=True)

    print("Linear Regression MAE:", metrics['lr_mae'])
    print("Linear Regression MSE:", metrics['lr_mse'])
    print("Random Forest MAE:", metrics['rf_mae'])
    print("Random Forest MSE:", metrics['rf_mse'])


if __name__ == '__main__':
    main()

