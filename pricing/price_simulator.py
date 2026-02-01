import numpy as np
import pandas as pd
import os
import sys

# Robust import for models/train function and features
try:
    # If run as package
    from ..models.demand_model import train_models
    from ..features.feature_engineering import prepare_features
except Exception:
    try:
        # Absolute import
        from models.demand_model import train_models
        from ..features.feature_engineering import prepare_features
    except Exception:
        # Fallback: load local files by path to avoid shadowing
        import importlib.util
        project_root = os.path.dirname(os.path.dirname(__file__))
        dm_path = os.path.join(project_root, 'models', 'demand_model.py')
        fe_path = os.path.join(project_root, 'features', 'feature_engineering.py')
        if not os.path.exists(dm_path):
            raise ImportError(f'Could not locate local demand_model.py at {dm_path}')
        spec = importlib.util.spec_from_file_location('local_demand_model', dm_path)
        dm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dm)
        train_models = dm.train_models

        if not os.path.exists(fe_path):
            raise ImportError(f'Could not locate local feature_engineering.py at {fe_path}')
        spec2 = importlib.util.spec_from_file_location('local_feature_engineering', fe_path)
        fe = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(fe)
        prepare_features = fe.prepare_features

# -------------------------------
# Price Simulation Function
# -------------------------------
def recommend_price(
    model,
    base_context: dict,
    price_range: tuple = (80, 120),
    step: int = 1
):
    """
    base_context example:
    {
        "day_of_week": 5,
        "month": 7,
        "promotion": 1
    }
    """

    prices = np.arange(price_range[0], price_range[1] + step, step)
    simulations = []

    for price in prices:
        row = {
            "price": price,
            "promotion": base_context["promotion"],
        }

        # Cyclical features
        row["price_squared"] = price ** 2
        row["price_log"] = np.log(price)

        row["dow_sin"] = np.sin(2 * np.pi * base_context["day_of_week"] / 7)
        row["dow_cos"] = np.cos(2 * np.pi * base_context["day_of_week"] / 7)

        row["month_sin"] = np.sin(2 * np.pi * base_context["month"] / 12)
        row["month_cos"] = np.cos(2 * np.pi * base_context["month"] / 12)

        X_sim = pd.DataFrame([row])

        # Align features to the model's expected feature names/order if available
        if hasattr(model, 'feature_names_in_'):
            required = list(model.feature_names_in_)
            # Add missing cols with 0 and reorder
            for c in required:
                if c not in X_sim.columns:
                    X_sim[c] = 0
            X_sim = X_sim[required]
        else:
            # Fallback canonical order used in training
            expected = [
                "price",
                "price_squared",
                "price_log",
                "promotion",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
            ]
            for c in expected:
                if c not in X_sim.columns:
                    X_sim[c] = 0
            X_sim = X_sim[expected]

        # Ensure numeric dtype
        X_sim = X_sim.astype(float)

        predicted_demand = model.predict(X_sim)[0]
        predicted_demand = max(predicted_demand, 0)

        revenue = price * predicted_demand

        simulations.append({
            "price": price,
            "predicted_demand": predicted_demand,
            "revenue": revenue
        })

    sim_df = pd.DataFrame(simulations)
    best_row = sim_df.loc[sim_df["revenue"].idxmax()]

    return best_row, sim_df


def get_default_models(csv_path: str = None):
    """Train and return default models (Linear and RandomForest).

    If csv_path is None, look in project data/ and project root for sales data.
    """
    if csv_path is None:
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

    Lr_model, Rf_model = train_models(csv_path)
    return Lr_model, Rf_model


if __name__ == '__main__':
    # Quick demo: train models and recommend prices for a sample context
    lr, rf = get_default_models()
    context = {'day_of_week': 5, 'month': 7, 'promotion': 1}
    print('Evaluating Linear Regression model...')
    best_lr, df_lr = recommend_price(lr, context, price_range=(80, 120), step=1)
    print('Best price (LR):', best_lr.to_dict())

    print('Evaluating Random Forest model...')
    best_rf, df_rf = recommend_price(rf, context, price_range=(80,120), step=1)
    print('Best price (RF):', best_rf.to_dict())