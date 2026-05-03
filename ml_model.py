"""
ml_model.py — RandomForest Workload Prediction
===============================================
Trains a RandomForest regression model on PlanetLab-derived CPU demand
data and returns normalised predictions for all tasks in the batch.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "workload.csv")


def train_model(batch_size: int = 100, dataset: str = "planetlab") -> list[float]:
    """
    Load workload data, train RandomForest, and return CPU demand predictions.

    Parameters
    ----------
    batch_size : int
        Number of task records to use (10–1000).
    dataset : str
        'planetlab' → read from CSV; 'live' → stochastic generation.

    Returns
    -------
    list[float]
        Normalised predicted CPU loads in [0.0, 1.0].
    """
    if dataset == "live":
        # Stochastic generation mirroring PlanetLab distribution
        rng = np.random.default_rng()
        cpu_demands = np.clip(rng.beta(a=2, b=5, size=batch_size), 0.0, 1.0)
        return cpu_demands.tolist()

    # Load PlanetLab-derived dataset
    df = pd.read_csv(DATASET_PATH).head(batch_size)
    if df.empty:
        raise ValueError("workload.csv is empty or batch_size too small.")

    X = df[["task_id"]].values
    y = df["cpu_demand"].values

    # Normalise targets
    scaler = MinMaxScaler()
    y_norm = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Train / test split (reproducible)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_norm, test_size=0.2, random_state=42
    )

    # Train RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    # Evaluate on held-out set
    y_pred_test = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred_test)
    rmse = mean_squared_error(y_te, y_pred_test) ** 0.5
    print(f"[ML] MAE={mae:.4f}  RMSE={rmse:.4f}  (test split, n={len(y_te)})")

    # Generate predictions for full batch
    predictions = model.predict(X)
    predictions = np.clip(predictions, 0.0, 1.0)
    return predictions.tolist()
