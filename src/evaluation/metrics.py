"""
metrics.py -- Evaluation metrics for the recommendation benchmark.

All metrics follow the same interface:
    metric(y_true, y_pred) -> float

These work with both raw numpy arrays and pandas Series,
so they can be used with Surprise predictions or custom models.
"""

import numpy as np


def rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt( (1/N) * sum( (y_true_i - y_pred_i)^2 ) )

    Lower is better. Standard metric for rating prediction tasks.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error.

    MAE = (1/N) * sum( |y_true_i - y_pred_i| )

    Lower is better. More robust to outliers than RMSE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def surprise_predictions_to_arrays(predictions):
    """
    Convert a list of Surprise Prediction objects into
    (y_true, y_pred) numpy arrays.

    Each Surprise Prediction has:
        .uid   - user id
        .iid   - item id
        .r_ui  - true rating
        .est   - predicted rating

    Returns:
        tuple[np.ndarray, np.ndarray]: (true_ratings, predicted_ratings)
    """
    y_true = np.array([pred.r_ui for pred in predictions], dtype=np.float64)
    y_pred = np.array([pred.est for pred in predictions], dtype=np.float64)
    return y_true, y_pred


def evaluate_surprise_predictions(predictions) -> dict:
    """
    Given a list of Surprise Prediction objects, compute all metrics.

    Returns:
        dict with keys: 'rmse', 'mae', 'n_predictions'
    """
    y_true, y_pred = surprise_predictions_to_arrays(predictions)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n_predictions": len(y_true),
    }
