"""
classical.py -- Classical recommendation baselines using Surprise.

Models implemented:
    1. SVD       - Matrix factorization (Funk SVD)
    2. KNNBasic  - User-based collaborative filtering (cosine similarity)
    3. NMF       - Non-negative Matrix Factorization

All models share a common interface via the ClassicalBaseline wrapper:
    .fit(train_df)       -> trains on a pandas DataFrame
    .predict(test_df)    -> returns a list of Surprise Prediction objects
    .get_name()          -> returns the model name string

Hyperparameters are explicitly documented for reproducibility.
"""

import time
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise import Trainset


# ──────────────────────────────────────────────
# Hyperparameters (documented for reproducibility)
# ──────────────────────────────────────────────
HYPERPARAMS = {
    "SVD": {
        "n_factors": 100,       # Number of latent factors
        "n_epochs": 20,         # Number of SGD epochs
        "lr_all": 0.005,        # Learning rate for all parameters
        "reg_all": 0.02,        # Regularization term for all parameters
        "random_state": 42,
    },
    "KNN": {
        "k": 40,                # Number of nearest neighbors
        "sim_options": {
            "name": "cosine",   # Similarity measure
            "user_based": True, # User-based CF (vs item-based)
        },
    },
    "NMF": {
        "n_factors": 15,        # Number of latent factors
        "n_epochs": 50,         # Number of SGD epochs
        "random_state": 42,
    },
}


def _df_to_surprise_trainset(df: pd.DataFrame) -> Trainset:
    """
    Convert a pandas DataFrame with columns [user_id, item_id, rating]
    into a Surprise Trainset object.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["user_id", "item_id", "rating"]], reader
    )
    return data.build_full_trainset()


def _df_to_surprise_testset(df: pd.DataFrame) -> list:
    """
    Convert a pandas DataFrame with columns [user_id, item_id, rating]
    into a Surprise testset (list of tuples).
    """
    return list(
        df[["user_id", "item_id", "rating"]].itertuples(index=False, name=None)
    )


class ClassicalBaseline:
    """
    Wrapper around a Surprise algorithm with timing and a clean interface.
    """

    def __init__(self, name: str, algorithm):
        self.name = name
        self.algorithm = algorithm
        self.fit_time = 0.0
        self.predict_time = 0.0

    def get_name(self) -> str:
        return self.name

    def fit(self, train_df: pd.DataFrame):
        """Train the model on a ratings DataFrame."""
        trainset = _df_to_surprise_trainset(train_df)
        start = time.perf_counter()
        self.algorithm.fit(trainset)
        self.fit_time = time.perf_counter() - start
        print(f"  [{self.name}] Trained in {self.fit_time:.3f}s")
        return self

    def predict(self, test_df: pd.DataFrame) -> list:
        """Generate predictions for a test DataFrame. Returns Surprise Predictions."""
        testset = _df_to_surprise_testset(test_df)
        start = time.perf_counter()
        predictions = self.algorithm.test(testset)
        self.predict_time = time.perf_counter() - start
        print(f"  [{self.name}] Predicted {len(predictions):,} ratings in {self.predict_time:.3f}s")
        return predictions


# ──────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────

def build_svd() -> ClassicalBaseline:
    """Build an SVD baseline with documented hyperparameters."""
    params = HYPERPARAMS["SVD"]
    algo = SVD(
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        lr_all=params["lr_all"],
        reg_all=params["reg_all"],
        random_state=params["random_state"],
    )
    return ClassicalBaseline("SVD", algo)


def build_knn() -> ClassicalBaseline:
    """Build a KNN (user-based, cosine) baseline with documented hyperparameters."""
    params = HYPERPARAMS["KNN"]
    algo = KNNBasic(
        k=params["k"],
        sim_options=params["sim_options"],
        verbose=False,
    )
    return ClassicalBaseline("KNN-User", algo)


def build_nmf() -> ClassicalBaseline:
    """Build an NMF baseline with documented hyperparameters."""
    params = HYPERPARAMS["NMF"]
    algo = NMF(
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        random_state=params["random_state"],
    )
    return ClassicalBaseline("NMF", algo)


def get_all_baselines() -> list[ClassicalBaseline]:
    """Return a list of all classical baselines."""
    return [build_svd(), build_knn(), build_nmf()]
