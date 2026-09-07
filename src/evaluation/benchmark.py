"""
benchmark.py -- Reproducible benchmark harness for classical baselines.

Runs all classical models on the same train/test split and records:
    - RMSE
    - MAE
    - Fit time (seconds)
    - Predict time (seconds)

Results are saved as both CSV and a formatted console table.

Usage:
    python src/evaluation/benchmark.py
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classical import get_all_baselines, HYPERPARAMS
from src.evaluation.metrics import evaluate_surprise_predictions


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_split():
    """Load the pre-computed train/test CSVs."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"[OK] Loaded train ({len(train):,}) and test ({len(test):,}) sets")
    return train, test


def run_benchmark(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all classical baselines and collect results.

    Returns a DataFrame with columns:
        Model, RMSE, MAE, Fit_Time_s, Predict_Time_s
    """
    baselines = get_all_baselines()
    results = []

    print("\n" + "=" * 60)
    print("  CLASSICAL BASELINES BENCHMARK")
    print("=" * 60)

    for model in baselines:
        print(f"\n--- {model.get_name()} ---")

        # Train
        model.fit(train_df)

        # Predict
        predictions = model.predict(test_df)

        # Evaluate
        metrics = evaluate_surprise_predictions(predictions)

        results.append({
            "Model": model.get_name(),
            "RMSE": round(metrics["rmse"], 4),
            "MAE": round(metrics["mae"], 4),
            "Fit_Time_s": round(model.fit_time, 3),
            "Predict_Time_s": round(model.predict_time, 3),
            "N_Predictions": metrics["n_predictions"],
        })

    results_df = pd.DataFrame(results)
    return results_df


def save_results(results_df: pd.DataFrame):
    """Save results to CSV and print a formatted table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "classical_baselines.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")


def print_results_table(results_df: pd.DataFrame):
    """Print a nicely formatted results table."""
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'Fit(s)':>10} {'Predict(s)':>12}")
    print("  " + "-" * 54)

    for _, row in results_df.iterrows():
        print(
            f"  {row['Model']:<12} "
            f"{row['RMSE']:>8.4f} "
            f"{row['MAE']:>8.4f} "
            f"{row['Fit_Time_s']:>10.3f} "
            f"{row['Predict_Time_s']:>12.3f}"
        )

    # Highlight best
    best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
    best_mae = results_df.loc[results_df["MAE"].idxmin()]
    print("  " + "-" * 54)
    print(f"  Best RMSE : {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
    print(f"  Best MAE  : {best_mae['Model']} ({best_mae['MAE']:.4f})")
    print("=" * 70)


def print_hyperparams():
    """Print all hyperparameters for reproducibility documentation."""
    print("\n--- Hyperparameters (for reproducibility) ---")
    for model_name, params in HYPERPARAMS.items():
        print(f"  {model_name}: {params}")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    total_start = time.perf_counter()

    # Load data
    train, test = load_split()

    # Print hyperparams
    print_hyperparams()

    # Run benchmark
    results_df = run_benchmark(train, test)

    # Save & display
    save_results(results_df)
    print_results_table(results_df)

    total_time = time.perf_counter() - total_start
    print(f"\n[DONE] Total benchmark time: {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
