"""
benchmark_quantum.py -- Benchmark comparing Classical Baselines and Quantum Kernel Models.

Runs both:
    1. Classical models: SVD, KNN-User, NMF
    2. Quantum models:
        - QuantumKernelModel(n_qubits=4, trainable=False)  # untrained baseline
        - QuantumKernelModel(n_qubits=4, trainable=True)   # trained kernel

Evaluates on the exact same train/test split, with a configurable test slice
(default: 2,000 rows, since quantum prediction evaluates ~1 circuit per test row).

Usage:
    python src/evaluation/benchmark_quantum.py [--test-slice 2000]
"""

import sys
import time
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classical import get_all_baselines
from src.models.quantum import QuantumKernelModel
from src.evaluation.metrics import evaluate_surprise_predictions

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_split(test_slice_size: int = 2000):
    """Load pre-split train and test sets, slicing test set to test_slice_size."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    full_test = pd.read_csv(PROCESSED_DIR / "test.csv")

    if test_slice_size and test_slice_size < len(full_test):
        test_slice = full_test.iloc[:test_slice_size].copy().reset_index(drop=True)
        print(f"[OK] Using test slice: {len(test_slice):,} rows (from {len(full_test):,} total)")
    else:
        test_slice = full_test

    print(f"[OK] Training set: {len(train):,} rows")
    return train, test_slice


def run_benchmark(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run classical and 4-qubit quantum models through the benchmark harness."""
    models = [
        # Classical baselines
        *get_all_baselines(),
        # Quantum kernel models
        QuantumKernelModel(n_qubits=4, trainable=False),
        QuantumKernelModel(n_qubits=4, trainable=True),
    ]

    results = []

    print("\n" + "=" * 70)
    print("  HYBRID QUANTUM-CLASSICAL BENCHMARK (4 QUBITS)")
    print("=" * 70)

    for model in models:
        m_name = model.get_name()
        print(f"\n--- {m_name} ---")

        # Train
        model.fit(train_df)

        # Predict
        predictions = model.predict(test_df)

        # Evaluate
        metrics = evaluate_surprise_predictions(predictions)

        results.append({
            "Model": m_name,
            "Type": "Classical" if "Quantum" not in m_name else "Quantum",
            "RMSE": round(metrics["rmse"], 4),
            "MAE": round(metrics["mae"], 4),
            "Fit_Time_s": round(model.fit_time, 3),
            "Predict_Time_s": round(model.predict_time, 3),
            "N_Predictions": metrics["n_predictions"],
        })

    results_df = pd.DataFrame(results)
    return results_df


def save_and_print_results(results_df: pd.DataFrame):
    """Save benchmark results to CSV and print summary table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "quantum_4q_benchmark.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")

    print("\n" + "=" * 75)
    print("  BENCHMARK RESULTS (TEST SLICE: 2,000 ROWS)")
    print("=" * 75)
    print(f"  {'Model':<30} {'Type':<10} {'RMSE':>8} {'MAE':>8} {'Fit(s)':>8} {'Pred(s)':>8}")
    print("  " + "-" * 71)

    for _, row in results_df.iterrows():
        print(
            f"  {row['Model']:<30} "
            f"{row['Type']:<10} "
            f"{row['RMSE']:>8.4f} "
            f"{row['MAE']:>8.4f} "
            f"{row['Fit_Time_s']:>8.2f} "
            f"{row['Predict_Time_s']:>8.2f}"
        )

    print("  " + "-" * 71)
    best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
    best_mae = results_df.loc[results_df["MAE"].idxmin()]
    print(f"  Best RMSE : {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
    print(f"  Best MAE  : {best_mae['Model']} ({best_mae['MAE']:.4f})")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum-Classical Benchmark")
    parser.add_argument("--test-slice", type=int, default=2000, help="Number of test rows to evaluate")
    args = parser.parse_args()

    total_start = time.perf_counter()
    train, test_slice = load_split(test_slice_size=args.test_slice)

    results_df = run_benchmark(train, test_slice)
    save_and_print_results(results_df)

    total_time = time.perf_counter() - total_start
    print(f"\n[DONE] Total benchmark elapsed: {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
