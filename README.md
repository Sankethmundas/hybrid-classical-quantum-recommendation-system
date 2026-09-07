# Hybrid Classical-Quantum Recommendation System

A rigorous, reproducible benchmark comparing hybrid quantum-classical and classical-only recommendation models on the MovieLens 100K dataset.

## Project Goals

1. **Reproducible Benchmarking** — Clean, standardised train/test splits, identical metrics (RMSE, MAE), and documented hyperparameters across all models.
2. **Runtime-per-Qubit Scaling Analysis** — Plot how RMSE and wall-clock time change as qubit count scales from 4 → 10 → 16 on a laptop simulator.

## Tech Stack

| Layer | Tool |
|-------|------|
| Quantum kernels | PennyLane (default.qubit simulator) |
| Classical baselines | Scikit-Surprise (SVD, KNN, NMF) |
| ML utilities | Scikit-learn, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |

## Dataset

**MovieLens 100K** — 100 000 ratings from 943 users on 1 682 movies (1–5 scale).  
Downloaded and cleaned automatically by `src/data/load_data.py`.

## Project Structure

```
HYBRID SYSTEM/
├── data/                   # Raw & processed datasets (git-ignored)
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   └── load_data.py    # Download, clean, split MovieLens 100K
│   ├── models/             # Classical & quantum model implementations
│   ├── evaluation/         # Metrics, benchmark harness
│   └── utils/              # Helpers (config, logging, timing)
├── notebooks/              # Exploratory analysis & result plots
├── results/                # Saved metrics, plots, tables
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download & clean the dataset
python src/data/load_data.py
```

## License

MIT
