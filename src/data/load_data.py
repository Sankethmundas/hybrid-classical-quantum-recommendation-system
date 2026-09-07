"""
load_data.py — Download, clean, and split the MovieLens 100K dataset.

Reproducibility guarantees:
    - Fixed random seed (SEED = 42) for all splits
    - 80/20 stratified train/test split (stratified by user)
    - All cleaning steps documented and deterministic

Usage:
    python src/data/load_data.py
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.20  # 80/20 split
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ZIP_PATH = RAW_DIR / "ml-100k.zip"
EXTRACTED_DIR = RAW_DIR / "ml-100k"


def download_movielens():
    """Download MovieLens 100K zip if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"[OK] ZIP already exists: {ZIP_PATH}")
        return

    print(f"[DOWNLOADING] MovieLens 100K from {MOVIELENS_URL} ...")
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(MOVIELENS_URL, stream=True, timeout=120, verify=False)
    response.raise_for_status()

    total_bytes = 0
    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total_bytes += len(chunk)

    print(f"[OK] Downloaded {total_bytes / 1e6:.1f} MB -> {ZIP_PATH}")


def extract_zip():
    """Extract the ZIP archive if the folder doesn't exist yet."""
    if EXTRACTED_DIR.exists():
        print(f"[OK] Already extracted: {EXTRACTED_DIR}")
        return

    print(f"[EXTRACTING] {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"[OK] Extracted to {EXTRACTED_DIR}")


def load_ratings() -> pd.DataFrame:
    """
    Load the u.data ratings file (tab-separated).

    Columns: user_id, item_id, rating, timestamp
    """
    data_path = EXTRACTED_DIR / "u.data"
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": np.int32,
            "item_id": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )
    print(f"[OK] Loaded {len(df):,} ratings from {data_path.name}")
    return df


def load_movies() -> pd.DataFrame:
    """
    Load the u.item movie-metadata file.

    We keep: item_id, title, release_date, genres (binary columns).
    """
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    col_names = [
        "item_id", "title", "release_date", "video_release_date",
        "imdb_url",
    ] + genre_cols

    data_path = EXTRACTED_DIR / "u.item"
    df = pd.read_csv(
        data_path,
        sep="|",
        header=None,
        names=col_names,
        encoding="latin-1",
    )
    # Drop columns we won't use
    df.drop(columns=["video_release_date", "imdb_url"], inplace=True)
    print(f"[OK] Loaded {len(df):,} movies from {data_path.name}")
    return df


def load_users() -> pd.DataFrame:
    """
    Load the u.user user-demographics file.

    Columns: user_id, age, gender, occupation, zip_code
    """
    data_path = EXTRACTED_DIR / "u.user"
    df = pd.read_csv(
        data_path,
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )
    print(f"[OK] Loaded {len(df):,} users from {data_path.name}")
    return df


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the ratings DataFrame:
        1. Drop duplicate (user, item) pairs (keep first)
        2. Drop rows with missing values
        3. Validate rating range [1, 5]
        4. Sort by timestamp for temporal consistency
    """
    before = len(df)

    # 1. Duplicates
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    # 2. Missing values
    df = df.dropna()

    # 3. Rating range
    df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]

    # 4. Sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    after = len(df)
    dropped = before - after
    print(f"[OK] Cleaning: {before:,} -> {after:,} ratings ({dropped} dropped)")
    return df


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train/test sets, stratified by user_id.

    Each user's ratings are split individually so that every user
    appears in both train and test sets (important for fair RMSE).
    """
    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []

    for _, group in df.groupby("user_id"):
        n = len(group)
        n_test = max(1, int(n * test_size))  # At least 1 test rating per user
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    print(
        f"[OK] Split: {len(train_df):,} train / {len(test_df):,} test "
        f"({test_size:.0%} test, seed={seed})"
    )
    return train_df, test_df


def save_processed(
    ratings: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
):
    """Save all processed DataFrames as CSV files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ratings.to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    users.to_csv(PROCESSED_DIR / "users.csv", index=False)

    print(f"[OK] Saved processed files to {PROCESSED_DIR}")


def print_summary(ratings, train, test, movies, users):
    """Print a quick summary of the dataset."""
    print("\n" + "=" * 55)
    print("  MOVIELENS 100K — DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total ratings (cleaned) : {len(ratings):>8,}")
    print(f"  Training set            : {len(train):>8,}")
    print(f"  Test set                : {len(test):>8,}")
    print(f"  Unique users            : {ratings['user_id'].nunique():>8,}")
    print(f"  Unique movies           : {ratings['item_id'].nunique():>8,}")
    print(f"  Rating range            :    {ratings['rating'].min():.1f} – {ratings['rating'].max():.1f}")
    print(f"  Mean rating             :    {ratings['rating'].mean():.3f}")
    print(f"  Sparsity                :    {1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()):.2%}")
    print(f"  Movies with metadata    : {len(movies):>8,}")
    print(f"  Users with demographics : {len(users):>8,}")
    print(f"  Random seed             : {SEED}")
    print(f"  Test fraction           : {TEST_SIZE:.0%}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n[START] MovieLens 100K Data Pipeline\n")

    # Step 1: Download
    download_movielens()

    # Step 2: Extract
    extract_zip()

    # Step 3: Load
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    # Step 4: Clean
    ratings = clean_ratings(ratings)

    # Step 5: Stratified split
    train, test = stratified_train_test_split(ratings)

    # Step 6: Save
    save_processed(ratings, train, test, movies, users)

    # Step 7: Summary
    print_summary(ratings, train, test, movies, users)

    print("[DONE] Data pipeline complete. Ready for modelling.\n")


if __name__ == "__main__":
    main()
