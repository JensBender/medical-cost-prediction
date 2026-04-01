"""
Deterministic model training script for baseline models.

Steps:
  1.  MLflow Settings: Set up experiment tracking.
  2.  Preprocessed Data Loading: Parquet to Pandas.

For model training experiments, in-depth model evaluation, error analysis, and detailed rationale, see:
notebooks/2_modeling.ipynb

Usage:
    .venv-train/Scripts/python scripts/train_baseline.py
"""

import pandas as pd
import mlflow


# Main Baseline Model Training 
def main():
    # --- 1. MLflow Settings --- 
    # Set the tracking URI to point to your running MLflow UI server
    print("Step 1: Setting up MLflow experiment tracking...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # --- 2. Preprocessed Data Loading ---
    print("Step 2: Loading preprocessed data...")
    df_train_preprocessed = pd.read_parquet("data/training_data_preprocessed.parquet")
    df_val_preprocessed = pd.read_parquet("data/validation_data_preprocessed.parquet")
    print(f"  Loaded 'training_data_preprocessed.parquet' with {len(df_train_preprocessed):,} rows and {len(df_train_preprocessed.columns):,} columns")
    print(f"  Loaded 'validation_data_preprocessed.parquet' with {len(df_val_preprocessed):,} rows and {len(df_val_preprocessed.columns):,} columns")


if __name__ == "__main__":
    main()