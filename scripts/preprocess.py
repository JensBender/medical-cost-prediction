"""
Deterministic data preprocessing: raw MEPS SAS data → preprocessed parquet splits.

This script contains only the production steps needed for reproducible
preprocessing via ``dvc repro``. For exploratory data analysis and preprocessing
experiments, see ``notebooks/1_eda_and_preprocessing.py``.

Usage::

    .venv-train/Scripts/python scripts/preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.constants import (
    RANDOM_STATE,
    ID_COLUMN,
    WEIGHT_COLUMN,
    TARGET_COLUMN,
    RAW_COLUMNS_TO_KEEP,
    RAW_BINARY_FEATURES,
    MEPS_MISSING_CODES,
    PIPELINE_NUMERICAL_FEATURES,
    PIPELINE_BINARY_FEATURES,
    PIPELINE_NOMINAL_FEATURES,
    PIPELINE_REQUIRED_FEATURES,
    PIPELINE_OPTIONAL_FEATURES
)
from src.pipeline import create_preprocessing_pipeline


# ============================================================
# Configuration
# ============================================================

# Paths (relative to project root)
RAW_DATA_PATH = "data/h251.sas7bdat"
OUTPUT_DIR = "data"


# ============================================================
# Helper: Distribution-informed stratification for splitting
# ============================================================

def create_stratification_bins(y):
    """Create stratification bins that preserve the zero-inflated, heavy-tailed
    cost distribution across train/val/test splits.

    Uses non-linear percentile boundaries (50th, 80th, 95th, 99th, 99.9th)
    to ensure extreme high-cost cases are balanced across all splits.
    """
    strata = pd.Series(index=y.index, dtype=int)

    # Bin 0: Zero costs (the hurdle)
    is_zero = y == 0
    strata[is_zero] = 0

    # Bins 1–6: Non-linear quantiles for positive spenders
    positive_y = y[~is_zero]
    bins = [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0]
    strata[~is_zero] = pd.qcut(positive_y, q=bins, labels=False, duplicates="drop") + 1

    return strata


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

def main():
    # --- 1. Data Loading ---
    print("Step 1/11: Loading raw MEPS data...")
    df = pd.read_sas(RAW_DATA_PATH, format="sas7bdat", encoding="latin1")
    print(f"  Loaded {len(df):,} rows × {len(df.columns):,} columns")

    # --- 2. Column Selection ---
    print("Step 2/11: Selecting columns...")
    df = df[RAW_COLUMNS_TO_KEEP]
    print(f"  Kept {len(df.columns)} columns")

    # --- 3. Row Filtering (adults with positive weights) ---
    print("Step 3/11: Filtering target population...")
    n_before = len(df)
    df = df[(df[WEIGHT_COLUMN] > 0) & (df["AGE23X"] >= 18)].copy()
    print(f"  Kept {len(df):,} of {n_before:,} rows")

    # --- 4. Data Type Handling ---
    print("Step 4/11: Handling data types...")
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df.set_index(ID_COLUMN, inplace=True)

    # --- 5. Missing Value Standardization ---
    print("Step 5/11: Standardizing missing values...")
    # Recover implied values from survey skip patterns
    df.loc[df["ADSMOK42"] == -1, "ADSMOK42"] = 2        # Never smokers → No
    df.loc[(df["JTPAIN31_M18"] == -1) & (df["ARTHDX"] == 1), "JTPAIN31_M18"] = 1  # Arthritis skip → Yes
    # Convert remaining MEPS codes to NaN
    df.replace(MEPS_MISSING_CODES, np.nan, inplace=True)
    print(f"  Total missing values: {df.isnull().sum().sum():,}")

    # --- 6. Binary Standardization (MEPS 1/2 → 1/0) ---
    print("Step 6/11: Standardizing binary features to 0/1...")
    df[RAW_BINARY_FEATURES] = df[RAW_BINARY_FEATURES].replace({2: 0})

    # --- 7. Feature Engineering ---
    print("Step 7/11: Engineering features...")
    # Recent Life Transition flag
    marital_transitions = [7, 8, 9, 10]
    employment_transitions = [2, 3]
    df["RECENT_LIFE_TRANSITION"] = (
        df["MARRY31X"].isin(marital_transitions) | df["EMPST31"].isin(employment_transitions)
    ).astype(float)
    df.loc[df["MARRY31X"].isna() & df["EMPST31"].isna(), "RECENT_LIFE_TRANSITION"] = np.nan

    # Collapse marital status transition categories into stable counterparts
    marital_map = {7: 1, 8: 2, 9: 3, 10: 4}
    df["MARRY31X_GRP"] = df["MARRY31X"].replace(marital_map)

    # Collapse employment status categories (2, 3, 4 → 0: Not Employed)
    employment_map = {2: 0, 3: 0, 4: 0}
    df["EMPST31_GRP"] = df["EMPST31"].replace(employment_map)

    # --- 8. Train/Val/Test Split (80/10/10 stratified) ---
    print("Step 8/11: Splitting data...")
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # First split: 80% train, 20% temp
    y_strata = create_stratification_bins(y)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y_strata
    )

    # Second split: temp → 10% val, 10% test
    temp_strata = create_stratification_bins(y_temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_strata
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # --- 9. Preprocessing Pipeline (fit on train, transform all) ---
    print("Step 9/11: Running preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(
        PIPELINE_REQUIRED_FEATURES,
        PIPELINE_OPTIONAL_FEATURES,
        PIPELINE_NUMERICAL_FEATURES,
        PIPELINE_NOMINAL_FEATURES,
        PIPELINE_BINARY_FEATURES,
        strict=False,
    )
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    print(f"  Output features: {X_train_preprocessed.shape[1]}")

    # --- 10. Verification ---
    print("Step 10/11: Verifying preprocessing results...")
    for name, raw, processed in [
        ("Train", X_train, X_train_preprocessed),
        ("Val",   X_val,   X_val_preprocessed),
        ("Test",  X_test,  X_test_preprocessed),
    ]:
        rows_ok = len(raw) == len(processed)
        nulls = processed.isnull().sum().sum()
        numeric_ok = processed.apply(pd.api.types.is_numeric_dtype).all()
        status = "✅" if (rows_ok and nulls == 0 and numeric_ok) else "❌"
        print(f"  {name:5}: {status}  rows={len(processed):,}  nulls={nulls}  all_numeric={numeric_ok}")

    # --- 11. Save Parquet Files ---
    print("Step 11/11: Saving preprocessed data...")
    for split_name, X_proc, y_split, X_raw in [
        ("training",    X_train_preprocessed, y_train, X_train),
        ("validation",  X_val_preprocessed,   y_val,   X_val),
        ("test",        X_test_preprocessed,  y_test,  X_test),
    ]:
        # Merge preprocessed features, target variable, and sample weights
        df_out = pd.concat([X_proc, y_split, X_raw[WEIGHT_COLUMN]], axis=1)
        path = f"{OUTPUT_DIR}/{split_name}_data_preprocessed.parquet"
        df_out.to_parquet(path)
        print(f"  Saved {path} ({df_out.shape[0]:,} rows × {df_out.shape[1]} cols)")

    print("\n✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
