"""
Deterministic data preprocessing: raw MEPS data (SAS) → preprocessed data (parquet).

This script contains only the production steps needed for reproducible
preprocessing via ``dvc repro``. For exploratory data analysis and preprocessing
experiments, see ``notebooks/1_eda_and_preprocessing.py``.

Usage::

    .venv-train/Scripts/python scripts/preprocess.py
"""

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from src.constants import (
    ID_COLUMN,
    WEIGHT_COLUMN,
    TARGET_COLUMN,
    RAW_BINARY_FEATURES,
    RAW_COLUMNS_TO_KEEP,
    MEPS_MISSING_CODES,
    MARRY31X_TRANSITION_CODES,
    EMPST31_TRANSITION_CODES,
    MARRY31X_COLLAPSE_MAP,
    EMPST31_COLLAPSE_MAP,
    RANDOM_STATE,
    PIPELINE_NUMERICAL_FEATURES,
    PIPELINE_BINARY_FEATURES,
    PIPELINE_NOMINAL_FEATURES,
    PIPELINE_REQUIRED_FEATURES,
    PIPELINE_OPTIONAL_FEATURES
)
from src.pipeline import create_preprocessing_pipeline
from src.utils import create_stratification_bins


# Paths (relative to project root)
RAW_DATA_PATH = "data/h251.sas7bdat"
OUTPUT_DIR = "data"


# Main Preprocessing 
def main():
    # --- 1. Data Loading ---
    print("Step 1/11: Loading raw MEPS data...")
    df = pd.read_sas(RAW_DATA_PATH, format="sas7bdat", encoding="latin1")
    n_rows_raw = len(df)
    n_cols_raw = len(df.columns)
    print(f"  Loaded {n_rows_raw:,} rows and {n_cols_raw:,} columns")

    # --- 2. Variable Selection ---
    print("Step 2/11: Selecting variables...")
    df = df[RAW_COLUMNS_TO_KEEP]
    print(f"  Kept {len(df.columns)} of {n_cols_raw:,} columns")

    # --- 3. Target Population Filtering (adults with positive weights) ---
    print("Step 3/11: Filtering target population...")
    df = df[(df[WEIGHT_COLUMN] > 0) & (df["AGE23X"] >= 18)].copy()
    print(f"  Kept {len(df):,} of {n_rows_raw:,} rows")

    # --- 4. Data Type Handling ---
    print("Step 4/11: Handling data types...")
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df.set_index(ID_COLUMN, inplace=True)
    print("  Converted ID to string and set as index")

    # --- 5. Missing Value Standardization ---
    print("Step 5/11: Standardizing missing values...")
    # Recover implied values from survey skip patterns
    df.loc[df["ADSMOK42"] == -1, "ADSMOK42"] = 2  # Converts -1 "Never Smoker" → 2 "No"
    df.loc[(df["JTPAIN31_M18"] == -1) & (df["ARTHDX"] == 1), "JTPAIN31_M18"] = 1  # Converts -1 for joint pain to 1 "Yes" if they have Arthritis
    
    # Convert remaining MEPS codes to NaN
    df.replace(MEPS_MISSING_CODES, np.nan, inplace=True)
    print(f"  Recovered missing values from survey skip patterns and converted {df.isnull().sum().sum():,} remaining MEPS missing codes to np.nan")

    # --- 6. Binary Feature Standardization (MEPS 1/2 → 1/0) ---
    print("Step 6/11: Standardizing binary features...")
    df[RAW_BINARY_FEATURES] = df[RAW_BINARY_FEATURES].replace({2: 0})
    print("  Binary features standardized to 0/1")

    # --- 7. Feature Engineering (Stateless) ---
    print("Step 7/11: Engineering stateless features...")
    # Recent Life Transition flag
    df["RECENT_LIFE_TRANSITION"] = (
        df["MARRY31X"].isin(MARRY31X_TRANSITION_CODES) | df["EMPST31"].isin(EMPST31_TRANSITION_CODES)
    ).astype(float)
    df.loc[df["MARRY31X"].isna() & df["EMPST31"].isna(), "RECENT_LIFE_TRANSITION"] = np.nan

    # Collapse marital status transition categories into stable counterparts 
    # Map 7→1 (Married), 8→2 (Widowed), 9→3 (Divorced), 10→4 (Separated)
    df["MARRY31X_GRP"] = df["MARRY31X"].replace(MARRY31X_COLLAPSE_MAP)

    # Collapse employment status categories 
    # Map 2, 3, 4 → 0 (Not Employed)
    df["EMPST31_GRP"] = df["EMPST31"].replace(EMPST31_COLLAPSE_MAP)
    print("  Created RECENT_LIFE_TRANSITION feature and collapsed sparse categories for marrital and employment transitions")

    # --- 8. Train-Validation-Test Split (80/10/10 stratified) ---
    print("Step 8/11: Performing train-validation-test split (80/10/10) with a distribution-informed stratified split...")
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
    print(f"  Split the data into training ({len(X_train):,}), validation ({len(X_val):,}), and test ({len(X_test):,})")

    # --- 9. Preprocessing Pipeline (stateful; fit on train, transform all) ---
    print("Step 9/11: Running preprocessing pipeline with feature standardization, validation, imputation, medical feature engineering, scaling, and encoding...")
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
    print(f"  Created preprocessed training, validation, and test DataFrames with {len(X_train_preprocessed.columns)} output features")

    # --- 10. Verification ---
    print("Step 10/11: Verifying preprocessing results...")
    for name, raw, processed in [
        ("Train", X_train, X_train_preprocessed),
        ("Val", X_val, X_val_preprocessed),
        ("Test", X_test, X_test_preprocessed),
    ]:
        rows_ok = len(raw) == len(processed)
        nulls = processed.isnull().sum().sum()
        numeric_ok = processed.apply(pd.api.types.is_numeric_dtype).all()
        status = "✅" if (rows_ok and nulls == 0 and numeric_ok) else "❌"
        print(f"  {name:5}: {status}  rows={len(processed):,}  nulls={nulls}  all_numeric={numeric_ok}")

    # --- 11. Data Persistence (save parquet files) ---
    print("Step 11/11: Saving preprocessed data...")
    for split_name, X_proc, y_split, X_raw in [
        ("training", X_train_preprocessed, y_train, X_train),
        ("validation", X_val_preprocessed, y_val, X_val),
        ("test", X_test_preprocessed, y_test, X_test),
    ]:
        # Merge preprocessed features, target variable, and sample weights
        df_out = pd.concat([X_proc, y_split, X_raw[WEIGHT_COLUMN]], axis=1)
        path = f"{OUTPUT_DIR}/{split_name}_data_preprocessed.parquet"
        df_out.to_parquet(path)
        print(f"  Saved {path} ({df_out.shape[0]:,} rows × {df_out.shape[1]} cols)")

    print("\n✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
