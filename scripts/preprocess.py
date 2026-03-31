"""
Deterministic data preprocessing from raw MEPS SAS data to model-ready parquet files.

This script implements the production-ready, reproducible version of all data preparation, 
cleaning, preprocessing, and feature engineering steps.

Steps:
  1.  Data Loading: SAS to Pandas.
  2.  Variable Selection: Keep only candidate features, target variable, ID, and weights.
  3.  Population Filtering: Adults (>=18) with positive weights.
  4.  Data Type Handling: Convert IDs to String and assign as index.
  5.  Missing Value Standardization: Recover survey skip patterns and convert missings to np.nan.
  6.  Binary Standardization: Recode MEPS 1/2 codes to 0/1.
  7.  Feature Engineering (Stateless): Create life transition flag and collapse sparse categories.
  8.  Train-Val-Test Split: 80/10/10 stratified split based on target distribution.
  9.  Preprocessing Pipeline (Stateful): Feature standardization, validation imputation, medical 
      feature engineering, scaling, and encoding.
  10. Data Verification: Automated checks for row integrity, missing values, data types and scaling.
  11. Data Persistence: Export to parquet files for model training.

For preprocessing experiments, exploratory data analysis, and detailed rationale, see:
notebooks/1_eda_and_preprocessing.ipynb

Usage:
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

    # --- 10. Data Verification ---
    print("Step 10/11: Verifying preprocessed data...")
    for name, raw, processed in [
        ("Train", X_train, X_train_preprocessed),
        ("Val", X_val, X_val_preprocessed),
        ("Test", X_test, X_test_preprocessed),
    ]:
        # Verify equal row counts between raw and processed data
        rows_match = "✅" if len(raw) == len(processed) else "❌"
        # Verify absence of missing values
        no_nulls = "✅" if processed.isnull().sum().sum() == 0 else "❌"
        # Verify all preprocessed features are numeric (floats/ints)
        all_numeric = "✅" if processed.apply(pd.api.types.is_numeric_dtype).all() else "❌"
        # Verify scaled features have mean=0, std=1
        scaled_features = preprocessor.named_steps["feature_scaler_encoder"].named_transformers_["numerical_scaler"].get_feature_names_out()
        means = processed[scaled_features].mean()
        stds = processed[scaled_features].std(ddof=0)
        is_mean_0 = np.allclose(means, 0, atol=1e-7 if name == "Train" else 0.1)  # small tolerance (1e-7) allows minor floating-point precision errors on train, large tolerance (0.1) allows minor distributions drift on Val and Test
        is_std_1 = np.allclose(stds, 1, atol=1e-7 if name == "Train" else 0.1)  
        scaled = "✅" if (is_mean_0 and is_std_1) else "❌"
        print(f"  {name:5}: Rows Match: {rows_match} | No Missing Values: {no_nulls} | All Numeric: {all_numeric} | Scaled (M=0, Std=1): {scaled}")

    # --- 11. Data Persistence (save parquet files) ---
    print("Step 11/11: Saving preprocessed data...")
    # Merge preprocessed X features, y target variable, and sample weights
    df_train_preprocessed = pd.concat([X_train_preprocessed, y_train, X_train[WEIGHT_COLUMN]], axis=1)
    df_val_preprocessed = pd.concat([X_val_preprocessed, y_val, X_val[WEIGHT_COLUMN]], axis=1)
    df_test_preprocessed = pd.concat([X_test_preprocessed, y_test, X_test[WEIGHT_COLUMN]], axis=1)
    # Save as .parquet files (preserves index, data types, is faster, and requires less storage space than .csv)
    df_train_preprocessed.to_parquet(f"{OUTPUT_DIR}/training_data_preprocessed.parquet")
    df_val_preprocessed.to_parquet(f"{OUTPUT_DIR}/validation_data_preprocessed.parquet")
    df_test_preprocessed.to_parquet(f"{OUTPUT_DIR}/test_data_preprocessed.parquet")
    print(f"  Saved preprocessed features, target variable, and sample weights as .parquet files in {OUTPUT_DIR} directory")

    print("\n✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
