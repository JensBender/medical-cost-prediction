# Third-party library imports
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Local imports
from src.transformers import (
    MissingValueChecker, 
    RobustSimpleImputer,
    MedicalFeatureDeriver
)

# Ensure that the output of all scikit-learn transformers is a Pandas DataFrame
set_config(transform_output="pandas")


# --- Helper functions to create pipelines ---
# Data preprocessing pipeline 
def create_preprocessing_pipeline(
    required_features, 
    optional_features, 
    numerical_features, 
    ordinal_features=None, 
    nominal_features=None, 
    binary_features=None, 
    strict=True
):
    """
    Creates a scikit-learn pipeline for data preprocessing with four steps:
    1. Missing Value Check: Identifies missing values using `MissingValueChecker`.
       It raises a `MissingValueError` for required columns and logs a warning for optional columns.
    2. Missing Value Imputation: Replaces missing values using a `ColumnTransformer` with `RobustSimpleImputer`. 
       - Median imputation for numerical features.
       - Mode imputation for categorical features.
    3. Feature Engineering: Uses `MedicalFeatureDeriver` to create new domain-specific features.

    Args:
        required_features (list): Columns that must not contain missing values.
        optional_features (list): Columns where missing values are tolerated and then imputed.
        numerical_features (list): Names of numerical columns for median imputation.
        ordinal_features (list, optional): Numerical columns for ordinal encoding.
        nominal_features (list, optional): Columns for one-hot encoding.
        binary_features (list, optional): Columns to pass through (already 0/1).
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A complete data preprocessing pipeline.
    """
    # Combine categorical features for imputation
    categorical_features = (ordinal_features or []) + (nominal_features or []) + (binary_features or [])

    return Pipeline(steps=[
        ("missing_value_checker", MissingValueChecker(required_features, optional_features, strict=strict)),
        ("missing_value_imputer", ColumnTransformer(
            transformers=[
                ("numerical_imputer", RobustSimpleImputer(strategy="median"), numerical_features),
                ("categorical_imputer", RobustSimpleImputer(strategy="most_frequent"), categorical_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )),
        ("medical_feature_deriver", MedicalFeatureDeriver()),
        ("feature_transformer", ColumnTransformer(
            transformers=[
                ("numerical_scaler", StandardScaler(), numerical_features + MedicalFeatureDeriver.OUTPUT_FEATURES),
                ("ordinal_encoder", OrdinalEncoder(), ordinal_features or []),
                ("nominal_encoder", OneHotEncoder(drop="first", sparse_output=False), nominal_features or []),
                ("binary_passthrough", "passthrough", binary_features or [])
            ],
            remainder="drop",
            verbose_feature_names_out=False
        ))
    ])


# Missing value handling pipeline (component/sub-pipeline of the data preprocessing pipeline)
def create_missing_value_handling_pipeline(
    required_features, 
    optional_features, 
    numerical_features, 
    ordinal_features=None, 
    nominal_features=None, 
    binary_features=None, 
    strict=True
):
    """
    Creates a scikit-learn pipeline for missing value handling with two steps:
    1. Missing Value Check: Identifies missing values using `MissingValueChecker`.
       It raises a `MissingValueError` for required columns and logs a warning for optional columns.
    2. Imputation: Replaces missing values using a `ColumnTransformer` with `RobustSimpleImputer`. 
       - Median imputation for numerical features.
       - Mode imputation for categorical features.

    Args:
        required_features (list): Columns that must not contain missing values.
        optional_features (list): Columns where missing values are tolerated and then imputed.
        numerical_features (list): Names of numerical columns for median imputation.
        ordinal_features (list, optional): Ordinal columns for mode imputation.
        nominal_features (list, optional): Nominal columns for mode imputation.
        binary_features (list, optional): Binary columns for mode imputation.
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline configured for missing value handling.
    """
    # Combine categorical features for imputation
    categorical_features = (ordinal_features or []) + (nominal_features or []) + (binary_features or [])
    return Pipeline(steps=[
        ("missing_value_checker", MissingValueChecker(required_features, optional_features, strict=strict)),
        ("missing_value_imputer", ColumnTransformer(
            transformers=[
                ("numerical_imputer", RobustSimpleImputer(strategy="median"), numerical_features),
                ("categorical_imputer", RobustSimpleImputer(strategy="most_frequent"), categorical_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False  # Preserves input column names instead of adding prefix 
        ))
    ])
