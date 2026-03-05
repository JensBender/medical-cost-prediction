# Third-party library imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

# Local imports
from src.transformers import (
    MissingValueChecker, 
    RobustSimpleImputer,
    MedicalFeatureDeriver,
    RobustStandardScaler,
    RobustOneHotEncoder,
    RobustOrdinalEncoder
)

# Ensure that the output of all scikit-learn transformers is a Pandas DataFrame
set_config(transform_output="pandas")


# --- Helper functions to create pipelines ---
# Data preprocessing pipeline 
def create_preprocessing_pipeline(
    required_features, 
    optional_features, 
    numerical_features, 
    ordinal_features, 
    nominal_features, 
    binary_features, 
    strict=True
):
    """
    Creates a scikit-learn pipeline for data preprocessing with four steps:
    1. Missing Value Check: Identifies missing values using `MissingValueChecker`.
       It raises a `MissingValueError` for required columns and logs a warning for optional columns.
    2. Missing Value Imputation: Replaces missing values using a `ColumnTransformer` with `RobustSimpleImputer`. 
       - Median imputation for numerical features.
       - Mode imputation for categorical features.
    3. Feature Engineering: Uses `MedicalFeatureDeriver` to aggregate binary indicators into counts:
       - `CHRONIC_COUNT`: Sum of chronic medical condition flags.
       - `LIMITATION_COUNT`: Sum of functional limitation flags.
    4. Feature Scaling and Encoding: Transforms all features into a model-ready format:
       - `RobustStandardScaler`: Scales numerical features (raw and engineered) to mean 0 and variance 1.
       - `RobustOneHotEncoder`: Converts nominal features into binary dummy variables (dropping first).
       - `RobustOrdinalEncoder`: Encodes ordinal features while preserving their inherent order.
       - Binary Passthrough: Preserves original binary features without modification.

    Args:
        required_features (list): Columns that must not contain missing values.
        optional_features (list): Columns where missing values are tolerated and then imputed.
        numerical_features (list): Numerical column names for median imputation and scaling.
        ordinal_features (list): Ordinal column names for mode imputation and ordinal encoding.
        nominal_features (list): Nominal column names for mode imputation and one-hot encoding.
        binary_features (list): Binary column names for mode imputation and to pass through encoder (already 0/1).
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A complete data preprocessing pipeline.
    """
    categorical_features = ordinal_features + nominal_features + binary_features

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
        ("feature_scaler_encoder", ColumnTransformer(
            transformers=[
                ("numerical_scaler", RobustStandardScaler(), numerical_features + MedicalFeatureDeriver.OUTPUT_FEATURES),
                ("ordinal_encoder", RobustOrdinalEncoder(), ordinal_features),
                ("nominal_encoder", RobustOneHotEncoder(drop="first", sparse_output=False), nominal_features),
                ("binary_passthrough", "passthrough", binary_features)
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
    ordinal_features, 
    nominal_features, 
    binary_features, 
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
        numerical_features (list): Numerical columns for median imputation.
        ordinal_features (list): Ordinal columns for mode imputation.
        nominal_features (list): Nominal columns for mode imputation.
        binary_features (list): Binary columns for mode imputation.
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline configured for missing value handling.
    """
    # Combine categorical features for imputation
    categorical_features = ordinal_features + nominal_features + binary_features
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
