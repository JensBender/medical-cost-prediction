# Third-party library imports
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer

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
def create_preprocessing_pipeline(required_features, optional_features, numerical_features, categorical_features, strict=True):
    """
    Creates a scikit-learn pipeline for data preprocessing with three steps:
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
        categorical_features (list): Names of categorical columns for mode imputation.
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A complete data preprocessing pipeline.
    """
    return Pipeline(steps=[
        ("missing_value_checker", MissingValueChecker(required_features, optional_features, strict=strict)),
        ("missing_value_imputer", ColumnTransformer(
            transformers=[
                ("numerical_imputer", RobustSimpleImputer(strategy="median"), numerical_features),
                ("categorical_imputer", RobustSimpleImputer(strategy="most_frequent"), categorical_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False  # Preserves input column names instead of adding prefix 
        )),
        ("medical_feature_deriver", MedicalFeatureDeriver())
    ])


# Missing value handling pipeline (component/sub-pipeline of the data preprocessing pipeline)
def create_missing_value_handling_pipeline(required_features, optional_features, numerical_features, categorical_features, strict=True):
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
        categorical_features (list): Names of categorical columns for mode imputation.
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline configured for missing value handling.
    """
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
