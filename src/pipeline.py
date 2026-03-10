# Third-party library imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

# Local imports
from src.constants import (
    CATEGORY_LABELS_PIPELINE,
    NOMINAL_CATEGORIES,
    NOMINAL_DROP_CATEGORIES 
)
from src.transformers import (
    MissingValueChecker, 
    CategoricalLabelStandardizer,
    RobustSimpleImputer,
    MedicalFeatureDeriver,
    RobustStandardScaler,
    RobustOneHotEncoder
)

# Ensure that the output of all scikit-learn transformers is a Pandas DataFrame
set_config(transform_output="pandas")


# --- Helper functions to create pipelines ---
# Data preprocessing pipeline 
def create_preprocessing_pipeline(
    required_features, 
    optional_features, 
    numerical_features, 
    nominal_features, 
    binary_features, 
    strict=True
):
    """
    Creates a scikit-learn pipeline for data preprocessing with five steps:
    1. Categorical Label Standardization: Uses `CategoricalLabelStandardizer` to normalize 
       inputs (accepts both numeric codes and string labels) and ensures 
       nominal features use descriptive strings for human-readable encoded names. 
    2. Missing Value Check: Identifies missing values using `MissingValueChecker`.
       It raises a `MissingValueError` for required columns and logs a warning for optional columns.
    3. Missing Value Imputation: Replaces missing values using a `ColumnTransformer` with `RobustSimpleImputer`. 
       - Median imputation for numerical features (including ordinal).
       - Mode imputation for categorical features.
    4. Feature Engineering: Uses `MedicalFeatureDeriver` to aggregate binary indicators into counts:
       - `CHRONIC_COUNT`: Sum of chronic medical condition flags.
       - `LIMITATION_COUNT`: Sum of functional limitation flags.
    5. Feature Scaling and Encoding: Transforms all features into a model-ready format:
       - `RobustStandardScaler`: Scales numerical features (raw, ordinal, and engineered) to mean 0 and variance 1.
       - `RobustOneHotEncoder`: Converts nominal features into binary dummy variables (dropping preferred baselines).
       - Binary Passthrough: Preserves original binary features without modification.

    Args:
        required_features (list): Columns that must not contain missing values.
        optional_features (list): Columns where missing values are tolerated and then imputed.
        numerical_features (list): Numerical and ordinal column names for median imputation and scaling.
        nominal_features (list): Nominal column names for mode imputation and one-hot encoding.
        binary_features (list): Binary column names for mode imputation and to pass through encoder (already 0/1).
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A complete data preprocessing pipeline.
    """
    categorical_features = nominal_features + binary_features

    return Pipeline(steps=[
        ("categorical_label_standardizer", CategoricalLabelStandardizer(binary_features, nominal_features, categorical_label_map=CATEGORY_LABELS_PIPELINE)),
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
                ("nominal_encoder", RobustOneHotEncoder(drop=NOMINAL_DROP_CATEGORIES, categories=NOMINAL_CATEGORIES, sparse_output=False), nominal_features),
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
    nominal_features, 
    binary_features, 
    strict=True
):
    """
    Creates a scikit-learn pipeline for missing value handling with three steps:
    1. Categorical Label Standardization: Uses `CategoricalLabelStandardizer` to normalize 
       inputs into string labels for nominal features and numeric codes for binary. 
    2. Missing Value Check: Identifies missing values using `MissingValueChecker`.
       It raises a `MissingValueError` for required columns and logs a warning for optional columns.
    3. Imputation: Replaces missing values using a `ColumnTransformer` with `RobustSimpleImputer`. 
       - Median imputation for numerical features.
       - Mode imputation for categorical features.

    Args:
        required_features (list): Columns that must not contain missing values.
        optional_features (list): Columns where missing values are tolerated and then imputed.
        numerical_features (list): Numerical columns (including ordinal) for median imputation.
        nominal_features (list): Nominal columns for mode imputation.
        binary_features (list): Binary columns for mode imputation.
        strict (bool, optional): If True, pipeline raises error for missing required values. Defaults to True.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline configured for missing value handling.
    """
    # Combine categorical features for imputation
    categorical_features = nominal_features + binary_features
    return Pipeline(steps=[
        ("categorical_label_standardizer", CategoricalLabelStandardizer(binary_features, nominal_features, categorical_label_map=CATEGORY_LABELS_PIPELINE)),
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
