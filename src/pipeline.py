# Third-party library imports
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Local imports
from src.transformers import MissingValueChecker

# Ensure that the output of all scikit-learn transformers is a Pandas DataFrame
set_config(transform_output="pandas")


# Helper function to create the data preprocessing pipeline
def create_preprocessing_pipeline(required_features, optional_features, numerical_features, categorical_features, strict=True):
    """
    Creates a scikit-learn pipeline for data preprocessing.
    In this version, it only handles missing values.
    """
    return Pipeline(steps=[
        ("missing_value_checker", MissingValueChecker(required_features, optional_features, strict=strict)),
        ("missing_value_imputer", ColumnTransformer(
            transformers=[
                ("numerical_imputer", SimpleImputer(strategy="median"), numerical_features),
                ("categorical_imputer", SimpleImputer(strategy="most_frequent"), categorical_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False  # Preserves input column names instead of adding prefix 
        ))
    ])
