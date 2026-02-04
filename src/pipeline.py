from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Ensure that the output of all scikit-learn transformers is a Pandas DataFrame
set_config(transform_output="pandas")


# Helper function to create the data preprocessing pipeline
def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Creates a scikit-learn pipeline for data preprocessing.
    In this version, it only handles missing values by using a ColumnTransformer to impute the median 
    for numerical features and the mode for categorical features.
    """
    
    # Numerical transformer: Median imputation
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    # Categorical transformer: Mode imputation
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    # Bundle preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop" 
    )
    
    return preprocessor

