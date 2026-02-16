from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import check_is_fitted
import pandas as pd


# --- Custom error classes --- 
# For missing values in required columns of the X input DataFrame (in MissingValueChecker)
class MissingValueError(ValueError):
    pass

# For mismatch between expected and actual columns in X input DataFrame because of missing columns, unexpected columns, or wrong column order 
class ColumnMismatchError(ValueError):
    pass


# --- Custom transformer classes --- 
# Custom transformer class to check missing values ---
class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, required_features, optional_features=None):
        # Default optional_features to an empty list if not provided
        if optional_features is None:
            optional_features = []

        # Ensure "required_features" and "optional_features" are lists
        if not isinstance(required_features, list):
            raise TypeError("'required_features' must be a list of feature names.")
        if not isinstance(optional_features, list):
            raise TypeError("'optional_features' must be a list of feature names.")

        # Ensure required features list is not empty
        if not required_features:
            raise ValueError("'required_features' cannot be an empty list. It must specify the names of the required features.")

        self.required_features = required_features
        self.optional_features = optional_features
    
    def _validate_input(self, X):
        # Ensure X input is DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")          
        
        # Ensure DataFrame has no missing columns
        input_columns = set(X.columns)
        expected_columns = set(self.required_features + self.optional_features)
        missing_columns = expected_columns - input_columns 
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")

        # Ensure DataFrame has no unexpected columns
        unexpected_columns = input_columns - expected_columns
        if unexpected_columns:
            raise ColumnMismatchError(f"Input X contains the following columns that are neither defined in 'required_features' nor in 'optional_features': {', '.join(unexpected_columns)}.")

    def _check_missing_values(self, X):
        # --- Required features ---
        # Calculate total number of missing values  
        n_missing_total_required = X[self.required_features].isnull().sum().sum()
        # Calculate number of rows with missing values  
        n_missing_rows_required = X[self.required_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_required = X[self.required_features].isnull().sum().to_dict()
        # Raise error  
        if n_missing_total_required > 0:
            values = "value" if n_missing_total_required == 1 else "values"
            rows = "row" if n_missing_rows_required == 1 else "rows"
            raise MissingValueError(
                f"{n_missing_total_required} missing {values} found in required features "
                f"across {n_missing_rows_required} {rows}. Please provide missing {values}.\n"
                f"Missing values by column: {n_missing_by_column_required}"
            )

        # --- Optional features ---
        if not self.optional_features:
            return

        # Calculate total number of missing values 
        n_missing_total_optional = X[self.optional_features].isnull().sum().sum()        
        # Calculate number of rows with missing values 
        n_missing_rows_optional = X[self.optional_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_optional = X[self.optional_features].isnull().sum().to_dict()
        # Display warning message
        if n_missing_total_optional > 0:
            values = "value" if n_missing_total_optional == 1 else "values"
            rows = "row" if n_missing_rows_optional == 1 else "rows"
            print(
                f"Warning: {n_missing_total_optional} missing {values} found in optional features "
                f"across {n_missing_rows_optional} {rows}. Missing {values} will be imputed.\n"
                f"Missing values by column: {n_missing_by_column_optional}"
            )
            
    def fit(self, X, y=None):
        # Validate input 
        self._validate_input(X)  

        # Check missing values
        self._check_missing_values(X)

        # Raise MissingValueError if an optional feature has only missing values
        for optional_feature in self.optional_features:
            if X[optional_feature].isnull().all():
                raise MissingValueError(f"'{optional_feature}' cannot be only missing values. Please ensure at least one non-missing value.")

        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self 

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input 
        self._validate_input(X)    
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      
        
        # Check missing values
        self._check_missing_values(X)

        return X
