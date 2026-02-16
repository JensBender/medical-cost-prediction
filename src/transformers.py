from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import check_is_fitted
import pandas as pd


# --- Custom error classes --- 
# For missing values in required columns of the X input DataFrame (in MissingValueChecker)
class MissingValueError(ValueError):
    def __init__(self, message, missing_columns=None, failed_indices=None):
        super().__init__(message)
        self.missing_columns = missing_columns or []
        self.failed_indices = failed_indices or []

# For mismatch between expected and actual columns in X input DataFrame (missing, unexpected, or wrong order)
class ColumnMismatchError(ValueError):
    pass


# --- Custom transformer classes --- 
# Custom transformer class to check missing values ---
class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, required_features, optional_features=None, strict=True):
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
        self.strict = strict
    
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
        """Internal helper to identify missing values and either raise errors or print warnings."""
        # Required features 
        missing_mask_required = X[self.required_features].isnull()
        n_missing_total_required = missing_mask_required.sum().sum()
        
        if n_missing_total_required > 0:
            # Identify columns and row indices with missing values
            failed_columns = missing_mask_required.columns[missing_mask_required.any()].tolist()
            failed_indices = X.index[missing_mask_required.any(axis=1)].tolist()
            n_missing_rows_required = len(failed_indices)
            
            # Format row index report (truncate to top 5 for the message string)
            truncated_indices = failed_indices[:5]
            index_report = str(truncated_indices) + ("..." if n_missing_rows_required > 5 else "")
            
            # Grammatical helpers
            values_word = "value" if n_missing_total_required == 1 else "values"
            rows_word = "row" if n_missing_rows_required == 1 else "rows"
            
            # Craft detailed summary message
            msg = (
                f"Missing Value Error: {n_missing_total_required} missing {values_word} found in required features "
                f"across {n_missing_rows_required} {rows_word}.\n"
                f"Affected Features: {failed_columns}\n"
                f"Affected Row Indices: {index_report}"
            )
            
            if self.strict:  
                raise MissingValueError(msg, missing_columns=failed_columns, failed_indices=failed_indices)
            else:
                print(f"Warning: {msg}\nThese will be imputed.")

        # Optional features
        if not self.optional_features:
            return

        missing_mask_optional = X[self.optional_features].isnull()
        n_missing_total_optional = missing_mask_optional.sum().sum()

        if n_missing_total_optional > 0:
            failed_columns_opt = missing_mask_optional.columns[missing_mask_optional.any()].tolist()
            n_missing_rows_optional = missing_mask_optional.any(axis=1).sum()
            
            # Grammatical helpers
            values_word = "value" if n_missing_total_optional == 1 else "values"
            rows_word = "row" if n_missing_rows_optional == 1 else "rows"
            
            print(
                f"Warning: {n_missing_total_optional} missing {values_word} found in optional features "
                f"across {n_missing_rows_optional} {rows_word}.\n"
                f"Affected Features: {failed_columns_opt}\n"
                f"Missing values will be imputed."
            )
            
    def fit(self, X, y=None):
        # Validate input 
        self._validate_input(X)  

        # Ensure no feature is 100% missing (imputer would fail)
        all_features = self.required_features + self.optional_features
        for feature in all_features:
            if X[feature].isnull().all():
                raise MissingValueError(f"'{feature}' contains only missing values. At least one non-missing value is required to fit the imputer.")

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


