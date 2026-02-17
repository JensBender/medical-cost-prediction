from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import check_is_fitted
from .constants import DISPLAY_LABELS
import pandas as pd


# --- Custom error classes --- 
# For missing values in required features of the X input DataFrame (in MissingValueChecker)
class MissingValueError(ValueError):
    """Custom error for missing values in required features.
    
    Attributes:
        details (dict): Structured information about the error for API/programmatic usage.
    """
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}

    def to_dict(self):
        """Returns a dictionary representation of the error for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "details": self.details
        }

# For mismatch between expected and actual columns in the X input DataFrame (missing, unexpected, or wrong order)
class ColumnMismatchError(ValueError):
    """Custom error for mismatch between expected and actual columns.
    
    Attributes:
        details (dict): Structured information about the mismatch for API/programmatic usage.
    """
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}

    def to_dict(self):
        """Returns a dictionary representation of the error for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "details": self.details
        }


# --- Custom transformer classes --- 
# Transformer class to check missing values ---
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
            details = {
                "missing_columns": list(missing_columns),
                "expected_columns": list(expected_columns),
                "actual_columns": list(input_columns)
            }
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.", details=details)

        # Ensure DataFrame has no unexpected columns
        unexpected_columns = input_columns - expected_columns
        if unexpected_columns:
            details = {
                "unexpected_columns": list(unexpected_columns),
                "expected_columns": list(expected_columns),
                "actual_columns": list(input_columns)
            }
            raise ColumnMismatchError(f"Input X contains the following columns that are neither defined in 'required_features' nor in 'optional_features': {', '.join(unexpected_columns)}.", details=details)

    def _check_missing_values(self, X):
        """Internal helper to identify missing values and either raise errors or print warnings."""
        # Required features 
        missing_mask_required = X[self.required_features].isnull()
        n_missing_required = missing_mask_required.sum().sum()
        
        if n_missing_required > 0:
            # Identify columns and row indices with missing values
            failed_columns = missing_mask_required.columns[missing_mask_required.any()].tolist()
            failed_indices = X.index[missing_mask_required.any(axis=1)].tolist()
            n_missing_rows_required = len(failed_indices)
            
            # Format row index report (truncate to top 5 for the message string)
            truncated_indices = failed_indices[:5]
            index_report = str(truncated_indices) + ("..." if n_missing_rows_required > 5 else "")
            
            # Grammatical helpers
            values_word = "value" if n_missing_required == 1 else "values"
            rows_word = "row" if n_missing_rows_required == 1 else "rows"
            
            # Identify display labels for failed columns
            failed_labels = [DISPLAY_LABELS.get(col, col) for col in failed_columns]

            # Craft detailed summary message
            msg = (
                f"Missing Value Error: {n_missing_required} missing {values_word} found in required features "
                f"across {n_missing_rows_required} {rows_word}.\n"
                f"Affected Features: {failed_columns} ({failed_labels})\n"
                f"Affected Row Indices: {index_report}"
            )
            
            if self.strict:  
                details = {
                    "total_missing_values": int(n_missing_required),
                    "total_affected_rows": int(n_missing_rows_required),
                    "affected_features": failed_columns,
                    "affected_labels": failed_labels,
                    "affected_row_indices": [str(idx) for idx in failed_indices]
                }
                raise MissingValueError(msg, details=details)
            else:
                print(f"Warning: {msg}\nThese will be imputed.")

        # Optional features
        if not self.optional_features:
            return

        missing_mask_optional = X[self.optional_features].isnull()
        n_missing_optional = missing_mask_optional.sum().sum()

        if n_missing_optional > 0:
            failed_columns_opt = missing_mask_optional.columns[missing_mask_optional.any()].tolist()
            failed_indices_opt = X.index[missing_mask_optional.any(axis=1)].tolist()
            n_missing_rows_optional = len(failed_indices_opt)
            
            # Format row index report (truncate to top 5 for the message string)
            truncated_indices_opt = failed_indices_opt[:5]
            index_report_opt = str(truncated_indices_opt) + ("..." if n_missing_rows_optional > 5 else "")

            # Grammatical helpers
            values_word = "value" if n_missing_optional == 1 else "values"
            rows_word = "row" if n_missing_rows_optional == 1 else "rows"
            
            # Identify display labels for categorical features
            failed_labels_opt = [DISPLAY_LABELS.get(col, col) for col in failed_columns_opt]

            print(
                f"Warning: {n_missing_optional} missing {values_word} found in optional features "
                f"across {n_missing_rows_optional} {rows_word}.\n"
                f"Affected Features: {failed_columns_opt} ({failed_labels_opt})\n"
                f"Affected Row Indices: {index_report_opt}\n"
                f"These will be imputed."
            )
            
    def fit(self, X, y=None):
        # Validate input 
        self._validate_input(X)  

        # Ensure no feature is 100% missing (imputer would fail)
        all_features = self.required_features + self.optional_features
        for feature in all_features:
            if X[feature].isnull().all():
                details = {
                    "feature": feature,
                    "label": DISPLAY_LABELS.get(feature, feature),
                    "reason": "Column is 100% missing"
                }
                raise MissingValueError(
                    f"'{feature}' contains only missing values. At least one non-missing value is required to fit the imputer.",
                    details=details
                )

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
            details = {
                "expected_order": self.feature_names_in_,
                "actual_order": X.columns.tolist(),
                "reason": "Column order mismatch or different set of columns"
            }
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().", details=details)      
        
        # Check missing values 
        self._check_missing_values(X)

        return X


