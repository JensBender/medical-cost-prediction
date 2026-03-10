from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import validation as sklearn_validation
import pandas as pd
import logging

# Set up logger
logger = logging.getLogger(__name__)


# --- Custom error classes --- 
# For missing values in required features of the provided DataFrame (in MissingValueChecker)
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

# For missing columns in the provided DataFrame
class MissingColumnError(ValueError):
    """Custom error for missing columns.
    
    Attributes:
        details (dict): Structured information about the missing columns for API/programmatic usage.
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
class CategoricalLabelStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardizes categorical feature labels to ensure pipeline robustness and interpretability.

    This transformer acts as a "normalization layer" for categorical inputs, enabling 
    flexible input formats (both numeric codes and string labels) for pipeline users. 
    Standardization is applied based on categorical type:
    
    - Binary Features: Standardizes to numeric 0/1 codes. This allows these 
      features to be passed through the feature scaler and encoder downstream 
      in the pipeline without further transformation.
    - Nominal Features: Standardizes to descriptive string labels. This ensures 
      that the one-hot encoder generates human-readable feature names (e.g., 
      'REGION23_Midwest' instead of 'REGION23_2.0').
    
    Validation Logic:
    - Performs case-insensitive matching for string-to-numeric conversion of 
      binary features.
    - Preserves categorical values that already match the target format.
    - Maintains original values (or NaNs) if no mapping is found.
    - Raises `TypeError` if the provided input is not a pandas DataFrame.
    """
    def __init__(self, binary_features=None, nominal_features=None, categorical_label_map=None):
        self.binary_features = binary_features or []
        self.nominal_features = nominal_features or []
        self.categorical_label_map = categorical_label_map

    def fit(self, X, y=None):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")
        
        # Use provided map or empty dict as fallback
        label_map = self.categorical_label_map or {}
        
        X = X.copy()
        for col in X.columns:
            if col in label_map:
                mapping = label_map[col]
                
                if col in self.binary_features:
                    # Binary features: Ensure 0/1 numeric codes
                    # Create reverse mapping (String -> Int)
                    rev_map = {str(v).lower(): k for k, v in mapping.items()}
                    
                    def standardize_binary(val):
                        if pd.isna(val):
                            return val
                        # Try string lookup first (case-insensitive)
                        str_val = str(val).lower()
                        if str_val in rev_map:
                            return rev_map[str_val]
                        # Otherwise assume it's already a numeric code
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return val

                    X[col] = X[col].map(standardize_binary)
                
                elif col in self.nominal_features:
                    # Nominal features: Ensure descriptive string labels
                    def standardize_nominal(val):
                        if pd.isna(val):
                            return val
                        # If it's already a string in our mapping values, keep it
                        if val in mapping.values():
                            return val
                        # Otherwise try to map from numeric code
                        return mapping.get(val, val)

                    X[col] = X[col].map(standardize_nominal)
        
        return X

    def get_feature_names_out(self, input_features=None):
        sklearn_validation.check_is_fitted(self)
        return self.feature_names_in_


class MissingValueChecker(BaseEstimator, TransformerMixin):
    """
    Validates the presence of missing values across required and optional features.

    In strict mode (default), any missing values in 'required_features' will raise a 
    MissingValueError, making it suitable for production/deployment validation. 
    In non-strict mode, these are instead logged as warnings, allowing the 
    pipeline to continue (typically for training where imputation is acceptable).
    Missing values in 'optional_features' always trigger a warning and are 
    expected to be handled by downstream imputation.
    """
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
            raise TypeError("The provided input X must be a pandas DataFrame.")          
        
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
            raise MissingColumnError(f"The provided DataFrame is missing the following columns: {', '.join(missing_columns)}.", details=details)

        # Log unexpected columns but do not raise an error 
        unexpected_columns = input_columns - expected_columns
        if unexpected_columns:
            logger.warning(
                f"Unexpected Column Warning: The provided DataFrame contains columns that are not in the feature list and will be ignored.\n"
                f"- Unexpected Columns: {', '.join(unexpected_columns)}."
            )

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
            
            # Format failed columns and indices report (truncate to top 5 for the message string)
            failed_columns_report = str(failed_columns[:5]) + ("..." if len(failed_columns) > 5 else "")
            failed_indices_report = str(failed_indices[:5]) + ("..." if n_missing_rows_required > 5 else "")
            
            # Grammatical helpers
            values_word = "value" if n_missing_required == 1 else "values"
            rows_word = "row" if n_missing_rows_required == 1 else "rows"
            
            # Craft detailed summary message
            msg = (
                f"{n_missing_required} missing {values_word} found in required features "
                f"across {n_missing_rows_required} {rows_word}. {'' if self.strict else 'These will be imputed.'}\n"
                f"- Affected Features: {failed_columns_report}\n"
                f"- Affected Row Indices: {failed_indices_report}"
            )
            
            if self.strict:  
                details = {
                    "n_missing": int(n_missing_required),
                    "n_missing_rows": int(n_missing_rows_required),
                    "affected_features": failed_columns,
                    "affected_row_indices": [str(idx) for idx in failed_indices]
                }
                raise MissingValueError(msg, details=details)
            else:
                logger.warning(f"Missing Value Warning: {msg}")

        # Optional features
        if not self.optional_features:
            return

        missing_mask_optional = X[self.optional_features].isnull()
        n_missing_optional = missing_mask_optional.sum().sum()

        if n_missing_optional > 0:
            failed_columns_opt = missing_mask_optional.columns[missing_mask_optional.any()].tolist()
            failed_indices_opt = X.index[missing_mask_optional.any(axis=1)].tolist()
            n_missing_rows_optional = len(failed_indices_opt)
            
            # Format failed columns and indices report (truncate to top 5 for the message string)
            failed_columns_opt_report = str(failed_columns_opt[:5]) + ("..." if len(failed_columns_opt) > 5 else "")
            failed_indices_opt_report = str(failed_indices_opt[:5]) + ("..." if n_missing_rows_optional > 5 else "")

            # Grammatical helpers
            values_word = "value" if n_missing_optional == 1 else "values"
            rows_word = "row" if n_missing_rows_optional == 1 else "rows"
            
            logger.warning(
                f"Missing Value Warning: {n_missing_optional} missing {values_word} found in optional features "
                f"across {n_missing_rows_optional} {rows_word}. These will be imputed.\n"
                f"- Affected Features: {failed_columns_opt_report}\n"
                f"- Affected Row Indices: {failed_indices_opt_report}"
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
        sklearn_validation.check_is_fitted(self)
        
        # Validate input 
        self._validate_input(X)    
        
        # Check missing values 
        self._check_missing_values(X)

        return X

    def get_feature_names_out(self, input_features=None):
        sklearn_validation.check_is_fitted(self)
        # Returns the input feature names
        return self.feature_names_in_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        self._check_missing_values(X)
        return X


class RobustSimpleImputer(SimpleImputer):
    """
    A wrapper for SimpleImputer that handles empty DataFrames gracefully.

    Instead of raising a ValueError (SimpleImputer's default behavior) when 
    encountering an empty DataFrame during .transform(), this class returns 
    the empty DataFrame as-is. This is useful for production pipelines where 
    batch processing might produce zero-row outputs.

    Validation Logic:
    Consistent with other transformers in this pipeline, this class requires 
    input 'X' to be a pandas DataFrame and will raise a TypeError otherwise.

    Note:
        During transform(), if 'X' is empty (X.empty is True), the original 
        input is returned without imputation.
    """
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")
        
        if X.empty:
            return X
            
        # Perform imputation
        # Note: strategy="most_frequent" with mixed categorical column types (strings and numbers) in same imputer converts numeric dtypes to object
        result = super().transform(X)
        
        # Restore numeric dtypes where possible
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                try:
                    result[col] = pd.to_numeric(result[col])
                except (ValueError, TypeError):
                    continue  # Keep as object/string if non-numeric (e.g., nominal labels)
            return result
        return result


class MedicalFeatureDeriver(BaseEstimator, TransformerMixin):
    """
    Derives medical features from raw indicators based on medical domain knowledge.

    This transformer calculates aggregate counts of chronic conditions and 
    functional limitations. It is designed to be placed AFTER imputation in 
    the pipeline to ensure a deterministic derivation from complete data.

    Generated Features:
    - `CHRONIC_COUNT`: Integer sum of binary flags for chronic conditions
      (e.g., Blood Pressure, Cholesterol, Diabetes, etc.).
    - `LIMITATION_COUNT`: Integer sum of binary flags for functional 
      limitations (e.g., ADL, IADL, Cognitive limitations).

    Validation Logic:
    This transformer performs strict input validation during both `fit` and `transform`:
    - Raises `TypeError` if the input is not a pandas DataFrame.
    - Raises `MissingColumnError` if any source features are missing from the input.
    - Raises `MissingValueError` if any source features contain NaNs, ensuring
      deterministic and non-biased feature derivation.
    """
    
    # Define input features used to derive new features
    CHRONIC_CONDITION_FEATURES = [
        "HIBPDX", "CHOLDX", "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"
    ]

    FUNCTIONAL_LIMITATION_FEATURES = [
        "ADLHLP31", "IADLHP31", "WLKLIM31", "COGLIM31", "JTPAIN31_M18"
    ]

    # Features created by this transformer
    OUTPUT_FEATURES = ["CHRONIC_COUNT", "LIMITATION_COUNT"]

    def _validate_input(self, X):
        # Ensure X input is DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")          
        
        # Ensure DataFrame has no missing columns
        input_columns = set(X.columns)
        expected_columns = set(self.CHRONIC_CONDITION_FEATURES + self.FUNCTIONAL_LIMITATION_FEATURES)
        missing_columns = expected_columns - input_columns 
        if missing_columns:
            details = {
                "missing_columns": list(missing_columns),
                "expected_columns": list(expected_columns),
                "actual_columns": list(input_columns)
            }
            raise MissingColumnError(f"The provided DataFrame is missing the following columns: {', '.join(missing_columns)}.", details=details)

        # Ensure input features have no missing values
        missing_mask = X[list(expected_columns)].isnull()
        n_missing = missing_mask.sum().sum()
        if n_missing > 0:
            # Identify input features and row indices with missing values
            missing_features = missing_mask.columns[missing_mask.any()].tolist()
            missing_rows = X.index[missing_mask.any(axis=1)].tolist()
            n_missing_rows = len(missing_rows)
            
            # Create error message (truncate to top 5 features and rows for the message string)
            missing_features_msg = str(missing_features[:5]) + ("..." if len(missing_features) > 5 else "")
            missing_rows_msg = str(missing_rows[:5]) + ("..." if n_missing_rows > 5 else "")
            values_word = "value" if n_missing == 1 else "values"
            msg = (
                f"MedicalFeatureDeriver found {n_missing} missing {values_word}, but requires complete data for all source features used to derive new features. Make sure to handle missing values first.\n"
                f"- Affected Features: {missing_features_msg}\n"
                f"- Affected Row Indices: {missing_rows_msg}"
            )    

            # Create error detail        
            details = {
                "n_missing": int(n_missing),
                "affected_features": missing_features,
                "affected_row_indices": [str(idx) for idx in missing_rows]
            }

            raise MissingValueError(msg, details=details)

    def fit(self, X, y=None):
        # Validate input 
        self._validate_input(X)
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        # Store output feature names
        self.feature_names_out_ = X.columns.tolist() + self.OUTPUT_FEATURES
        return self

    def transform(self, X):
        # Ensure .fit() happened before
        sklearn_validation.check_is_fitted(self)
        
        # Validate input 
        self._validate_input(X)
        
        # Performance optimization: Ensure binary indicators are treated as floats before summation. 
        # This prevents the overhead computation if columns were converted from float to object by a previous step.
        feature_subset = X[self.CHRONIC_CONDITION_FEATURES + self.FUNCTIONAL_LIMITATION_FEATURES].apply(pd.to_numeric, errors="coerce")
            
        return X.assign(
            # Derive Chronic Conditions Count
            CHRONIC_COUNT=feature_subset[self.CHRONIC_CONDITION_FEATURES].sum(axis=1),
            # Derive Functional Limitations Count
            LIMITATION_COUNT=feature_subset[self.FUNCTIONAL_LIMITATION_FEATURES].sum(axis=1)
        )

    def get_feature_names_out(self, input_features=None):
        sklearn_validation.check_is_fitted(self)
        # Return the output feature names, including derived ones
        return self.feature_names_out_


class RobustStandardScaler(StandardScaler):
    """
    A wrapper for StandardScaler that handles empty DataFrames gracefully.

    Instead of raising a ValueError (StandardScaler's default behavior) when 
    encountering an empty DataFrame during .transform(), this class returns 
    the empty DataFrame as-is. This is useful for production pipelines where 
    batch processing might produce zero-row outputs.

    Validation Logic:
    Consistent with other transformers in this pipeline, this class requires 
    input 'X' to be a pandas DataFrame and will raise a TypeError otherwise.

    Note:
        During transform(), if 'X' is empty (X.empty is True), the original 
        input is returned without scaling.
    """
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")
    
        if X.empty:
            return X
        return super().transform(X)


class RobustOneHotEncoder(OneHotEncoder):
    """
    A wrapper for OneHotEncoder that handles empty DataFrames gracefully.

    Instead of raising a ValueError (OneHotEncoder's default behavior) when 
    encountering an empty DataFrame during .transform(), this class returns 
    the empty DataFrame as-is. This is useful for production pipelines where 
    batch processing might produce zero-row outputs.

    Validation Logic:
    Consistent with other transformers in this pipeline, this class requires 
    input 'X' to be a pandas DataFrame and will raise a TypeError otherwise.

    Note:
        During transform(), if 'X' is empty (X.empty is True), the original 
        input is returned without one-hot encoding.
    """
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")
    
        if X.empty:
            return X
        return super().transform(X)


class OutlierRemover3SD(BaseEstimator, TransformerMixin):
    """
    Identifies and removes outliers based on the 3 Standard Deviations (3SD) rule.

    This transformer calculates the mean and standard deviation for specified 
    numerical columns during `fit`. In `transform`, it filters out rows where 
    any of the specified columns contain a value more than 3 standard 
    deviations away from the mean.

    Detection Logic:
    A value is considered an outlier if:
    abs(value - mean) > 3 * std

    Attributes:
        stats_ (pd.DataFrame): Statistics (mean, std, cutoffs, and outlier counts) 
          calculated for each column during `fit`.
        outliers_ (int): Total number of rows identified as outliers across all 
          specified columns during the last operation.

    Note:
        This transformer removes entire rows from the DataFrame. Missing values 
        (NaNs) are treated as outliers and will be removed. It is recommended 
        to handle missing values before applying this transformer.
    """
    def fit(self, X, numerical_columns):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")

        # Convert single column string to list
        if isinstance(numerical_columns, str):
            self.numerical_columns_ = [numerical_columns]
        else:
            self.numerical_columns_ = numerical_columns
            
        # Warn if any missing values 
        # NaN comparisons silently evaluate to False, which causes those rows to be counted as outliers
        n_missing_by_column = X[self.numerical_columns_].isnull().sum()
        n_missing_columns = n_missing_by_column[n_missing_by_column > 0]
        if not n_missing_columns.empty:
            missing_summary = ", ".join(f"{col} ({n})" for col, n in n_missing_columns.items())
            logger.warning(
                f"Missing Value Warning: The provided numerical features contain missing values. These will be counted as outliers. Handle missing values before calling fit() to ensure correct outlier handling.\n"
                f"- Affected Features: {missing_summary}"
            )
        
        # Calculate statistics (mean, std, cutoff values) for each column
        self.stats_ = pd.DataFrame(index=self.numerical_columns_)
        self.stats_["mean"] = X[self.numerical_columns_].mean()
        self.stats_["std"] = X[self.numerical_columns_].std()
        self.stats_["lower_cutoff"] = self.stats_["mean"] - 3 * self.stats_["std"]
        self.stats_["upper_cutoff"] = self.stats_["mean"] + 3 * self.stats_["std"]
        
        # Create masks for filtering outliers 
        self.masks_ = (X[self.numerical_columns_] >= self.stats_["lower_cutoff"]) & (X[self.numerical_columns_] <= self.stats_["upper_cutoff"])  # masks by column
        self.final_mask_ = self.masks_.all(axis=1)  # single mask across all columns
     
        # Calculate number of outliers
        self.stats_["n_outliers"] = (~self.masks_).sum()  # by column
        self.stats_["pct_outliers"] = self.stats_["n_outliers"] / len(X) * 100  # by column
        self.outliers_ = (~self.final_mask_).sum()  # across all columns
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self

    def transform(self, X):
        # Ensure .fit() happened before
        sklearn_validation.check_is_fitted(self)
        
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")

        # Create masks for X (can be a different X than during fit; e.g. X_train, X_test)
        self.masks_ = (X[self.numerical_columns_] >= self.stats_["lower_cutoff"]) & (X[self.numerical_columns_] <= self.stats_["upper_cutoff"])  # masks by column
        self.final_mask_ = self.masks_.all(axis=1)  # single mask across all columns
        
        # Remove outliers based on the final mask
        return X[self.final_mask_]

    def get_feature_names_out(self, input_features=None):
        sklearn_validation.check_is_fitted(self)
        # Return the input feature names
        return self.feature_names_in_

    def fit_transform(self, X, numerical_columns):
        # Perform both fit and transform 
        return self.fit(X, numerical_columns).transform(X)


class OutlierRemoverIQR(BaseEstimator, TransformerMixin):
    """
    Identifies and removes outliers based on the Interquartile Range (IQR) method.

    This transformer calculates the lower and upper bounds for specified numerical 
    columns during `fit` using the formula: Q1 - 1.5*IQR and Q3 + 1.5*IQR. In 
    `transform`, it filters out rows where any of the specified columns contain 
    a value outside these bounds.

    Detection Logic:
    1. Calculate Q1 (25th percentile) and Q3 (75th percentile).
    2. Calculate IQR = Q3 - Q1.
    3. Bounds: [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
    
    Attributes:
        stats_ (pd.DataFrame): Statistics (quartiles, IQR, cutoffs, and outlier counts) 
          calculated for each column during `fit`.
        outliers_ (int): Total number of rows identified as outliers across all 
          specified columns during the last operation.

    Note:
        This transformer removes entire rows from the DataFrame. Missing values 
        (NaNs) are treated as outliers and will be removed. It is recommended 
        to handle missing values before applying this transformer.
    """
    def fit(self, X, numerical_columns):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")

        # Convert single column string to list
        if isinstance(numerical_columns, str):
            self.numerical_columns_ = [numerical_columns]
        else:
            self.numerical_columns_ = numerical_columns
        
        # Warn if any missing values 
        # NaN comparisons silently evaluate to False, which causes those rows to be counted as outliers
        n_missing_by_column = X[self.numerical_columns_].isnull().sum()
        n_missing_columns = n_missing_by_column[n_missing_by_column > 0]
        if not n_missing_columns.empty:
            missing_summary = ", ".join(f"{col} ({n})" for col, n in n_missing_columns.items())
            logger.warning(
                f"Missing Value Warning: The provided numerical features contain missing values. These will be counted as outliers. Handle missing values before calling fit() to ensure correct outlier handling.\n"
                f"- Affected Features: {missing_summary}"
            )
        
        # Calculate statistics (quartiles, interquartile range, cutoff values) for each column
        self.stats_ = pd.DataFrame(index=self.numerical_columns_)
        self.stats_["Q1"] = X[self.numerical_columns_].quantile(0.25)
        self.stats_["Q3"] = X[self.numerical_columns_].quantile(0.75)
        self.stats_["IQR"] = self.stats_["Q3"] - self.stats_["Q1"]
        self.stats_["lower_cutoff"] = self.stats_["Q1"] - 1.5 * self.stats_["IQR"]
        self.stats_["upper_cutoff"] = self.stats_["Q3"] + 1.5 * self.stats_["IQR"]

        # Create masks for filtering outliers 
        self.masks_ = (X[self.numerical_columns_] >= self.stats_["lower_cutoff"]) & (X[self.numerical_columns_] <= self.stats_["upper_cutoff"])  # masks by column
        self.final_mask_ = self.masks_.all(axis=1)  # single mask across all columns

        # Calculate number of outliers
        self.stats_["n_outliers"] = (~self.masks_).sum()  # by column
        self.stats_["pct_outliers"] = self.stats_["n_outliers"] / len(X) * 100  # by column
        self.outliers_ = (~self.final_mask_).sum()  # across all columns
               
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self

    def transform(self, X):
        # Ensure .fit() happened before
        sklearn_validation.check_is_fitted(self)

        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The provided input X must be a pandas DataFrame.")

        # Create masks for X (can be a different X than during fit; e.g. X_train, X_test)
        self.masks_ = (X[self.numerical_columns_] >= self.stats_["lower_cutoff"]) & (X[self.numerical_columns_] <= self.stats_["upper_cutoff"])  # masks by column
        self.final_mask_ = self.masks_.all(axis=1)  # single mask across all columns
        
        # Remove outliers based on the final mask
        return X[self.final_mask_]

    def get_feature_names_out(self, input_features=None):
        sklearn_validation.check_is_fitted(self)
        # Return the input feature names
        return self.feature_names_in_

    def fit_transform(self, X, numerical_columns):
        # Perform both fit and transform
        return self.fit(X, numerical_columns).transform(X)
