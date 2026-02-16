from sklearn.base import BaseEstimator, TransformerMixin 


# --- Custom error classes --- 
# For missing values in critical columns of the X input DataFrame (in MissingValueChecker)
class MissingValueError(ValueError):
    pass

# For mistmatch between expected and actual columns in X input DataFrame because of missing columns, unexpected columns, or wrong column order 
class ColumnMismatchError(ValueError):
    pass


# --- Custom transformer classes --- 
# Custom transformer class to check missing values ---
class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, critical_features, non_critical_features):
        # Ensure "critical_features" and "non_critical_features" are lists
        if not isinstance(critical_features, list):
            raise TypeError("'critical_features' must be a list of feature names.")
        if not isinstance(non_critical_features, list):
            raise TypeError("'non_critical_features' must be a list of feature names.")

        # Ensure lists are not empty
        if not critical_features:
            raise ValueError("'critical_features' cannot be an empty list. It must specify the names of the critical features.")
        if not non_critical_features:
            raise ValueError("'non_critical_features' cannot be an empty list. It must specify the names of the non-critical features.")

        self.critical_features = critical_features
        self.non_critical_features = non_critical_features
    
    def _validate_input(self, X):
        # Ensure X input is DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")          
        
        # Ensure DataFrame has no missing columns
        input_columns = set(X.columns)
        expected_columns = set(self.critical_features + self.non_critical_features)
        missing_columns = expected_columns - input_columns 
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")

        # Ensure DataFrame has no unexpected columns
        unexpected_columns = input_columns - expected_columns
        if unexpected_columns:
            raise ColumnMismatchError(f"Input X contains the following columns that are neither defined in 'critical_features' nor in 'non_critical_features: {', '.join(unexpected_columns)}.")

    def _check_missing_values(self, X):
        # --- Critical features ---
        # Calculate total number of missing values  
        n_missing_total_critical = X[self.critical_features].isnull().sum().sum()
        # Calculate number of rows with missing values  
        n_missing_rows_critical = X[self.critical_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_critical = X[self.critical_features].isnull().sum().to_dict()
        # Raise error  
        if n_missing_total_critical > 0:
            values = "value" if n_missing_total_critical == 1 else "values"
            rows = "row" if n_missing_rows_critical == 1 else "rows"
            raise MissingValueError(
                f"{n_missing_total_critical} missing {values} found in critical features "
                f"across {n_missing_rows_critical} {rows}. Please provide missing {values}.\n"
                f"Missing values by column: {n_missing_by_column_critical}"
            )

        # --- Non-critical features ---
        # Calculate total number of missing values 
        n_missing_total_noncritical = X[self.non_critical_features].isnull().sum().sum()        
        # Calculate number of rows with missing values 
        n_missing_rows_noncritical = X[self.non_critical_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_noncritical = X[self.non_critical_features].isnull().sum().to_dict()
        # Display warning message
        if n_missing_total_noncritical > 0:
            values = "value" if n_missing_total_noncritical == 1 else "values"
            rows = "row" if n_missing_rows_noncritical == 1 else "rows"
            print(
                f"Warning: {n_missing_total_noncritical} missing {values} found in non-critical features "
                f"across {n_missing_rows_noncritical} {rows}. Missing {values} will be imputed.\n"
                f"Missing values by column: {n_missing_by_column_noncritical}"
            )
            
    def fit(self, X, y=None):
        # Validate input 
        self._validate_input(X)  

        # Check missing values
        self._check_missing_values(X)

        # Raise MissingValueError if a non-critical feature has only missing values
        for non_critical_feature in self.non_critical_features:
            if X[non_critical_feature].isnull().all():
                raise MissingValueError(f"'{non_critical_feature}' cannot be only missing values. Please ensure at least one non-missing value.")

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