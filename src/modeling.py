# Standard library imports
import contextlib  # to train model without MLflow tracking using a null context
import json
import time
from pathlib import Path

# Third-party library imports
import joblib
import mlflow
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor  # for median baseline prediction
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.constants import TARGET_COLUMN, RANDOM_STATE

# Paths (relative to project root)
RAW_DATA_PATH = "data/h251.sas7bdat"
TRAIN_DATA_PATH = "data/training_data_preprocessed.parquet"
VAL_DATA_PATH = "data/validation_data_preprocessed.parquet"
TEST_DATA_PATH = "data/test_data_preprocessed.parquet"


# =========================
# Model Definitions
# =========================

def get_baseline_models():
    """
    Define baseline machine learning models to predict out-of-pocket medical costs with distribution-aware 
    hyperparameters.

    Returns:
        dict: A dictionary mapping model names (str) to Scikit-learn estimators or pipelines.
    """
    baseline_models = {
        "Median Prediction": DummyRegressor(strategy="median"),  # Always predict median as a benchmark
        "Linear Regression": TransformedTargetRegressor(      
            regressor=LinearRegression(),  # Simple, linear predictions as an interpretable baseline
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "Elastic Net": TransformedTargetRegressor(
            regressor=Pipeline([
                ("polynomials", PolynomialFeatures(degree=2, include_bias=False)),  # Intercept (bias) handled by model 
                ("model", ElasticNet())
            ]),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "Decision Tree": TransformedTargetRegressor(
            regressor=DecisionTreeRegressor(
                criterion="absolute_error",  # Optimize for MAE of log-costs; corresponds to predicting the Median of raw costs
                max_depth=12,          # Limits tree depth to prevent overfitting (Default: None)
                min_samples_split=100, # Prevents splitting on tiny, noisy groups of patients (Default: 2)
                min_samples_leaf=50,   # Ensures each leaf's cost prediction has enough support (Default: 1)
                max_features="sqrt",   # Uses random feature subset for each split to prevent overfitting on individual features (Default: None)
                random_state=RANDOM_STATE  
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "Random Forest": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                criterion="absolute_error",  # Optimize for MAE of log-costs; corresponds to predicting the Median of raw costs
                n_estimators=200,      # More trees for more stable estimates (Default: 100)
                max_depth=16,          # Limits tree depth to prevent overfitting (Default: None)
                min_samples_split=50,  # Prevents splitting on tiny, noisy groups of patients (Default: 2)
                min_samples_leaf=25,   # Ensures each leaf's cost prediction has enough support (Default: 1)
                max_features="sqrt",   # Uses random feature subset for each split to prevent overfitting on individual features (Default: 1.0)
                n_jobs=-1,             # Use all CPU cores to speed up training
                random_state=RANDOM_STATE 
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "XGBoost": TransformedTargetRegressor(
            regressor=XGBRegressor(
                objective="reg:absoluteerror", # Optimized for MAE of log-costs; corresponds to predicting the Median raw costs (Default: "reg:squarederror")
                n_estimators=600,      # More rounds with lower learning rate for smooth fitting (Default: 100)
                learning_rate=0.05,    # Smaller steps to prevent overshooting noisy targets (Default: 0.3)
                min_child_weight=10,   # Requires more "support" per node to split (Default: 1)
                subsample=0.8,         # Uses random row subset (80%) to prevent overfitting (Default: 1)
                colsample_bytree=0.5,  # Uses random feature subset (50%) for each split to prevent overfitting on individual features (Default: 1)
                reg_lambda=2.0,        # L2 regularization on leaf weights to avoid sharp peaks (Default: 1)
                tree_method="hist",    # Uses histogram algorithm for significantly faster training (Default: auto)
                n_jobs=-1,             # Use all CPU cores to speed up training
                random_state=RANDOM_STATE  
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "Support Vector Machine": TransformedTargetRegressor(
            regressor=SVR(
                kernel="rbf",    # Handles non-linearity and feature interactions (Default: same)
                C=10.0,          # Slightly higher regularization strength for log-costs (Default: 1.0)
                cache_size=1000  # Increase memory budget (1GB) to speed up training time (Default: 200)
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )
    }
    
    return baseline_models


# =========================
# Metrics
# =========================

def weighted_median_absolute_error(y_true, y_pred, sample_weight):
    """
    Computes the population-representative Median Absolute Error.
    
    Args:
        y_true (array-like): True target variable values.
        y_pred (array-like): Predicted target variable values.
        sample_weight (array-like): Weights for population-level estimates.

    Returns:
        float: The weighted median absolute error.
    """
    # Calculate absolute errors and ensure inputs are numpy arrays
    abs_errors = np.abs(np.array(y_true) - np.array(y_pred))
    weights = np.array(sample_weight)
    
    # Sort errors and weights by error magnitude
    sorted_idx = np.argsort(abs_errors)
    errors_sorted = abs_errors[sorted_idx]
    weights_sorted = weights[sorted_idx]
    
    # Find the value where cumulative weight reaches 50%
    cumulative_weight = np.cumsum(weights_sorted)
    cutoff = 0.5 * np.sum(weights_sorted)
    
    return errors_sorted[np.searchsorted(cumulative_weight, cutoff)]


# =============================
# Model Training & Evaluation
# =============================

def train_and_evaluate(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    w_train=None, w_val=None, 
    track_mlflow=False,
    model_name="model", 
    log_model=False,
    calculate_train_metrics=True
):
    """
    Train and evaluate a single machine learning model with optional MLflow experiment tracking.

    Args:
        model (estimator): The Scikit-learn estimator or pipeline to be trained.
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Target variable for training data.
        X_val (pd.DataFrame): Preprocessed validation features.
        y_val (pd.Series): Target variable for validation data.
        w_train (pd.Series, optional): Sample weights for training data. Defaults to None.
        w_val (pd.Series, optional): Sample weights for validation data. Defaults to None.
        track_mlflow (bool, optional): Whether to track experiment with MLflow. Defaults to False. 
        model_name (str, optional): Display name of the model for MLflow experiment tracking. Defaults to "model".
        log_model (bool, optional): Whether to log the fitted model as an artifact to MLflow. Defaults to False.
        calculate_train_metrics (bool, optional): Whether to calculate training performance metrics. Defaults to True.

    Returns:
        dict: A dictionary containing the evaluation results:
            - "val_mdae" (float): Validation Median Absolute Error.
            - "val_mae" (float): Validation Mean Absolute Error.
            - "val_r2" (float): Validation Coefficient of Determination.
            - "train_mdae" (float): Training Median Absolute Error (if calculate_train_metrics is True).
            - "train_mae" (float): Training Mean Absolute Error (if calculate_train_metrics is True).
            - "train_r2" (float): Training Coefficient of Determination (if calculate_train_metrics is True).
            - "training_time" (float): Training time in seconds.
            - "fitted_model" (estimator): The trained model object.
            - "y_val_pred" (np.ndarray): The predicted values on the validation set.
    """
    # Ensure weights are passed to model even when wrapped in a Pipeline (for Elastic Net)
    fit_params = {}
    if w_train is not None:
        # Normalize weights so mean is 1.0 (prevents numerical instability in algorithms like SVR)
        w_train_norm = w_train / w_train.mean()
        
        reg = getattr(model, "regressor", model)
        key = f"{reg.steps[-1][0]}__sample_weight" if isinstance(reg, Pipeline) else "sample_weight"
        fit_params[key] = w_train_norm

    # Use a real MLflow run or a no-op context depending on track_mlflow
    run_context = (
        mlflow.start_run(run_name=model_name)
        if track_mlflow
        else contextlib.nullcontext()
    )

    with run_context:
        if track_mlflow:
            # Tag raw data source
            mlflow.set_tag("data_source", RAW_DATA_PATH.split("/")[-1])

            # Log data lineage
            data_train = mlflow.data.from_pandas(
                X_train.assign(**{TARGET_COLUMN: y_train}), 
                targets=TARGET_COLUMN,
                source=TRAIN_DATA_PATH, 
                name="training_data"
            )
            data_val = mlflow.data.from_pandas(
                X_val.assign(**{TARGET_COLUMN: y_val}), 
                targets=TARGET_COLUMN,
                source=VAL_DATA_PATH, 
                name="validation_data"
            )
            mlflow.log_input(data_train, context="training")
            mlflow.log_input(data_val, context="validation")

            # Log model hyperparameters
            mlflow.log_params(model.get_params())

        # Fit model on training data
        start_time = time.time()  # Measure training time
        model.fit(X_train, y_train, **fit_params)
        training_time = time.time() - start_time

        # Predict on validation data
        y_val_pred = model.predict(X_val)

        # Calculate evaluation metrics on validation data 
        val_mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)  
        val_mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)

        results = {
            "val_mdae": val_mdae,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "training_time": training_time,
            "fitted_model": model,
            "y_val_pred": y_val_pred,
        }

        # Calculate evaluation metrics on training data for overfitting analysis
        if calculate_train_metrics:
            y_train_pred = model.predict(X_train)
            train_mdae = weighted_median_absolute_error(y_train, y_train_pred, sample_weight=w_train)
            train_mae = mean_absolute_error(y_train, y_train_pred, sample_weight=w_train)
            train_r2 = r2_score(y_train, y_train_pred, sample_weight=w_train)
            
            results.update({
                "train_mdae": train_mdae,
                "train_mae": train_mae,
                "train_r2": train_r2
            })

        if track_mlflow:
            # Log metrics
            mlflow.log_metrics({
                "val_mdae": val_mdae,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "training_time": training_time
            })
            if calculate_train_metrics:
                mlflow.log_metrics({
                    "train_mdae": train_mdae,
                    "train_mae": train_mae,
                    "train_r2": train_r2
                })

           # Log fitted model artifact
            if log_model:
                mlflow.sklearn.log_model(model, "model")

    return results


# =========================
# Modeling Utilities
# =========================

def get_core_model_params(fitted_model):
    """
    Extract JSON-friendly core model params for persistence.

    - Unwrap TransformedTargetRegressor to its inner regressor.
    - For Pipeline models, persist only step parameters (e.g., polynomials__degree),
      not estimator objects stored under top-level keys like "polynomials" or "model".
    """
    core_model = getattr(fitted_model, "regressor_", getattr(fitted_model, "regressor", fitted_model))

    if isinstance(core_model, Pipeline):
        params = {}
        for step_name, step in core_model.named_steps.items():
            step_params = step.get_params(deep=False)
            for param_name, param_value in step_params.items():
                params[f"{step_name}__{param_name}"] = param_value
        return params

    return core_model.get_params(deep=False)


# =========================
# Model Persistence
# =========================

def save_model(model, filepath, verbose=True):
    """
    Save a trained model or pipeline or a results dictionary to a file using joblib.

    Args:
        model: The model object or pipeline object or results dictionary to be saved.
        filepath (str or Path): The destination file path (e.g., 'models/baseline.joblib').
        verbose (bool): Whether to print a success message.
    """
    try:
        # Ensure the parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, filepath)
        if verbose:
            print(f"Successfully saved model to '{filepath}'.")
    except Exception as e:
        print(f"Error while saving model: {e}")


def load_model(filepath, verbose=True):
    """
    Load a trained model or pipeline or a results dictionary from a file using joblib.

    Args:
        filepath (str or Path): The file path to load from.
        verbose (bool): Whether to print a success message.

    Returns:
        The loaded object (model, pipeline, or dictionary).
    """
    try:
        model = joblib.load(filepath)
        if verbose:
            print(f"Successfully loaded model from '{filepath}'.")
        return model
    except Exception as e:
        print(f"Error while loading model: {e}")
        return None


def save_metrics(metrics, filepath, verbose=True):
    """
    Save performance metrics to a JSON file, automatically converting 
    NumPy types to Python floats for serialization.

    Args:
        metrics (dict or list): Metrics to save. Usually a dict of dicts 
            or a list of dicts.
        filepath (str or Path): Path to save the JSON file.
        verbose (bool): Whether to print a success message.
    """
    try:
        def clean_value(x):
            """Recursively convert NumPy scalars to Python floats for JSON serialization."""
            if isinstance(x, dict):
                return {key: clean_value(value) for key, value in x.items()}
            if isinstance(x, list):
                return [clean_value(i) for i in x]
            if hasattr(x, "dtype"):  # for single NumPy float
                return float(x)
            return x

        clean_metrics = clean_value(metrics)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(clean_metrics, f, indent=4)
        if verbose:
            print(f"Successfully saved metrics to '{filepath}'.")
    except Exception as e:
        print(f"Error while saving metrics: {e}")


def load_metrics(filepath, verbose=True):
    """
    Load metrics from a JSON file.

    Args:
        filepath (str or Path): The file path to load from.
        verbose (bool): Whether to print a success message.

    Returns:
        dict or list: The loaded metrics or None if an error occurred.
    """
    try:
        with open(filepath, "r") as f:
            metrics = json.load(f)
        if verbose:
            print(f"Successfully loaded metrics from '{filepath}'.")
        return metrics
    except Exception as e:
        print(f"Error while loading metrics: {e}")
        return None
