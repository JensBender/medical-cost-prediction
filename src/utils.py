# Standard library imports
import json
from pathlib import Path

# Third-party library imports
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def add_table_caption(styler, caption, font_size="14px", font_weight="bold", text_align="left"):
    """
    Adds a caption to a Pandas DataFrame for notebook display using a Pandas Styler object
    and styled HTML.
    Args:
        styler (pandas.io.formats.style.Styler): The Styler object to modify.
        caption (str): The text to display as the table title.
        font_size (str): CSS font-size value (e.g., "14px").
        font_weight (str): CSS font-weight value (e.g., "bold").
        text_align (str): CSS text-align value (e.g., "left").
    Returns:
        pandas.io.formats.style.Styler: The modified Styler object with the caption applied.
    """
    return styler.set_caption(caption).set_table_styles([{
        "selector": "caption", 
        "props": [
            ("font-size", font_size), 
            ("font-weight", font_weight), 
            ("text-align", text_align),
            ("color", "#4A4A4A"),
            ("margin-bottom", "8px")
        ]
    }])


def create_stratification_bins(y):
    """Create stratification bins that preserve the zero-inflated, heavy-tailed
    cost distribution across train/val/test splits.

    Uses non-linear percentile boundaries (50th, 80th, 95th, 99th, 99.9th)
    to ensure extreme high-cost cases are balanced across all splits.
    """
    # Initialize strata series 
    strata = pd.Series(index=y.index, dtype=int)
    
    # Bin 0: Zero Costs (Handle the hurdle separately)
    is_zero = (y == 0)
    strata[is_zero] = 0
    
    # Custom non-linear quantiles for positive values to capture the tail
    positive_y = y[~is_zero]
    bins = [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0]
    
    # Assign positive spenders to bins 1 through 6 
    # note: labels=False returns the bin indices (0-5) instead of Interval objects (e.g., [0, 150.5]). We add 1 to shift the indices to 1-6 with 0 being reserved for the zero cost bin.
    # note: duplicates="drop" avoids errors if multiple quantiles (e.g., 0.95 and 0.99) result in the same cost value.
    strata[~is_zero] = pd.qcut(positive_y, q=bins, labels=False, duplicates="drop") + 1
    
    return strata


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


def save_model(model, filepath, verbose=True):
    """
    Save a trained model or pipeline or a results dictionary to a file using joblib.

    Args:
        model: The model object or pipeline object or results dictionary to be saved.
        filepath (str or Path): The destination file path (e.g., 'models/baseline.joblib').
        verbose (bool): Whether to print a success message.
    """
    try:
        # Ensure the parent directory exists (e.g., if saving to 'models/baseline.joblib')
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


def save_metrics(metrics, filepath, verbose=True):
    """
    Save performance metrics to a JSON file, automatically converting 
    NumPy types to Python floats for serialization.

    Supports nested dictionaries (e.g., baseline models names with metrics) and 
    lists of dictionaries (e.g., hyperparameter tuning runs with metrics).

    Args:
        metrics (dict or list): Metrics to save. Usually a dict of dicts 
            or a list of dicts.
        filepath (str or Path): Path to save the JSON file.
        verbose (bool): Whether to print a success message.
    """
    try:
        # Clean metrics: Ensure NumPy floats are converted to Python floats for JSON
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
