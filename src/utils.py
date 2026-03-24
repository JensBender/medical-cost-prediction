from pathlib import Path
import numpy as np 
import joblib


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


def save_model(model, filepath):
    """
    Save a trained model or pipeline or a results dictionary to a file using joblib.

    Args:
        model: The model object or pipeline object or results dictionary to be saved.
        filepath (str or Path): The destination file path (e.g., 'models/baseline.joblib').
    """
    try:
        # Ensure the parent directory exists (e.g., if saving to 'models/baseline.joblib')
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        print(f"Successfully saved model to '{filepath}'.")
    except Exception as e:
        print(f"Error while saving model: {e}")