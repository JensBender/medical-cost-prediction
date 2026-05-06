# =========================
# Survey Statistics
# =========================
# Statistical utility functions that account for MEPS survey weights. 
# These exist because numpy/pandas do not provide weighted  
# equivalents for quantiles and standard deviation.
#
# NOT DVC-tracked — safe to modify without triggering pipeline reruns.

import numpy as np
import pandas as pd


def weighted_quantile(variable, weights, quantile):
    """
    Compute a weighted quantile using a sorted cumulative-weight CDF.

    Works with both pandas Series and numpy arrays as inputs.

    Args:
        variable (array-like): The values to compute the quantile over.
        weights (array-like): Survey weights (positive, same length as variable).
        quantile (float or array-like): Quantile(s) to compute, in [0, 1].

    Returns:
        float or np.ndarray: The weighted quantile value(s).
    """
    values = np.asarray(variable, dtype=float)
    w = np.asarray(weights, dtype=float)
    sorter = np.argsort(values)
    values, w = values[sorter], w[sorter]
    cumulative_weight = np.cumsum(w) - 0.5 * w  # midpoint convention
    cumulative_weight /= np.sum(w)              # normalize to [0, 1]
    return np.interp(quantile, cumulative_weight, values)


def weighted_std(variable, weights):
    """
    Compute the weighted standard deviation.

    Args:
        variable (array-like): The values to compute the standard deviation over.
        weights (array-like): Survey weights (positive, same length as variable).

    Returns:
        float: The weighted standard deviation.
    """
    values = np.asarray(variable, dtype=float)
    w = np.asarray(weights, dtype=float)
    weighted_mean = np.average(values, weights=w)
    weighted_variance = np.average((values - weighted_mean) ** 2, weights=w)
    return np.sqrt(weighted_variance)


def create_stratification_bins(y):
    """
    Create stratification bins that preserve the zero-inflated, heavy-tailed
    cost distribution across train/val/test splits.

    Uses non-linear percentile boundaries (50th, 80th, 95th, 99th, 99.9th)
    to ensure extreme high-cost cases are balanced across all splits.

    Args:
        y (pd.Series): Target variable (out-of-pocket costs).

    Returns:
        pd.Series: Integer bin labels aligned with the input index.
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
    strata[~is_zero] = pd.qcut(positive_y, q=bins, labels=False, duplicates="drop") + 1

    return strata
