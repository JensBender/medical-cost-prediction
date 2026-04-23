import pandas as pd


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
    strata[~is_zero] = pd.qcut(positive_y, q=bins, labels=False, duplicates="drop") + 1
    
    return strata
