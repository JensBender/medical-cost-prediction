"""SHAP setup and repeatable explanation calls."""

import numpy as np
import pandas as pd
import shap

from src.constants import RANDOM_STATE


def calculate_max_evals(permutation_rounds, n_features):
    """Convert complete forward/backward permutation rounds to a mask budget."""
    if not isinstance(permutation_rounds, (int, np.integer)) or permutation_rounds < 1:
        raise ValueError("permutation_rounds must be a positive integer.")
    if not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    return permutation_rounds * (2 * n_features + 1)


def build_shap_explainer(predictor, background, *, random_state=RANDOM_STATE):
    """Build once from a CostPredictor and an already selected background sample.

    The caller loads or samples the background. Keep every supplied row: repeated
    rows in a survey-weighted sample represent their greater population weight.
    """
    if not isinstance(background, pd.DataFrame) or background.empty:
        raise ValueError("SHAP background must be a non-empty DataFrame.")
    background = background.loc[:, predictor.input_features]
    numpy_random_state = np.random.get_state()
    try:
        masker = shap.maskers.Independent(background, max_samples=len(background))
        return shap.Explainer(
            predictor.predict_median_cost,
            masker,
            algorithm="permutation",
            feature_names=predictor.input_features,
            seed=random_state,
        )
    finally:
        # SHAP's constructor also seeds NumPy's global random generator.
        np.random.set_state(numpy_random_state)


def calculate_shap_explanation(explainer, X, *, max_evals, random_state=RANDOM_STATE):
    """Calculate a SHAP explanation in 2023 USD for one or more input rows 
    using an existing explainer.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("SHAP inputs must be provided as a pandas DataFrame.")
    features = explainer.feature_names
    missing_features = [feature for feature in features if feature not in X.columns]
    if missing_features:
        raise ValueError(f"SHAP input is missing preprocessor input features: {missing_features}")

    # Repeat the same feature permutations for the same ordered inputs.
    # Restore the random state afterward so other code is unaffected.
    numpy_random_state = np.random.get_state()
    try:
        np.random.seed(random_state)
        return explainer(X.loc[:, features], max_evals=max_evals, silent=True)
    finally:
        np.random.set_state(numpy_random_state)
