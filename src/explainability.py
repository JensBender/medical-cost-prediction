import numpy as np
import pandas as pd
import shap

from src.constants import RANDOM_STATE


def build_shap_explainer(predictor, background, *, random_state=RANDOM_STATE):
    """Build a reusable permutation SHAP explainer for q50 predictions.

    ``background`` must be a non-empty DataFrame containing every predictor input
    feature. These are the cleaned columns in the form expected as input by the
    preprocessing pipeline. The explainer keeps every supplied row, including
    duplicates that represent greater population weight in the survey-weighted
    background.

    SHAP resets NumPy's global random state while building the explainer. This
    function restores the previous state afterward to prevent side effects on
    unrelated NumPy random sampling. 
    """
    if not isinstance(background, pd.DataFrame) or background.empty:
        raise ValueError("SHAP background must be a non-empty DataFrame.")
    missing_features = [feature for feature in predictor.input_features if feature not in background.columns]
    if missing_features:
        raise ValueError(
            "SHAP background is missing predictor input features: "
            f"{missing_features}"
        )
    background = background.loc[:, predictor.input_features]
    previous_numpy_random_state = np.random.get_state()
    try:
        background_masker = shap.maskers.Independent(
            background,
            max_samples=len(background),
        )
        return shap.Explainer(
            predictor.predict_median_cost,
            background_masker,
            algorithm="permutation",
            feature_names=predictor.input_features,
            seed=random_state,
        )
    finally:
        np.random.set_state(previous_numpy_random_state)


def calculate_shap_explanation(
    explainer,
    explanation_input,
    *,
    max_evals,
    random_state=RANDOM_STATE,
):
    """Explain q50 predictions for one or more preprocessor-input rows.

    ``explanation_input`` must be a DataFrame containing every feature expected by
    the explainer. The function orders those columns. The same inputs and
    ``random_state`` produce the same SHAP values. ``max_evals`` controls the
    permutation mask budget. SHAP values and base values are in 2023 USD because
    the predictor returns q50 in that unit.
    """
    if not isinstance(explanation_input, pd.DataFrame):
        raise TypeError("SHAP inputs must be provided as a pandas DataFrame.")
    input_features = explainer.feature_names
    missing_features = [feature for feature in input_features if feature not in explanation_input.columns]
    if missing_features:
        raise ValueError(
            "SHAP input is missing preprocessor input features: "
            f"{missing_features}"
        )

    # SHAP uses NumPy's global random state for feature permutations. Reset it for
    # reproducible explanations, then restore it to avoid affecting unrelated
    # NumPy random sampling.
    previous_numpy_random_state = np.random.get_state()
    try:
        np.random.seed(random_state)
        return explainer(
            explanation_input.loc[:, input_features],
            max_evals=max_evals,
            silent=True,
        )
    finally:
        np.random.set_state(previous_numpy_random_state)


def calculate_max_evals(permutation_rounds, n_features):
    """Return the ``max_evals`` given the SHAP permutation rounds and number of
    features.

    Each round evaluates one fully masked input, followed by one forward and one
    backward step per feature: ``2 * n_features + 1`` evaluations. Both arguments
    must be positive integers.
    """
    if (not isinstance(permutation_rounds, (int, np.integer)) or permutation_rounds < 1):
        raise ValueError("permutation_rounds must be a positive integer.")
    if not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    return permutation_rounds * (2 * n_features + 1)
