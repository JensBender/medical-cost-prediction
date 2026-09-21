import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.explainability import (
    build_shap_explainer,
    calculate_shap_explanation,
    calculate_max_evals,
)

pytestmark = pytest.mark.unit


def _assert_numpy_random_state_is_unchanged(random_state_before):
    """Internal assertion helper for tests that must preserve NumPy's random state."""
    actual_state = np.random.get_state()
    assert actual_state[0] == random_state_before[0]
    np.testing.assert_array_equal(actual_state[1], random_state_before[1])
    assert actual_state[2:] == random_state_before[2:]


def test_build_shap_explainer_preserves_selected_background(predictor):
    """Keep the full 225-row weighted background.

    SHAP otherwise subsamples backgrounds larger than its default maximum of 100
    rows, which would change the selected background and its repeated weighted rows.
    """
    shap_background = pd.DataFrame(
        {
            "a": [0.0, 2.0, 1.0] * 75,
            "b": [1.0, 0.0, 3.0] * 75,
        }
    )

    explainer = build_shap_explainer(predictor, shap_background)

    np.testing.assert_array_equal(
        explainer.masker.data,
        shap_background.to_numpy(),
    )


def test_build_shap_explainer_reports_missing_predictor_input_features(
    predictor,
):
    """Report predictor input features missing from the SHAP background."""
    # The predictor is configured for a and b, so b is required.
    shap_background = pd.DataFrame({"a": [1.0]})

    with pytest.raises(ValueError, match=r"missing predictor input features.*'b'"):
        build_shap_explainer(predictor, shap_background)


def test_build_shap_explainer_restores_numpy_random_state(predictor):
    """Leave NumPy's global random state unchanged after building the explainer."""
    shap_background = pd.DataFrame(
        {
            "a": [0.0, 2.0, 1.0],
            "b": [1.0, 0.0, 3.0],
        }
    )
    random_state_before = np.random.get_state()

    build_shap_explainer(predictor, shap_background)

    _assert_numpy_random_state_is_unchanged(random_state_before)


def test_shap_contributions_sum_to_median_predictions(predictor):
    """Reconstruct each q50 prediction from its SHAP base value and contributions."""
    shap_background = pd.DataFrame(
        {
            "a": [0.0, 2.0, 1.0],
            "b": [1.0, 0.0, 3.0],
        }
    )
    explanation_input = pd.DataFrame(
        {
            "a": [3.0, 6.0],
            "b": [4.0, 1.0],
        }
    )
    explainer = build_shap_explainer(predictor, shap_background)
    explanation = calculate_shap_explanation(
        explainer,
        explanation_input,
        max_evals=5,
    )

    reconstructed_median_costs = (
        explanation.base_values + explanation.values.sum(axis=1)
    )
    expected_median_costs = predictor.predict_median_cost(explanation_input)

    np.testing.assert_allclose(
        reconstructed_median_costs,
        expected_median_costs,
    )


def test_calculate_shap_explanation_is_repeatable_after_other_random_work(
    predictor,
):
    """Return the same explanation after unrelated code advances NumPy's RNG."""
    shap_background = pd.DataFrame(
        {
            "a": [0.0, 2.0, 1.0],
            "b": [1.0, 0.0, 3.0],
        }
    )
    explanation_input = pd.DataFrame(
        {
            "a": [3.0, 6.0],
            "b": [4.0, 1.0],
        }
    )
    explainer = build_shap_explainer(predictor, shap_background)

    first_explanation = calculate_shap_explanation(
        explainer,
        explanation_input,
        max_evals=5,
    )

    # Advance the global RNG to confirm that each explanation resets its own seed.
    np.random.random(13)

    second_explanation = calculate_shap_explanation(
        explainer,
        explanation_input,
        max_evals=5,
    )

    np.testing.assert_array_equal(
        first_explanation.values,
        second_explanation.values,
    )
    np.testing.assert_array_equal(
        first_explanation.base_values,
        second_explanation.base_values,
    )


def test_calculate_shap_explanation_restores_numpy_random_state(predictor):
    """Leave NumPy's global random state unchanged after an explanation."""
    shap_background = pd.DataFrame(
        {
            "a": [0.0, 2.0, 1.0],
            "b": [1.0, 0.0, 3.0],
        }
    )
    explanation_input = pd.DataFrame(
        {
            "a": [3.0, 6.0],
            "b": [4.0, 1.0],
        }
    )
    explainer = build_shap_explainer(predictor, shap_background)
    random_state_before = np.random.get_state()

    calculate_shap_explanation(
        explainer,
        explanation_input,
        max_evals=5,
    )

    _assert_numpy_random_state_is_unchanged(random_state_before)


def test_calculate_shap_explanation_restores_random_state_after_failure():
    """Restore NumPy's random state even when the explainer raises an error."""

    class FailingExplainer:
        """Advance NumPy's RNG, then simulate an explanation failure."""

        feature_names = ["a", "b"]

        def __call__(self, input_rows, *, max_evals, silent):
            np.random.random(5)
            raise RuntimeError("expected failure")

    explanation_input = pd.DataFrame({"a": [1], "b": [2]})
    random_state_before = np.random.get_state()

    with pytest.raises(RuntimeError, match="expected failure"):
        calculate_shap_explanation(
            FailingExplainer(),
            explanation_input,
            max_evals=5,
        )

    _assert_numpy_random_state_is_unchanged(random_state_before)


@pytest.mark.parametrize(
    "permutation_rounds, expected_max_evals",
    [
        pytest.param(1, 55, id="one_round"),
        pytest.param(3, 165, id="three_rounds"),
    ],
)
def test_calculate_max_evals_counts_complete_permutation_rounds(
    permutation_rounds,
    expected_max_evals,
):
    """Count one masked state plus forward and backward steps for 27 features."""
    actual_max_evals = calculate_max_evals(permutation_rounds, n_features=27)

    assert actual_max_evals == expected_max_evals


def test_calculate_max_evals_rejects_fractional_permutation_rounds():
    """Reject a fraction of a round because both permutation passes must finish."""
    with pytest.raises(ValueError, match="positive integer"):
        calculate_max_evals(1.5, 27)


def test_runtime_modules_import_without_training_dependencies():
    """Import prediction and explainability without loading modeling, MLflow, or DVC."""
    import_check = (
        "import sys\n"
        "import src.prediction\n"
        "import src.explainability\n"
        'assert "src.modeling" not in sys.modules\n'
        'assert "mlflow" not in sys.modules\n'
        'assert "dvc" not in sys.modules\n'
    )

    # Use a fresh process so imports from earlier tests cannot affect sys.modules.
    subprocess.run(
        [sys.executable, "-c", import_check],
        check=True,
    )
