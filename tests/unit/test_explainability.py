import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.explainability import build_shap_explainer, calculate_max_evals, calculate_shap_explanation

pytestmark = pytest.mark.unit


def assert_random_state_equal(expected):
    actual = np.random.get_state()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_shap_preserves_background_and_repeats_after_other_random_work(predictor):
    # More than SHAP's default 100 background rows, including deliberate duplicates.
    background = pd.DataFrame({'a': [0., 2., 1.] * 75, 'b': [1., 0., 3.] * 75})
    state = np.random.get_state()
    explainer = build_shap_explainer(predictor, background)
    assert_random_state_equal(state)
    np.testing.assert_array_equal(explainer.masker.data, background.to_numpy())
    rows = pd.DataFrame({'b': [4., 1.], 'a': [3., 6.]})
    first = calculate_shap_explanation(explainer, rows, max_evals=5)
    assert_random_state_equal(state)
    np.random.random(13)
    state = np.random.get_state()
    second = calculate_shap_explanation(explainer, rows, max_evals=5)
    assert_random_state_equal(state)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.base_values, second.base_values)
    np.testing.assert_allclose(first.base_values + first.values.sum(axis=1), predictor.predict_median_cost(rows))


def test_random_state_is_restored_after_explainer_failure():
    class FailingExplainer:
        feature_names = ['a', 'b']

        def __call__(self, *args, **kwargs):
            np.random.random(5)
            raise RuntimeError('expected failure')

    state = np.random.get_state()
    with pytest.raises(RuntimeError, match='expected failure'):
        calculate_shap_explanation(FailingExplainer(), pd.DataFrame({'a': [1], 'b': [2]}), max_evals=5)
    assert_random_state_equal(state)


def test_mask_budget_matches_rounds_and_rejects_partial_rounds():
    assert calculate_max_evals(1, 27) == 55
    assert calculate_max_evals(3, 27) == 165
    with pytest.raises(ValueError, match='positive integer'):
        calculate_max_evals(1.5, 27)


def test_runtime_imports_do_not_load_training_dependencies():
    subprocess.run([
        sys.executable, '-c',
        'import sys; import src.prediction, src.explainability; '
        'assert "src.modeling" not in sys.modules; '
        'assert "mlflow" not in sys.modules; assert "dvc" not in sys.modules',
    ], check=True)
