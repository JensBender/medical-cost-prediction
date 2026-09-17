import numpy as np
import pandas as pd
import pytest

from src.prediction import postprocess_quantile_predictions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "input_quantiles, expected_quantiles",
    [
        # Replace the negative q25 with zero.
        pytest.param(
            [-10, 20, 30, 40], [0, 20, 30, 40], id="negative_cost",
        ),
        # Fix quantile crossing by raising q75 to q50.
        pytest.param(
            [10, 30, 20, 40], [10, 30, 30, 40], id="quantile_crossing",
        ),
    ],
)
def test_postprocessing_fixes_negative_costs_and_quantile_crossing(
    input_quantiles, expected_quantiles,
):
    input_quantiles = np.array([input_quantiles], dtype=float)
    original_quantiles = input_quantiles.copy()
    expected_quantiles = np.array([expected_quantiles], dtype=float)

    actual_quantiles = postprocess_quantile_predictions(input_quantiles)

    np.testing.assert_array_equal(actual_quantiles, expected_quantiles)
    # Ensure input quantiles remain unchanged.
    np.testing.assert_array_equal(input_quantiles, original_quantiles)


@pytest.mark.parametrize(
    "input_quantiles",
    [
        pytest.param(
            np.array([1, 2, 3, 4]),  # must be [[1, 2, 3, 4]], not [1, 2, 3, 4]
            id="missing_row_dimension",
        ),
        pytest.param(
            np.array([[1, 2, 3], [4, 5, 6]]),
            id="too_few_columns",
        ),
        pytest.param(
            np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            id="too_many_columns",
        ),
    ],
)
def test_postprocessing_requires_rows_with_four_quantile_columns(input_quantiles):
    """Reject inputs that are not a 2D array with four columns (for q25, q50, q75, q90)."""
    with pytest.raises(ValueError, match="q25/q50/q75/q90"):  # ValueError message mentions the four quantiles
        postprocess_quantile_predictions(input_quantiles)


def test_predict_quantiles_selects_and_orders_preprocessor_input_features(
    predictor,
):
    """Pass only the configured preprocessor input features, in their required order."""
    prediction_input = pd.DataFrame(
        {
            "b": [1, 2],
            "unused": [99, 99],
            "a": [3, 6],
        }
    )
    expected_quantiles = np.array(
        [
            [2, 4, 6, 8],   # The fake model maps a=3, b=1 to these quantiles.
            [4, 8, 10, 12],  # The fake model maps a=6, b=2 to these quantiles.
        ]
    )

    actual_quantiles = predictor.predict_quantiles(prediction_input)

    np.testing.assert_array_equal(actual_quantiles, expected_quantiles)


def test_array_prediction_uses_configured_feature_order(predictor):
    """Treat array columns as already ordered a followed by b."""
    input_features = np.array([[3, 1], [6, 2]])
    expected_quantiles = np.array([[2, 4, 6, 8], [4, 8, 10, 12]])

    actual_quantiles = predictor.predict_quantiles(input_features)

    np.testing.assert_array_equal(actual_quantiles, expected_quantiles)


def test_median_prediction_returns_q50(predictor):
    """Return q50, the second of the four predicted quantiles."""
    input_features = np.array([[3, 1], [6, 2]])
    expected_median_costs = np.array([4, 8])

    actual_median_costs = predictor.predict_median_cost(input_features)

    np.testing.assert_array_equal(actual_median_costs, expected_median_costs)


def test_prediction_rejects_missing_or_wrong_shape_inputs(predictor):
    with pytest.raises(ValueError, match='missing features'):
        predictor.predict_quantiles(pd.DataFrame({'a': [1]}))
    with pytest.raises(ValueError, match='shape'):
        predictor.predict_quantiles(np.ones((1, 3)))


def test_prediction_rejects_wrong_model_quantiles(predictor, monkeypatch):
    monkeypatch.setattr(predictor.model, 'predict', lambda X: np.ones((len(X), 3)))
    with pytest.raises(ValueError, match='q25/q50/q75/q90'):
        predictor.predict_quantiles(np.ones((1, 2)))


def test_prediction_rejects_wrong_model_row_count(predictor, monkeypatch):
    monkeypatch.setattr(predictor.model, 'predict', lambda X: np.ones((len(X) + 1, 4)))
    with pytest.raises(ValueError, match='one prediction row per input row'):
        predictor.predict_quantiles(np.ones((1, 2)))
