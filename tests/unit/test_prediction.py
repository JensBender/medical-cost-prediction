import numpy as np
import pandas as pd
import pytest

from src.prediction import postprocess_quantile_predictions

pytestmark = pytest.mark.unit


def test_postprocessing_clips_and_enforces_quantile_order():
    raw = np.array([[-3., 2., 1., 5.], [-4., -2., -3., -1.]])
    np.testing.assert_array_equal(postprocess_quantile_predictions(raw), [[0, 2, 2, 5], [0, 0, 0, 0]])
    assert raw[0, 0] == -3  # Caller-owned raw predictions are preserved.


@pytest.mark.parametrize(
    "raw",
    [np.ones(4), np.ones((2, 3)), np.ones((2, 5))],
)
def test_postprocessing_rejects_invalid_quantile_shapes(raw):
    with pytest.raises(ValueError, match="q25/q50/q75/q90"):
        postprocess_quantile_predictions(raw)


def test_prediction_aligns_columns_and_matches_ordered_arrays(predictor):
    X = pd.DataFrame({'b': [1, 5], 'unused': [99, 99], 'a': [3, 2]})
    expected = [[2, 4, 6, 8], [0, 7, 9, 11]]
    np.testing.assert_array_equal(predictor.predict_quantiles(X), expected)
    np.testing.assert_array_equal(predictor.predict_quantiles(X[['a', 'b']].to_numpy()), expected)
    np.testing.assert_array_equal(predictor.predict_median_cost(X), [4, 7])


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
