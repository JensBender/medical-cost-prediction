"""Shared test setup for prediction and SHAP explainability tests.

The fake preprocessor and model are small substitutes for the fitted preprocessor 
and model. Their simple outputs let tests check CostPredictor and SHAP behavior 
without training models or loading saved artifacts.
"""

import numpy as np
import pytest

from src.prediction import CostPredictor


class FakePreprocessor:
    """Provide the transform method CostPredictor expects.

    Convert the input DataFrame to a NumPy array, keeping its values and column
    order. No fitting, scaling, imputation, or feature engineering takes place.
    """

    def transform(self, X):
        return X.to_numpy()


class FakeQuantileModel:
    """Use fixed arithmetic to stand in for a model's four quantile outputs.

    For input columns a and b, return a-b, a+b, a+b+2, and a+b+4 as the
    q25/q50/q75/q90 columns. These are test values, not learned quantiles.
    For example, a=3 and b=1 produce [2, 4, 6, 8], so tests can check the
    expected predictions and q50 selection.
    """

    def predict(self, X):
        a, b = X.T
        return np.column_stack([a - b, a + b, a + b + 2, a + b + 4])


@pytest.fixture
def predictor():
    """Give each test a fresh CostPredictor using the fake preprocessor and model."""
    return CostPredictor(FakePreprocessor(), FakeQuantileModel(), ['a', 'b'])
