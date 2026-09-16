"""Shared model inference in 2023 dollars, before API/UI formatting."""

import numpy as np
import pandas as pd


def postprocess_quantile_predictions(y_pred):
    """Enforce non-negative, monotonic quantiles (q25 <= q50 <= q75 <= q90)."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0)
    return np.maximum.accumulate(y_pred, axis=1)


class CostPredictor:
    """Reuse fitted artifacts for q25/q50/q75/q90 predictions and the SHAP callable.

    The model must return the four quantiles in that order, with any inverse
    target transformation already applied (as our TransformedTargetRegressor does).
    No artifacts are loaded and no predictions are stored by this class.
    """

    def __init__(self, preprocessor, model, input_features):
        self.preprocessor = preprocessor
        self.model = model
        self.input_features = list(input_features)
        if not self.input_features or len(set(self.input_features)) != len(self.input_features):
            raise ValueError("Input feature names must be non-empty and unique.")

    def predict_quantiles(self, X):
        """Predict cleaned q25/q50/q75/q90 from a DataFrame or ordered SHAP array."""
        if isinstance(X, pd.DataFrame):
            missing_features = [feature for feature in self.input_features if feature not in X.columns]
            if missing_features:
                raise ValueError(f"Prediction input is missing features: {missing_features}")
            X = X.loc[:, self.input_features]
        else:
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[1] != len(self.input_features):
                raise ValueError(f"Prediction input must have shape (n_rows, {len(self.input_features)}).")
            X = pd.DataFrame(X, columns=self.input_features)

        predictions = self.model.predict(self.preprocessor.transform(X))
        if np.shape(predictions) != (len(X), 4):
            raise ValueError("The quantile model must return q25/q50/q75/q90 for each row.")
        return postprocess_quantile_predictions(predictions)

    def predict_median_cost(self, X):
        """Return postprocessed q50 for SHAP to explain the full prediction path."""
        return self.predict_quantiles(X)[:, 1]
