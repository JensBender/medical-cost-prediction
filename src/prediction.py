"""Core quantile-model inference in 2023 dollars, before API/UI formatting."""

import numpy as np
import pandas as pd


def postprocess_quantile_predictions(y_pred):
    """Enforce non-negative, monotonic quantiles.

    Ensure predictions have one row per observation and four columns in
    q25/q50/q75/q90 order. Negative costs are clipped to zero. Quantile
    crossing is resolved by raising each later quantile to the preceding
    quantile when needed (q25 <= q50 <= q75 <= q90).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2 or y_pred.shape[1] != 4:
        raise ValueError(
            "Quantile predictions must have shape (n_rows, 4) with "
            "q25/q50/q75/q90 columns."
        )
    y_pred = np.maximum(y_pred, 0)
    return np.maximum.accumulate(y_pred, axis=1)


class CostPredictor:
    """Run the core q25/q50/q75/q90 inference path with fitted artifacts.

    The class holds an already-loaded fitted preprocessor, fitted quantile
    model, and required preprocessor-input feature order. It aligns input
    features, applies preprocessing, predicts all four quantiles, applies the
    model's inverse target transformation, and postprocesses the quantiles.

    ``predict_median_cost`` additionally selects q50 for SHAP explanations.

    The model must return q25/q50/q75/q90 in that order. The model is nested in 
    a TransformedTargetRegressor, which applies the inverse target transformation 
    during ``model.predict``. 
    
    This class does not load or persist artifacts and does not perform API 
    validation, inflation adjustment, or output formatting.
    """

    def __init__(self, preprocessor, model, input_features):
        self.preprocessor = preprocessor
        self.model = model
        self.input_features = list(input_features)
        if not self.input_features or len(set(self.input_features)) != len(self.input_features):
            raise ValueError("Input feature names must be non-empty and unique.")

    def predict_quantiles(self, X):
        """Return postprocessed q25/q50/q75/q90 predictions in 2023 dollars.

        DataFrame columns are selected and ordered using ``input_features``.
        Array inputs, such as SHAP's masked inputs, must already use that order.
        The method then applies the fitted preprocessor, calls the fitted model,
        and enforces non-negative, monotonic quantiles.
        """
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

        predictions = postprocess_quantile_predictions(
            self.model.predict(self.preprocessor.transform(X))
        )
        if len(predictions) != len(X):
            raise ValueError(
                "The quantile model must return one prediction row per input row."
            )
        return predictions

    def predict_median_cost(self, X):
        """Return q50 from the complete postprocessed inference path for SHAP."""
        return self.predict_quantiles(X)[:, 1]
