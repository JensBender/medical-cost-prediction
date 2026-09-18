"""Core quantile-model inference in 2023 dollars, before API/UI formatting."""

import numpy as np
import pandas as pd


def postprocess_quantile_predictions(y_pred):
    """Return non-negative quantiles in increasing order without changing the input.

    The input must have shape (n_rows, 4). The caller must supply the columns
    in q25/q50/q75/q90 order; this function checks the shape, not the column
    meanings. Negative costs are clipped to zero. Quantile crossing occurs when
    a later quantile is lower than the preceding one. Raise it to that value to
    enforce monotonic quantiles: q25 <= q50 <= q75 <= q90. The result is a new
    NumPy array of the same shape.
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
    """Predict out-of-pocket costs using a fitted preprocessor and quantile model.

    Create this object once with the loaded preprocessor, model, and input
    feature names. Each prediction orders the input features, applies the
    preprocessor, calls the model, and makes the four quantiles non-negative
    and increasing. ``predict_median_cost`` selects q50 from these results
    for SHAP explanations.

    The model must return q25/q50/q75/q90 in 2023 dollars, in that order.
    Our saved model is a TransformedTargetRegressor, whose ``predict``
    method converts predictions from log-transformed costs back to dollars.

    The caller loads the artifacts. This class does not load or save them.
    API input validation, inflation adjustment, and response formatting
    happen outside this class.
    """

    def __init__(self, preprocessor, model, input_features):
        self.preprocessor = preprocessor
        self.model = model
        self.input_features = list(input_features)
        if not self.input_features or len(set(self.input_features)) != len(self.input_features):
            raise ValueError("Input feature names must be non-empty and unique.")

    def predict_quantiles(self, X):
        """Return postprocessed q25/q50/q75/q90 predictions in 2023 dollars.

        For regular prediction calls, pass a pandas DataFrame. The method selects
        and orders its columns using ``input_features``.

        NumPy array support exists only for SHAP, which calls
        ``predict_median_cost`` with masked arrays. These arrays must already
        follow the ``input_features`` order. This method restores their column
        names in a DataFrame before running the preprocessor. Arrays are not 
        intended for general prediction calls.

        The method then applies the fitted preprocessor, calls the fitted model,
        and enforces non-negative, monotonic quantiles. The result is a NumPy
        array with shape (n_rows, 4), with one column for each quantile.
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
        """Run preprocessing, prediction, and postprocessing, then select q50.

        Return a NumPy array with shape (n_rows,) containing median cost
        predictions in 2023 dollars. SHAP uses this method to explain the
        prediction after all these steps.
        """
        return self.predict_quantiles(X)[:, 1]
