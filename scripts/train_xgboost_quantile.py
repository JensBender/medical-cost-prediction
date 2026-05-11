"""
Train native XGBoost quantile regression for prediction ranges.

Outputs:
    - models/xgb_quantile_model.joblib
    - models/xgb_quantile_predictions.joblib
    - models/xgb_quantile_metrics.json
    - models/xgb_quantile_params.json
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, r2_score
from xgboost import XGBRegressor

from src.constants import RANDOM_STATE, TARGET_COLUMN, WEIGHT_COLUMN
from src.modeling import (
    load_metrics,
    save_metrics,
    save_model,
    weighted_median_absolute_error,
)


QUANTILE_ALPHAS = [0.25, 0.50, 0.75, 0.90]
QUANTILE_COLUMNS = [f"q{int(alpha * 100):02d}" for alpha in QUANTILE_ALPHAS]


def load_xyw(path):
    """Load preprocessed data and split into features, target, and survey weights."""
    df = pd.read_parquet(path)
    X = df.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    w = df[WEIGHT_COLUMN]
    return X, y, w


def get_xgb_quantile_params(point_params_path):
    """Use tuned point-model structure as the starting point for quantile regression."""
    tuned_params = load_metrics(point_params_path, verbose=False) or {}
    keep_params = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    ]
    quantile_params = {key: tuned_params[key] for key in keep_params if key in tuned_params}
    quantile_params.update({
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILE_ALPHAS,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    })
    return quantile_params


def monotonic_quantile_predictions(y_pred):
    """Clip predictions to valid dollars and enforce q25 <= q50 <= q75 <= q90."""
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    y_pred = np.clip(y_pred, 0, None)
    return np.maximum.accumulate(y_pred, axis=1)


def weighted_interval_coverage(y_true, lower, upper, sample_weight):
    """Population-weighted share of observations inside an interval."""
    covered = (np.asarray(y_true) >= np.asarray(lower)) & (np.asarray(y_true) <= np.asarray(upper))
    return np.average(covered, weights=sample_weight)


def evaluate_quantiles(y_true, y_pred_quantiles, sample_weight):
    """Evaluate median accuracy and quantile calibration."""
    y_pred_quantiles = monotonic_quantile_predictions(y_pred_quantiles)
    q25, q50, q75, q90 = [y_pred_quantiles[:, i] for i in range(len(QUANTILE_ALPHAS))]

    metrics = {
        "q50_mdae": weighted_median_absolute_error(y_true, q50, sample_weight=sample_weight),
        "q50_mae": mean_absolute_error(y_true, q50, sample_weight=sample_weight),
        "q50_r2": r2_score(y_true, q50, sample_weight=sample_weight),
        "q25_q75_coverage": weighted_interval_coverage(y_true, q25, q75, sample_weight),
        "q90_coverage": np.average(np.asarray(y_true) <= q90, weights=sample_weight),
        "mean_interval_width": np.average(q75 - q25, weights=sample_weight),
        "mean_cushion_width": np.average(q90 - q50, weights=sample_weight),
    }

    for i, (alpha, column) in enumerate(zip(QUANTILE_ALPHAS, QUANTILE_COLUMNS)):
        metrics[f"{column}_pinball_loss"] = mean_pinball_loss(
            y_true,
            y_pred_quantiles[:, i],
            alpha=alpha,
            sample_weight=sample_weight,
        )

    return metrics


def train_xgboost_quantile(args):
    """Train quantile model and persist artifacts."""
    X_train, y_train, w_train = load_xyw(args.train_data)
    X_val, y_val, w_val = load_xyw(args.val_data)

    params = get_xgb_quantile_params(args.point_params)
    model = TransformedTargetRegressor(
        regressor=XGBRegressor(**params),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    print("Training XGBoost quantile regression...")
    start_time = time.time()
    model.fit(X_train, y_train, sample_weight=w_train / w_train.mean())
    training_time = time.time() - start_time

    y_train_pred = monotonic_quantile_predictions(model.predict(X_train))
    y_val_pred = monotonic_quantile_predictions(model.predict(X_val))
    train_metrics = evaluate_quantiles(y_train, y_train_pred, w_train)
    val_metrics = evaluate_quantiles(y_val, y_val_pred, w_val)

    metrics = {
        "XGBoost Quantile": {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "training_time": training_time,
        }
    }
    val_predictions = pd.DataFrame(y_val_pred, index=X_val.index, columns=QUANTILE_COLUMNS)

    save_model(model, args.model_output, verbose=False)
    save_model(val_predictions, args.predictions_output, verbose=False)
    save_metrics(metrics, args.metrics_output, verbose=False)
    save_metrics(params, args.params_output, verbose=False)

    print(
        "XGBoost Quantile  ->  "
        f"q50 MdAE: ${val_metrics['q50_mdae']:,.2f} | "
        f"q25-q75 coverage: {val_metrics['q25_q75_coverage']:.1%} | "
        f"q90 coverage: {val_metrics['q90_coverage']:.1%} | "
        f"training: {training_time:.1f}s"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="data/training_data_preprocessed.parquet")
    parser.add_argument("--val-data", default="data/validation_data_preprocessed.parquet")
    parser.add_argument("--point-params", default="models/xgb_tuned_params.json")
    parser.add_argument("--model-output", default="models/xgb_quantile_model.joblib")
    parser.add_argument("--predictions-output", default="models/xgb_quantile_predictions.joblib")
    parser.add_argument("--metrics-output", default="models/xgb_quantile_metrics.json")
    parser.add_argument("--params-output", default="models/xgb_quantile_params.json")
    return parser.parse_args()


if __name__ == "__main__":
    train_xgboost_quantile(parse_args())
