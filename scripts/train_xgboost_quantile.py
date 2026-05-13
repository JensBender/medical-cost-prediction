"""
Reproducible model training and evaluation script for XGBoost quantile regression.

This script trains a multi-quantile XGBoost model using the hyperparameters from 
the best tuned point-estimate model. It predicts four quantiles (q25, q50, q75, q90)
to provide prediction ranges (q25-q75) and a cushion (q90) for financial planning, 
logs the experiment to MLflow, and persists the model results.

Workflow:
  1.  MLflow Setup: Initialize experiment tracking for "Quantile Regression".
  2.  Preprocessed Data Loading: Load Parquet datasets into memory.
  3.  Feature-Target Separation: Separate features, target variable, and sample weights.
  4.  Model Configuration: Load tuned hyperparameters and adapt them for quantile regression.
  5.  Training: Fit the multi-quantile model on log-transformed targets.
  6.  Predictions: Generate and post-process predictions (non-negative, monotonic).
  7.  Evaluation: Compute median accuracy, interval coverage, and interval width metrics.
  8.  Model Persistence: Save the fitted model, evaluation metrics, hyperparameters, and
      predicted values.

Artifacts:
  - models/xgb_quantile_model.joblib: Fitted model.
  - models/xgb_quantile_metrics.json: Evaluation metrics.
  - models/xgb_quantile_params.json: Hyperparameters used for training.
  - models/xgb_quantile_predictions.joblib: Validation set predictions for all quantiles.

Reference:
    For quantile regression exploration and detailed rationale, see:
    notebooks/2_modeling.ipynb

Usage:
    1. Start the MLflow UI server (in a separate terminal): ./run_mlflow_ui.sh
    2. Run: ./.venv-train/Scripts/python scripts/train_xgboost_quantile.py
"""

# Standard library imports
import time
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import mlflow
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.constants import TARGET_COLUMN, WEIGHT_COLUMN
from src.modeling import TRAIN_DATA_PATH, VAL_DATA_PATH, weighted_median_absolute_error, save_model, save_metrics, load_metrics

# Suppress benign MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


def main():
    # --- 1. MLflow Setup ---
    print("Step 1: Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Points to running MLflow UI server
    mlflow.set_experiment("Quantile Regression")
    print(f"  Set up 'Quantile Regression' experiment in MLflow with tracking URI '{mlflow.get_tracking_uri()}'")

    # --- 2. Preprocessed Data Loading ---
    print("Step 2: Loading preprocessed data...")
    df_train = pd.read_parquet(TRAIN_DATA_PATH)
    df_val = pd.read_parquet(VAL_DATA_PATH)
    print(f"  Loaded '{TRAIN_DATA_PATH}' with {len(df_train):,} rows and {len(df_train.columns):,} columns")
    print(f"  Loaded '{VAL_DATA_PATH}' with {len(df_val):,} rows and {len(df_val.columns):,} columns")

    # --- 3. Feature-Target Separation ---
    print("Step 3: Separating features and target...")
    X_train = df_train.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
    y_train = df_train[TARGET_COLUMN]
    w_train = df_train[WEIGHT_COLUMN]
    X_val = df_val.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
    y_val = df_val[TARGET_COLUMN]
    w_val = df_val[WEIGHT_COLUMN]
    del df_train, df_val  # Free up memory
    print("  Separated data into X features, y target variable, and w sample weights")

    # --- 4. Model Configuration ---
    print("Step 4: Configuring XGBoost multi-quantile model parameters...")
    QUANTILES = [0.25, 0.50, 0.75, 0.90]

    tuned_params = load_metrics("models/xgb_tuned_params.json", verbose=False)
    keep_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "tree_method",
            "n_jobs",
            "random_state",
    ]
    xgb_quantile_params = {k: tuned_params[k] for k in keep_params}
    xgb_quantile_params.update({
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILES,
    })
    print(f"  Loaded hyperparameters of best tuned model and updated them for {len(QUANTILES)} quantiles: {QUANTILES}")

    # --- 5. Model Training ---
    print("Step 5: Training XGBoost quantile regression model...")
    # Train on log-costs: quantiles are invariant to monotonic transformations, and the log scale
    # stabilizes tree-splitting logic by preventing extreme outliers from dominating the partition search.
    xgb_quantile_model = TransformedTargetRegressor(
        regressor=XGBRegressor(**xgb_quantile_params),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()

    with mlflow.start_run(run_name="XGBoost Quantile"):
        mlflow.set_tag("stage", "quantile_training")
        mlflow.log_params(xgb_quantile_params)

        start_time = time.time()
        xgb_quantile_model.fit(X_train, y_train, sample_weight=w_train_norm)
        training_time = time.time() - start_time

        print(f"  Completed training in {training_time:.1f} s")

        # --- 6. Predictions ---
        print("Step 6: Predicting on training and validation set...")
        # Predict on training and validation set
        y_train_pred_raw = xgb_quantile_model.predict(X_train)
        y_val_pred_raw = xgb_quantile_model.predict(X_val)

        # Ensure non-negative predictions
        y_train_pred_non_negative = np.maximum(y_train_pred_raw, 0)
        y_val_pred_non_negative = np.maximum(y_val_pred_raw, 0)

        # Ensure monotonic predictions (q25 <= q50 <= q75 <= q90) by pulling lower estimates up (more conservative for financial planning)
        y_train_pred = np.maximum.accumulate(y_train_pred_non_negative, axis=1)
        y_val_pred = np.maximum.accumulate(y_val_pred_non_negative, axis=1)
        print(f"  Generated predictions for {len(y_train_pred):,} train and {len(y_val_pred):,} validation samples and ensured non-negative and monotonic predictions")

        # --- 7. Evaluation ---
        print("Step 7: Evaluating model performance...")
        # Unpack quantiles
        y_train_pred_q25, y_train_pred_q50, y_train_pred_q75, y_train_pred_q90 = y_train_pred.T
        y_val_pred_q25, y_val_pred_q50, y_val_pred_q75, y_val_pred_q90 = y_val_pred.T

        # Evaluate median prediction
        train_q50_mdae = weighted_median_absolute_error(y_train, y_train_pred_q50, sample_weight=w_train)
        train_q50_mae = mean_absolute_error(y_train, y_train_pred_q50, sample_weight=w_train)
        train_q50_r2 = r2_score(y_train, y_train_pred_q50, sample_weight=w_train)

        val_q50_mdae = weighted_median_absolute_error(y_val, y_val_pred_q50, sample_weight=w_val)
        val_q50_mae = mean_absolute_error(y_val, y_val_pred_q50, sample_weight=w_val)
        val_q50_r2 = r2_score(y_val, y_val_pred_q50, sample_weight=w_val)

        # Evaluate coverage (share of population whose actual cost is within the predicted range)
        train_q25_q75_coverage = np.average((y_train >= y_train_pred_q25) & (y_train <= y_train_pred_q75), weights=w_train)
        train_q90_coverage = np.average(y_train <= y_train_pred_q90, weights=w_train)
        val_q25_q75_coverage = np.average((y_val >= y_val_pred_q25) & (y_val <= y_val_pred_q75), weights=w_val)
        val_q90_coverage = np.average(y_val <= y_val_pred_q90, weights=w_val)

        # Evaluate interval precision
        train_q25_q75_width = np.average(y_train_pred_q75 - y_train_pred_q25, weights=w_train)
        train_q50_q90_width = np.average(y_train_pred_q90 - y_train_pred_q50, weights=w_train)  # "Safety Cushion" width
        val_q25_q75_width = np.average(y_val_pred_q75 - y_val_pred_q25, weights=w_val)
        val_q50_q90_width = np.average(y_val_pred_q90 - y_val_pred_q50, weights=w_val)

        print(f"  Median MdAE       ->  Train: ${train_q50_mdae:.2f} | Val: ${val_q50_mdae:.2f}")
        print(f"  Median MAE        ->  Train: ${train_q50_mae:.2f} | Val: ${val_q50_mae:.2f}")
        print(f"  Median R2         ->  Train: {train_q50_r2:.2f} | Val: {val_q50_r2:.2f}")
        print(f"  q25-q75 coverage  ->  Train: {train_q25_q75_coverage:.1%} | Val: {val_q25_q75_coverage:.1%}")
        print(f"  q90 coverage      ->  Train: {train_q90_coverage:.1%} | Val: {val_q90_coverage:.1%}")
        print(f"  Avg Range Width   ->  Train: ${train_q25_q75_width:.0f} | Val: ${val_q25_q75_width:.0f}")
        print(f"  Avg Cushion Width ->  Train: ${train_q50_q90_width:.0f} | Val: ${val_q50_q90_width:.0f}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_q50_mdae": train_q50_mdae,
            "train_q50_mae": train_q50_mae,
            "train_q50_r2": train_q50_r2,
            "train_q25_q75_coverage": train_q25_q75_coverage,
            "train_q90_coverage": train_q90_coverage,
            "train_q25_q75_width": train_q25_q75_width,
            "train_q50_q90_width": train_q50_q90_width,
            "val_q50_mdae": val_q50_mdae,
            "val_q50_mae": val_q50_mae,
            "val_q50_r2": val_q50_r2,
            "val_q25_q75_coverage": val_q25_q75_coverage,
            "val_q90_coverage": val_q90_coverage,
            "val_q25_q75_width": val_q25_q75_width,
            "val_q50_q90_width": val_q50_q90_width,
            "training_time": training_time,
        })

    # --- 8. Model Persistence ---
    print("Step 8: Persisting model results...")
    # 8.1. Save fitted model as .joblib file
    save_model(xgb_quantile_model, "models/xgb_quantile_model.joblib", verbose=False)
    print("  Saved XGBoost quantile regression model to 'models/xgb_quantile_model.joblib'")

    # 8.2. Save evaluation metrics as JSON
    xgb_quantile_metrics = {
        "XGBoost (Quantile)": {
            "train_q50_mdae": train_q50_mdae,
            "train_q50_mae": train_q50_mae,
            "train_q50_r2": train_q50_r2,
            "train_q25_q75_coverage": train_q25_q75_coverage,
            "train_q90_coverage": train_q90_coverage,
            "train_q25_q75_width": train_q25_q75_width,
            "train_q50_q90_width": train_q50_q90_width,
            "val_q50_mdae": val_q50_mdae,
            "val_q50_mae": val_q50_mae,
            "val_q50_r2": val_q50_r2,
            "val_q25_q75_coverage": val_q25_q75_coverage,
            "val_q90_coverage": val_q90_coverage,
            "val_q25_q75_width": val_q25_q75_width,
            "val_q50_q90_width": val_q50_q90_width,
            "training_time": training_time,
        }
    }
    save_metrics(xgb_quantile_metrics, "models/xgb_quantile_metrics.json", verbose=False)
    print("  Saved evaluation metrics of XGBoost quantile regression to 'models/xgb_quantile_metrics.json'")

    # 8.3. Save hyperparameters as JSON
    save_metrics(xgb_quantile_params, "models/xgb_quantile_params.json", verbose=False)
    print("  Saved hyperparameters of XGBoost quantile regression to 'models/xgb_quantile_params.json'")

    # 8.4. Save predicted values as .joblib file
    save_model(y_val_pred, "models/xgb_quantile_predictions.joblib", verbose=False)
    print("  Saved predicted values of XGBoost quantile regression to 'models/xgb_quantile_predictions.joblib'")

    print("\n[OK] XGBoost quantile regression complete.")


if __name__ == "__main__":
    main()