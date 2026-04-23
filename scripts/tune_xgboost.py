"""
Reproducible hyperparameter tuning script for XGBoost Regressor.

This script performs a randomized search over a defined hyperparameter space 
with MLflow experiment tracking. It uses log-transformed targets 
(TransformedTargetRegressor) and evaluates each configuration on the training 
and validation set using weighted MdAE, MAE, and R². It then retrains the best 
model and persists the fitted model, evaluation metrics, hyperparameters, 
predictions, and the full randomized search history.

Workflow:
  1.  MLflow Setup: Initialize experiment tracking for "XGBoost Tuning".
  2.  Preprocessed Data Loading: Load Parquet datasets into memory.
  3.  Feature-Target Separation: Separate features, target variable, and sample weights.
  4.  Hyperparameter Search: Evaluate N_ITER random configurations using 
      ParameterSampler. Track each trial as an MLflow child run with 
      training/validation metrics and training time.
  5.  Best Model: Retrain the best configuration with full MLflow logging.
  6.  Model Persistence: Save the best tuned model as a Joblib file, evaluation metrics 
      as JSON, parameters as JSON, and full random search history as JSON.

Reference:
    For tuning exploration and detailed rationale, see:
    notebooks/2_modeling.ipynb

Usage:
    1. Start the MLflow UI server (in a separate terminal): ./run_mlflow_ui.sh
    2. Run: ./.venv-train/Scripts/python scripts/tune_xgboost.py
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
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.constants import TARGET_COLUMN, WEIGHT_COLUMN, RANDOM_STATE
from src.modeling import train_and_evaluate, TRAIN_DATA_PATH, VAL_DATA_PATH, weighted_median_absolute_error, save_model, save_metrics, get_core_model_params
from src.params import XGB_PARAM_DISTRIBUTIONS, XGB_N_ITER

# Suppress benign MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


def main():
    # --- 1. MLflow Setup ---
    print("Step 1: Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("XGBoost Tuning")
    print(f"  Set up 'XGBoost Tuning' experiment in MLflow with tracking URI '{mlflow.get_tracking_uri()}'")

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

    # --- 4. Randomized Search ---
    print(f"Step 4: Running randomized search ({XGB_N_ITER} iterations)...")
    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()
    param_list = list(ParameterSampler(XGB_PARAM_DISTRIBUTIONS, n_iter=XGB_N_ITER, random_state=RANDOM_STATE))

    tuning_history = []
    best_mdae = np.inf
    best_idx = -1
    search_start = time.time()

    # Start MLflow parent run
    with mlflow.start_run(run_name="XGBoost Randomized Search"):
        mlflow.set_tag("stage", "tuning")
        mlflow.log_param("n_iterations", XGB_N_ITER)

        for i, params in enumerate(param_list):
            # Build model: XGBoost wrapped in Target Log-Transformer
            model = TransformedTargetRegressor(
                regressor=XGBRegressor(
                    objective="reg:absoluteerror",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    **params
                ),
                func=np.log1p,
                inverse_func=np.expm1
            )

            # Train with normalized sample weights
            iter_start = time.time()
            model.fit(X_train, y_train, sample_weight=w_train_norm)
            training_time = time.time() - iter_start

            # Predict on training and validation set
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Evaluate with raw survey weights
            train_mdae = weighted_median_absolute_error(y_train, y_train_pred, sample_weight=w_train)
            train_mae = mean_absolute_error(y_train, y_train_pred, sample_weight=w_train)
            train_r2 = r2_score(y_train, y_train_pred, sample_weight=w_train)

            val_mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)
            val_mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
            val_r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)

            tuning_history.append({
                "params": params,
                "train_mdae": train_mdae,
                "train_mae": train_mae,
                "train_r2": train_r2,
                "val_mdae": val_mdae,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "training_time": training_time
            })

            # Track best configuration
            if val_mdae < best_mdae:
                best_mdae = val_mdae
                best_idx = i

            # Log each iteration to MLflow as a child run
            with mlflow.start_run(run_name=f"Trial {i+1:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "val_mdae": val_mdae,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                    "train_mdae": train_mdae,
                    "train_mae": train_mae,
                    "train_r2": train_r2,
                    "training_time": training_time
                })

            # Progress logging 
            print(f"  [{i+1:3d}/{XGB_N_ITER}] MdAE: {val_mdae:8.2f} | est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']:.3f}, sub={params['subsample']:.2f}, col={params['colsample_bytree']:.2f} | fit: {training_time:5.1f} s")

        mlflow.log_metric("random_search_time", time.time() - search_start)

    total_search_time = time.time() - search_start
    print(f"  Random search completed in {total_search_time:.0f} s")

    # --- 5. Best Model: Retrain with MLflow Logging ---
    print("Step 5: Retraining best model...")
    best_params = param_list[best_idx]

    best_xgb_model = TransformedTargetRegressor(
        regressor=XGBRegressor(
            objective="reg:absoluteerror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **best_params
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )

    best_xgb_result = train_and_evaluate(
        best_xgb_model,
        X_train, y_train,
        X_val, y_val,
        w_train, w_val,
        track_mlflow=True,
        model_name="XGBoost (Tuned)"
    )
    print(f"  Best Tuned XGBoost  →  MdAE: {best_xgb_result['val_mdae']:.2f} | MAE: {best_xgb_result['val_mae']:.2f} | "
          f"R²: {best_xgb_result['val_r2']:.4f} | Training Time: {best_xgb_result['training_time']:.2f}s")

    # --- 6. Model Persistence ---
    print("Step 6: Persisting hyperparameter tuning results...")

    save_model(best_xgb_result["fitted_model"], "models/xgb_tuned_model.joblib", verbose=False)
    print("  Saved best model to 'models/xgb_tuned_model.joblib'")

    tuned_metrics = {
        "XGBoost (Tuned)": {
            "val_mdae": best_xgb_result["val_mdae"],
            "val_mae": best_xgb_result["val_mae"],
            "val_r2": best_xgb_result["val_r2"],
            "train_mdae": best_xgb_result["train_mdae"],
            "train_mae": best_xgb_result["train_mae"],
            "train_r2": best_xgb_result["train_r2"],
            "training_time": best_xgb_result["training_time"]
        }
    }
    save_metrics(tuned_metrics, "models/xgb_tuned_metrics.json", verbose=False)
    print("  Saved evaluation metrics of best model to 'models/xgb_tuned_metrics.json'")

    save_metrics(tuning_history, "models/xgb_tuning_history.json", verbose=False)
    print("  Saved evaluation metrics of all models to 'models/xgb_tuning_history.json'")

    save_metrics(get_core_model_params(best_xgb_result["fitted_model"]), "models/xgb_tuned_params.json", verbose=False)
    print("  Saved hyperparameters of best model to 'models/xgb_tuned_params.json'")
    
    save_model(best_xgb_result["y_val_pred"], "models/xgb_tuned_predictions.joblib", verbose=False)
    print("  Saved predicted values of best model to 'models/xgb_tuned_predictions.joblib'")

    print("\n✅ XGBoost hyperparameter tuning complete.")


if __name__ == "__main__":
    main()
