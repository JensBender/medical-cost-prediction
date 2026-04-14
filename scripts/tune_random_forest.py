"""
Reproducible hyperparameter tuning script for Random Forest Regressor.

This script performs a randomized search over a defined hyperparameter space 
with MLflow experiment tracking, evaluates each configuration on the training 
and validation set using weighted MdAE, MAE, and R², and persists metrics, 
the best model and predicted values.

Workflow:
  1.  MLflow Setup: Initialize experiment tracking for "RF Tuning".
  2.  Preprocessed Data Loading: Load preprocessed Parquet datasets into memory.
  3.  Feature-Target Separation: Separate features, target variable, and sample weights.
  4.  Hyperparameter Search: Evaluate N_ITER random configurations using 
      ParameterSampler on the holdout validation set with weighted scoring.
  5.  Best Model: Retrain the best configuration with full MLflow logging.
  6.  Model Persistence: Save the best tuned model as a Joblib file, evaluation metrics
      as JSON, and tuning search results as CSV.

Reference:
    For tuning exploration and detailed rationale, see:
    notebooks/2_modeling.ipynb

Usage:
    ./.venv-train/Scripts/python scripts/tune_random_forest.py
"""

# Standard library imports
import time
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.constants import TARGET_COLUMN, WEIGHT_COLUMN, RANDOM_STATE
from src.modeling import train_and_evaluate
from src.params import RF_PARAM_DISTRIBUTIONS, RF_N_ITER
from src.utils import weighted_median_absolute_error, save_model, save_metrics

# Suppress benign MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Paths (relative to project root)
TRAIN_DATA_PATH = "data/training_data_preprocessed.parquet"
VAL_DATA_PATH = "data/validation_data_preprocessed.parquet"


def main():
    # --- 1. MLflow Setup ---
    print("Step 1: Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Random Forest Tuning")
    print(f"  Set up 'Random Forest Tuning' experiment in MLflow with tracking URI '{mlflow.get_tracking_uri()}'")

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
    print(f"Step 4: Running randomized search ({RF_N_ITER} iterations)...")
    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()
    param_list = list(ParameterSampler(RF_PARAM_DISTRIBUTIONS, n_iter=RF_N_ITER, random_state=RANDOM_STATE))

    tuning_metrics = []
    best_mdae = np.inf
    search_start = time.time()

    # Start MLflow parent run (to group all iterations as child runs for better organization in UI)
    with mlflow.start_run(run_name="Random Forest Randomized Search"):
        mlflow.set_tag("stage", "tuning")
        mlflow.log_param("n_iterations", RF_N_ITER)

        for i, params in enumerate(param_list):
            # Build model: RandomForest wrapped in TransformedTargetRegressor(log1p)
            model = TransformedTargetRegressor(
                regressor=RandomForestRegressor(
                    criterion="absolute_error",
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

            # Predict on training and validation set (predictions in raw dollars)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Evaluate with raw survey weights
            train_mdae = weighted_median_absolute_error(y_train, y_train_pred, sample_weight=w_train)
            train_mae = mean_absolute_error(y_train, y_train_pred, sample_weight=w_train)
            train_r2 = r2_score(y_train, y_train_pred, sample_weight=w_train)

            val_mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)
            val_mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
            val_r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)

            tuning_metrics.append({
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
            print(f"  [{i+1:3d}/{RF_N_ITER}] MdAE: {val_mdae:8.2f} | trees={params['n_estimators']}, depth={params['max_depth']}, leaf={params['min_samples_leaf']}, feats={params['max_features']}, samples={params['max_samples']:.2f}, split={params['min_samples_split']} | training: {training_time:5.1f} s")

    total_search_time = time.time() - search_start
    print(f"  Random search complete in {total_search_time:.0f} s")

    # --- 5. Best Model: Retrain with MLflow Logging ---
    print("Step 5: Retraining best configuration with MLflow logging...")
    best_params = param_list[best_idx]
    print(f"  Best hyperparameters (iter {best_idx+1}): {best_params}")

    best_rf_model = TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            criterion="absolute_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **best_params
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )

    best_rf_result = train_and_evaluate(
        best_rf_model,
        X_train, y_train,
        X_val, y_val,
        w_train, w_val,
        track_mlflow=True,
        model_name="Random Forest (Tuned)",
        log_model=True
    )
    print(f"  Tuned RF  →  MdAE: {best_rf_result['val_mdae']:.2f} | MAE: {best_rf_result['val_mae']:.2f} | "
          f"R²: {best_rf_result['val_r2']:.4f} | Time: {best_rf_result['training_time']:.2f}s")

    # --- 6. Model Persistence ---
    print("Step 6: Persisting hyperparameter tuning results...")

    # Save best fitted model as .joblib file
    save_model(best_rf_result["fitted_model"], "models/rf_tuned_model.joblib", verbose=False)
    print("  Saved best model to 'models/rf_tuned_model.joblib'")

    # Save evaluation metrics of best model as JSON
    tuned_metrics = {
        "Random Forest (Tuned)": {
            "val_mdae": best_rf_result["val_mdae"],
            "val_mae": best_rf_result["val_mae"],
            "val_r2": best_rf_result["val_r2"],
            "train_mdae": best_rf_result["train_mdae"],
            "train_mae": best_rf_result["train_mae"],
            "train_r2": best_rf_result["train_r2"]
        }
    }
    save_metrics(tuned_metrics, "models/rf_tuned_metrics.json", verbose=False)
    print("  Saved evaluation metrics to 'models/rf_tuned_metrics.json'")

    # Save metrics of all randomly searched models as JSON 
    save_metrics(tuning_metrics, "models/rf_tuning_history.json", verbose=False)
    print("  Saved metrics of all randomly searched models to 'models/rf_tuning_history.json'")

    # Save best hyperparameters as JSON for reproducibility
    save_metrics(best_params, "models/rf_tuned_params.json", verbose=False)
    print("  Saved best hyperparameters to 'models/rf_tuned_params.json'")
    
    # Save predictions of best model as .joblib file
    save_model(best_rf_result["y_val_pred"], "models/rf_tuned_predictions.joblib", verbose=False)
    print("  Saved predicted values of best model to 'models/rf_tuned_predictions.joblib'")

    print("\n✅ Random Forest hyperparameter tuning complete.")


if __name__ == "__main__":
    main()
