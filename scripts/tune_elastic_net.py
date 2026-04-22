"""
Reproducible hyperparameter tuning script for Elastic Net Regressor.

This script performs a randomized search over a defined hyperparameter space 
with MLflow experiment tracking. It uses a pipeline with second-degree polynomial 
features and log-transformed targets (TransformedTargetRegressor). It evaluates 
each configuration on the training and validation set using weighted MdAE, MAE, and R². It then retrains the best 
configuration using the train_and_evaluate function and persists the fitted 
model, evaluation metrics, hyperparameters, validation predictions, and the 
full randomized search history.

Workflow:
  1.  MLflow Setup: Initialize experiment tracking for "Elastic Net Tuning".
  2.  Preprocessed Data Loading: Load preprocessed Parquet datasets into memory.
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
    ./.venv-train/Scripts/python scripts/tune_elastic_net.py
"""

# Standard library imports
import time
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.constants import TARGET_COLUMN, WEIGHT_COLUMN, RANDOM_STATE
from src.modeling import train_and_evaluate
from src.params import EN_PARAM_DISTRIBUTIONS, EN_N_ITER
from src.utils import weighted_median_absolute_error, save_model, save_metrics, get_core_model_params

# Suppress benign MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Paths (relative to project root)
TRAIN_DATA_PATH = "data/training_data_preprocessed.parquet"
VAL_DATA_PATH = "data/validation_data_preprocessed.parquet"


def main():
    # --- 1. MLflow Setup ---
    print("Step 1: Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Elastic Net Tuning")
    print(f"  Set up 'Elastic Net Tuning' experiment in MLflow with tracking URI '{mlflow.get_tracking_uri()}'")

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
    print(f"Step 4: Running randomized search ({EN_N_ITER} iterations)...")
    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()
    param_list = list(ParameterSampler(EN_PARAM_DISTRIBUTIONS, n_iter=EN_N_ITER, random_state=RANDOM_STATE))

    tuning_history = []
    best_mdae = np.inf  # positive infinity
    best_idx = -1
    search_start = time.time()

    # Start MLflow parent run (to group all iterations as child runs for better organization in UI)
    with mlflow.start_run(run_name="Elastic Net Randomized Search"):
        mlflow.set_tag("stage", "tuning")
        mlflow.log_param("n_iterations", EN_N_ITER)

        for i, params in enumerate(param_list):
            # Build model: Elastic Net with Polynomial Features wrapped in Target Log-Transformer
            model = TransformedTargetRegressor(
                regressor=Pipeline([
                    ("polynomials", PolynomialFeatures(degree=2, include_bias=False)),  # include_bias=False lets ElasticNet handle the intercept
                    ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=2000))
                ]),
                func=np.log1p,
                inverse_func=np.expm1
            )
            # Set hyperparameters for the internal model in the pipeline
            model.regressor.set_params(**params)

            # Train with normalized sample weights
            iter_start = time.time()
            # We need to pass the sample_weight correctly to the last step of the pipeline
            model.fit(X_train, y_train, model__sample_weight=w_train_norm)
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
            with mlflow.start_run(run_name=f"Trial {i+1:03d}", nested=True):  # :03d displays 3-digit integer with leading zeros
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
            squares_label = "off" if params["polynomials__interaction_only"] else "on "  # interaction_only=True means turning off squared features
            print(f"  [{i+1:3d}/{EN_N_ITER}] MdAE: {val_mdae:8.2f} | alpha={params['model__alpha']:.2f}, l1_ratio={params['model__l1_ratio']:.2f}, squares={squares_label:3} | fit: {training_time:5.1f} s")

        mlflow.log_metric("random_search_time", time.time() - search_start)

    total_search_time = time.time() - search_start
    print(f"  Random search completed in {total_search_time:.0f} s")

    # --- 5. Best Model: Retrain with MLflow Logging ---
    print("Step 5: Retraining best model...")
    best_params = param_list[best_idx]

    best_en_model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("polynomials", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=5000))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )
    best_en_model.regressor.set_params(**best_params)

    best_en_result = train_and_evaluate(
        best_en_model,
        X_train, y_train,
        X_val, y_val,
        w_train, w_val,
        track_mlflow=True,
        model_name="Elastic Net (Tuned)"
    )
    print(f"  Best Tuned Elastic Net  →  MdAE: {best_en_result['val_mdae']:.2f} | MAE: {best_en_result['val_mae']:.2f} | "
          f"R²: {best_en_result['val_r2']:.4f} | Training Time: {best_en_result['training_time']:.2f}s")

    # --- 6. Model Persistence ---
    print("Step 6: Persisting hyperparameter tuning results...")

    # Save best fitted model as .joblib file
    save_model(best_en_result["fitted_model"], "models/en_tuned_model.joblib", verbose=False)
    print("  Saved best model to 'models/en_tuned_model.joblib'")

    # Save evaluation metrics of best model as JSON
    tuned_metrics = {
        "Elastic Net (Tuned)": {
            "val_mdae": best_en_result["val_mdae"],
            "val_mae": best_en_result["val_mae"],
            "val_r2": best_en_result["val_r2"],
            "train_mdae": best_en_result["train_mdae"],
            "train_mae": best_en_result["train_mae"],
            "train_r2": best_en_result["train_r2"],
            "training_time": best_en_result["training_time"]
        }
    }
    save_metrics(tuned_metrics, "models/en_tuned_metrics.json", verbose=False)
    print("  Saved evaluation metrics of best model to 'models/en_tuned_metrics.json'")

    # Save evaluation metrics of all randomly searched models as JSON 
    save_metrics(tuning_history, "models/en_tuning_history.json", verbose=False)
    print("  Saved evaluation metrics of all models to 'models/en_tuning_history.json'")

    # Save hyperparameters as JSON
    save_metrics(get_core_model_params(best_en_result["fitted_model"]), "models/en_tuned_params.json", verbose=False)
    print("  Saved hyperparameters of best model to 'models/en_tuned_params.json'")
    
    # Save predictions of best model as .joblib file
    save_model(best_en_result["y_val_pred"], "models/en_tuned_predictions.joblib", verbose=False)
    print("  Saved predicted values of best model to 'models/en_tuned_predictions.joblib'")

    print("\n✅ Elastic Net hyperparameter tuning complete.")


if __name__ == "__main__":
    main()
