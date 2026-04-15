"""
Reproducible model training and evaluation script for baseline models.

This script trains multiple baseline models, logs the experiments to MLflow, and persists the model results.

Workflow:
  1.  MLflow Setup: Initialize experiment tracking for "Baseline Models".
  2.  Data Loading: Load preprocessed Parquet datasets into Memory.
  3.  Feature-Target Separation: Separate features, target variable, and sample weights.
  4.  Training and Evaluation: Fit baseline models (e.g., Linear Regression, Random Forest, XGBoost)
      on the training data using fixed random states. Predict and evaluate metrics on the validation data.
  5.  Model Persistence: Save fitted models as individual Joblib files (DVC-tracked), evaluation metrics 
      as a collective JSON file (Git-tracked), and predicted values as a collective Joblib file (DVC-tracked).

Reference:
    For model training exploration, in-depth model evaluation, error analysis, and detailed rationale, see:
    notebooks/2_modeling.ipynb

Usage:
    ./.venv-train/Scripts/python scripts/train_baseline.py
"""

# Standard library imports
import warnings

# Thrid-party imports
import pandas as pd
import mlflow

# Local imports
from src.constants import TARGET_COLUMN, WEIGHT_COLUMN
from src.modeling import get_baseline_models, train_and_evaluate
from src.utils import save_model, save_metrics, get_core_model_params

# Suppress benign MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Paths (relative to project root)
TRAIN_DATA_PATH = "data/training_data_preprocessed.parquet"
VAL_DATA_PATH = "data/validation_data_preprocessed.parquet"


# Main Baseline Model Training 
def main():
    # --- 1. MLflow Setup --- 
    print("Step 1: Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # Points to running MLflow UI server
    mlflow.set_experiment("Baseline Models")
    print(f"  Set up 'Baseline Models' experiment in MLflow with tracking URI '{mlflow.get_tracking_uri()}'")

    # --- 2. Preprocessed Data Loading ---
    print("Step 2: Loading preprocessed data...")
    df_train_preprocessed = pd.read_parquet(TRAIN_DATA_PATH)
    df_val_preprocessed = pd.read_parquet(VAL_DATA_PATH)
    print(f"  Loaded '{TRAIN_DATA_PATH}' with {len(df_train_preprocessed):,} rows and {len(df_train_preprocessed.columns):,} columns")
    print(f"  Loaded '{VAL_DATA_PATH}' with {len(df_val_preprocessed):,} rows and {len(df_val_preprocessed.columns):,} columns")

    # --- 3. Feature-Target Separation ---
    print("Step 3: Separating features and target...")
    X_train_preprocessed = df_train_preprocessed.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
    y_train = df_train_preprocessed[TARGET_COLUMN]
    w_train = df_train_preprocessed[WEIGHT_COLUMN]
    X_val_preprocessed = df_val_preprocessed.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
    y_val = df_val_preprocessed[TARGET_COLUMN]
    w_val = df_val_preprocessed[WEIGHT_COLUMN]
    del df_train_preprocessed, df_val_preprocessed  # Free up memory
    print("  Separated data into X features, y target variable, and w sample weights")

    # --- 4. Model Training and Evaluation ---
    print("Step 4: Training and evaluating baseline models...")    
    baseline_models = get_baseline_models()
    baseline_results = {}
    for model_name, model in baseline_models.items():
        print(f"  Training {model_name}...")
        result = train_and_evaluate(
            model, 
            X_train_preprocessed, y_train, 
            X_val_preprocessed, y_val, 
            w_train, w_val,
            track_mlflow=True,
            model_name=model_name
        )
        baseline_results[model_name] = result
        print(f"    {model_name:<20} → MdAE: {result['val_mdae']:8.2f} | MAE: {result['val_mae']:8.2f} | "
              f"R²: {result['val_r2']:.4f} | Training Time: {result['training_time']:.2f}s")

    # --- 5. Model Persistence ---
    print("Step 5: Persisting baseline models...")
    for model_name, result in baseline_results.items():        
        # Consistent model identifier for filenames
        model_id = (model_name.lower().replace(" ", "_")
                    .replace("support_vector_machine", "svm")
                    .replace("linear_regression", "lr")
                    .replace("elastic_net", "en")
                    .replace("decision_tree", "tree")
                    .replace("random_forest", "rf")
                    .replace("xgboost", "xgb")
                    .replace("median_prediction", "median"))
        
        # 5.1. Save fitted model as .joblib file
        model_path = f"models/{model_id}_baseline_model.joblib"
        save_model(result["fitted_model"], model_path, verbose=False)
        print(f"  Saved fitted {model_name} model to '{model_path}'")
        
        # 5.2. Save evaluation metrics as JSON
        metrics_dict = {
            model_name: {
                "val_mdae": result["val_mdae"],
                "val_mae": result["val_mae"],
                "val_r2": result["val_r2"],
                "train_mdae": result["train_mdae"],
                "train_mae": result["train_mae"],
                "train_r2": result["train_r2"],
                "training_time": result["training_time"]
            }
        }
        metrics_path = f"models/{model_id}_baseline_metrics.json"
        save_metrics(metrics_dict, metrics_path, verbose=False)
        print(f"  Saved evaluation metrics of {model_name} to '{metrics_path}'")

        # 5.3. Save hyperparameters as JSON
        params_path = f"models/{model_id}_baseline_params.json"
        save_metrics(get_core_model_params(result["fitted_model"]), params_path, verbose=False)
        print(f"  Saved hyperparameters of {model_name} to '{params_path}'")
        
        # 5.4. Save predicted values as .joblib file
        pred_path = f"models/{model_id}_baseline_predictions.joblib"
        save_model(result["y_val_pred"], pred_path, verbose=False)
        print(f"  Saved predicted values of {model_name} to '{pred_path}'")

    print("\n✅ Baseline model training complete.")


if __name__ == "__main__":
    main()