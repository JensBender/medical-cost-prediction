# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Medical Cost Prediction
#     language: python
#     name: medical_cost_prediction
# ---

# %% [markdown]
# <div style="text-align:center; background-color:#fff6e4; padding:20px; border:5px solid #f5ecda; border-radius:8px;">
#     <div style="font-size:36px; font-weight:bold; color:#4A4A4A;">
#         Medical Cost Prediction
#     </div>
#     <div style="font-size:24px; font-weight:bold; color:#4A4A4A;">
#         Part 2: Modeling
#     </div>
#     <div style="font-size:14px; font-weight:normal; color:#666; margin-top:16px;">
#         Author: Jens Bender <br> 
#         Created: March 2026<br>
#         Last updated: August 2026
#     </div>
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Setup</h1>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     <strong>Notebook Settings</strong>
# </div>

# %%
# Automatically reload local modules before each cell run (prevents having to restart kernel after changes)
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     <strong>Imports</strong>
# </div>

# %%
# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.compose import TransformedTargetRegressor  # to log-transform target
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Custom transformers
from src.transformers import MedicalFeatureDeriver

# Models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Model selection
from sklearn.model_selection import ParameterSampler

# Model evaluation
from sklearn.metrics import (
    mean_absolute_error, 
    mean_pinball_loss,
    r2_score
)
import time  # to measure training time

# Model explainability
import shap 

# Local imports
from src.modeling import (
    get_baseline_models,
    train_and_evaluate,
    weighted_median_absolute_error,
    save_model,
    load_model,
    save_metrics,
    load_metrics,
    get_core_model_params,
    postprocess_quantile_predictions
)
from src.stats import (
    weighted_quantile,
    weighted_std,
    create_stratification_bins
)
from src.params import (
    EN_PARAM_DISTRIBUTIONS,
    RF_PARAM_DISTRIBUTIONS, 
    XGB_PARAM_DISTRIBUTIONS 
)
from src.constants import (
    ID_COLUMN,
    WEIGHT_COLUMN,
    TARGET_COLUMN,
    RANDOM_STATE,
    PIPELINE_NUMERICAL_FEATURES,
    PIPELINE_BINARY_FEATURES,
    PIPELINE_NOMINAL_FEATURES,
)
from src.display import (
    DISPLAY_LABELS, 
    METRIC_LABELS,
    MODEL_DISPLAY_LABELS,
    CATEGORY_LABELS_EDA,
    POP_COLOR,
    SAMPLE_COLOR,
    TYPICAL_RANGE_COLOR,
    SAFETY_CUSHION_COLOR,
    add_table_caption
)

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Data Loading</h1>
# </div>
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Load the model-ready data from the <code>.parquet</code> files into Pandas DataFrames.
# </div>

# %%
df_train_preprocessed = pd.read_parquet("../data/training_data_model_ready.parquet")
df_val_preprocessed = pd.read_parquet("../data/validation_data_model_ready.parquet")
df_test_preprocessed = pd.read_parquet("../data/test_data_model_ready.parquet")

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> 
#     📌 Inspect the data.
# </div>

# %%
def inspect_df(df):
    """
    Inspect a DataFrame and return its shape and data integrity status.

    Args:
        df (pd.DataFrame): The DataFrame to be validated.

    Returns:
        list: A list containing:
            - tuple: Shape of the DataFrame (rows, columns).
            - str: Status icon for numerical-only columns (✅ if all numerical, ❌ otherwise).
            - str: Status icon for missing values (✅ if none, ❌ otherwise).
            - str: Status icon for infinite values (✅ if none, ❌ otherwise).
            - str: Status icon for constant columns (✅ if none, ❌ otherwise).
            - str: Status icon for unique index/ID (✅ if unique, ❌ otherwise).
            - str: Status icon for target variable presence (✅ if TARGET_COLUMN exists, ❌ otherwise).
            - str: Status icon for weight variable presence (✅ if WEIGHT_COLUMN exists, ❌ otherwise).
    """
    shape = df.shape
    no_missings = "✅" if not df.isna().any().any() else "❌"
    all_numerical = "✅" if df.select_dtypes(exclude=[np.number]).empty else "❌"
    no_infinites = "✅" if not np.isinf(df.select_dtypes(include=[np.number])).any().any() else "❌"
    no_constants = "✅" if (df.nunique(dropna=False) > 1).all() else "❌"
    unique_id = "✅" if (df.index.is_unique and df.index.name == ID_COLUMN) else "❌"
    target_present = "✅" if TARGET_COLUMN in df.columns else "❌"
    weights_present = "✅" if WEIGHT_COLUMN in df.columns else "❌"

    return [
        shape,
        all_numerical,
        no_missings,
        no_infinites,
        no_constants,
        unique_id,
        target_present,
        weights_present,
    ]

data_inspection = pd.DataFrame(
    {
        "Training": inspect_df(df_train_preprocessed),
        "Validation": inspect_df(df_val_preprocessed),
        "Test": inspect_df(df_test_preprocessed),
    },
    index=[
        "Shape",
        "All Numerical",
        "No Missings",
        "No Infinites",
        "No Constants",
        "Unique ID",
        "Target Present",
        "Weights Present",
    ],
)
display(data_inspection.style.pipe(add_table_caption, "Data Inspection"))

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> 
#     📌 Split the preprocessed data into X features, y target variable, and sample weights (w).
# </div>

# %%
X_train_preprocessed = df_train_preprocessed.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
y_train = df_train_preprocessed[TARGET_COLUMN]
w_train = df_train_preprocessed[WEIGHT_COLUMN]

X_val_preprocessed = df_val_preprocessed.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
y_val = df_val_preprocessed[TARGET_COLUMN]
w_val = df_val_preprocessed[WEIGHT_COLUMN]

X_test_preprocessed = df_test_preprocessed.drop([TARGET_COLUMN, WEIGHT_COLUMN], axis=1)
y_test = df_test_preprocessed[TARGET_COLUMN]
w_test = df_test_preprocessed[WEIGHT_COLUMN]

# Delete redundant DataFrames to free up memory
del df_train_preprocessed, df_val_preprocessed, df_test_preprocessed

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Baseline Models</h1>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Training</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Train 6 baseline models with distribution-aware baseline hyperparameters.  
#     <ul>
#         <li>Linear Regression (lr)</li>
#         <li>Elastic Net Regression (en)</li>
#         <li>Decision Tree Regressor (tree)</li>
#         <li>Random Forest Regressor (rf)</li>
#         <li>XGBoost Regressor (xgb)</li>
#         <li>Support Vector Regressor (svr)</li>
#     </ul>
#     Training Setup:
#     <ul>
#         <li>Train on preprocessed data (standardized, imputed, feature engineered, scaled, and encoded).</li>
#         <li>Train on all candidate features (27 raw, 40 preprocessed).</li>
#         <li>Use sample weights for population representativeness. Normalize weights (mean=1.0) to maintain relative importance while ensuring numerical stability during training (especially for svr).</li>
#         <li>Apply log-transformation of target variable for all baseline models using <code>TransformedTargetRegressor</code>. Use <code>log1p</code> instead of <code>log</code> to handle zeros in target (<code>log(0)</code> is undefined).</li>
#         <li>Implement polynomial features for elastic net regression using second-degree <code>PolynomialFeatures</code> with a small <code>Pipeline</code>.</li>
#         <li>For each model, store the following artifacts: the fitted model as a <code>.joblib</code> file, evaluation metrics as a <code>.json</code> file, model parameters as a <code>.json</code> file, predictions as a <code>.joblib</code> file.</li>
#     </ul>  
#     For more details, see <a href="../src/modeling.py">src/modeling.py</a>.
#     <br><br>
#     Note: This notebook is used for prototyping, the production training run was executed via the reproducible script <code><a href="../scripts/train_baseline.py">scripts/train_baseline.py</a></code>.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Train and evaluate each baseline model and store model results.
# </div> 

# %%
# Build baseline models (using helper function from "src/modeling.py")
baseline_models = get_baseline_models()

# Train and evaluate linear regression model (example usage of train_and_evaluate) 
# lr_results = train_and_evaluate(baseline_models["Linear Regression"], X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val)
# lr_metrics = pd.DataFrame([lr_results])[["mdae", "mae", "r2", "training_time"]]
# display(lr_metrics.rename(columns=METRIC_LABELS).style.pipe(add_table_caption, "Linear Regression: Metrics").format("{:.2f}").hide()) 


def train_and_evaluate_all_models(models, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
    """
    Train and evaluate multiple models and consolidate their results.

    Args:
        models (dict): A dictionary mapping model names (str) to model objects (estimators).
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Target variable for training data.
        X_val (pd.DataFrame): Preprocessed validation features.
        y_val (pd.Series): Target variable for validation data.
        w_train (pd.Series, optional): Sample weights for training data. Defaults to None.
        w_val (pd.Series, optional): Sample weights for validation data. Defaults to None.

    Returns:
        dict: A dictionary of evaluation results for each model, where keys are model names and
              values are the dictionaries returned by the `train_and_evaluate` function.
    """
    print("Training and evaluating baseline models...")    
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        result = train_and_evaluate(model, X_train, y_train, X_val, y_val, w_train, w_val)
        results[model_name] = result
        print(f"  {model_name} trained in {result['training_time']:.2f} sec (MdAE: {result['val_mdae']:.2f})")        
    return results


def persist_all_models(model_results):
    """
    Save baseline model results in various files:
      1.  Saves each fitted model object individually as a .joblib file.
      2.  Aggregates all performance metrics into a single JSON file.
      3.  Saves predictions of all models on the validation data into a single .joblib file.
    Args:
        model_results (dict): A nested dictionary mapping model names to results 
            dictionaries (containing 'fitted_model', training and validation metrics 
            (e.g. 'val_mdae', 'train_mdae'), and 'y_val_pred').
    """
    print("Persisting baseline models...")
    all_metrics = {}
    all_predictions = {}
    for model_name, result in model_results.items():        
        # Save fitted model as .joblib file 
        model_id = model_name.lower().replace(" ", "_")
        model_path = f"models/{model_id}_baseline.joblib"
        save_model(result["fitted_model"], model_path, verbose=False)
        print(f"  Saved fitted {model_name} model to '{model_path}'")
        
        # Collect evaluation metrics of all models in single dictionary
        all_metrics[model_name] = {
            "val_mdae": result["val_mdae"],
            "val_mae": result["val_mae"],
            "val_r2": result["val_r2"],
            "train_mdae": result["train_mdae"],
            "train_mae": result["train_mae"],
            "train_r2": result["train_r2"]
        }
        
        # Collect predicted values of all models in single dictionary
        all_predictions[model_name] = result["y_val_pred"]

    # Save evaluation metrics as JSON 
    save_metrics(all_metrics, "models/baseline_metrics.json", verbose=False)
    print(f"  Saved model evaluation metrics to 'models/baseline_metrics.json'")
    
    # Save predictions as .joblib file 
    save_model(all_predictions, "models/baseline_predictions.joblib", verbose=False)
    print(f"  Saved predicted values of all baseline models to 'models/baseline_predictions.joblib'")


# Train and evaluate baseline models
# baseline_results = train_and_evaluate_all_models(baseline_models, X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val)

# Save baseline model results
# persist_all_models(baseline_results)


# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Evaluation</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     🎯 Evaluate model performance on the validation dataset.  
#     <ul>
#         <li>Primary Metric:
#             <ul>
#                 <li>Median Absolute Error (Target: MdAE < $500)</li>
#             </ul>
#         </li>
#         <li>Secondary Metrics:
#             <ul>
#                 <li>Mean Absolute Error (MAE)</li>
#                 <li>Coefficient of Determination (R²)</li>
#             </ul>     
#         </li>
#         <li>Additional Diagnostics:
#             <ul>
#                 <li>Metrics Comparison Tables</li>
#                 <li>Overfitting Analysis</li>
#             </ul>
#         </li>
#     </ul>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Metric Comparison Table</strong><br>
#     📌 Compare evaluation metrics of all baseline models on the validation data.
# </div> 

# %%
# Load baseline model metrics from JSON files 
baseline_models_to_evaluate = ["median", "lr", "en", "tree", "rf", "xgb", "svm"]
baseline_metrics = {}
for model in baseline_models_to_evaluate:
    metrics = load_metrics(f"../models/{model}_baseline_metrics.json")
    baseline_metrics.update(metrics)

# Display metric comparison table
display(
    pd.DataFrame(baseline_metrics).T[["val_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x).replace(" (Baseline)", ""))
    .style
    .pipe(add_table_caption, "Baseline Model Metrics (Validation Data)")
    .format("{:.2f}")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <strong>Note on Negative $r^2$ Scores</strong>
#     <br>
#     Indicates the model performs worse than always predicting the mean. This is common in medical cost prediction for several reasons:
#     <ul>
#         <li><strong>Sensitivity to Outliers</strong>: $R^2$ uses squared errors. Since medical costs are extremely heavy-tailed, even a few large mispredictions on high-cost individuals can cause the squared error to explode.</li>
#         <li><strong>Log-Transformation</strong>: While log-transforming handles skewness, small errors in "log-space" become exponential errors when converted back to raw dollars.</li>
#         <li><strong>Sample Weights</strong>: Weighted $R^2$ penalizes errors more heavily on observations that represent larger portions of the US population.</li>
#     </ul>
#     Observation: The relatively small MdAE (~\$250) vs. large MAE (~\$1000) confirms that the baseline models predict typical costs well, but fail on high-cost outliers.
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Overfitting Analysis</strong><br>
#     📌 Compare training vs. validation MdAE (primary metric) to identify overfitting.
# </div> 
# %%
# Extract train MdAE, val MdAE, and calculate difference
overfitting_data = []
for model_name, metrics in baseline_metrics.items():
        overfitting_data.append({
            "Model": MODEL_DISPLAY_LABELS.get(model_name, model_name).replace(" (Baseline)", ""),
            "MdAE (Val)": metrics["val_mdae"],
            "MdAE (Train)": metrics["train_mdae"],
            "Delta": metrics["val_mdae"] - metrics["train_mdae"],
            "Delta %": ((metrics["val_mdae"] - metrics["train_mdae"]) / metrics["train_mdae"]) * 100
        })

# Display overfitting table
display(
    pd.DataFrame(overfitting_data)
    .style
    .hide()
    .set_properties(subset=["Model"], **{"font-weight": "bold"})
    .pipe(add_table_caption, "Baseline Models: Overfitting Analysis")
    .format({"MdAE (Train)": "{:.2f}", "MdAE (Val)": "{:.2f}", "Delta": "{:.2f}", "Delta %": "{:+.1f}%"})
)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Log-Scale Metric Comparison Table</strong><br>
#     📌 Recalculate metrics in log-space to diagnose model learning without the outlier error "explosion" effect on the raw dollar scale.
# </div> 

# %%
# Log-transform true values
y_val_log = np.log1p(y_val)

# Map model names to display labels
model_name_map = {
    "median": "Median Prediction",
    "lr": "Linear Regression",
    "en": "Elastic Net",
    "tree": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "svm": "Support Vector Machine"
}

# Evaluate all baseline models on log-scale
baseline_models_to_evaluate = ["median", "lr", "en", "tree", "rf", "xgb", "svm"]
log_metrics = {}
for model in baseline_models_to_evaluate:
    # Load predicted values from .joblib file (use load_model for binary files)
    y_val_pred = load_model(f"../models/{model}_baseline_predictions.joblib", verbose=False)
    
    # Log-transform predictions (they were inverse-transformed to dollars by TransformedTargetRegressor)
    y_val_pred_log = np.log1p(y_val_pred)

    # Calculate weighted metrics in log-space
    log_metrics[model] = {
        "MdAE (Log)": weighted_median_absolute_error(y_val_log, y_val_pred_log, sample_weight=w_val),
        "MAE (Log)": mean_absolute_error(y_val_log, y_val_pred_log, sample_weight=w_val),
        "R² (Log)": r2_score(y_val_log, y_val_pred_log, sample_weight=w_val)
    }

# Display log-scale comparison table
display(
    pd.DataFrame(log_metrics).T
    .rename(index=model_name_map)
    .style
    .pipe(add_table_caption, "Baseline Model Metrics (Log-Scale)")
    .format("{:.2f}")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px; margin-bottom:16px;">
#     💡 <strong>Insights:</strong>
#     <ul style="margin-top:8px; margin-bottom:8px">
#         <li><strong>The Log-Scale "North Star":</strong> While R² on the raw dollar scale is near zero (or negative), the <b>Log-Scale R² is ~0.30</b> across all top models. This confirms the features have strong predictive signal for healthcare costs and that the negative raw R² is simply a scaling artifact caused by rare high-cost "black swan" events.</li>
#         <li><strong>MdAE Priority:</strong> For our typical app user, <b>MdAE is the most meaningful success metric</b>. The data confirms that predicting the "typical experience" is statistically distinct from predicting the catastrophic extreme costs.</li>
#         <li><strong>MdAE vs. R² Trade-off:</strong> Elastic Net has the <b>best MdAE (163)</b> but a weak Log R² (0.09), while XGBoost has the <b>best Log R² (0.30)</b> but a higher MdAE (281). Elastic Net's polynomial features concentrate predictions tightly around the median, excelling for the typical user but compressing the prediction range. XGBoost captures more of the full cost structure but hasn't been optimized for median accuracy yet — a gap that tuning can close.</li>
#         <li><strong>Median Prediction Sanity Check:</strong> The naive "always predict the median" baseline achieves MdAE = 248. Notably, <b>XGBoost (281), Decision Tree (271), and SVM (291) perform worse than this naive baseline on MdAE</b> despite having strong log-scale signal. Their <code>absolute_error</code> objectives minimize mean errors, not median errors — tuning should address this misalignment.</li>
#         <li><strong>Overfitting Paradox:</strong> XGBoost and SVM show extreme overfitting (+98% to +191% MdAE gap). Their aggressive "memorization" of training data (Train MdAE < 142) fails to generalize, confirming they require heavy regularization to handle the noisy nature of healthcare costs.</li>
#         <li><strong>Stability of Regularized Linear Models:</strong> Elastic Net's low overfitting (+6.6%) combined with its top-tier validation MdAE (163) suggests that L1/L2 regularization is highly effective at denoising medical feature sets, often outperforming complex non-linear models that haven't been properly constrained.</li>
#     </ul>
#     <hr style="height: 1px; border: none; background-color: #e0f0e0; margin: 12px 0;">
#     🎯 <strong>Selected Models for Hyperparameter Tuning:</strong>
#     <ol style="margin-top:8px; margin-bottom:0px">
#         <li><strong>Elastic Net:</strong> Current MdAE champion (163). Its polynomial features and L1/L2 regularization handle correlated medical inputs well. Tuning goal: Improve tail accuracy (R²) without sacrificing MdAE leadership and maintain low overfitting (+6.6% delta).</li>
#         <li><strong>XGBoost:</strong> Strong predictive signal (Log R² = 0.30, best Log MAE = 1.89). Its gradient boosting captures non-linear health interactions that simpler models miss. Tuning goal: Close the massive +98% overfitting gap through aggressive regularization to translate its high Log R² into robust raw-dollar predictions.</li>
#         <li><strong>Random Forest:</strong> Best raw MAE (958) and tied-best Log R² (0.30). A stable ensemble learner that provides an essential diversity check against XGBoost's boosting bias. Tuning goal: Push MdAE below 200 via leaf/split constraints; serves as the primary non-linear benchmark if XGBoost remains volatile.</li>
#     </ol>
#     <br>
#     <strong>Not selected:</strong> Linear Regression (dominated by Elastic Net; same family but less flexible), Decision Tree (dominated by Random Forest), SVM (worst MdAE, slow training, hardest to tune).
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">LLM Benchmark</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Note: This notebook is used for prototyping, the entire benchmarking run was executed via the reproducible script <code><a href="../scripts/benchmark_llm.py">scripts/benchmark_llm.py</a></code>.
# </div>

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Setup 
# </div>

# %%
# Standard library imports
import os
import re
import sys
import json

# Third-party imports
from google import genai
from pydantic import BaseModel, Field
from typing import Annotated
from dotenv import load_dotenv

# Local imports
from src.constants import (
    RAW_COLUMNS_TO_KEEP, RAW_BINARY_FEATURES,
    MEPS_MISSING_CODES,
    MARRY31X_TRANSITION_CODES, EMPST31_TRANSITION_CODES,
    MARRY31X_COLLAPSE_MAP, EMPST31_COLLAPSE_MAP,
)

# Load environment variables from .env file
load_dotenv()

# Configuration
LLM_MODEL = "gemini-3-flash-preview"  # "gemini-3.1-flash-lite-preview" | "gemma-4-31b-it"
LLM_TEMPERATURE = 0          # Almost deterministic model outputs (except for tiny variations due to floating-point math)
LLM_THINKING_LEVEL = "high"  # Reasoning depth 
BATCH_SIZE = 25              # User profiles per API call (fits well within context window)
DELAY_SECONDS = 4            # Seconds between API calls to stay within free-tier limit (5 RPM for gemini-3-flash)
MAX_ATTEMPTS = 5             # Maximum times to try API call before giving up

# Paths (relative to /notebooks directory)
RAW_DATA_PATH = "../data/h251.sas7bdat"
TRAIN_PREPROCESSOR_INPUT_DATA_PATH = "../data/training_data_preprocessor_input.parquet"
VAL_PREPROCESSOR_INPUT_DATA_PATH = "../data/validation_data_preprocessor_input.parquet"
TEST_PREPROCESSOR_INPUT_DATA_PATH = "../data/test_data_preprocessor_input.parquet"
TRAIN_MODEL_READY_DATA_PATH = "../data/training_data_model_ready.parquet"
VAL_MODEL_READY_DATA_PATH = "../data/validation_data_model_ready.parquet"
TEST_MODEL_READY_DATA_PATH = "../data/test_data_model_ready.parquet"

# Human-Readable Label Maps
SEX_LABELS = {1: "Male", 0: "Female"}
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
MARITAL_LABELS = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married"}
INCOME_LABELS = {1: "Poor/Negative", 2: "Near Poor", 3: "Low Income", 4: "Middle Income", 5: "High Income"}
EDUCATION_LABELS = {1: "No Degree", 2: "GED", 3: "High School Diploma", 4: "Bachelor's Degree", 5: "Master's Degree", 6: "Doctorate", 7: "Other Degree"}
INSURANCE_LABELS = {1: "Private Insurance", 2: "Public Insurance Only (Medicare/Medicaid)", 3: "Uninsured"}
EMPLOYMENT_LABELS = {1: "Employed", 0: "Not Employed"}
HEALTH_SCALE = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
YES_NO = {1: "Yes", 0: "No"}

CHRONIC_CONDITIONS = {
    "HIBPDX": "High Blood Pressure",
    "CHOLDX": "High Cholesterol",
    "DIABDX_M18": "Diabetes",
    "CHDDX": "Coronary Heart Disease",
    "STRKDX": "Stroke",
    "CANCERDX": "Cancer",
    "ARTHDX": "Arthritis",
    "ASTHDX": "Asthma",
}

FUNCTIONAL_LIMITATIONS = {
    "ADLHLP31": "Needs help with personal care (bathing, dressing)",
    "IADLHP31": "Needs help with daily tasks (bills, medications, shopping)",
    "WLKLIM31": "Difficulty walking or climbing stairs",
    "COGLIM31": "Difficulty concentrating, remembering, or making decisions",
    "JTPAIN31_M18": "Joint pain, aching, or stiffness",
}


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Define structured output schema.
# </div> 

# %%
class PredictionBatch(BaseModel):
    """
    Schema for a batch of LLM cost predictions.
    Annotated with Field(ge=0) to ensure costs are never negative.
    """
    costs: list[Annotated[float, Field(ge=0)]]


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Prepare raw MEPS data for LLM input.
# </div> 

# %%
def prepare_human_readable_split_data(split_data_path=VAL_MODEL_READY_DATA_PATH, split_label="validation"):
    """
    Recover human-readable feature values for a preprocessed split (like val or test).

    The saved parquet contains scaled/encoded features (after StandardScaler and
    OneHotEncoder). This function reloads the raw MEPS SAS file, applies the same
    cleaning steps 1-7 as preprocess.py (but NOT the sklearn pipeline), then filters
    to only the requested split rows by matching DUPERSID indices. It manually
    includes 'Race' for fairness audit while ensuring it remains excluded from model
    training and LLM benchmarking.

    Args:
        split_data_path (str): Path to the preprocessed split parquet file.
        split_label (str): Human-readable split name for progress messages.

    Returns:
        tuple: (df_raw_split, y_split, w_split) where df_raw_split has human-readable
               feature values, y_split is the target, and w_split are sample weights.
               All aligned by DUPERSID index in parquet row order.
    """
    # Load preprocessed split data to get row IDs, target, and weights
    df_split = pd.read_parquet(split_data_path)
    split_ids = set(df_split.index.astype(str))
    y_split = df_split[TARGET_COLUMN]
    w_split = df_split[WEIGHT_COLUMN]

    # --- Data Preparation (mirrors preprocess.py steps 1-7) ---
    # Step 1: Load raw MEPS data
    print("  Loading raw MEPS SAS data...")
    df = pd.read_sas(RAW_DATA_PATH, format="sas7bdat", encoding="latin1")

    # Step 2: Variable selection
    print("  Selecting variables...")
    # Manually include 'Race' for fairness monitoring only; strictly excluded from model training and LLM benchmarking.
    columns_to_load = list(set(RAW_COLUMNS_TO_KEEP + ["RACETHX"]))
    df = df[columns_to_load]

    # Step 3: Population filtering (adults with positive weights)
    print("  Filtering target population...")
    df = df[(df[WEIGHT_COLUMN] > 0) & (df["AGE23X"] >= 18)].copy()

    # Step 4: Data type handling
    print("  Handling data types...")
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df.set_index(ID_COLUMN, inplace=True)

    # Step 5: Missing value standardization
    print("  Standardizing missing values...")
    # Recover implied values from survey skip patterns
    df.loc[df["ADSMOK42"] == -1, "ADSMOK42"] = 2    # -1 "Never Smoker" → 2 "No"
    df.loc[(df["JTPAIN31_M18"] == -1) & (df["ARTHDX"] == 1), "JTPAIN31_M18"] = 1
    # Convert remaining MEPS codes to NaN
    df.replace(MEPS_MISSING_CODES, np.nan, inplace=True)

    # Step 6: Binary standardization (MEPS 1/2 → 1/0)
    print("  Standardizing binary features...")
    df[RAW_BINARY_FEATURES] = df[RAW_BINARY_FEATURES].replace({2: 0})

    # Step 7: Feature engineering (stateless)
    print("  Engineering stateless features...")
    df["RECENT_LIFE_TRANSITION"] = (
        df["MARRY31X"].isin(MARRY31X_TRANSITION_CODES) | df["EMPST31"].isin(EMPST31_TRANSITION_CODES)
    ).astype(float)
    df.loc[df["MARRY31X"].isna() & df["EMPST31"].isna(), "RECENT_LIFE_TRANSITION"] = np.nan
    df["MARRY31X_GRP"] = df["MARRY31X"].replace(MARRY31X_COLLAPSE_MAP)
    df["EMPST31_GRP"] = df["EMPST31"].replace(EMPST31_COLLAPSE_MAP)

    # Filter to requested split rows and align to preprocessed data row order
    print(f"  Filtering rows to match preprocessed {split_label} data...")
    df_raw_split = df.loc[df.index.isin(split_ids)].reindex(y_split.index)
    n_matched = df_raw_split.index.isin(split_ids).sum()
    n_complete = df_raw_split.notna().all(axis=1).sum()
    print(f"  Matched {n_matched:,} of {len(split_ids):,} rows of the preprocessed {split_label} data ({n_complete:,} complete, {n_matched - n_complete:,} with missing values)")

    return df_raw_split, y_split, w_split


# Example usage: Prepare validation data for LLM benchmarking
# df_raw_val, y_val, w_val = prepare_human_readable_split_data(VAL_MODEL_READY_DATA_PATH, "validation")

# Align all arrays by common indices
# common_ids = df_raw_val.dropna(how="all").index.intersection(y_val.index)
# df_raw_val = df_raw_val.loc[common_ids]
# y_val = y_val.loc[common_ids]
# w_val = w_val.loc[common_ids]
# df_raw_val.head()


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Create natural language profiles for LLM input.
# </div> 

# %%
def row_to_profile(row):
    """
    Convert a single row of cleaned (pre-pipeline) data to a natural language profile
    that we feed as input to the LLM. Profiles use a bulleted list of explicit 
    feature names with corresponding values to maximize clarity during batch inference.

    Missing values (NaN) are intentionally omitted from the profile rather than
    imputed. This simulates a real-world "just ask an LLM" scenario where a user
    would simply not mention information they don't know or don't want to provide.
    This establishes a fair benchmark for the LLM's performance on natural,
    unstructured input compared to the app's structured and imputed results.
    """
    lines = []

    # --- Demographics ---
    if pd.notna(row.get("AGE23X")):
        lines.append(f"- Age: {int(row['AGE23X'])}")
    if pd.notna(row.get("SEX")):
        lines.append(f"- Sex: {SEX_LABELS.get(int(row['SEX']), 'Unknown')}")
    if pd.notna(row.get("REGION23")):
        lines.append(f"- U.S. Region: {REGION_LABELS.get(int(row['REGION23']), 'Unknown')}")
    if pd.notna(row.get("MARRY31X_GRP")):
        lines.append(f"- Marital Status: {MARITAL_LABELS.get(int(row['MARRY31X_GRP']), 'Unknown')}")
    if pd.notna(row.get("FAMSZE23")):
        lines.append(f"- Family Size: {int(row['FAMSZE23'])}")

    # --- Socioeconomic ---
    if pd.notna(row.get("POVCAT23")):
        lines.append(f"- Family Income: {INCOME_LABELS.get(int(row['POVCAT23']), 'Unknown')}")
    if pd.notna(row.get("HIDEG")):
        lines.append(f"- Education: {EDUCATION_LABELS.get(int(row['HIDEG']), 'Unknown')}")
    if pd.notna(row.get("EMPST31_GRP")):
        lines.append(f"- Employment: {EMPLOYMENT_LABELS.get(int(row['EMPST31_GRP']), 'Unknown')}")

    # --- Insurance & Access ---
    if pd.notna(row.get("INSCOV23")):
        lines.append(f"- Insurance: {INSURANCE_LABELS.get(int(row['INSCOV23']), 'Unknown')}")
    if pd.notna(row.get("HAVEUS42")):
        lines.append(f"- Has Usual Source of Healthcare: {YES_NO.get(int(row['HAVEUS42']), 'Unknown')}")

    # --- Health & Lifestyle ---
    if pd.notna(row.get("RTHLTH31")):
        lines.append(f"- Self-Rated Physical Health: {HEALTH_SCALE.get(int(row['RTHLTH31']), 'Unknown')}")
    if pd.notna(row.get("MNHLTH31")):
        lines.append(f"- Self-Rated Mental Health: {HEALTH_SCALE.get(int(row['MNHLTH31']), 'Unknown')}")
    if pd.notna(row.get("ADSMOK42")):
        lines.append(f"- Current Smoker: {YES_NO.get(int(row['ADSMOK42']), 'Unknown')}")

    # --- Chronic Conditions (list only diagnosed) ---
    conditions = [
        label for var, label in CHRONIC_CONDITIONS.items()
        if pd.notna(row.get(var)) and int(row[var]) == 1
    ]
    lines.append(f"- Diagnosed Chronic Conditions: {', '.join(conditions) if conditions else 'None'}")

    # --- Functional Limitations (list only present) ---
    limitations = [
        label for var, label in FUNCTIONAL_LIMITATIONS.items()
        if pd.notna(row.get(var)) and int(row[var]) == 1
    ]
    lines.append(f"- Functional Limitations: {', '.join(limitations) if limitations else 'None'}")

    return "\n".join(lines)

    
# Example usage: Create natural language profiles for LLM input
# profiles = [row_to_profile(row) for _, row in df_raw_val.head(5).iterrows()]
# print(profiles[0])

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Build LLM prompt with prompt-batching.
# </div> 

# %%
# System Prompt
# Ensures LLM and the domain-specifc ML model solve the same problem by defining costs explicitly.
# This sets a higher bar compared to real LLM chatbot usage by providing expert-level clarity in prompt.
SYSTEM_PROMPT = """\
You are a healthcare cost estimation expert for the United States.

You will be given demographic and health profiles of US adults. For each profile, \
predict their total annual out-of-pocket healthcare costs for the year 2023 in US dollars.

Out-of-pocket costs include deductibles, copays, and coinsurance for: \
office visits, prescriptions, hospital stays, ER visits, dental, vision, \
home health care, and medical equipment.
Out-of-pocket costs EXCLUDE monthly insurance premiums and over-the-counter medications.

For each profile, provide your best single-number estimate (in dollars), 
returned in the requested list format."""


def build_batch_prompt(profiles, start_idx):
    """
    Build a prompt containing multiple profiles for prompt-batching.
    
    Rationale: Bundling multiple profiles into a single request maximizes 
    throughput under RPM-constrained free tier, reduces total latency by 
    minimizing round-trips, and improves token efficiency. 
    
    Trade-offs: Large batches can suffer from "lost in the middle" effects 
    (reduced attention to middle profiles) or cross-profile information 
    leakage/anchoring (e.g., first prediction influences subsequent 
    predictions). A batch size of 25 is chosen as a "sweet spot" that 
    maintains high prediction quality and reliable JSON arrays while reducing 
    total latency and improving token efficiency.
    """
    profile_texts = []
    for i, profile in enumerate(profiles):
        profile_texts.append(f"Profile {start_idx + i + 1}:\n{profile}")

    n = len(profiles)
    return (
        f"Predict the total annual out-of-pocket healthcare costs (in 2023 US dollars) "
        f"for each of the following {n} US adults.\n\n"
        + "\n\n".join(profile_texts)
        + f"\n\nReturn the {n} estimates as an ordered array."
    )

    
# Example usage: Build a prompt containing multiple profiles for prompt-batching
# batch_prompt = build_batch_prompt(profiles[:3], start_idx=0)
# print(batch_prompt)


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Query LLM with batch-prompt in single API request.
# </div> 

# %%
def query_llm_batch(client, profiles, start_idx, batch_num):
    """Send a batch of profiles in a single prompt to the LLM API with retry logic."""
    batch_prompt = build_batch_prompt(profiles, start_idx)

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=batch_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=LLM_TEMPERATURE,
                    thinking_config=genai.types.ThinkingConfig(thinking_level=LLM_THINKING_LEVEL),                   
                    # Use structured JSON output
                    response_mime_type="application/json",
                    response_schema=PredictionBatch,
                ),
            )
            return parse_llm_response(response, len(profiles))

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_ATTEMPTS - 1:  
                wait_time = DELAY_SECONDS * (2 ** attempt)  # 20 sec after first failed attempt, 40 after 2nd, 80 after 3rd, 160 after 4th
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"    ⚠️ Rate limited (attempt {attempt + 1}/{MAX_ATTEMPTS}). Waiting {wait_time}s...")
                else:
                    print(f"    ⚠️ API error (attempt {attempt + 1}/{MAX_ATTEMPTS}): {error_msg[:120]}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Dont't wait after last attempt failed
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"    ❌ Rate limited (final attempt {MAX_ATTEMPTS}/{MAX_ATTEMPTS}).")
                else:
                    print(f"    ❌ API error (final attempt {MAX_ATTEMPTS}/{MAX_ATTEMPTS}): {error_msg[:120]}.")

    print(f"    ❌ Batch {batch_num} failed after {MAX_ATTEMPTS} attempts")
    return [np.nan] * len(profiles)

    
# Example usage: Query a single batch of profiles via LLM API
# api_key = os.environ.get("GEMINI_API_KEY")  
# client = genai.Client(api_key=api_key)
# batch_results = query_llm_batch(client, profiles[:BATCH_SIZE], start_idx=0, batch_num=1)
# client.close()
# print(batch_results)


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Parse LLM response.
# </div> 

# %%
def parse_llm_response(response, expected_count):
    """
    Extract predictions from the LLM response object.

    Handles the parsed Pydantic object if available, falling back to 
    manual string parsing if the structured output failed.

    Division of Labor:
      1. Data Integrity (Pydantic): Ensures JSON is valid, values are floats, 
         and costs are non-negative (Field ge=0). Errors here trigger a
         ValidationError caught in the try/except block.
      2. Contextual Alignment (Manual): Ensures the LLM didn't "hallucinate" 
         extra values or omit profiles. If the count mismatches, the entire 
         batch is discarded (returned as NaNs) to prevent data shifting, where 
         a single skipped profile would cause all subsequent predictions to 
         be misaligned with ground-truth labels.
    """
    try:
        # Preferred: Use the SDK's parsed field (v1.0+)
        if hasattr(response, "parsed") and response.parsed:
            predictions = response.parsed.costs
            if len(predictions) == expected_count:
                return predictions
            
            print(f"    ⚠️  Count mismatch in parsed output: Expected {expected_count}, got {len(predictions)}. Returning NaNs for this batch.")
            return [np.nan] * expected_count

        # Fallback: Manual parsing of raw text if structured output is missing
        text = response.text.strip()
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            predictions = json.loads(match.group())
            if isinstance(predictions, list) and len(predictions) == expected_count:
                return [float(p) for p in predictions]

    except (Exception) as e:
        # Capture specific validation/parsing errors for easier debugging
        err_msg = str(e).replace('\n', ' ')
        print(f"    ⚠️  Parse/Validation error: {err_msg[:150]}... Returning NaNs for this batch.")

    print(f"    ❌ Unparseable or mismatched response. Returning NaNs for this batch.")
    return [np.nan] * expected_count

    
# Example usage: Extract costs from an LLM response object
# costs = parse_llm_response(response, expected_count=BATCH_SIZE)
# print(costs)


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Compare performance of general intelligence LLM with specialized ML baseline models.
# </div> 

# %%
# Load LLM metrics from JSON file
llm_metrics = load_metrics("../models/llm_benchmark_metrics.json")

# Load baseline model metrics from JSON files 
baseline_models_to_evaluate = ["median", "lr", "en", "tree", "rf", "xgb", "svm"]
baseline_metrics = {}
for model in baseline_models_to_evaluate:
    metrics = load_metrics(f"../models/{model}_baseline_metrics.json")
    baseline_metrics.update(metrics)

# Combine all metrics 
all_metrics = {**llm_metrics, **baseline_metrics}

# Create DataFrame and calculate Overfitting (MdAE %Δ)
comparison_df = pd.DataFrame(all_metrics).T
comparison_df["overfitting_mdae"] = (
    (comparison_df["val_mdae"] - comparison_df["train_mdae"]) / comparison_df["train_mdae"] * 100
)

# Display metric comparison table
display(
    comparison_df[["val_mdae", "overfitting_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x).replace(" (Baseline)", ""))
    .sort_values("MdAE")
    .style
    .pipe(add_table_caption, "General LLM vs. Specialized ML Models")
    .format({
        "MdAE": "${:.2f}",
        "MAE": "${:.2f}",
        "R²": "{:.2f}",
        "Overfitting (MdAE Δ)": "{:+.1f}%"
    }, na_rep="N/A")
    .highlight_min(subset=["MdAE", "Overfitting (MdAE Δ)", "MAE"], color="#d4edda")
    .highlight_max(subset=["R²"], color="#d4edda")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul style="margin-top:8px; margin-bottom:0px">
#         <li><strong>Specialized ML Crushes General Intelligence:</strong> The best specialized model (Elastic Net, MdAE=\$163) outperforms the LLM (MdAE=\$600) by a factor of 3.7x. For the typical user, the domain-specific model is far more accurate.</li>
#         <li><strong>The LLM "Sanity Check" Failure:</strong> Notably, Gemini performs significantly worse than the naive "Median Prediction" baseline (MdAE \$600 vs. \$248). This indicates the LLM lacks a grounded statistical understanding of typical US healthcare costs, potentially overestimating based on "catastrophic" outliers.</li>
#         <li><strong>The R² Paradox:</strong> Despite poor median accuracy, the LLM achieves the best R² (0.11), while ML models are near-zero. This suggests the LLM's high-variance predictions capture the high-cost "tails" better than the conservative ML models, which prioritize the typical case (MdAE) over outlier variance (R²).</li>
#         <li><strong>Proof of Value:</strong> This benchmark justifies the entire project. Even a state-of-the-art LLM with expert instructions cannot match a model trained on the specific distribution of US medical expenditures.</li>
#     </ul>
# </div>
# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Hyperparameter Tuning</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Tune the hyperparameters of Elastic Net, Random Forest, and XGBoost using randomized search on the fixed holdout validation set. 
#     <br><br>
#     <b>Tuning Framework:</b>
#     <ul>
#         <li><b>Search Strategy:</b> Manual loop with <code>ParameterSampler</code> to avoid <code>sample_weight</code> routing issues.</li>
#         <li><b>Target Transform:</b> <code>TransformedTargetRegressor(log1p)</code> to handle skewness and optimize in log-space.</li>
#         <li><b>Sample Weights:</b> Normalized weights (mean=1.0) for training; raw survey weights for evaluation.</li>
#         <li><b>Scoring:</b> Weighted Median Absolute Error (MdAE) on raw-dollar predictions.</li>
#         <li><b>Iterations:</b> Small number (2-5) in notebook for prototyping. Scale to 50-100 in production scripts (e.g., <code>scripts/tune_random_forest.py</code>).</li>
#     </ul>
#     <b>Why not <code>RandomizedSearchCV</code>?</b><br>
#     Avoids <code>sample_weight</code> routing complexities for nested objects (<code>TransformedTargetRegressor</code>, <code>Pipeline</code>) and ensures the weighted MdAE is calculated explicitly and correctly on the validation set.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Elastic Net</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Tune <code>ElasticNet</code> hyperparameters.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Documentation:</b> Refer to the official <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html" target="_blank">ElasticNet documentation</a> for hyperparameter details.</li>
#         <li><b>Search Space:</b> Refer to <code><a href="../src/params.py">src/params.py</a></code> for rationale and parameter distributions.</li>
#         <li><b>Production Script:</b> This notebook is for prototyping; the production run was executed via <code><a href="../scripts/tune_elastic_net.py">scripts/tune_elastic_net.py</a></code>.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Define the hyperparameter search space.
# </div>

# %%
# Generate random parameter combinations
N_ITER = 5  # Small for prototyping
en_param_list = list(ParameterSampler(EN_PARAM_DISTRIBUTIONS, n_iter=N_ITER, random_state=RANDOM_STATE))

print(f"Generated {len(en_param_list)} random hyperparameter combinations")
print(f"Example: {en_param_list[0]}")


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Perform randomized search and persist model artifacts (best model weights as <code>.joblib</code>, metrics as <code>.json</code>, parameters as <code>.json</code>, predictions as <code>.joblib</code>, and full tuning history as <code>.json</code>).
# </div>

# %%
def tune_elastic_net(X_train, y_train, X_val, y_val, w_train, w_val, param_list, random_state=RANDOM_STATE):
    """
    Perform randomized search for Elastic Net hyperparameters and persist results.
    """
    n_iter = len(param_list)
    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()
    
    # Run randomized search
    print(f"Tuning Elastic Net ({n_iter} iterations)...")    
    en_tuning_metrics = []
    best_mdae = np.inf
    best_idx = -1
    
    for i, params in enumerate(param_list):
        # Build model: Elastic Net with Polynomial Features wrapped in Target Log-Transformer
        en_model = TransformedTargetRegressor(
            regressor=Pipeline([
                ("polynomials", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", ElasticNet(random_state=random_state, max_iter=2000))
            ]),
            func=np.log1p,
            inverse_func=np.expm1
        )
        # Set hyperparameters for the internal model in the pipeline
        en_model.regressor.set_params(**params)
        
        # Train with normalized sample weights
        start_time = time.time()
        en_model.fit(X_train, y_train, model__sample_weight=w_train_norm)
        training_time = time.time() - start_time
        
        # Predict on training and validation set (predictions are in raw dollars due to inverse_func)
        y_train_pred = en_model.predict(X_train)
        y_val_pred = en_model.predict(X_val)
        
        # Evaluate with raw survey weights
        train_mdae = weighted_median_absolute_error(y_train, y_train_pred, sample_weight=w_train)
        train_mae = mean_absolute_error(y_train, y_train_pred, sample_weight=w_train)
        train_r2 = r2_score(y_train, y_train_pred, sample_weight=w_train)
    
        val_mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)
        
        en_tuning_metrics.append({
            "params": params, 
            "train_mdae": train_mdae, 
            "train_mae": train_mae, 
            "train_r2": train_r2,
            "val_mdae": val_mdae, 
            "val_mae": val_mae, 
            "val_r2": val_r2,
            "training_time": training_time
        })
        
        if val_mdae < best_mdae:
            best_mdae = val_mdae
            best_idx = i

        # Progress logging 
        squares_label = "off" if params["polynomials__interaction_only"] else "on "
        print(f"  [{i+1:3d}/{n_iter}] MdAE: {val_mdae:8.2f} | alpha={params['model__alpha']:.4f}, l1={params['model__l1_ratio']:.2f}, squares={squares_label:3} | fit: {training_time:5.1f} s")
    
    # Retrain best model 
    print("\nRetraining best model...")
    best_params = param_list[best_idx]
    best_en_model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("polynomials", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", ElasticNet(random_state=random_state, max_iter=5000))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )
    best_en_model.regressor.set_params(**best_params)
    
    best_en_result = train_and_evaluate(
        best_en_model, 
        X_train, y_train, 
        X_val, y_val, 
        w_train, w_val
    )
    print(f"  Best Tuned Elastic Net →  MdAE: {best_en_result['val_mdae']:.2f} | MAE: {best_en_result['val_mae']:.2f} | "
          f"R²: {best_en_result['val_r2']:.4f} | Training Time: {best_en_result['training_time']:.2f}s")

    # Persist results
    print("Step 6: Persisting hyperparameter tuning results...")

    save_metrics(en_tuning_metrics, "../models/en_tuning_history.json", verbose=False)
    print("  Saved tuned Elastic Net history to 'models/en_tuning_history.json'")
    
    save_model(best_en_result["fitted_model"], "../models/en_tuned_model.joblib", verbose=False)
    print("  Saved best model to 'models/en_tuned_model.joblib'")
    
    save_metrics({ "Elastic Net (Tuned)": {
        "val_mdae": best_en_result["val_mdae"],
        "val_mae": best_en_result["val_mae"],
        "val_r2": best_en_result["val_r2"],
        "train_mdae": best_en_result["train_mdae"],
        "train_mae": best_en_result["train_mae"],
        "train_r2": best_en_result["train_r2"],
        "training_time": best_en_result["training_time"]
    }}, "../models/en_tuned_metrics.json", verbose=False)
    print("  Saved evaluation metrics of best model to 'models/en_tuned_metrics.json'")
    
    save_metrics(get_core_model_params(best_en_result["fitted_model"]), "../models/en_tuned_params.json", verbose=False)
    print("  Saved hyperparameters of best model to 'models/en_tuned_params.json'")
    
    save_model(best_en_result["y_val_pred"], "../models/en_tuned_predictions.joblib", verbose=False)
    print("  Saved predicted values of best model to 'models/en_tuned_predictions.joblib'")
    
    print("\n✅ Elastic Net hyperparameter tuning complete.")
    
    return en_tuning_metrics, best_en_result


# Run tuning
# en_tuning_metrics, best_en_results = tune_elastic_net(X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val, en_param_list)


# %%
# Load tuned Elastic Net metrics from JSON file
en_tuning_history = load_metrics("../models/en_tuning_history.json")

# Display metric comparison table  
en_tuning_df = pd.DataFrame(en_tuning_history)
en_tuning_df = en_tuning_df.sort_values("val_mdae")  # Sorts by primary metric
en_params_df = pd.json_normalize(en_tuning_df["params"])
en_display_df = pd.concat([en_tuning_df[["val_mdae", "val_mae", "val_r2"]], en_params_df], axis=1) 

display(
    en_display_df
    .style
    .pipe(add_table_caption, "Elastic Net: Hyperparameter Tuning Results")
    .format({"val_mdae": "{:.2f}", "val_mae": "{:.2f}", "val_r2": "{:.4f}", "model__alpha": "{:.4f}", "model__l1_ratio": "{:.2f}"})
    .highlight_min(subset=["val_mdae", "val_mae"], color="#d4edda")
    .highlight_max(subset=["val_r2"], color="#d4edda")
    .hide()
)

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Random Forest</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Tune <code>RandomForestRegressor</code> hyperparameters.
#     <ul style="margin-top:8px; margin-bottom:0px">
#         <li><b>Objective:</b> <code>criterion="absolute_error"</code> to minimize L1 loss on log-costs.</li>
#         <li><b>Key Params:</b> Control variance via <code>min_samples_leaf</code> and <code>max_features</code>.</li>
#         <li><b>Documentation:</b> Refer to the official scikit-learn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">RandomForestRegressor documentation</a> for hyperparameter details.</li>
#         <li><b>Search Space:</b> Refer to <code><a href="../src/params.py">src/params.py</a></code> for rationale and parameter distributions.</li>
#         <li><b>Production Script:</b> This notebook is for prototyping; the production run was executed via <code><a href="../scripts/tune_random_forest.py">scripts/tune_random_forest.py</a></code>.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Define the hyperparameter search space.
# </div>

# %%
# Generate random parameter combinations
N_ITER = 2  # Small for prototyping
rf_param_list = list(ParameterSampler(RF_PARAM_DISTRIBUTIONS, n_iter=N_ITER, random_state=RANDOM_STATE))

print(f"Generated {len(rf_param_list)} random hyperparameter combinations")
print(f"Example: {rf_param_list[0]}")


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Perform randomized search and persist model artifacts (best model weights as <code>.joblib</code>, metrics as <code>.json</code>, parameters as <code>.json</code>, predictions as <code>.joblib</code>, and full tuning history as <code>.json</code>).
# </div>

# %%
def tune_random_forest(X_train, y_train, X_val, y_val, w_train, w_val, param_list, random_state=RANDOM_STATE):
    """
    Perform randomized search for Random Forest hyperparameters and persist results.
    """
    n_iter = len(param_list)
    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()
    
    # Run randomized search
    print(f"Tuning random forest ({n_iter} iterations)...")    
    rf_tuning_metrics = []
    best_mdae = np.inf
    best_idx = -1
    
    for i, params in enumerate(param_list):
        # Build model: RandomForest wrapped in log-transform
        rf_model = TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                criterion="absolute_error",
                n_jobs=-1,
                random_state=random_state,
                **params
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        # Train with normalized sample weights
        start_time = time.time()  # To measure training time
        rf_model.fit(X_train, y_train, sample_weight=w_train_norm)
        training_time = time.time() - start_time
        
        # Predict on training and validation set (predictions are in raw dollars due to inverse_func)
        y_train_pred = rf_model.predict(X_train)
        y_val_pred = rf_model.predict(X_val)
        
        # Evaluate with raw survey weights
        train_mdae = weighted_median_absolute_error(y_train, y_train_pred, sample_weight=w_train)
        train_mae = mean_absolute_error(y_train, y_train_pred, sample_weight=w_train)
        train_r2 = r2_score(y_train, y_train_pred, sample_weight=w_train)
    
        val_mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)
        
        rf_tuning_metrics.append({
            "params": params, 
            "train_mdae": train_mdae, 
            "train_mae": train_mae, 
            "train_r2": train_r2,
            "val_mdae": val_mdae, 
            "val_mae": val_mae, 
            "val_r2": val_r2,
            "training_time": training_time
        })
        
        if val_mdae < best_mdae:
            best_mdae = val_mdae
            best_idx = i

        # Progress logging 
        print(f"  [{i+1:3d}/{n_iter}] MdAE: {val_mdae:8.2f} | trees={params['n_estimators']}, depth={params['max_depth']}, leaf={params['min_samples_leaf']}, feats={params['max_features']}, samples={params['max_samples']:.2f}, split={params['min_samples_split']} | fit: {training_time:5.1f} s")
    
    # Retrain best model 
    print("\nRetraining best model...")
    best_params = param_list[best_idx]
    best_rf_model = TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            criterion="absolute_error",
            n_jobs=-1,
            random_state=random_state,
            **best_params
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    best_rf_result = train_and_evaluate(
        best_rf_model, 
        X_train, y_train, 
        X_val, y_val, 
        w_train, w_val
    )
    print(f"  Best Tuned Random Forest →  MdAE: {best_rf_result['val_mdae']:.2f} | MAE: {best_rf_result['val_mae']:.2f} | "
          f"R²: {best_rf_result['val_r2']:.4f} | Training Time: {best_rf_result['training_time']:.2f}s")

    # Persist results
    print("Step 6: Persisting hyperparameter tuning results...")

    save_metrics(rf_tuning_metrics, "../models/rf_tuning_history.json", verbose=False)
    print("  Saved tuned random forest history to 'models/rf_tuning_history.json'")
    
    save_model(best_rf_result["fitted_model"], "../models/rf_tuned_model.joblib", verbose=False)
    print("  Saved best model to 'models/rf_tuned_model.joblib'")
    
    save_metrics({ "Random Forest (Tuned)": {
        "val_mdae": best_rf_result["val_mdae"],
        "val_mae": best_rf_result["val_mae"],
        "val_r2": best_rf_result["val_r2"],
        "train_mdae": best_rf_result["train_mdae"],
        "train_mae": best_rf_result["train_mae"],
        "train_r2": best_rf_result["train_r2"],
        "training_time": best_rf_result["training_time"]
    }}, "../models/rf_tuned_metrics.json", verbose=False)
    print("  Saved evaluation metrics of best model to 'models/rf_tuned_metrics.json'")
    
    save_metrics(get_core_model_params(best_rf_result["fitted_model"]), "../models/rf_tuned_params.json", verbose=False)
    print("  Saved hyperparameters of best model to 'models/rf_tuned_params.json'")
    
    save_model(best_rf_result["y_val_pred"], "../models/rf_tuned_predictions.joblib", verbose=False)
    print("  Saved predicted values of best model to 'models/rf_tuned_predictions.joblib'")
    
    print("\n✅ Random Forest hyperparameter tuning complete.")
    
    return rf_tuning_metrics, best_rf_result


# Run tuning
# rf_tuning_metrics, best_rf_results = tune_random_forest(X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val, rf_param_list)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate tuning results. 
# </div>

# %%
# Load tuned random forest metrics from JSON file
rf_tuning_metrics = load_metrics("../models/rf_tuning_history.json")

# Display metric comparison table  
rf_tuning_metrics_df = pd.DataFrame(rf_tuning_metrics)
rf_tuning_metrics_df = rf_tuning_metrics_df.sort_values("val_mdae")  # Sorts by primary metric
params_df = pd.json_normalize(rf_tuning_metrics_df["params"])
rf_tuning_metrics_display = pd.concat([rf_tuning_metrics_df[["val_mdae", "val_mae", "val_r2"]], params_df], axis=1) 
display(
    rf_tuning_metrics_display
    .style
    .pipe(add_table_caption, "Random Forest: Hyperparameter Tuning Results")
    .format({"val_mdae": "{:.2f}", "val_mae": "{:.2f}", "val_r2": "{:.4f}", "max_samples": "{:.2f}", "max_features": "{}"})
    .highlight_min(subset=["val_mdae", "val_mae"], color="#d4edda")
    .highlight_max(subset=["val_r2"], color="#d4edda")
    .hide()
)


# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">XGBoost</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Tune <code>XGBRegressor</code> hyperparameters.
#     <ul style="margin-top:8px; margin-bottom:0px">
#         <li><b>Objective:</b> <code>objective="reg:absoluteerror"</code> to minimize L1 loss on log-costs.</li>
#         <li><b>Speed:</b> Uses <code>tree_method="hist"</code> for efficient histogram-based splitting.</li>
#         <li><b>Documentation:</b> Refer to the official <a href="https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor" target="_blank">XGBoost documentation</a> for hyperparameter details.</li>
#         <li><b>Search Space:</b> Refer to <code><a href="../src/params.py">src/params.py</a></code> for rationale and parameter distributions.</li>
#         <li><b>Production Script:</b> This notebook is for prototyping; the production run was executed via <code><a href="../scripts/tune_xgboost.py">scripts/tune_xgboost.py</a></code>.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Define the hyperparameter search space.
# </div>

# %%
# Generate random parameter combinations
N_ITER = 5  # Small for prototyping
xgb_param_list = list(ParameterSampler(XGB_PARAM_DISTRIBUTIONS, n_iter=N_ITER, random_state=RANDOM_STATE))

print(f"Generated {len(xgb_param_list)} random hyperparameter combinations")
print(f"Example: {xgb_param_list[0]}")


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Perform randomized search and persist model artifacts (best model weights as <code>.joblib</code>, metrics as <code>.json</code>, parameters as <code>.json</code>, predictions as <code>.joblib</code>, and full tuning history as <code>.json</code>).
# </div>

# %%
def tune_xgboost(X_train, y_train, X_val, y_val, w_train, w_val, param_list, random_state=RANDOM_STATE):
    """
    Perform randomized search for XGBoost hyperparameters and persist results.
    """
    n_iter = len(param_list)
    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()
    
    # Run randomized search
    print(f"Tuning XGBoost ({n_iter} iterations)...")    
    xgb_tuning_metrics = []
    best_mdae = np.inf
    best_idx = -1
    
    for i, params in enumerate(param_list):
        # Build model: XGBoost wrapped in log-transform
        xgb_model = TransformedTargetRegressor(
            regressor=XGBRegressor(
                objective="reg:absoluteerror",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
                **params
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        # Train with normalized sample weights
        start_time = time.time()  # Measure training time
        xgb_model.fit(X_train, y_train, sample_weight=w_train_norm)
        training_time = time.time() - start_time
        
        # Predict on training and validation set (predictions are in raw dollars due to inverse_func)
        y_train_pred = xgb_model.predict(X_train)
        y_val_pred = xgb_model.predict(X_val)
        
        # Evaluate with raw survey weights
        train_mdae = weighted_median_absolute_error(y_train, y_train_pred, sample_weight=w_train)
        train_mae = mean_absolute_error(y_train, y_train_pred, sample_weight=w_train)
        train_r2 = r2_score(y_train, y_train_pred, sample_weight=w_train)
    
        val_mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        val_r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)
        
        xgb_tuning_metrics.append({
            "params": params, 
            "train_mdae": train_mdae, 
            "train_mae": train_mae, 
            "train_r2": train_r2,
            "val_mdae": val_mdae, 
            "val_mae": val_mae, 
            "val_r2": val_r2,
            "training_time": training_time
        })
        
        if val_mdae < best_mdae:
            best_mdae = val_mdae
            best_idx = i

        print(f"  [{i+1:3d}/{n_iter}] MdAE: {val_mdae:8.2f} | estimators={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']:.3f}, sub={params['subsample']:.2f}, col={params['colsample_bytree']:.2f} | training: {training_time:5.1f} s")
    
    # Retrain best model 
    print("\nRetraining best model...")
    best_params = param_list[best_idx]
    best_xgb_model = TransformedTargetRegressor(
        regressor=XGBRegressor(
            objective="reg:absoluteerror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            **best_params
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    best_xgb_results = train_and_evaluate(
        best_xgb_model, 
        X_train, y_train, 
        X_val, y_val, 
        w_train, w_val
    )
    print(f"  Best Tuned XGBoost  →  MdAE: {best_xgb_results['val_mdae']:.2f} | MAE: {best_xgb_results['val_mae']:.2f} | "
          f"R²: {best_xgb_results['val_r2']:.4f} | Training Time: {best_xgb_results['training_time']:.2f}s")

    # Persist results
    print("Step 6: Persisting hyperparameter tuning results...")

    save_metrics(xgb_tuning_metrics, "../models/xgb_tuning_history.json", verbose=False)
    print("  Saved tuned XGBoost history to 'models/xgb_tuning_history.json'")
    
    save_model(best_xgb_results["fitted_model"], "../models/xgb_tuned_model.joblib", verbose=False)
    print("  Saved best model to 'models/xgb_tuned_model.joblib'")
    
    save_metrics({ "XGBoost (Tuned)": {
        "val_mdae": best_xgb_results["val_mdae"],
        "val_mae": best_xgb_results["val_mae"],
        "val_r2": best_xgb_results["val_r2"],
        "train_mdae": best_xgb_results["train_mdae"],
        "train_mae": best_xgb_results["train_mae"],
        "train_r2": best_xgb_results["train_r2"],
        "training_time": best_xgb_results["training_time"]
    }}, "../models/xgb_tuned_metrics.json", verbose=False)
    print("  Saved evaluation metrics of best model to 'models/xgb_tuned_metrics.json'")
    
    save_metrics(get_core_model_params(best_xgb_results["fitted_model"]), "../models/xgb_tuned_params.json", verbose=False)
    print("  Saved hyperparameters of best model to 'models/xgb_tuned_params.json'")
    
    save_model(best_xgb_results["y_val_pred"], "../models/xgb_tuned_predictions.joblib", verbose=False)
    print("  Saved predicted values of best model to 'models/xgb_tuned_predictions.joblib'")
    
    print("\n✅ XGBoost hyperparameter tuning complete.")
    
    return xgb_tuning_metrics, best_xgb_results


# Run tuning
# xgb_tuning_metrics, best_xgb_results = tune_xgboost(X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val, xgb_param_list)


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate tuning results. 
# </div>

# %%
# Load tuned XGBoost metrics from JSON file
xgb_tuning_history = load_metrics("../models/xgb_tuning_history.json")

# Display metric comparison table  
xgb_tuning_df = pd.DataFrame(xgb_tuning_history)
xgb_tuning_df = xgb_tuning_df.sort_values("val_mdae")  # Sorts by primary metric
xgb_params_df = pd.json_normalize(xgb_tuning_df["params"])
xgb_display_df = pd.concat([xgb_tuning_df[["val_mdae", "val_mae", "val_r2"]], xgb_params_df], axis=1) 

display(
    xgb_display_df
    .style
    .pipe(add_table_caption, "XGBoost: Hyperparameter Tuning Results")
    .format({"val_mdae": "{:.2f}", "val_mae": "{:.2f}", "val_r2": "{:.4f}", "learning_rate": "{:.3f}", "subsample": "{:.2f}", "colsample_bytree": "{:.2f}", "reg_lambda": "{:.2f}", "reg_alpha": "{:.2f}"})
    .highlight_min(subset=["val_mdae", "val_mae"], color="#d4edda")
    .highlight_max(subset=["val_r2"], color="#d4edda")
    .hide()
)


# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Evaluation</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     🎯 Evaluate Model Performance:
#     <ul>
#         <li>Metrics Comparison Tables (MdAE, MAE, R²)</li>
#         <li>Overfitting Analysis</li>
#         <li>Heteroscedasticity (Residuals vs. Predicted)</li> 
#         <li>Stratified Error Analysis (Model Reliability & Fairness Audit)</li>
#         <li>(optionally) Feature Dependencies (Residuals vs. Features)</li> 
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Metric Comparison Tables</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Compare all baseline and tuned models.
# </div> 

# %%
# Load LLM metrics 
llm_metrics = load_metrics("../models/llm_benchmark_metrics.json", verbose=False)

# Load baseline model metrics
baseline_models_to_evaluate = ["median", "lr", "en", "tree", "rf", "xgb", "svm"]
baseline_metrics = {}
for model in baseline_models_to_evaluate:
    metrics = load_metrics(f"../models/{model}_baseline_metrics.json", verbose=False)
    baseline_metrics.update(metrics)

# Load tuned model metrics
tuned_models_to_evaluate = ["en", "rf", "xgb"]
tuned_metrics = {}
for model in tuned_models_to_evaluate:
    metrics = load_metrics(f"../models/{model}_tuned_metrics.json", verbose=False)
    tuned_metrics.update(metrics)

# Combine all metrics 
all_metrics = {**llm_metrics, **baseline_metrics, **tuned_metrics}

# Create DataFrame and calculate Overfitting (MdAE %Δ)
comparison_df = pd.DataFrame(all_metrics).T
comparison_df["overfitting_mdae"] = (
    (comparison_df["val_mdae"] - comparison_df["train_mdae"]) / comparison_df["train_mdae"] * 100
)

# Define custom order for model comparison: Benchmarks -> Baselines -> Tuned Model Pairs
custom_order = [
    "Median Prediction (Baseline)",
    f"LLM ({LLM_MODEL})",
    "Decision Tree (Baseline)",
    "Support Vector Machine (Baseline)",
    "Linear Regression (Baseline)",
    "Elastic Net (Baseline)",
    "Elastic Net (Tuned)",
    "Random Forest (Baseline)",
    "Random Forest (Tuned)",
    "XGBoost (Baseline)",
    "XGBoost (Tuned)"
]

# Display metric comparison table
display(
    comparison_df.reindex([m for m in custom_order if m in comparison_df.index])[["val_mdae", "overfitting_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x))
    .style
    .pipe(add_table_caption, "Tuned Model Metrics")
    .format({
        "MdAE": "${:.2f}",
        "Overfitting (MdAE Δ)": "{:+.1f}%",
        "MAE": "${:.2f}",
        "R²": "{:.2f}"
    }, na_rep="N/A")
    .highlight_min(subset=["MdAE", "Overfitting (MdAE Δ)", "MAE"], color="#d4edda")
    .highlight_max(subset=["R²"], color="#d4edda")
)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Compare curated finalists: Best tuned models, LLM benchmark, median benchmark, and linear regression (interpretable baseline).
# </div> 

# %%
# Define the curated list of finalists 
finalists = [
    f"LLM ({LLM_MODEL})",
    "Median Prediction (Baseline)",
    "Linear Regression (Baseline)",
    "Elastic Net (Tuned)",
    "Random Forest (Tuned)",
    "XGBoost (Tuned)"
]

# Display finalists comparison table
display(
    comparison_df.loc[[f for f in finalists if f in comparison_df.index], ["val_mdae", "overfitting_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x))
    .sort_values("MdAE")
    .reset_index().rename(columns={"index": "Model"})
    .style
    .hide()
    .set_properties(subset=["Model"], **{"font-weight": "bold"})
    .pipe(add_table_caption, "Tuned Model Metrics (Finalists)")
    .format({
        "MdAE": "${:.2f}",
        "Overfitting (MdAE Δ)": "{:+.1f}%",
        "MAE": "${:.2f}",
        "R²": "{:.2f}"
    }, na_rep="N/A")
    .highlight_min(subset=["MdAE", "Overfitting (MdAE Δ)", "MAE"], color="#d4edda")
    .highlight_max(subset=["R²"], color="#d4edda")
)

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Overfitting Analysis</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Compare training vs. validation MdAE (primary metric) to identify overfitting.
# </div> 
# %%
# Extract train MdAE, val MdAE, and calculate difference
model_families = ["Elastic Net", "Random Forest", "XGBoost"]
overfitting_data = []

for family in model_families:
    for suffix in [" (Baseline)", " (Tuned)"]:
        model_key = f"{family}{suffix}"
        if model_key in all_metrics:
            metrics = all_metrics[model_key]
            overfitting_data.append({
                "Model": MODEL_DISPLAY_LABELS.get(model_key, model_key),
                "MdAE (Val)": metrics["val_mdae"],
                "MdAE (Train)": metrics["train_mdae"],
                "Delta": metrics["val_mdae"] - metrics["train_mdae"],
                "Delta %": ((metrics["val_mdae"] - metrics["train_mdae"]) / metrics["train_mdae"]) * 100
            })

# Display overfitting table
display(
    pd.DataFrame(overfitting_data)
    .style
    .hide()  # Hides index
    .set_properties(subset=["Model"], **{"font-weight": "bold"})
    .pipe(add_table_caption, "Overfitting Analysis")
    .format({"MdAE (Train)": "{:.2f}", "MdAE (Val)": "{:.2f}", "Delta": "{:.2f}", "Delta %": "{:+.1f}%"})
    .highlight_min(subset=["Delta %"], color="#d4edda")
)

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Heteroscedasticity</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Visualize residuals vs. predicted values to assess whether prediction error variance changes with predicted cost level.
# </div>

# %%
# Load predictions of all tuned models (on validation data)
print("Loading tuned model predictions...")
tuned_model_predictions = {
    "Elastic Net (Tuned)": load_model("../models/en_tuned_predictions.joblib", verbose=False),
    "Random Forest (Tuned)": load_model("../models/rf_tuned_predictions.joblib", verbose=False),
    "XGBoost (Tuned)": load_model("../models/xgb_tuned_predictions.joblib", verbose=False)
}

# Display predicted cost ranges by model
pred_ranges = []
for model_key, y_pred in tuned_model_predictions.items():
    pred_ranges.append({
        "Model": MODEL_DISPLAY_LABELS.get(model_key, model_key),
        "Min Prediction": np.min(y_pred),
        "Max Prediction": np.max(y_pred)
    })

pred_ranges_df = pd.DataFrame(pred_ranges).set_index("Model")
display(pred_ranges_df.style.format("${:,.2f}").pipe(add_table_caption, "Predicted Cost Range"))


# %%
def plot_residuals_vs_predicted(y_true, predictions_dict, weights, n_bins=15, n_cols=2, save_to_file=None):
    """
    Plots residuals vs. predicted values scatter plots for multiple models 
    with binned median trend and IQR bands.
    
    Creates a facet grid plot (N x n_cols) showing residuals against 
    predicted values. Overlays binned median (robust trend) and 
    interquartile range bands to visualize heteroscedasticity.
    
    Args:
        y_true (pd.Series): Actual target values.
        predictions_dict (dict): {model_name: y_pred_array} for each model.
        weights (pd.Series): Sample weights for weighted statistics.
        n_bins (int): Number of bins for the trend line.
        n_cols (int): Number of columns in the facet grid plot.
        save_to_file (str, optional): Full path to save the plot.
    """
    n_models = len(predictions_dict)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for i, (model_key, y_pred) in enumerate(predictions_dict.items()):
        ax = axes_flat[i]
        model_label = MODEL_DISPLAY_LABELS.get(model_key, model_key)
        
        # Standardize core data as numpy arrays
        y_pred = np.array(y_pred)
        residuals = np.array(y_true) - y_pred
        w = np.array(weights)
        
        # Scale scatter points by weights
        s_weights = 1 + (w / w.max()) * 39  # reasonable range between 1 and 40

        # Scatter plot 
        ax.scatter(
            y_pred, residuals, 
            alpha=0.25, 
            color=SAMPLE_COLOR,  # data points represent survey respondents (sample)
            s=s_weights,         # larger points represents more people in the population
            edgecolors="none", 
            rasterized=True  
        )
        
        # Reference line at 0
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        
        # Binned median trend line and IQR bands 
        # Each bin represents ~6.7% of the population which is ~100 respondents in sample (validation set size n=1477)
        bin_probs = np.linspace(0, 0.99, n_bins + 1)
        bin_edges = weighted_quantile(y_pred, w, bin_probs)  # Uses weighted quantiles to ensure each bin represents population 
        
        bin_centers = []
        bin_medians = []
        bin_q25 = []
        bin_q75 = []
        
        for b in range(n_bins):
            mask = (y_pred >= bin_edges[b]) & (y_pred < bin_edges[b + 1])
            if mask.sum() >= 10:  # Require minimum sample size for stable statistics
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                bin_residuals = residuals[mask]
                bin_weights = w[mask]
                
                # Use weighted quantiles for binned statistics
                bin_medians.append(weighted_quantile(bin_residuals, bin_weights, 0.5))
                bin_q25.append(weighted_quantile(bin_residuals, bin_weights, 0.25))
                bin_q75.append(weighted_quantile(bin_residuals, bin_weights, 0.75))
        
        # Plot trend line and range bands
        ax.plot(bin_centers, bin_medians, color=POP_COLOR, linewidth=2, label="Median Residual")
        ax.fill_between(bin_centers, bin_q25, bin_q75, alpha=0.15, color=POP_COLOR, label="IQR (25th–75th)")
        
        # Formatting
        ax.set_title(model_label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Cost")
        
        # Predictions and residuals axis limits for "zoomed-in" view (ignoring extreme outliers)
        pred_max = weighted_quantile(y_pred, w, 0.99)
        res_max = weighted_quantile(residuals, w, 0.95)
        res_min = weighted_quantile(residuals, w, 0.01)

        ax.set_xlim(0, pred_max * 1.01)
        ax.set_ylim(res_min * 1.5, res_max * 1.3)
        
        # Format ticks: -$500 instead of $-500
        currency_fmt = plt.FuncFormatter(lambda x, _: f"{'-' if x < 0 else ''}${abs(x):,.0f}")
        ax.xaxis.set_major_formatter(currency_fmt)
        
        # Only show y-label on the first column of each row
        if i % n_cols == 0:
            ax.set_ylabel("Residual (Actual − Predicted)")
            ax.yaxis.set_major_formatter(currency_fmt)
            ax.legend(loc="upper left", fontsize=9, frameon=True)
            
        sns.despine(ax=ax)
    
    # Remove empty subplots if n_models < n_rows * n_cols
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes_flat[j])
    
    fig.suptitle("Heteroscedasticity: Residuals vs. Predicted", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2)
    
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)
    
    plt.show()


# Plot heteroscedasticity 
plot_residuals_vs_predicted(
    y_val, 
    tuned_model_predictions, 
    w_val,
    save_to_file="../figures/evaluation/heteroscedasticity.png"
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Errors grow with predicted costs ("fan shape"):</strong> All three models show wider error spread as predictions increase. This is expected, because expensive medical events (surgeries, hospitalizations) are rare and hard to predict from survey data alone.</li>
#         <li><strong>Errors are mostly underestimates:</strong> Residuals skew heavily upward, which means the model underpredicts actual costs. A few individuals incur extreme costs that no model can fully anticipate, so the models miss on the high side far more often than they overpredict.</li>
#         <li><strong>Elastic Net can't separate low vs. high costs:</strong> With a max prediction of only \$217, Elastic Net compresses nearly everyone into a narrow "low-cost" band. Its median residual climbs steeply, meaning the higher it tries to predict, the more it underestimates. It effectively treats the entire population as low-risk.</li>
#         <li><strong>XGBoost differentiates best:</strong> XGBoost predictions span up to \$2,114, roughly 10× the range of Elastic Net and 1.7× Random Forest. A wider prediction range means the model better separates low-cost from high-cost individuals.</li>
#         <li><strong>Tree models are less biased:</strong> Both RF and XGBoost keep their median residual near zero across most of their prediction range, indicating less systematic bias. Elastic Net's median drifts sharply upward, confirming it systematically underpredicts.</li>
#         <li><strong>Mid-range uncertainty peak ("inverted-U"):</strong> RF and XGBoost share a distinctive pattern: the IQR band widens through mid-range predictions, then narrows again at the highest predictions. This suggests that when tree models confidently predict high costs, those predictions are relatively well-calibrated.</li>
#     </ul>
#     <hr style="border: 0; border-top: 1px solid #e0f0e0; margin: 15px 0;">
#     <strong>Decision: Select XGBoost for Production</strong> <br>
#     Elastic Net is the best model if the product only needs one middle estimate. It has the lowest tuned validation MdAE, so it is the point-estimate champion. However, the production app is not a single-number estimator. It is a planning tool that needs a plan-around estimate, a typical range, and a safety cushion. For that product goal, XGBoost is the better production choice.
#     <ul style="margin-top:8px">
#         <li><b>Point estimate vs. product usefulness:</b> Elastic Net wins MdAE by keeping most predictions close to the low-cost middle of the population. That lowers the typical error, but it also compresses predictions into a narrow band (max prediction: \$217). This makes it hard for the app to distinguish someone with low expected costs from someone with meaningfully higher financial risk.</li>
#         <li><b>XGBoost separates risk levels better:</b> XGBoost predictions span up to \$2,114, roughly 10× wider than Elastic Net. This wider spread is useful because the app needs to tell users not only "what is typical?" but also "how much cushion should I consider?"</li>
#         <li><b>Quantile regression fit:</b> XGBoost supports multi-quantile regression, so one model can produce the plan-around estimate (median/q50), the typical range (q25-q75), and the safety cushion (q90). Elastic Net would need a more complicated setup and still show weaker ability to separate low-risk and high-risk profiles.</li>
#         <li><b>Trade-off:</b> Accept a higher median error from XGBoost than Elastic Net because the production app values calibrated ranges and risk separation, not just the lowest possible single-number MdAE. Elastic Net remains a useful benchmark and challenger model for median/q50 accuracy.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Stratified Error Analysis</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Compare model performance across different population segments or groups. 
#     <br><br>
#     <strong>Column selection</strong>: To ensure a robust yet focused stratified analysis, features are selected based on four core criteria:
#     <ul style="margin-top:8px">
#         <li><strong>Statistical Stability:</strong> Groups should have sufficient sample size (target n ≥ 30) to ensure MdAE metrics are stable and not driven by outliers.</li>
#         <li><strong>Feature Importance:</strong> Top-performing features that the model logic relies on must be audited for functional consistency (Model Reliability).</li>
#         <li><strong>Legal Importance:</strong> Legally protected groups (Age, Sex, Race) or proxy variables for legally protected groups (e.g., walking limitations as a proxy for disability) must be monitored for disparate impact (Fairness Audit).</li>
#         <li><strong>Stakeholder Importance:</strong> Focus on segments where prediction errors have high financial or health consequences (e.g., high spenders).</li>
#     </ul>
#     📌 <strong>Goal:</strong> Avoid detecting false alarms through over-testing by auditing high-impact columns rather than all features.
# </div>
# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Stratified Error Analysis Table</strong> <br>
#     📌 Recover raw feature values, stratify selected features and out-of-pocket costs, calculate weighted MdAE by group, and display results table for all models.
# </div> 

# %%
# --- Prepare Features ---
# Recover raw features of validation data for stratification
# We need the original categorical codes (before pipeline one-hot encoding) to group the data
print("Recovering raw validation features...")
df_raw_val, y_val_true, w_val_weights = prepare_human_readable_split_data(VAL_MODEL_READY_DATA_PATH, "validation")

# Create chronic conditions count  
chronic_cols = list(CHRONIC_CONDITIONS.keys())
df_raw_val["CHRONIC_COUNT"] = df_raw_val[chronic_cols].sum(axis=1).astype(int)
df_raw_val["CHRONIC_COUNT_GRP"] = df_raw_val["CHRONIC_COUNT"].apply(lambda x: f"{x} Condition" if x == 1 else (f"{x} Conditions" if x < 4 else "4+ Conditions"))  # Merge 4 or more due to small group sample sizes

# Create age groups for a more stable and interpretable Fairness Audit
age_bins = [18, 35, 50, 65, 120]
age_labels = ["18-34", "35-49", "50-64", "65+"]
df_raw_val["AGE_GRP"] = pd.cut(df_raw_val["AGE23X"], bins=age_bins, labels=age_labels, right=False)


# --- Prepare Target Variable ---
# Load predictions of all tuned models (on validation data)
print("Loading tuned model predictions...")
tuned_model_predictions = {
    "Elastic Net (Tuned)": load_model("../models/en_tuned_predictions.joblib", verbose=False),
    "Random Forest (Tuned)": load_model("../models/rf_tuned_predictions.joblib", verbose=False),
    "XGBoost (Tuned)": load_model("../models/xgb_tuned_predictions.joblib", verbose=False)
}
print(f"  Loaded predictions for {len(tuned_model_predictions)} tuned models on the validation set")

# Create medical cost ranges for reporting 
# Note: Use the same bins as in the train-val-test split, but merge the Top 5% cost bins (for n>30 subgroup sample size)
COST_BIN_LABELS = {
    0: "Zero Costs",
    1: "Low Spend (0-50%)",
    2: "Moderate (50-80%)",
    3: "High Spend (80-95%)",
    4: "Very High Spend (Top 5%)"
}
actual_cost_bin_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}
# For predicted costs, also merge the Zero Costs bin (0) into Low Spend (1) for n>30 subgroup sample size because tree model predictions are almost never zero
predicted_cost_bin_map = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}

df_raw_val["ACTUAL_COSTS"] = create_stratification_bins(y_val_true).map(actual_cost_bin_map) 


# --- Define Stratified Error Configurations ---
# 1. Model Reliability: Performance across selected groups and cost levels
reliability_configs = [
    {"col": "ACTUAL_COSTS", "label": "Out-of-Pocket Costs (Actual)", "category_map": COST_BIN_LABELS},        # Reliability across actual cost ranges
    {"col": "PREDICTED_COSTS", "label": "Out-of-Pocket Costs (Predicted)", "category_map": COST_BIN_LABELS},  # Reliability across predicted cost ranges
    {"col": "RTHLTH31", "label": DISPLAY_LABELS["RTHLTH31"], "category_map": CATEGORY_LABELS_EDA["RTHLTH31"]}, # Stability across self-reported health levels
    {"col": "INSCOV23", "label": DISPLAY_LABELS["INSCOV23"], "category_map": CATEGORY_LABELS_EDA["INSCOV23"]}, # Stability across insurance types
    {"col": "CHRONIC_COUNT_GRP", "label": DISPLAY_LABELS["CHRONIC_COUNT"], "category_map": None}        # Stability across medical complexity
]

# 2. Fairness Audit: Performance across protected groups and vulnerable populations
# 2.1 Legally Protected Groups (Sex, Age, and Race)
legally_protected_configs = [
    {"col": "SEX", "label": DISPLAY_LABELS["SEX"], "category_map": CATEGORY_LABELS_EDA["SEX"]},            
    {"col": "AGE_GRP", "label": "Age Group", "category_map": None},                                         
    {"col": "RACETHX", "label": DISPLAY_LABELS["RACETHX"], "category_map": CATEGORY_LABELS_EDA["RACETHX"]}, 
]

# 2.2 Vulnerable & Proxy Groups (ethically sensitive groups or proxy variables for protected groups)
vulnerable_and_proxy_configs = [
    {"col": "MNHLTH31", "label": DISPLAY_LABELS["MNHLTH31"], "category_map": CATEGORY_LABELS_EDA["MNHLTH31"]},  # Ethically sensitive mental health
    {"col": "POVCAT23", "label": DISPLAY_LABELS["POVCAT23"], "category_map": CATEGORY_LABELS_EDA["POVCAT23"]},  # Family income as a proxy for socioeconomic status 
    {"col": "HIDEG", "label": DISPLAY_LABELS["HIDEG"], "category_map": CATEGORY_LABELS_EDA["HIDEG"]},           # Education as a proxy for socioeconomic status
    {"col": "REGION23", "label": DISPLAY_LABELS["REGION23"], "category_map": CATEGORY_LABELS_EDA["REGION23"]},  # Region for geographic equality
    {"col": "WLKLIM31", "label": DISPLAY_LABELS["WLKLIM31"], "category_map": CATEGORY_LABELS_EDA["WLKLIM31"]},  # Walking limitation as a proxy for disability
]

# Combine configurations 
stratified_error_configs = reliability_configs + legally_protected_configs + vulnerable_and_proxy_configs


# --- Stratified Error Analysis ---
# Calculate weighted MdAE for each model, column, and group
stratified_error_results = []

for model_key, y_val_pred in tuned_model_predictions.items():
    model_label = MODEL_DISPLAY_LABELS.get(model_key, model_key)
    
    for config in stratified_error_configs:
        col = config["col"]
        label = config["label"]
        category_map = config["category_map"]
        
        # For predicted costs: Use each model's own predictions
        if col == "PREDICTED_COSTS":
            y_val_pred_series = pd.Series(y_val_pred, index=y_val_true.index)  # Converts y_val_pred numpy array to Series to align on index
            col_bins = create_stratification_bins(y_val_pred_series).map(predicted_cost_bin_map)  
        else:
            col_bins = df_raw_val[col]
            
        # Calculate weighted MdAE for each subgroup of current column
        groups = sorted(col_bins.dropna().unique())
        for group in groups:
            mask = (col_bins == group)
            
            # Calculate weighted MdAE
            group_mdae = weighted_median_absolute_error(
                y_val_true[mask],  # Aligns via Index
                y_val_pred[mask],  # Aligns via Position (df_raw_val was reindexed to match the validation parquet)
                sample_weight=w_val_weights[mask]  # Aligns via Index
            )
            
            # Calculate weighted median actual cost
            group_median_actual = weighted_quantile(y_val_true[mask], w_val_weights[mask], 0.5)
            
            # Add results of current column to list
            stratified_error_results.append({
                "Model": model_label,
                "Column": label,
                "Group": category_map.get(int(group), group) if category_map else group,  # Maps group label 
                "MdAE": group_mdae,
                "Sample Size": mask.sum(),
                "Median Actual Cost": group_median_actual
            })

# Convert to DataFrame
subgroup_df = pd.DataFrame(stratified_error_results)

# Display results table (pivoted for model comparison)
# Pivot on Column, Group, Sample Size, and Median Actual Cost to keep metadata organized in separate columns
subgroup_pivoted_df = subgroup_df.pivot(
    index=["Column", "Group", "Sample Size", "Median Actual Cost"], 
    columns="Model", 
    values="MdAE"
)
ordered_index = pd.MultiIndex.from_frame(
    subgroup_df[["Column", "Group", "Sample Size", "Median Actual Cost"]].drop_duplicates()
)

display(
    subgroup_pivoted_df.reindex(ordered_index)
    .style
    .pipe(add_table_caption, "Tuned Models: Stratified Error Analysis")
    .format("${:.2f}")
    .format_index(formatter="{:,}", level="Sample Size")
    .format_index(formatter="${:,.2f}", level="Median Actual Cost")
    .background_gradient(cmap="RdYlGn_r", axis=1)  # Red-Yellow-Green reversed (lower MdAE is better) to make winning model pop out in green
)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Model Reliability Analysis</strong> <br>
#     📌 Visualize stratified error of actual and predicted out-of-pocket costs, as well as selected features across all tuned models.
# </div> 

# %%
def plot_subgroup_performance(df, column_labels, title, save_to_file=None):
    """
    Visualizes stratified Median Absolute Error (MdAE) as a faceted bar plot grid.
    
    Filters the analysis dataframe to specific columns and generates a standardized 
    visualization for stratified prediction performance analysis.

    Args:
        df (pd.DataFrame): Dataframe containing 'Column', 'Group', 'MdAE', and 'Sample Size'.
        column_labels (list): List of display labels (e.g., from reliability_configs) to plot.
        title (str): Main title for the figure.
        save_to_file (str, optional): Full path and filename to save the plot.
    """
    # Filter data to requested columns 
    plot_data = df[df["Column"].isin(column_labels)].copy()

    # Create faceted bar plot grid with model as hue
    g = sns.catplot(
        data=plot_data,
        kind="bar",
        x="MdAE",
        y="Group",
        hue="Model",
        col="Column",
        col_wrap=2,
        height=5,
        aspect=1.4,
        sharex=False,  # Independent x-axes to handle different error scales for each column
        sharey=False,  # Independent y-axes 
        palette="Set2",
        legend_out=False
    )

    # Calculate dynamic coordinates based on figure height (in inches) to ensure identical spacing across plots
    fig_height = g.fig.get_figheight()
    title_y = 1.0 - (0.18 / fig_height)
    legend_y = 1.0 - (0.50 / fig_height)
    subplots_top = 1.0 - (1.3 / fig_height)
    subplots_bottom = 0.80 / fig_height

    # Customize title and spacing
    g.set_titles("{col_name}", weight="bold", size=14)
    g.fig.suptitle(title, fontsize=18, weight="bold", y=title_y)
    plt.subplots_adjust(top=subplots_top, bottom=subplots_bottom, hspace=0.25, wspace=0.25)

    # Apply custom formatting to each subplot
    for ax in g.axes.flat:
        ax.set_ylabel("")
        ax.set_xlabel("Weighted MdAE")
        ax.set_xticklabels([])  # Removes redundant x-tick labels (since bars are annotated)
        
        # Get y-tick labels and update them with Sample Size and Median Actual Cost
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        if not any(y_labels):
            y_labels = [label.get_text() for label in ax.yaxis.get_ticklabels()]
        
        col_name = ax.get_title()
        new_labels = []
        for label in y_labels:
            subgroup_data = plot_data[(plot_data["Column"] == col_name) & (plot_data["Group"] == label)]
            if not subgroup_data.empty:
                first_row = subgroup_data.iloc[0]
                n = first_row["Sample Size"]
                med = first_row["Median Actual Cost"]
                clean_label = str(label).split(" (")[0]
                new_labels.append(f"{clean_label}\nn={n:,} | ${med:,.0f}")
            else:
                new_labels.append(str(label).split(" (")[0])
        
        # Set ticks explicitly first to avoid matplotlib UserWarning
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(new_labels, fontsize=9)
        
        for c in ax.containers:          
            value_labels = [f"${v:,.0f}" for v in c.datavalues]  # Formats value labels with thousand separator and no decimals
            ax.bar_label(c, labels=value_labels, padding=3, fontsize=9)
        ax.margins(x=0.15)
    
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, legend_y), ncol=3, frameon=True, title=None)
    
    # Add footnote
    # Compute final layout, then align footnote with the first subplot's leftmost label
    g.fig.canvas.draw()
    footnote_x = g.axes.flat[0].get_tightbbox(g.fig.canvas.get_renderer()).transformed(g.fig.transFigure.inverted()).x0

    # Add footnote at the bottom, aligned with the leftmost label/title
    g.fig.text(
        footnote_x, 
        0.01, 
        "Note: Dollar values in subgroup labels (e.g., $269) represent the actual median out-of-pocket costs of that subgroup.", 
        fontsize=9, 
        style="italic", 
        color="#555555",
        ha="left"
    )
    
    # Save to file
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)
        
    plt.show()


# Model reliability analysis 
reliability_labels = [c["label"] for c in reliability_configs]
plot_subgroup_performance(
    subgroup_df,
    reliability_labels,
    "Tuned Models: Subgroup Reliability (Validation)",
    save_to_file="../figures/evaluation/tuned_models_validation_subgroup_reliability.png"
)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Fairness Audit</strong> <br>
#     📌 Visualize stratified error for legally protected groups, vulnerable populations, and proxy attributes across all tuned models. 
#     <br><br>
#     Ensures that the model's prediction error (MdAE) does not disproportionately affect protected groups. While model reliability analysis ensures the model works functionally, the Fairness Audit ensures it works equitably. Detecting a disparity is a diagnostic signal (triggering investigation into "Legitimate Business Necessity") rather than an automatic failure of the model.
# </div> 

# %%
# Legally Protected Groups
legally_protected_labels = [c["label"] for c in legally_protected_configs]
plot_subgroup_performance(
    subgroup_df,
    legally_protected_labels,
    "Tuned Models: Subgroup Fairness - Protected Groups (Validation)",
    save_to_file="../figures/evaluation/tuned_models_validation_subgroup_fairness_protected.png"
)

# Vulnerable & Proxy Groups
vulnerable_and_proxy_labels = [c["label"] for c in vulnerable_and_proxy_configs]
plot_subgroup_performance(
    subgroup_df,
    vulnerable_and_proxy_labels,
    "Tuned Models: Subgroup Fairness - Vulnerable & Proxy Groups (Validation)",
    save_to_file="../figures/evaluation/tuned_models_validation_subgroup_fairness_vulnerable_proxy.png"
)


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights: Stratified Error Analysis (Model Comparison)</b> 
#     <br>
#     <div style="margin-top:10px;"><strong>Model Reliability</strong></div>
#     <ul>
#         <li><strong>Actual cost range:</strong> MdAE rises exponentially with higher actual costs. Models converge at the Top 5% (~\$9.5k error), highlighting the noise limit of healthcare data (e.g., random catastrophic events that cannot be predicted from survey demographics). Elastic Net struggles with Zero Costs (\$90 vs. \$29–\$31 for tree models) due to its linear assumptions.</li>
#         <li><strong>Predicted cost range:</strong> MdAE rises with predicted costs. Random Forest is the most reliable for "Very High Spend" predictions (\$751 MdAE) vs. Elastic Net (\$1,095), indicating tree-based models are better calibrated for identifying high-risk individuals.</li>
#         <li><strong>Health:</strong> Errors rise as physical health declines. At the "Poor" stratum, Elastic Net (\$799) produces nearly 2× the error of tree models (~\$420), which better capture the non-linear relationship for high-complexity patients.</li>
#         <li><strong>Insurance:</strong> Largest error for Private, followed by Public Insurance. Least error for Uninsured. Notably, Elastic Net (\$95) shows 3–4× the error of tree models (~\$30) for the Uninsured, failing to capture the sharp spending constraints of this near-zero cluster.</li>
#         <li><strong>Chronic conditions:</strong> Error rises with the number of conditions. At 4+ conditions, tree models plateau (~\$500), while Elastic Net jumps to \$799 (a 1.6× gap). This confirms tree models better capture the "cost saturation effect" of extreme medical complexity (e.g., due to combined doctor visits/medication or out-of-pocket maximums in insurance plan).</li>
#     </ul>
#     <div style="margin-top:10px;"><strong>Fairness Audit</strong></div>
#     <ul>
#         <li><strong>Sex:</strong> Female/Male disparity is consistent (~1.5×) across all architectures, likely reflecting utilization variance (e.g., reproductive care) rather than algorithmic bias. Elastic Net shows the highest disparity (2.1×) due to its exceptionally low male error (\$117).</li>
#         <li><strong>Age:</strong> Prediction error increases with age across all models (4–6× larger error for oldest vs. youngest adults). Architectures converge for the 65+ group (~\$470), while the rising error trend suggests clinical complexity rather than age-based bias.</li>
#         <li><strong>Race/Ethnicity:</strong> Error is highest for White (\$317–\$336) and lower for Hispanic and Black populations (~\$100–\$140). All models agree on this pattern, which likely reflects data properties: White spending is more spread out, while minority spending is more concentrated at lower levels.</li>
#         <li><strong>Family Income:</strong> Error rises with family income across all models (4–7× larger error for highest vs. lowest income group).</li>
#         <li><strong>Education:</strong> A strong monotonic gradient from No Degree to Doctorate across all models, though models diverge at the top: XGBoost (\$837) shows the highest Doctorate MdAE compared to Elastic Net (\$611) and RF (\$577). The overall gradient (No Degree to Doctorate: 8–12× ratio) confirms education acts as a proxy for income and private insurance quality, reflecting higher financial exposure and spending variance rather than a discriminatory harm against vulnerable populations.</li>
#         <li><strong>Mental health:</strong> Error rises as mental health declines. Elastic Net is the most precise for "Excellent" mental health (\$115) but the least precise for "Poor" mental health (\$644), where non-linear models better handle the complex comorbidities.</li>
#         <li><strong>Walking limitations:</strong> Prediction error is larger when having walking limitations compared to not (3–4× ratio). Elastic Net is better than tree-based models when walking limitations are absent but worse when they are present, reflecting that linear models struggle more with the medical complexity.</li>
#         <li><strong>Region:</strong> Smallest disparity of any audited dimension. South and West show slightly lower prediction error compared to Northeast and Midwest.</li>
#     </ul>
#     <div style="margin-top:10px;"><strong>Overall Assessment:</strong> All three model architectures produce the same directional disparities across protected groups, confirming these reflect data-generating characteristics (utilization patterns, access constraints) rather than architecture-specific algorithmic bias. Tree-based models (XGBoost, Random Forest) consistently outperform Elastic Net for high-complexity segments (poor health, many chronic conditions, walking limitations), while Elastic Net shows superior performance for low medical complexity segments. No evidence of discriminatory disparate impact against protected or vulnerable groups was found across any model. The models perform better for several marginalized groups (Hispanic/Black/Asian, low income, low education). Conversely, where prediction error is higher for protected or vulnerable groups (females, older adults, walking impaired), the disparity is justified by clinical complexity and higher utilization variance, satisfying the Legitimate Business Necessity defense. All three models are suitable for deployment as low-risk advisory tools under NIST/FTC transparency guidelines.</div>
# </div>
# %% [markdown]
# <div style="background-color:#f0f7ff; padding:15px; border:3px solid #cfe2ff; border-radius:6px; margin-bottom:16px;">
#     <strong>⚖️ Regulatory Compliance & Ethical AI</strong> <br>
#     This project is designed for the US Market (NIST, FTC) with a roadmap for EU Expansion (AI Act, GDPR). To ensure responsible and compliant deployment, performed stratified error analysis to detect algorithmic bias and use sample weights to ensure population representativeness.
#     <p style="margin-top:10px;">
#         For details on regulatory compliance and ethical AI, refer to: <a href="../docs/research/regulatory_compliance.md">docs/research/regulatory_compliance.md</a>
#     </p>
# </div>
#

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Quantile Regression</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <strong>"Plan Around + Safety Cushion" Approach</strong> <br> 
#     For the medical cost planner, adopt a "Plan Around + Safety Cushion" approach to give app users a helpful, accurate, and actionable prediction for next year's out-of-pocket costs.
# <br><br>
# <strong>Rationale: Why Prediction Ranges Matter</strong> <br>
# Standard models provide a point estimate (a single mean or median) that implies false precision. A better approach is to show a plan-around estimate, a typical range, and a safety cushion so users can act on the result without needing deep technical knowledge.
#
# *   <b>Heteroscedasticity:</b> Residual plots confirm a "fan shape" where error variance is not constant. As medical complexity increases, the spread of possible outcomes expands exponentially. A single number cannot capture this shifting uncertainty.
# *   <b>Stratified Error:</b> Prediction error varies across user segments, confirming that "average error" is misleading. Healthy users have very predictable costs (narrow range), while high-risk users face extreme uncertainty (wide range). A single number would over-prepare the healthy and under-prepare the sick. 
#
# <strong>The Solution: Quantile Regression</strong> <br>
# Instead of one "best guess", train models to predict specific percentiles of the cost distribution. This allows the medical cost planner to communicate uncertainty directly to the user without needing deep understanding of statistics:
#
# <table style="width:70%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;">
#     <thead>
#         <tr style="border-bottom: 2px solid #d0e7fa; text-align: left;">
#             <th style="padding: 8px; width: 18%;">Scenario</th>
#             <th style="padding: 8px; width: 18%;">Quantile</th>
#             <th style="padding: 8px; width: 28%;">Purpose</th>
#             <th style="padding: 8px; width: 36%;">Actionable Insight</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr style="border-bottom: 1px solid #eef7fe;">
#             <td style="padding: 8px;"><b>Plan Around</b></td>
#             <td style="padding: 8px;"><code>0.50</code> (Median)</td>
#             <td style="padding: 8px;">Half of similar people spend less; half spend more.</td>
#             <td style="padding: 8px;">Simple anchor for basic budgeting.</td>
#         </tr>
#         <tr style="border-bottom: 1px solid #eef7fe;">
#             <td style="padding: 8px;"><b>Typical Range</b></td>
#             <td style="padding: 8px;"><code>0.25</code> to <code>0.75</code></td>
#             <td style="padding: 8px;">Middle 50% of similar profiles.</td>
#             <td style="padding: 8px;">Expected range for FSA/HSA planning.</td>
#         </tr>
#         <tr>
#             <td style="padding: 8px;"><b>Safety Cushion</b></td>
#             <td style="padding: 8px;"><code>0.90</code></td>
#             <td style="padding: 8px;">Higher-cost year.</td>
#             <td style="padding: 8px;">Emergency fund and risk planning.</td>
#         </tr>
#     </tbody>
# </table>
# Recommendation: Display q50 as "Plan around", q25-q75 as the "Typical range", and q90 as the "Safety cushion". By using this approach, the typical range will be different for a low-risk user (with plan-around cost < \$1,000) compared to a high-risk user (plan-around cost ≥ \$1,000). For example, a low-risk user may see a tight range like \$50 – \$450, while a high-risk user sees a wider range like \$1,100 – \$7,200, accurately reflecting their higher financial risk. For high-risk users, display a note on possible out-of-pocket insurance maximum. 
# <br><br>
#     <strong>Example Prediction Text</strong> (shown to app users)
# <table style="width:100%; border-spacing: 10px 0px; border-collapse: separate; margin-top: 10px;">
#     <tr>
#         <td style="width: 50%; padding: 0 10px; vertical-align: bottom;">
#             <strong>Low Cost Profile</strong><br>
#             <i>28-year-old with no chronic conditions</i>
#         </td>
#         <td style="width: 50%; padding: 0 10px; vertical-align: bottom;">
#             <strong>High Cost Profile</strong><br>
#             <i>68-year-old with multiple chronic conditions</i>
#         </td>
#     </tr>
#     <tr>
#         <td style="background-color: #fcfcfc; border: 1px solid #ddd; padding: 15px; vertical-align: top; border-radius: 4px;">
#             <b>Your Estimated Out-of-Pocket Costs for Next Year</b><br><br>
#             💰 <b>Plan around:</b> \$180<br>
#             📊 <b>Typical range:</b> \$50 – \$450<br>
#             🛡️ <b>Safety cushion:</b> plan up to \$900<br><br>
#             <span style="font-size: 0.85em; color: #555;">People with answers like yours often spend about \$180. Many spend between \$50 and \$450. If you want a cushion for a higher-cost year, planning up to \$900 would cover most similar cases.</span>
#         </td>
#         <td style="background-color: #fcfcfc; border: 1px solid #ddd; padding: 15px; vertical-align: top; border-radius: 4px;">
#             <b>Your Estimated Out-of-Pocket Costs for Next Year</b><br><br>
#             💰 <b>Plan around:</b> \$2,850<br>
#             📊 <b>Typical range:</b> \$1,100 – \$7,200<br>
#             🛡️ <b>Safety cushion:</b> plan up to \$9,500<br><br>
#             <span style="font-size: 0.85em; color: #555;">
#                 People with answers like yours often spend about \$2,850, but costs can change a lot from year to year. Many similar people spend between \$1,100 and \$7,200. If you want a cushion for a higher-cost year, planning up to \$9,500 would cover most similar cases. If you have insurance, check your plan's out-of-pocket maximum for covered in-network care; you likely won't need to set aside more than that yearly limit.
#             </span>
#         </td>
#     </tr>
# </table>
# </div>

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <strong>Implementation Decisions</strong>
# <br><br>
# <strong>1. Model Selection: XGBoost Only</strong> <br>
# Train quantile regression using XGBoost only (not Elastic Net or Random Forest). The goal is not just the lowest single-number error; the goal is a useful planning range:
#     <ul style="margin-top:8px">
#         <li><b>XGBoost differentiates best:</b> Predictions span up to \$2,114 (10× Elastic Net's \$217 and 1.7× Random Forest's \$1,273). A model that better separates low-cost from high-cost individuals is more likely to produce useful quantile ranges.</li>
#         <li><b>Native multi-quantile support:</b> XGBoost's <code>reg:quantileerror</code> objective trains all four quantiles (0.25, 0.50, 0.75, 0.90) in a single model via the <code>quantile_alpha</code> parameter. Elastic Net would require <code>sklearn.linear_model.QuantileRegressor</code> (separate model per quantile) and Random Forest would require the external <code>quantile-forest</code> package, both adding complexity without changing the conclusion.</li>
#         <li><b>Deployment efficiency:</b> A single multi-quantile XGBoost model produces one <code>.joblib</code> artifact instead of 4 separate model files.</li>
#         <li><b>Metrics to confirm before release:</b> Evaluate q50 MdAE, pinball loss for each quantile, interval coverage, interval width, and subgroup performance on the untouched test set. Coverage alone is not enough: intervals also need to be narrow enough to help users make decisions.</li>
#     </ul>
# <strong>2. Calibration: Conformalized Quantile Regression (CQR)</strong> <br>
# Raw quantile regression has no coverage guarantee. The predicted "Typical Range" (q25–q75) might actually contain only 40% or 60% of real outcomes, not the intended 50%. CQR adds a calibration step that adjusts the intervals to provide a finite-sample coverage guarantee.
# <ul>
#     <li><b>Phase 1 (QR):</b> Train and evaluate raw quantile regression. Measure empirical coverage on the validation set to establish a baseline.</li>
#     <li><b>Phase 2 (CQR - Optional):</b> If Phase 1 coverage deviates significantly from targets (e.g., > ±5%), add a CQR calibration layer to provide a finite-sample coverage guarantee (see <code><a href="../docs/specs/technical_specifications.md" target="_blank">technical specifications</a></code> success metrics).</li>
# </ul>
# <b>Calibration Set (~1,000 samples of train):</b> If Phase 2 is triggered, CQR requires a hold-out calibration set to compute conformity scores. Holding out ~1,000 samples from the training set (~8.5% of 11,814) is large enough for stable conformity score estimation while small enough to not substantially reduce training data quality.
# <br><br>
# <strong>3. Explainability (SHAP): Median Only</strong> <br>
# A quantile model predicts four numbers per user. SHAP values explain feature contributions for a <em>specific</em> prediction target. Those contributions differ across quantiles (e.g., "Diabetes: +1,200" for the median vs. "Diabetes: +3,800" for the 90th percentile). Showing multiple, contradictory SHAP explanations would confuse users.
# <br><br>
# <strong>Decision:</strong> Display SHAP values for the plan-around estimate (median/q50) only. This gives users a single, coherent explanation of their "most likely" cost drivers. The typical range and safety cushion are presented as context for better financial planning.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Training</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Train an XGBoost multi-quantile regression model that returns q25, q50, q75, and q90 predictions. 
#     <br><br>
#     Production Script: This notebook is for prototyping; the production run was executed via <code><a href="../scripts/train_xgboost_quantile.py">scripts/train_xgboost_quantile.py</a></code>.
# </div>

# %%
def train_xgboost_quantile():
    # --- 1. Model Configuration ---
    print("Step 1: Configuring XGBoost multi-quantile model parameters...")
    QUANTILES = [0.25, 0.50, 0.75, 0.90]

    tuned_params = load_metrics("../models/xgb_tuned_params.json", verbose=False)
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

    # --- 2. Model Training ---
    print("Step 2: Training XGBoost quantile regression model...")
    # Train on log-costs: quantiles are invariant to monotonic transformations, and the log scale
    # stabilizes tree-splitting logic by preventing extreme outliers from dominating the partition search.
    xgb_quantile_model = TransformedTargetRegressor(
        regressor=XGBRegressor(**xgb_quantile_params),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # Normalize training weights (mean=1.0) for numerical stability during model fitting
    w_train_norm = w_train / w_train.mean()

    start_time = time.time()
    xgb_quantile_model.fit(X_train_preprocessed, y_train, sample_weight=w_train_norm)
    training_time = time.time() - start_time

    print(f"  Completed training in {training_time:.1f} s")

    # --- 3. Predictions ---
    print("Step 3: Predicting on training and validation set...")
    # Predict on training and validation set
    y_train_pred_raw = xgb_quantile_model.predict(X_train_preprocessed)
    y_val_pred_raw = xgb_quantile_model.predict(X_val_preprocessed)

    # Ensure valid cost quantiles (non-negative and monotonic q25 <= q50 <= q75 <= q90)
    y_train_pred = postprocess_quantile_predictions(y_train_pred_raw)
    y_val_pred = postprocess_quantile_predictions(y_val_pred_raw)
    print(f"  Generated predictions for {len(y_train_pred):,} train and {len(y_val_pred):,} validation samples and ensured non-negative and monotonic predictions")

    # --- 4. Evaluation ---
    print("Step 4: Evaluating model performance...")
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
    train_q50_q90_width = np.average(y_train_pred_q90 - y_train_pred_q50, weights=w_train)  # Safety cushion width
    val_q25_q75_width = np.average(y_val_pred_q75 - y_val_pred_q25, weights=w_val)
    val_q50_q90_width = np.average(y_val_pred_q90 - y_val_pred_q50, weights=w_val)

    print(f"  Plan Around MdAE       →  Train: {f'${train_q50_mdae:,.2f}':>10} | Val: {f'${val_q50_mdae:,.2f}':>10}")
    print(f"  Plan Around MAE        →  Train: {f'${train_q50_mae:,.2f}':>10} | Val: {f'${val_q50_mae:,.2f}':>10}")
    print(f"  Plan Around R²         →  Train: {train_q50_r2:10.2f} | Val: {val_q50_r2:10.2f}")
    print(f"  Typical Range Coverage →  Train: {train_q25_q75_coverage:10.1%} | Val: {val_q25_q75_coverage:10.1%}")
    print(f"  Safety Cushion Coverage →  Train: {train_q90_coverage:10.1%} | Val: {val_q90_coverage:10.1%}")
    print(f"  Avg Typical Range Width →  Train: {f'${train_q25_q75_width:,.0f}':>10} | Val: {f'${val_q25_q75_width:,.0f}':>10}")
    print(f"  Avg Safety Cushion W.  →  Train: {f'${train_q50_q90_width:,.0f}':>10} | Val: {f'${val_q50_q90_width:,.0f}':>10}")

    # --- 5. Model Persistence ---
    print("Step 5: Persisting model results...")
    # 5.1. Save fitted model as .joblib file
    save_model(xgb_quantile_model, "../models/xgb_quantile_model.joblib", verbose=False)
    print("  Saved XGBoost quantile regression model to 'models/xgb_quantile_model.joblib'")

    # 5.2. Save evaluation metrics as JSON
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
    save_metrics(xgb_quantile_metrics, "../models/xgb_quantile_metrics.json", verbose=False)
    print("  Saved evaluation metrics of XGBoost quantile regression to 'models/xgb_quantile_metrics.json'")

    # 5.3. Save hyperparameters as JSON
    save_metrics(xgb_quantile_params, "../models/xgb_quantile_params.json", verbose=False)
    print("  Saved hyperparameters of XGBoost quantile regression to 'models/xgb_quantile_params.json'")

    # 5.4. Save predicted values as .joblib file
    save_model(y_val_pred, "../models/xgb_quantile_predictions.joblib", verbose=False)
    print("  Saved predicted values of XGBoost quantile regression to 'models/xgb_quantile_predictions.joblib'")

    print("\n✅ XGBoost quantile regression complete.")

    
# Commented out to avoid overwriting model artifacts during notebook reruns (training is handled by 'scripts/train_xgboost_quantile.py')
# train_xgboost_quantile()

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Evaluation</h2>
# </div> 

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Pinball Loss & Skill Score</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     💡 <b>What is Pinball Loss?</b>
#     <br>
#     Pinball loss is an asymmetric penalty function used to estimate the prediction performance for specific quantiles. Unlike standard metrics (MSE/MAE) that treat all errors the same, pinball loss penalizes errors differently based on our target:
#     <ul style="margin-top:10px">
#         <li><b>For q90:</b> Under-predicting is 9x more "expensive" (0.90 weight) than over-predicting (0.10). This forces the model to "overshoot" 90% of the data to create a safety cushion.</li>
#         <li><b>For q25:</b> Over-predicting is 3x more "expensive" (0.75 weight) than under-predicting (0.25), forcing the model toward the lower end of costs.</li>
#         <li><b>For q50:</b> Penalties are symmetric (0.50 on both sides). Because every error is multiplied by 0.5, the mean pinball loss is exactly 0.5 * MAE.</li>
#     </ul>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     💡 <b>What is a Quantile Skill Score (QSS)?</b>
#     <br>
#     The QSS measures the improvement of our model compared to a naive baseline (always predicting the same population-level quantile). E.g., a score of 11% for the q90 safety cushion means the model's intelligence (using health features) reduced the error penalty by 11% compared to always predicting the 90th percentile.
#     <br><br>
#     <b>Interpretation:</b> Across most forecasting domains (finance, weather, supply chain), QSS is often interpreted as follows.
#     <ul style="margin-top:10px">
#         <li><b>0% (Zero Skill):</b> The model is no better than a simple "guess" of the population average or quantile.</li>
#         <li><b>< 5% (Low Skill):</b> The model is barely better than a guess. This usually means the features you are using don't have a strong relationship with the outcome, or the data is extremely noisy.</li>
#         <li><b>5% – 15% (Moderate Skill):</b> This is the "sweet spot" for many complex real-world problems. It indicates the model has captured meaningful patterns and provides real value over a simple average.</li>
#         <li><b>> 15% (High Skill):</b> The model is very strong. This is typical for problems with clear physical or logical rules (like electricity demand based on temperature).</li>
#         <li><b>> 30% (Exceptional):</b> Rare in human-behavior data. If you see this in medical costs, you should double-check for "data leakage".</li>
#         <li><b>100%:</b> The model is a "Perfect Oracle" with zero prediction error.</li>
#     </ul>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate each quantile using the Pinball Loss and Skill Score.
# </div>

# %%
# Load model 
xgb_quantile_model = load_model("../models/xgb_quantile_model.joblib", verbose=False)

# Predict on training data
y_train_quantile_pred_raw = xgb_quantile_model.predict(X_train_preprocessed)
y_train_quantile_pred = postprocess_quantile_predictions(y_train_quantile_pred_raw)

# Load predictions on validation data
y_val_quantile_pred = load_model("../models/xgb_quantile_predictions.joblib", verbose=False)

quantiles = [0.25, 0.50, 0.75, 0.90]
pinball_results = []

for idx, q in enumerate(quantiles):
    # Model predictions
    y_train_pred_q = y_train_quantile_pred[:, idx]
    y_val_pred_q = y_val_quantile_pred[:, idx]
    
    # Naive baseline predictions (overall weighted quantile of training target)
    train_naive_val = weighted_quantile(y_train, w_train, q)
    y_train_naive = np.full_like(y_train, fill_value=train_naive_val)
    y_val_naive = np.full_like(y_val, fill_value=train_naive_val)
    
    # Calculate weighted pinball loss
    train_loss_model = mean_pinball_loss(y_train, y_train_pred_q, alpha=q, sample_weight=w_train)
    val_loss_model = mean_pinball_loss(y_val, y_val_pred_q, alpha=q, sample_weight=w_val)
    
    train_loss_naive = mean_pinball_loss(y_train, y_train_naive, alpha=q, sample_weight=w_train)
    val_loss_naive = mean_pinball_loss(y_val, y_val_naive, alpha=q, sample_weight=w_val)
    
    # Calculate Quantile Skill Score (QSS)
    train_qss = 1.0 - (train_loss_model / train_loss_naive)
    val_qss = 1.0 - (val_loss_model / val_loss_naive)
    
    # Train/Val delta (overfitting measure)
    delta_percent = ((val_loss_model - train_loss_model) / train_loss_model) * 100
    
    pinball_results.append({
        "Quantile": f"q{int(q*100)}",
        "Model Pinball Loss": val_loss_model,
        "Pinball Loss (Train)": train_loss_model,
        "Pinball Delta %": delta_percent,
        "Skill Score": val_qss,
        "Skill Score (Train)": train_qss,
    })

pinball_df = pd.DataFrame(pinball_results)
display(
    pinball_df.style
    .hide()
    .pipe(add_table_caption, "XGBoost Quantile Regression: Pinball Loss & Skill Scores (Validation)")
    .format("${:,.2f}", subset=["Model Pinball Loss", "Pinball Loss (Train)"])
    .format("{:.2%}", subset=["Skill Score", "Skill Score (Train)"])
    .format("{:+.2f}%", subset=["Pinball Delta %"])
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Higher Skill at Upper Quantiles:</strong> The Quantile Skill Score triples from q25 (3.7%) to q75 (11.1%) on the validation set. This shows the model's features are more informative for predicting high compared to low out-of-pocket costs. At the low end, costs are mostly "noise" (random office visits, one-off prescriptions). There isn't much "skill" to be had, because low costs are somewhat random for everyone.</li>
#         <li><strong>q90 Overfitting:</strong> Pinball loss deltas are small for q25–q75 (absolute delta ≤ 1.3%), but the q90 delta jumps to +5.8%. More tellingly, the q90 QSS drops from 18.7% (train) to 10.8% (val), a 7.8 percentage point gap, far larger than the ~3% gaps at other quantiles. This indicates that the model's tail predictions are sensitive to training data outliers, and the apparent q90 superiority on train does not fully generalize.</li>
#         <li><strong>q75 vs. q90 Pinball Loss:</strong> While the absolute Pinball Loss peaks at q75 (\$580), it drops at q90 (\$502) due to the asymmetric penalty structure: at q90, over-predictions are penalized at only 10% weight, allowing the model to provide a conservative safety cushion with lower overall penalty.</li>
#         <li><strong>Practical Implication:</strong> On validation data, q75 (11.1%) and q90 (10.8%) achieve nearly identical skill scores. The safety cushion (q90) remains useful as a conservative upper bound for budgeting, but users should be aware that its precision is no better than the typical range upper bound (q75). Its value lies in the asymmetric "better safe than sorry" design, not in superior predictive accuracy.</li>
#     </ul>
# </div>


# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Quantile Calibration</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     💡 <b>What is Quantile Calibration?</b>
#     <br>
#     Quantile regression models should be well calibrated, meaning each predicted quantile behaves like the probability level it claims. A well-calibrated q25 should be greater than or equal to the actual cost for about 25% of the population, q50 for about 50%, q75 for about 75%, and q90 for about 90%.
#     <br><br>
#     <b>Quantile Coverage vs. Interval Coverage</b> <br> 
#     Quantile coverage checks each predicted quantile directly. Interval coverage checks whether actual costs fall between two endpoints, such as q25 and q75. Reporting both matters because the q25–q75 interval can have acceptable 50% coverage even when both endpoints are shifted in the same direction.
#     <br><br>
#     <b>Calibration Error</b> <br>
#     Calibration error is empirical coverage minus the nominal quantile level. Positive values mean the quantile is too high/conservative; negative values mean it is too low/aggressive. For this project, errors within about 5% are acceptable validation diagnostics; errors beyond about 10% would usually require recalibration or a clearer release warning.
#     <br><br>
#     <b>Metric Confidence Intervals</b> <br>
#     Bootstrap confidence intervals are calculated for metrics on the validation data. They are for the metrics, not prediction intervals for individual users. Each bootstrap sample resamples rows with replacement while keeping actual cost, predicted quantiles, and survey weight together.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Compare nominal quantile levels with empirical weighted coverage.
# </div>

# %%
N_BOOTSTRAP = 1000


def get_bootstrap_ci(samples, confidence=0.95):
    """
    Get a percentile confidence interval from bootstrap metric samples.

    Args:
        samples (array-like): Recomputed metric values from bootstrap resamples.
        confidence (float): Confidence level for the percentile interval. Defaults to 95%.

    Returns:
        tuple: Lower and upper confidence interval bounds.
    """
    alpha = 1 - confidence
    return np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def generate_quantile_metric_bootstrap_samples(
    y_true,
    y_pred,
    weights,
    quantiles,
    n_bootstrap=1000,
    random_state=RANDOM_STATE,
):
    """
    Generate bootstrap samples for key quantile regression metrics.

    This function returns the bootstrap distribution for each metric, not the
    summarized confidence intervals. Use get_bootstrap_ci() to convert
    any returned metric sample into a confidence interval.

    Metrics returned:
        - q25/q50/q75/q90 empirical coverage, based on quantiles
        - q50 MdAE
        - q50 MAE
        - q50 R²
        - q25-q75 interval coverage
        - q25-q75 average interval width
        - q50-q90 average safety cushion width

    Args:
        y_true (array-like): Actual costs.
        y_pred (np.ndarray): Predictions with one column per quantile.
        weights (array-like): Survey weights.
        quantiles (list): Quantile levels matching prediction columns.
        n_bootstrap (int): Number of bootstrap resamples.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Recomputed metric samples for calibration and product metrics.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    n_obs = len(y_true)

    bootstrap_samples = {
        **{f"q{int(q * 100)}_coverage": np.empty(n_bootstrap) for q in quantiles},
        "q50_mdae": np.empty(n_bootstrap),
        "q50_mae": np.empty(n_bootstrap),
        "q50_r2": np.empty(n_bootstrap),
        "q25_q75_coverage": np.empty(n_bootstrap),
        "q25_q75_width": np.empty(n_bootstrap),
        "q50_q90_width": np.empty(n_bootstrap),
    }

    for sample_idx in range(n_bootstrap):
        row_idx = rng.integers(0, n_obs, size=n_obs)
        y_boot = y_true[row_idx]
        pred_boot = y_pred[row_idx]
        w_boot = weights[row_idx]

        q25_pred, q50_pred, q75_pred, q90_pred = pred_boot.T

        for quantile_idx, q in enumerate(quantiles):
            bootstrap_samples[f"q{int(q * 100)}_coverage"][sample_idx] = np.average(
                y_boot <= pred_boot[:, quantile_idx],
                weights=w_boot,
            )

        bootstrap_samples["q50_mdae"][sample_idx] = weighted_median_absolute_error(
            y_boot,
            q50_pred,
            sample_weight=w_boot,
        )
        bootstrap_samples["q50_mae"][sample_idx] = mean_absolute_error(
            y_boot,
            q50_pred,
            sample_weight=w_boot,
        )
        bootstrap_samples["q50_r2"][sample_idx] = r2_score(
            y_boot,
            q50_pred,
            sample_weight=w_boot,
        )
        bootstrap_samples["q25_q75_coverage"][sample_idx] = np.average(
            (y_boot >= q25_pred) & (y_boot <= q75_pred),
            weights=w_boot,
        )
        bootstrap_samples["q25_q75_width"][sample_idx] = np.average(q75_pred - q25_pred, weights=w_boot)
        bootstrap_samples["q50_q90_width"][sample_idx] = np.average(q90_pred - q50_pred, weights=w_boot)

    return bootstrap_samples


val_quantile_metric_bootstrap_samples = generate_quantile_metric_bootstrap_samples(
    y_val,
    y_val_quantile_pred,
    w_val,
    quantiles,
    n_bootstrap=N_BOOTSTRAP,
)

quantile_coverage_results = []

for idx, q in enumerate(quantiles):
    quantile_label = f"q{int(q * 100)}"
    train_coverage = np.average(y_train <= y_train_quantile_pred[:, idx], weights=w_train)
    val_coverage = np.average(y_val <= y_val_quantile_pred[:, idx], weights=w_val)
    ci_lower, ci_upper = get_bootstrap_ci(val_quantile_metric_bootstrap_samples[f"{quantile_label}_coverage"])

    quantile_coverage_results.append({
        "Quantile": quantile_label,
        "Nominal Level": q,
        "Coverage (Val, 95% CI)": f"{val_coverage:.1%} [{ci_lower:.1%}, {ci_upper:.1%}]",
        "Calibration Error": val_coverage - q,
        "Coverage (Train)": train_coverage,
        "Empirical Coverage (Val)": val_coverage,
    })

quantile_coverage_df = pd.DataFrame(quantile_coverage_results)

display(
    quantile_coverage_df.style
    .hide()
    .pipe(add_table_caption, "XGBoost Quantile Calibration (Validation)")
    .format("{:.1%}", subset=[
        "Nominal Level",
        "Coverage (Train)",
    ])
    .format("{:+.1%}", subset=["Calibration Error"])
    .hide(axis="columns", subset=["Empirical Coverage (Val)"])
)

# %%
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot([0, 1], [0, 1], color="#4A4A4A", linestyle="--", linewidth=1.5, label="Perfect calibration")
ax.plot(
    quantile_coverage_df["Nominal Level"],
    quantile_coverage_df["Coverage (Train)"],
    marker="o",
    linewidth=2,
    color=SAMPLE_COLOR,
    label="Training",
)
ax.plot(
    quantile_coverage_df["Nominal Level"],
    quantile_coverage_df["Empirical Coverage (Val)"],
    marker="o",
    linewidth=2,
    color=POP_COLOR,
    label="Validation",
)

for _, row in quantile_coverage_df.iterrows():
    calibration_error = row["Calibration Error"]
    nominal_level = row["Nominal Level"]
    empirical_coverage = row["Empirical Coverage (Val)"]

    ax.vlines(
        x=nominal_level,
        ymin=nominal_level,
        ymax=empirical_coverage,
        color=POP_COLOR,
        linewidth=1.2,
        alpha=0.55,
    )
    ax.annotate(
        row["Quantile"],
        xy=(nominal_level, empirical_coverage),
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )

    ax.annotate(
        f"{calibration_error:+.1%}",
        xy=(nominal_level, nominal_level + calibration_error / 2),
        xytext=(-12, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color=POP_COLOR,
    )

percent_fmt = plt.FuncFormatter(lambda x, _: f"{x:.0%}")
ax.xaxis.set_major_formatter(percent_fmt)
ax.yaxis.set_major_formatter(percent_fmt)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Nominal Quantile Level")
ax.set_ylabel("Coverage (Weighted)")
ax.set_title("XGBoost Quantile Calibration: Nominal vs. Empirical Coverage", fontsize=13, fontweight="bold", pad=20)
ax.set_xticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
ax.grid(alpha=0.25)
ax.legend(frameon=False)

fig.text(
    0.01,
    0.01,
    "Note: Annotated percentages show calibration error on the validation data.",
    ha="left",
    fontsize=9,
    style="italic",
    color="#555555",
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Good Overall Calibration:</strong> q50, q75, and q90 are close to their nominal levels and the 95% confidence intervals contain the nominal levels. This confirms good calibration for the plan-around estimate and safety cushion.</li>
#         <li><strong>q25 is Slightly Conservative:</strong> q25 covers about 30.4% instead of 25% on the validation data. Since the nominal 25% fall outside the 95% CI [27.6%, 33.4%], this over-coverage is statistically significant. Decision: Monitor, Not Block. The q25 systematic shift is worth revisiting on the final holdout test set.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Product Metrics</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px; margin-bottom:12px;">
#     ℹ️ <b>Product-Facing Evaluation Metrics</b> <br>
#     The calibration section above checks whether each quantile endpoint is statistically trustworthy. This section translates those endpoints into the user-facing prediction estimates: plan-around estimate (q50), typical range (q25–q75), and safety cushion (q90). A prediction model must be both reliable and specific enough to be useful for out-of-pocket cost planning.
#     <br><br>
#     <b>Release Gate vs. Product Target</b> <br>
#     Defined release gates and product target performance for each metric, or clarified whenever the metric is only for additional diagnostic purposes.
#     <ul style="margin-top:8px">
#         <li><b>Release gate:</b> Minimum acceptable performance required to ship. Used for pass/fail decisions.</li>
#         <li><b>Product target:</b> Stronger performance that means the model has high useful for budgeting.</li>
#         <li><b>Diagnostic metric:</b> Monitored for model understanding, but not used as a pass/fail requirement.</li>
#     </ul>
#     <b>Product Coverage</b> <br>
#     Typical range coverage measures how often actual costs fall between q25 and q75. Safety cushion coverage is the q90 coverage already checked in the calibration section, repeated here because it is also a release metric for the product. Under-coverage means ranges are too narrow so that users encounter unexpectedly high costs more often than the app implies. Over-coverage means ranges are too wide, which is safer, but less useful for concrete budgeting.
#     <ul style="margin-top:8px">
#         <li><b>Typical range coverage (q25–q75):</b> Release gate = 45–55%; product target = 50%.</li>
#         <li><b>Safety cushion coverage (q90):</b> Release gate = 85–95%; product target = 90%.</li>
#     </ul>
#     <b>Interval Width</b> <br>
#     Interval width measures sharpness: how specific the predicted dollar range is. Coverage without reasonable width is not enough, because a model can meet coverage targets by returning overly broad ranges. Typical range width is q75 - q25. Safety cushion width is q90 - q50, measuring how much extra budget the safety cushion adds above the plan-around estimate. Reported widths are weighted averages across users, not the range for any single individual. 
#     <ul style="margin-top:8px">
#         <li><b>Typical range width (q25–q75):</b> Release gate &lt; \$1,500; product target &lt; \$1,000.</li>
#         <li><b>Safety cushion width (q50–q90):</b> Release gate &lt; \$3,500; product target &lt; \$2,500.</li>
#     </ul>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate XGBoost quantile regression metrics for the plan-around estimate (q50), typical range (q25-q75), and safety cushion (q90).
# </div>

# %%
xgb_quantile_metrics = load_metrics("../models/xgb_quantile_metrics.json")
metrics = xgb_quantile_metrics["XGBoost (Quantile)"]
y_val_pred_q25, y_val_pred_q50, y_val_pred_q75, y_val_pred_q90 = y_val_quantile_pred.T


def style_status_cells(value):
    """Color-code compact status cells."""
    if value == "Pass":
        return "background-color: #d4edda"
    if value == "Review":
        return "background-color: #fff3cd"
    if value == "Fail":
        return "background-color: #f8d7da"
    return ""


DOLLAR = r"\$"  # Escape dollar signs so Jupyter Notebook renders table cells correctly.


def format_metric_value(value, metric_format):
    """Format product metric values for display."""
    if metric_format == "percent":
        return f"{value:.1%}"
    if metric_format == "decimal":
        return f"{value:.2f}"
    if metric_format == "currency_0":
        return f"{DOLLAR}{value:,.0f}"
    return f"{DOLLAR}{value:,.2f}"


def format_metric_with_ci(value, samples, metric_format):
    """Format metric estimates with an inline bootstrap confidence interval."""
    ci_lower, ci_upper = get_bootstrap_ci(samples)
    return (
        f"{format_metric_value(value, metric_format)} "
        f"[{format_metric_value(ci_lower, metric_format)}, {format_metric_value(ci_upper, metric_format)}]"
    )


def format_metric_delta(train_value, val_value):
    """Format validation-vs-training relative delta."""
    if train_value == 0:
        return "n/a"
    return f"{((val_value - train_value) / train_value) * 100:+.1f}%"


PRODUCT_METRIC_GATES_AND_TARGETS = {
    "Plan Around MdAE (q50)": {
        "Release Gate": f"< {DOLLAR}500",
        "Product Target": f"< {DOLLAR}350",
    },
    "Plan Around MAE (q50)": {
        "Release Gate": "Diagnostic",
        "Product Target": "Diagnostic",
    },
    "Plan Around R² (q50)": {
        "Release Gate": "Diagnostic",
        "Product Target": "Diagnostic",
    },
    "Typical Range Coverage (q25–q75)": {
        "Release Gate": "45%–55%",
        "Product Target": "50%",
    },
    "Safety Cushion Coverage (q90)": {
        "Release Gate": "85%–95%",
        "Product Target": "90%",
    },
    "Typical Range Width": {
        "Release Gate": f"< {DOLLAR}1,500",
        "Product Target": f"< {DOLLAR}1,000",
    },
    "Safety Cushion Width": {
        "Release Gate": f"< {DOLLAR}3,500",
        "Product Target": f"< {DOLLAR}2,500",
    },
}


# Reshape for metrics display table: Metrics in index, Training/Validation in columns
product_metric_specs = [
    {
        "Metric": "Plan Around MdAE (q50)",
        "Training": metrics["train_q50_mdae"],
        "Validation": metrics["val_q50_mdae"],
        "Samples": val_quantile_metric_bootstrap_samples["q50_mdae"],
        "Format": "currency_2",
    },
    {
        "Metric": "Plan Around MAE (q50)",
        "Training": metrics["train_q50_mae"],
        "Validation": metrics["val_q50_mae"],
        "Samples": val_quantile_metric_bootstrap_samples["q50_mae"],
        "Format": "currency_2",
    },
    {
        "Metric": "Plan Around R² (q50)",
        "Training": metrics["train_q50_r2"],
        "Validation": metrics["val_q50_r2"],
        "Samples": val_quantile_metric_bootstrap_samples["q50_r2"],
        "Format": "decimal",
    },
    {
        "Metric": "Typical Range Coverage (q25–q75)",
        "Training": metrics["train_q25_q75_coverage"],
        "Validation": metrics["val_q25_q75_coverage"],
        "Samples": val_quantile_metric_bootstrap_samples["q25_q75_coverage"],
        "Format": "percent",
    },
    {
        "Metric": "Safety Cushion Coverage (q90)",
        "Training": metrics["train_q90_coverage"],
        "Validation": metrics["val_q90_coverage"],
        "Samples": val_quantile_metric_bootstrap_samples["q90_coverage"],
        "Format": "percent",
    },
    {
        "Metric": "Typical Range Width",
        "Training": metrics["train_q25_q75_width"],
        "Validation": metrics["val_q25_q75_width"],
        "Samples": val_quantile_metric_bootstrap_samples["q25_q75_width"],
        "Format": "currency_0",
    },
    {
        "Metric": "Safety Cushion Width",
        "Training": metrics["train_q50_q90_width"],
        "Validation": metrics["val_q50_q90_width"],
        "Samples": val_quantile_metric_bootstrap_samples["q50_q90_width"],
        "Format": "currency_0",
    },
]

metrics_display = []

for metric_spec in product_metric_specs:
    metric_format = metric_spec["Format"]
    validation_display = (
        format_metric_value(metric_spec["Validation"], metric_format)
        if metric_spec["Samples"] is None
        else format_metric_with_ci(metric_spec["Validation"], metric_spec["Samples"], metric_format)
    )

    metrics_display.append({
        "Metric": metric_spec["Metric"],
        "Validation (95% CI)": validation_display,
        "Release Gate": PRODUCT_METRIC_GATES_AND_TARGETS[metric_spec["Metric"]]["Release Gate"],
        "Product Target": PRODUCT_METRIC_GATES_AND_TARGETS[metric_spec["Metric"]]["Product Target"],
        "Training": format_metric_value(metric_spec["Training"], metric_format),
        "Val-Train Delta %": format_metric_delta(metric_spec["Training"], metric_spec["Validation"]),
    })

metrics_df = pd.DataFrame(metrics_display).set_index("Metric")
metrics_df.index.name = None

display(
    metrics_df.style
    .pipe(add_table_caption, "XGBoost Quantile Regression: Product Metrics")
)

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     💡 <b>Winkler Interval Score</b>
#     <br>
#     Coverage and width diagnose intervals separately. The Winkler interval score evaluates both together: it rewards narrow intervals when the actual cost falls inside the range and adds a penalty when the actual cost falls below q25 or above q75. Lower scores are better.
#     <br><br>
#     The interval skill score compares the model's Winkler interval score with a naive baseline that always predicts the same population q25–q75 interval from the training data. Positive skill means the model provides better typical ranges than a generic population range. 
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate the q25–q75 typical range using Winkler interval score and compare it with a naive population interval baseline.
# </div>

# %%
def interval_score(y_true, lower_pred, upper_pred, alpha, sample_weight=None):
    """
    Calculate the average interval score for prediction intervals.

    Lower scores are better. The score rewards narrow intervals when actual values fall 
    inside the interval and penalizes misses by distance outside the interval.

    Args:
        y_true (array-like): Actual values.
        lower_pred (array-like): Lower interval bounds.
        upper_pred (array-like): Upper interval bounds.
        alpha (float): Miss probability. For q25-q75, alpha=0.50.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Average interval score.
    """
    y_true = np.asarray(y_true)
    lower_pred = np.asarray(lower_pred)
    upper_pred = np.asarray(upper_pred)

    width = upper_pred - lower_pred
    below_penalty = np.where(y_true < lower_pred, (2 / alpha) * (lower_pred - y_true), 0)
    above_penalty = np.where(y_true > upper_pred, (2 / alpha) * (y_true - upper_pred), 0)
    scores = width + below_penalty + above_penalty

    if sample_weight is None:
        return np.mean(scores)
    return np.average(scores, weights=sample_weight)


def generate_interval_score_bootstrap_samples(
    y_true,
    lower_pred,
    upper_pred,
    weights,
    naive_lower,
    naive_upper,
    alpha=0.50,
    n_bootstrap=1000,
    random_state=RANDOM_STATE,
):
    """
    Generate bootstrap samples for model, naive, and skill-score interval metrics.

    This function returns the bootstrap distribution for each metric, not the
    summarized confidence intervals. Use get_bootstrap_ci() to convert
    any returned metric sample into a percentile confidence interval.

    Metrics returned:
        - XGBoost q25-q75 Winkler interval score
        - Naive population q25-q75 Winkler interval score
        - Interval skill score, comparing XGBoost against the naive baseline

    Args:
        y_true (array-like): Actual validation costs.
        lower_pred (array-like): Model lower interval bounds.
        upper_pred (array-like): Model upper interval bounds.
        weights (array-like): Validation survey weights.
        naive_lower (float): Naive lower interval bound.
        naive_upper (float): Naive upper interval bound.
        alpha (float): Miss probability.
        n_bootstrap (int): Number of bootstrap resamples.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Recomputed metric samples for model, naive, and skill-score interval metrics.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    lower_pred = np.asarray(lower_pred)
    upper_pred = np.asarray(upper_pred)
    weights = np.asarray(weights)
    n_obs = len(y_true)

    bootstrap_samples = {
        "model_interval_score": np.empty(n_bootstrap),
        "naive_interval_score": np.empty(n_bootstrap),
        "interval_skill_score": np.empty(n_bootstrap),
    }

    for sample_idx in range(n_bootstrap):
        row_idx = rng.integers(0, n_obs, size=n_obs)
        y_boot = y_true[row_idx]
        lower_boot = lower_pred[row_idx]
        upper_boot = upper_pred[row_idx]
        w_boot = weights[row_idx]

        model_score = interval_score(y_boot, lower_boot, upper_boot, alpha, sample_weight=w_boot)
        naive_score = interval_score(
            y_boot,
            np.full_like(y_boot, naive_lower, dtype=float),
            np.full_like(y_boot, naive_upper, dtype=float),
            alpha,
            sample_weight=w_boot,
        )

        bootstrap_samples["model_interval_score"][sample_idx] = model_score
        bootstrap_samples["naive_interval_score"][sample_idx] = naive_score
        bootstrap_samples["interval_skill_score"][sample_idx] = 1 - (model_score / naive_score)

    return bootstrap_samples


naive_q25 = weighted_quantile(y_train, w_train, 0.25)
naive_q75 = weighted_quantile(y_train, w_train, 0.75)

val_interval_score_bootstrap_samples = generate_interval_score_bootstrap_samples(
    y_val,
    y_val_pred_q25,
    y_val_pred_q75,
    w_val,
    naive_q25,
    naive_q75,
    n_bootstrap=N_BOOTSTRAP,
)

model_interval_score = interval_score(
    y_val,
    y_val_pred_q25,
    y_val_pred_q75,
    alpha=0.50,
    sample_weight=w_val,
)
naive_interval_score = interval_score(
    y_val,
    np.full_like(y_val, naive_q25, dtype=float),
    np.full_like(y_val, naive_q75, dtype=float),
    alpha=0.50,
    sample_weight=w_val,
)
interval_skill_score = 1 - (model_interval_score / naive_interval_score)

interval_score_specs = [
    {
        "Metric": "XGBoost Winkler Interval Score",
        "Validation": model_interval_score,
        "Samples": val_interval_score_bootstrap_samples["model_interval_score"],
        "Format": "currency_0",
    },
    {
        "Metric": "Naive Winkler Interval Score",
        "Validation": naive_interval_score,
        "Samples": val_interval_score_bootstrap_samples["naive_interval_score"],
        "Format": "currency_0",
    },
    {
        "Metric": "Interval Skill Score",
        "Validation": interval_skill_score,
        "Samples": val_interval_score_bootstrap_samples["interval_skill_score"],
        "Format": "percent",
    },
]

interval_score_display = []

for metric_spec in interval_score_specs:
    interval_score_display.append({
        "Metric": metric_spec["Metric"],
        "Estimate (95% CI)": format_metric_with_ci(
            metric_spec["Validation"],
            metric_spec["Samples"],
            metric_spec["Format"],
        ),
    })

interval_score_df = pd.DataFrame(interval_score_display).set_index("Metric")
interval_score_df.index.name = None

display(
    interval_score_df.style
    .pipe(add_table_caption, "Typical Range: Winkler Interval Score (Validation)")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Good Interval Calibration:</strong> The model achieves 48.6% coverage for the typical range (Target: 50%) and 88.7% for the safety cushion (Target: 90%). Their approximate 95% bootstrap CIs ([45.2%, 51.6%] and [86.7%, 90.6%]) fall within the product release tolerances (45%–55% and 85%–95%), supporting release readiness pending final holdout test confirmation.</li>
#         <li><strong>Excellent Generalization:</strong> The training values for MdAE (\$229.14), MAE (\$957.21), typical range width (\$917), and safety cushion width (\$2,019) all fall within the validation 95% bootstrap confidence intervals. This shows that the model generalizes exceptionally well and shows no significant overfitting.</li>
#         <li><strong>Manageable Typical Range Width for Budgeting:</strong> An average typical range width of \$891 (95% CI: [\$851, \$929]) provides users with a narrow, manageable window for standard HSA/FSA planning, while the \$1,980 safety cushion width (95% CI: [\$1,904, \$2,057]) offers a stable, realistic buffer for emergency fund planning.</li>
#         <li><strong>Typical Range Adds Value:</strong> The Winkler interval score for the q25–q75 typical range improves from \$3,707 for the naive population interval to \$3,376 for XGBoost. This corresponds to an 8.9% interval skill score. This confirms the typical range is not just well calibrated, it is more useful than showing every user the same generic range.</li>
#         <li><strong>MdAE vs. MAE:</strong> The large gap between MdAE (\$244.54) and MAE (\$954.70) reinforces that while the model is very precise for the "typical" user, rare high-cost outliers continue to drive the mean error, further justifying a "plan around + safety cushion" approach over simple point estimates.</li>
#         <li><strong>CQR Not Needed at Validation Stage:</strong> Conformalized quantile regression is not triggered because validation coverage falls within product tolerances. Revisit CQR if holdout test coverage falls outside release targets or if the q25 conservative bias worsens.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Heteroscedasticity</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Plot residuals vs. predicted values for the XGBoost quantile regression model's q50 (median) estimate on the validation set. Compare with the tuned XGBoost point-estimate model.
# </div>

# %%
# Load predictions 
xgb_tuned_pred = load_model("../models/xgb_tuned_predictions.joblib", verbose=False)
y_val_quantile_pred = load_model("../models/xgb_quantile_predictions.joblib", verbose=False)
y_val_pred_q50 = y_val_quantile_pred[:, 1]  # q50 is at index 1

# Plot heteroscedasticity of quantile vs. point-estimate model side-by-side
plot_residuals_vs_predicted(
    y_val, 
    {
        "XGBoost (Tuned Point-Estimate)": xgb_tuned_pred,
        "XGBoost Quantile (q50)": y_val_pred_q50
    }, 
    w_val,
    n_cols=2,
    save_to_file="../figures/evaluation/quantile_heteroscedasticity.png"
)


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Identical Error Profiles (No Degradation):</strong> The side-by-side comparison shows virtually identical residual spreads and binned median trends. This confirms that jointly training a single multi-quantile model (minimizing pinball loss across 4 quantiles) does not degrade the median prediction quality compared to a dedicated point-estimate model.</li>
#         <li><strong>Unbiased Median Predictions:</strong> For both models, the binned median residual line remains close to zero across the entire prediction range, demonstrating that predictions are stable and free of systematic bias for typical healthcare costs.</li>
#         <li><strong>Shared Fan-Shaped Uncertainty:</strong> Both plots display a widening "fan shape" (heteroscedasticity) and a heavy upward skew of positive residuals. This indicates that predicting out-of-pocket costs becomes increasingly uncertain as expected health risk rises, and both models systematically underpredict extreme catastrophic expenditures (outliers).</li>
#         <li><strong>Justification for Range Modeling:</strong> The presence of heteroscedasticity confirms that a single point estimate (q50) is insufficient for high-cost users. A range model with prediction intervals (q25-q75) and safety cushions (q90) is necessary to communicate this uncertainty to users.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Stratified Error Analysis</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px; margin-bottom:12px;">
#     ℹ️ <b>Subgroup Performance: Typical Range and Safety Cushion</b> <br>
#     Stratified error analysis for quantile regression extends the point-estimate stratified analysis from a single error metric to a comprehensive audit of central accuracy, interval calibration, and prediction uncertainty. The goal is to confirm that the model's plan-around estimate (q50), typical range (q25-q75), and safety cushion (q90) remain reliable and actionable across important user segments.
#     <ul style="margin-top:8px">
#         <li><b>Reuse stratification:</b> Keep the same reliability and fairness audit groups used for point-estimate models.</li>
#         <li><b>Quantile-specific predicted cost groups:</b> Replace the point-estimate <code>Predicted Costs</code> grouping with <code>Predicted Plan Around Costs</code> based on q50, and add <code>Predicted Safety Cushion Costs</code> based on q90 to audit the "plan up to" amount directly.</li>
#         <li><b>Reliability groups:</b> Actual cost tier, predicted plan-around cost tier (q50), predicted safety-cushion cost tier (q90), physical health, insurance status, and chronic condition count.</li>
#         <li><b>Fairness audit groups:</b> Sex, age group, race/ethnicity, mental health, family income, education, region, and walking limitation.</li>
#         <li><b>Metrics by group:</b> Plan-around MdAE (q50), typical range coverage/width (q25-q75), and safety cushion coverage/width (q90 and q50-q90).</li>
#         <li><b>Context columns:</b> Include sample size and weighted median actual cost to distinguish unfair or unreliable model behavior from genuinely higher-cost, higher-variance population segments.</li>
#         <li><b>Subgroup review bands:</b> Overall coverage targets are release gates; subgroup review bands are diagnostic ranges, not pass/fail gates. For subgroups with sufficient sample size (n ≥ 30), q25–q75 coverage should generally stay within 40–60%, and q90 coverage should generally stay within 80–97%. Subgroups with n &lt; 30 are flagged as review-only because their coverage estimates are less stable.</li>
#         <li><b>Width review guideline:</b> Wider intervals are acceptable for genuinely higher-risk groups, but concerning for low-risk groups if they do not improve coverage or reflect clearly higher actual costs.</li>
#         <li><b>Diagnostic flags:</b> Add diagnostic quality-control flags to identify groups that may need manual review before deployment. These flags are not automatic failure criteria; they highlight groups where the plan-around estimate, typical range, or safety cushion may be unreliable or insufficiently useful.</li>
#         <li><b>Interpretation principle:</b> Coverage alone is not enough. A useful interval must be calibrated and narrow enough to support budgeting decisions; a wide interval is acceptable only when it reflects real uncertainty for a higher-risk user group.</li>
#     </ul>
#     <table style="width:100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;">
#         <thead>
#             <tr style="background-color:#d0e7fa;">
#                 <th style="padding:8px; border:1px solid #b7d7ef; text-align:left;">Flag</th>
#                 <th style="padding:8px; border:1px solid #b7d7ef; text-align:left;">Review Condition</th>
#                 <th style="padding:8px; border:1px solid #b7d7ef; text-align:left;">Why It Matters</th>
#             </tr>
#         </thead>
#         <tbody>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>Small Sample</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">Subgroup sample size &lt; 30</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">Coverage and width metrics for this subgroup are less stable and should be treated as review-only subgroup metrics.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>Typical Range Undercoverage</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">q25–q75 coverage &lt; 40%</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">The typical range for this subgroup is too narrow and misses too many actual costs.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>Typical Range Overcoverage</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">q25–q75 coverage &gt; 60%</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">The typical range for this subgroup may be wider than needed for practical budgeting.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>Safety Cushion Undercoverage</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">q90 coverage &lt; 80%</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">The safety cushion for that subgroup underestimates actual costs, thus under-warning users.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>High Plan-Around Error</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">q50 MdAE &gt; 3× the overall MdAE</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">The plan-around estimate for that subgroup has an unusually high prediction error compared to the typical prediction error.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>Wide Low-Cost Typical Range</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">Low median actual cost and q25–q75 width &gt; overall average</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">Low-cost users may receive a typical range that is too broad to be actionable.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d0e7fa;"><code>Wide Low-Cost Cushion</code></td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">Low median actual cost and q50–q90 width &gt; overall average</td>
#                 <td style="padding:8px; border:1px solid #d0e7fa;">Low-cost users may be encouraged to over-budget without clear justification.</td>
#             </tr>
#         </tbody>
#     </table>
# </div>

# %%
# --- Stratification Setup ---
chronic_cols = list(CHRONIC_CONDITIONS.keys())
age_bins = [18, 35, 50, 65, 120]
age_labels = ["18-34", "35-49", "50-64", "65+"]

# Note: Use the same bins as in the point-estimate stratified analysis, but merge the Top 5% cost bins (for n>30 subgroup sample size)
COST_BIN_LABELS = {
    0: "Zero Costs",
    1: "Low Spend (0-50%)",
    2: "Moderate (50-80%)",
    3: "High Spend (80-95%)",
    4: "Very High Spend (Top 5%)"
}
actual_cost_bin_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}

# For predicted costs, also merge the Zero Costs bin (0) into Low Spend (1)
# Reason: To achieve n>30 subgroup sample size because model predictions are almost never zero
predicted_cost_bin_map = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}

quantile_reliability_configs = [
    {"col": "ACTUAL_COSTS", "label": "Out-of-Pocket Costs (Actual)", "category_map": COST_BIN_LABELS},
    {"col": "PREDICTED_MEDIAN_COSTS", "label": "Out-of-Pocket Costs (Plan Around, q50)", "category_map": COST_BIN_LABELS},
    {"col": "PREDICTED_CUSHION_COSTS", "label": "Out-of-Pocket Costs (Safety Cushion, q90)", "category_map": COST_BIN_LABELS},
    {"col": "RTHLTH31", "label": DISPLAY_LABELS["RTHLTH31"], "category_map": CATEGORY_LABELS_EDA["RTHLTH31"]},
    {"col": "INSCOV23", "label": DISPLAY_LABELS["INSCOV23"], "category_map": CATEGORY_LABELS_EDA["INSCOV23"]},
    {"col": "CHRONIC_COUNT_GRP", "label": DISPLAY_LABELS["CHRONIC_COUNT"], "category_map": None}
]

quantile_fairness_configs = [
    {"col": "SEX", "label": DISPLAY_LABELS["SEX"], "category_map": CATEGORY_LABELS_EDA["SEX"]},
    {"col": "AGE_GRP", "label": "Age Group", "category_map": None},
    {"col": "RACETHX", "label": DISPLAY_LABELS["RACETHX"], "category_map": CATEGORY_LABELS_EDA["RACETHX"]},
    {"col": "MNHLTH31", "label": DISPLAY_LABELS["MNHLTH31"], "category_map": CATEGORY_LABELS_EDA["MNHLTH31"]},
    {"col": "POVCAT23", "label": DISPLAY_LABELS["POVCAT23"], "category_map": CATEGORY_LABELS_EDA["POVCAT23"]},
    {"col": "HIDEG", "label": DISPLAY_LABELS["HIDEG"], "category_map": CATEGORY_LABELS_EDA["HIDEG"]},
    {"col": "REGION23", "label": DISPLAY_LABELS["REGION23"], "category_map": CATEGORY_LABELS_EDA["REGION23"]},
    {"col": "WLKLIM31", "label": DISPLAY_LABELS["WLKLIM31"], "category_map": CATEGORY_LABELS_EDA["WLKLIM31"]},
]

quantile_stratified_configs = quantile_reliability_configs + quantile_fairness_configs


def add_quantile_stratification_columns(df_raw, y_true, y_pred_q50, y_pred_q90):
    """
    Add reliability and fairness audit columns to a raw split dataframe (like val or test).
    """
    df_raw = df_raw.copy()
    df_raw["CHRONIC_COUNT"] = df_raw[chronic_cols].sum(axis=1).astype(int)
    df_raw["CHRONIC_COUNT_GRP"] = df_raw["CHRONIC_COUNT"].apply(
        lambda x: f"{x} Condition" if x == 1 else (f"{x} Conditions" if x < 4 else "4+ Conditions")
    )
    df_raw["AGE_GRP"] = pd.cut(df_raw["AGE23X"], bins=age_bins, labels=age_labels, right=False)
    df_raw["ACTUAL_COSTS"] = create_stratification_bins(y_true).map(actual_cost_bin_map)
    df_raw["PREDICTED_MEDIAN_COSTS"] = create_stratification_bins(y_pred_q50).map(predicted_cost_bin_map)
    df_raw["PREDICTED_CUSHION_COSTS"] = create_stratification_bins(y_pred_q90).map(predicted_cost_bin_map)
    return df_raw


def create_quantile_subgroup_df(df_raw, y_true, weights, y_pred_quantiles, configs):
    """
    Build subgroup metrics and diagnostic flags for a quantile model audit.
    """
    y_pred_q25, y_pred_q50, y_pred_q75, y_pred_q90 = y_pred_quantiles
    stratified_results = []

    for config in configs:
        col = config["col"]
        label = config["label"]
        category_map = config["category_map"]
        col_bins = df_raw[col]

        for group in sorted(col_bins.dropna().unique()):
            mask = (col_bins == group)
            y_group = y_true[mask]
            w_group = weights[mask]
            q25_group = y_pred_q25[mask]
            q50_group = y_pred_q50[mask]
            q75_group = y_pred_q75[mask]
            q90_group = y_pred_q90[mask]

            stratified_results.append({
                "Column": label,
                "Group": category_map.get(int(group), group) if category_map else group,
                "Sample Size": mask.sum(),
                "Median Actual Cost": weighted_quantile(y_group, w_group, 0.5),
                "Predicted Typical Low (q25)": np.average(q25_group, weights=w_group),
                "Predicted Plan Around (q50)": np.average(q50_group, weights=w_group),
                "Predicted Typical High (q75)": np.average(q75_group, weights=w_group),
                "Predicted Safety Cushion (q90)": np.average(q90_group, weights=w_group),
                "Plan Around MdAE (q50)": weighted_median_absolute_error(y_group, q50_group, sample_weight=w_group),
                "Typical Range Coverage (q25–q75)": np.average((y_group >= q25_group) & (y_group <= q75_group), weights=w_group),
                "Typical Range Width (q25–q75)": np.average(q75_group - q25_group, weights=w_group),
                "Safety Cushion Coverage (q90)": np.average(y_group <= q90_group, weights=w_group),
                "Safety Cushion Width (q50–q90)": np.average(q90_group - q50_group, weights=w_group),
            })

    subgroup_df = pd.DataFrame(stratified_results)
    overall_q50_mdae = weighted_median_absolute_error(y_true, y_pred_q50, sample_weight=weights)
    overall_median_actual_cost = weighted_quantile(y_true, weights, 0.5)
    overall_range_width = np.average(y_pred_q75 - y_pred_q25, weights=weights)
    overall_cushion_width = np.average(y_pred_q90 - y_pred_q50, weights=weights)

    def get_quantile_reliability_flags(subgroup):
        flags = []

        if subgroup["Sample Size"] < 30:
            flags.append("Small Sample")

        if subgroup["Typical Range Coverage (q25–q75)"] < 0.40:
            flags.append("Typical-Range Undercoverage")
        elif subgroup["Typical Range Coverage (q25–q75)"] > 0.60:
            flags.append("Typical-Range Overcoverage")

        if subgroup["Safety Cushion Coverage (q90)"] < 0.80:
            flags.append("Safety-Cushion Undercoverage")
        elif subgroup["Safety Cushion Coverage (q90)"] > 0.97:
            flags.append("Safety-Cushion Overcoverage")

        if subgroup["Plan Around MdAE (q50)"] > 3 * overall_q50_mdae:
            flags.append("High Plan-Around Error")

        is_low_cost_group = subgroup["Median Actual Cost"] <= overall_median_actual_cost
        if is_low_cost_group and subgroup["Typical Range Width (q25–q75)"] > overall_range_width:
            flags.append("Wide Low-Cost Typical-Range")
        if is_low_cost_group and subgroup["Safety Cushion Width (q50–q90)"] > overall_cushion_width:
            flags.append("Wide Low-Cost Safety-Cushion")

        return ", ".join(flags) if flags else "None"

    subgroup_df["Reliability Flags"] = subgroup_df.apply(get_quantile_reliability_flags, axis=1)
    return subgroup_df


# --- Prepare Validation Audit ---
print("Recovering raw validation features...")
df_raw_val, y_val_audit, w_val_audit = prepare_human_readable_split_data(VAL_MODEL_READY_DATA_PATH, "validation")

print("Loading XGBoost quantile predictions...")
y_val_quantile_pred = load_model("../models/xgb_quantile_predictions.joblib", verbose=False)
y_val_pred_q25_audit, y_val_pred_q50_audit, y_val_pred_q75_audit, y_val_pred_q90_audit = [
    pd.Series(values, index=y_val_audit.index)
    for values in y_val_quantile_pred.T
]
print(f"  Loaded predictions for {y_val_quantile_pred.shape[1]} quantiles on {len(y_val_quantile_pred)} validation set samples")

df_raw_val = add_quantile_stratification_columns(
    df_raw_val,
    y_val_audit,
    y_val_pred_q50_audit,
    y_val_pred_q90_audit,
)
quantile_subgroup_df = create_quantile_subgroup_df(
    df_raw_val,
    y_val_audit,
    w_val_audit,
    (y_val_pred_q25_audit, y_val_pred_q50_audit, y_val_pred_q75_audit, y_val_pred_q90_audit),
    quantile_stratified_configs,
)

display_columns = [
    "Column",
    "Group",
    "Sample Size",
    "Median Actual Cost",
    "Plan Around MdAE (q50)",
    "Typical Range Coverage (q25–q75)",
    "Typical Range Width (q25–q75)",
    "Safety Cushion Coverage (q90)",
    "Safety Cushion Width (q50–q90)",
    "Reliability Flags",
]

display(
    quantile_subgroup_df[display_columns]
    .style
    .hide()
    .pipe(add_table_caption, "XGBoost Quantile Regression: Stratified Error Analysis (Validation)")
    .format("{:,}", subset=["Sample Size"])
    .format("${:,.2f}", subset=["Median Actual Cost", "Plan Around MdAE (q50)", "Typical Range Width (q25–q75)", "Safety Cushion Width (q50–q90)"])
    .format("{:.1%}", subset=["Typical Range Coverage (q25–q75)", "Safety Cushion Coverage (q90)"])
    .apply(lambda row: ["background-color: #fff3cd" if row["Reliability Flags"] != "None" else "" for _ in row], axis=1)
)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>XGBoost Quantile Regression: Subgroup Reliability and Fairness</strong> <br>
#     📌 Visualize coverage and interval width side-by-side for each stratification column. Coverage measures calibration; width measures practical usefulness. Both are needed to determine whether the typical range and safety cushion are reliable for subgroup-level budgeting.
# </div>

# %%
def plot_quantile_subgroup_performance(df, column_labels, title, save_to_file=None):
    """
    Visualizes quantile regression performance across subgroups.

    Each subplot row is one stratification column. The left panel shows coverage for 
    the typical range (q25-q75) and safety cushion (q90). The right panel shows
    the corresponding range widths in USD.

    Args:
        df (pd.DataFrame): Stratified error analysis dataframe of quantile regression.
        column_labels (list): Ordered list of column labels to include.
        title (str): Main figure title.
        save_to_file (str, optional): Full path and filename to save the plot.
    """
    plot_data = df[df["Column"].isin(column_labels)].copy()
    n_rows = len(column_labels)
    max_width = plot_data[["Typical Range Width (q25–q75)", "Safety Cushion Width (q50–q90)"]].max().max()
    width_axis_max = np.ceil(max_width / 500) * 500

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(16, 4 * n_rows),
        squeeze=False,  # Guarantees axes is always a 2D array
        gridspec_kw={"width_ratios": [1.0, 1.15]}  # Right panel 15% wider than left panel
    )

    bar_height = 0.36
    colors = {
        "Range": TYPICAL_RANGE_COLOR,
        "Cushion": SAFETY_CUSHION_COLOR,
    }
    percent_fmt = plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    currency_fmt = plt.FuncFormatter(lambda x, _: f"${x:,.0f}")

    for row_idx, column_label in enumerate(column_labels):
        col_data = plot_data[plot_data["Column"] == column_label].copy()
        y_pos = np.arange(len(col_data))

        coverage_ax = axes[row_idx, 0]
        width_ax = axes[row_idx, 1]

        # Coverage (left panel)
        coverage_ax.axvspan(0.40, 0.60, color=TYPICAL_RANGE_COLOR, alpha=0.12, zorder=0)
        coverage_ax.axvspan(0.80, 0.97, color=SAFETY_CUSHION_COLOR, alpha=0.10, zorder=0)
        coverage_ax.axvline(0.50, color=TYPICAL_RANGE_COLOR, linestyle="--", linewidth=1, alpha=0.8)
        coverage_ax.axvline(0.90, color=SAFETY_CUSHION_COLOR, linestyle="--", linewidth=1, alpha=0.8)

        coverage_bars_range = coverage_ax.barh(
            y_pos - bar_height / 2,
            col_data["Typical Range Coverage (q25–q75)"],
            height=bar_height,
            color=colors["Range"],
            label="Typical Range (q25–q75)"
        )
        coverage_bars_cushion = coverage_ax.barh(
            y_pos + bar_height / 2,
            col_data["Safety Cushion Coverage (q90)"],
            height=bar_height,
            color=colors["Cushion"],
            label="Safety Cushion (q90)"
        )

        coverage_ax.bar_label(
            coverage_bars_range,
            labels=[f"{v:.1%}" for v in col_data["Typical Range Coverage (q25–q75)"]],
            padding=3,
            fontsize=8
        )
        coverage_ax.bar_label(
            coverage_bars_cushion,
            labels=[f"{v:.1%}" for v in col_data["Safety Cushion Coverage (q90)"]],
            padding=3,
            fontsize=8
        )
        coverage_ax.set_xlim(0, 1.01)  # x-axis up to 101% to prevent clipping
        coverage_ax.xaxis.set_major_formatter(percent_fmt)
        coverage_ax.set_title("Coverage", fontsize=12, fontweight="bold")

        # Width (right panel)
        width_bars_range = width_ax.barh(
            y_pos - bar_height / 2,
            col_data["Typical Range Width (q25–q75)"],
            height=bar_height,
            color=colors["Range"],
            label="Typical Range (q25–q75)"
        )
        width_bars_cushion = width_ax.barh(
            y_pos + bar_height / 2,
            col_data["Safety Cushion Width (q50–q90)"],
            height=bar_height,
            color=colors["Cushion"],
            label="Safety Cushion (q50–q90)"
        )

        width_ax.bar_label(
            width_bars_range,
            labels=[f"${v:,.0f}" for v in col_data["Typical Range Width (q25–q75)"]],
            padding=3,
            fontsize=8
        )
        width_ax.bar_label(
            width_bars_cushion,
            labels=[f"${v:,.0f}" for v in col_data["Safety Cushion Width (q50–q90)"]],
            padding=3,
            fontsize=8
        )
        width_ax.xaxis.set_major_formatter(currency_fmt)
        width_ax.set_xlim(0, width_axis_max)
        width_ax.set_title("Width", fontsize=12, fontweight="bold")

        # Shared row formatting
        coverage_ax.set_yticks(y_pos)
        yticklabels = [
            f"{str(g).split(' (')[0]}\nn={n:,} | ${med:,.0f}"
            for g, n, med in zip(col_data["Group"], col_data["Sample Size"], col_data["Median Actual Cost"])
        ]
        coverage_ax.set_yticklabels(yticklabels, fontsize=9)
        width_ax.set_yticks(y_pos)
        width_ax.set_yticklabels([])
        coverage_ax.invert_yaxis()
        width_ax.invert_yaxis()
        coverage_ax.set_ylabel(column_label, fontsize=11, fontweight="bold", labelpad=12)
        width_ax.set_ylabel("")

        for ax in [coverage_ax, width_ax]:
            ax.grid(axis="x", alpha=0.20)
            sns.despine(ax=ax, left=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=True)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.015)
    
    plt.tight_layout(rect=[0, 0.02, 1, 1], h_pad=2.0, w_pad=1.4)

    # Add footnote
    # Compute final layout, then align footnote with the first subplot's leftmost label
    fig.canvas.draw()
    footnote_x = axes[0, 0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0

    # Add footnote at the bottom, aligned with the leftmost label/title
    fig.text(
        footnote_x, 
        0.01, 
        "Note: Dollar values in subgroup labels (e.g., $269) represent the actual median out-of-pocket costs of that subgroup.", 
        fontsize=9, 
        style="italic", 
        color="#555555",
        ha="left"
    )

    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)

    plt.show()


quantile_reliability_labels = [c["label"] for c in quantile_reliability_configs]
plot_quantile_subgroup_performance(
    quantile_subgroup_df,
    quantile_reliability_labels,
    "XGBoost Quantile Regression: Subgroup Reliability (Validation)",
    save_to_file="../figures/evaluation/xgb_quantile_validation_subgroup_reliability.png"
)

quantile_fairness_labels = [c["label"] for c in quantile_fairness_configs]
plot_quantile_subgroup_performance(
    quantile_subgroup_df,
    quantile_fairness_labels,
    "XGBoost Quantile Regression: Subgroup Fairness (Validation)",
    save_to_file="../figures/evaluation/xgb_quantile_validation_subgroup_fairness.png"
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Actual Cost Tiers:</strong> Actual high spenders have typical range coverage of only 10.9% and cushion coverage of 48.7%; for the top 5% (median \$10,086) both drop to 0.0%, confirming the model severely underpredicts extreme costs. Conversely, the zero-cost group (n=307) shows only 24.9% typical range coverage but 100.0% cushion coverage. This is expected as individuals with \$0 actual costs rarely fall inside a range centered above zero. Low spenders show overcoverage at 70.0%, indicating that predicted ranges are wider than necessary for this group.</li>
#         <li><strong>Predicted Cost Tiers:</strong> When stratified by predicted plan-around (q50) or predicted cushion (q90), all groups are well-calibrated, remaining within subgroup review bands (typical range: 41.4%–58.2%; safety cushion: 86.5%–91.4%). This confirms that the model's own confidence segmentation is well-calibrated regardless of predicted spend level.</li>
#         <li><strong>Insurance:</strong> Private (49.0% / 90.8%) and public (51.2% / 89.2%) insurance groups are well-calibrated. However, the uninsured group has a typical range coverage of only 34.9% (below the lower 40% review-band boundary), indicating that predicted ranges are too narrow for the uninsured. This is likely because uninsured costs are more volatile and harder to predict from available features.</li>
#         <li><strong>Sex, Age, Race:</strong> Coverages are well within subgroup review bands (Typical: 44.7%–56.9%; Safety Cushion: 85.0%–95.2%). No demographic group is flagged for systematic calibration bias against protected subgroups.</li>
#         <li><strong>Width:</strong> Prediction interval widths scale proportionally with group risk: older cohorts (65+: \$1,408 typical range width), individuals with walking limitations (\$1,614), and groups with poorer health status receive wider intervals, aligning with their higher median costs and cost variance.</li>
#         <li><strong>Education:</strong> Doctorate degree holders fall below the lower 80% safety-cushion review-band boundary (78.1%, n=53). This group may contain higher cost variance not fully captured by available features.</li>
#         <li><strong>Mental Health:</strong> Individuals with 'Poor' perceived mental health (n=29) show typical range undercoverage (37.8%) and cushion undercoverage (71.7%). However, with n < 30, this group triggers the Small Sample flag and its metrics should be treated as review-only subgroup metrics rather than definitive evidence of miscalibration.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>XGBoost Quantile Regression: Predicted Costs by Subgroup</strong> <br>
#     📌 Visualize predicted out-of-pocket costs by subgroup. Unlike the coverage/width audit above, this diagnostic shows where the model places each subgroup on the dollar costs scale: plan-around estimate (q50), typical range (q25-q75), and safety cushion (q90). Use this to inspect risk separation and whether uncertainty grows in clinically plausible places.
# </div>

# %%
def _format_cost_label(value):
    # Escape $ so Matplotlib does not treat it as mathtext delimiters.
    return f"\\${value:,.0f}"


def _estimate_figure_x_right(plot_data, column_labels):
    """Provisional shared x-limit for cramped-label logic before final bbox trim."""
    col_max_q90 = plot_data.loc[plot_data["Column"].isin(column_labels), "Predicted Safety Cushion (q90)"].max()
    label_pad = max(col_max_q90 * 0.10, 400)
    x_right = np.ceil((col_max_q90 + label_pad) / 500) * 500
    if x_right < 1000:
        x_right = max(np.ceil((col_max_q90 + label_pad) / 100) * 100, 500)
    return x_right


def _panel_content_right(ax, renderer):
    """Rightmost data-coordinate extent of rendered artists in one subplot."""
    content_right = 0.0
    for artist in ax.get_children():
        try:
            bbox = artist.get_window_extent(renderer)
        except (ValueError, TypeError):
            continue
        bbox_data = bbox.transformed(ax.transData.inverted())
        if np.isfinite(bbox_data.x1):
            content_right = max(content_right, bbox_data.x1)
    return content_right


def _round_x_right(content_right, pad_frac=0.04, min_pad=200):
    if content_right <= 0:
        return None
    pad = max(content_right * pad_frac, min_pad)
    x_right = np.ceil((content_right + pad) / 500) * 500
    if x_right < 1000:
        x_right = max(np.ceil((content_right + pad) / 100) * 100, 500)
    return x_right


def _finalize_figure_xlim(axes, fig, pad_frac=0.02, min_pad=200):
    """Apply one shared xlim to every subplot so x-axes are directly comparable."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_content_right = max(_panel_content_right(ax, renderer) for ax in axes)
    x_right = _round_x_right(max_content_right, pad_frac=pad_frac, min_pad=min_pad)
    if x_right is None:
        return
    for ax in axes:
        ax.set_xlim(0, x_right)


def _annotate_quantile_timeline(ax, y, q25, q50, q75, q90, x_right):
    """
    Annotate a single quantile timeline. Uses split labels when there is room;
    falls back to one compact right-aligned summary when labels would overlap.
    """
    range_mid = (q25 + q75) / 2
    plan_label = _format_cost_label(q50)
    range_label = f"{_format_cost_label(q25)}-{_format_cost_label(q75)}"
    cushion_label = _format_cost_label(q90)
    span = max(q90 - q25, 1.0)
    min_sep = x_right * 0.07
    cramped = (
        span < min_sep * 2.5
        or (q75 + min_sep > q90)
        or (q50 + min_sep > range_mid and q50 - min_sep < range_mid)
    )

    if cramped:
        summary_x = q90 + max(x_right * 0.015, 75)
        ax.text(
            summary_x,
            y,
            f"{plan_label} | {range_label} | {cushion_label}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
            clip_on=True,
        )
        return

    ax.annotate(
        plan_label,
        (q50, y),
        xytext=(0, -11),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        clip_on=True,
    )
    ax.annotate(
        range_label,
        (range_mid, y),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TYPICAL_RANGE_COLOR,
        clip_on=True,
    )
    ax.annotate(
        cushion_label,
        (q90, y),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        color=SAFETY_CUSHION_COLOR,
        clip_on=True,
    )


def plot_quantile_subgroup_predictions(df, column_labels, title, save_to_file=None):
    """
    Visualizes quantile regression predicted costs across subgroups.

    Each subgroup is a single horizontal timeline (left to right):
    teal q25–q75 (typical range), purple q75–q90 (upper tail to cushion),
    a black marker at q50 (plan around), and a diamond at q90 (safety cushion).

    Annotations mirror the app copy: plan-around below the line, typical range
    endpoints above the teal segment, and q90 at the cushion marker. Cramped rows use
    one compact right-aligned summary to avoid overlap.

    Args:
        df (pd.DataFrame): Stratified error analysis dataframe of quantile regression.
        column_labels (list): Ordered list of column labels to include.
        title (str): Main figure title.
        save_to_file (str, optional): Full path and filename to save the plot.
    """
    from matplotlib.lines import Line2D

    plot_data = df[df["Column"].isin(column_labels)].copy()
    n_rows = len(column_labels)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(16, 3.0 * n_rows),
        squeeze=False,
    )

    currency_fmt = plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    cap_half_height = 0.14
    plan_color = "#222222"
    figure_x_right = _estimate_figure_x_right(plot_data, column_labels)
    panel_axes = [axes[row_idx, 0] for row_idx in range(n_rows)]

    for row_idx, column_label in enumerate(column_labels):
        ax = axes[row_idx, 0]
        col_data = plot_data[plot_data["Column"] == column_label].copy()
        y_pos = np.arange(len(col_data))
        ax.set_xlim(0, figure_x_right)

        for tick_idx, (_, row) in enumerate(col_data.iterrows()):
            y = y_pos[tick_idx]
            q25 = row["Predicted Typical Low (q25)"]
            q50 = row["Predicted Plan Around (q50)"]
            q75 = row["Predicted Typical High (q75)"]
            q90 = row["Predicted Safety Cushion (q90)"]

            ax.plot(
                [q25, q75],
                [y, y],
                color=TYPICAL_RANGE_COLOR,
                linewidth=9,
                alpha=0.9,
                solid_capstyle="round",
                zorder=2,
            )
            if q90 > q75:
                ax.plot(
                    [q75, q90],
                    [y, y],
                    color=SAFETY_CUSHION_COLOR,
                    linewidth=5,
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=2,
                )
            for cap_x in (q25, q75):
                ax.vlines(
                    cap_x,
                    y - cap_half_height,
                    y + cap_half_height,
                    color=TYPICAL_RANGE_COLOR,
                    linewidth=1.5,
                    alpha=0.85,
                    zorder=3,
                )
            ax.scatter(
                [q50],
                [y],
                color=plan_color,
                s=42,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )
            ax.scatter(
                [q90],
                [y],
                color=SAFETY_CUSHION_COLOR,
                marker="D",
                s=46,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )
            _annotate_quantile_timeline(ax, y, q25, q50, q75, q90, figure_x_right)

        yticklabels = [
            f"{str(g).split(' (')[0]}\nn={n:,} | ${med:,.0f}"
            for g, n, med in zip(col_data["Group"], col_data["Sample Size"], col_data["Median Actual Cost"])
        ]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(yticklabels, fontsize=9)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(currency_fmt)
        ax.set_title(column_label, loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.20)
        sns.despine(ax=ax, left=True)

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=plan_color,
            markeredgecolor=plan_color,
            markersize=8,
            label="Plan Around (q50)",
        ),
        Line2D([0], [0], color=TYPICAL_RANGE_COLOR, linewidth=8, label="Typical Range (q25–q75)"),
        Line2D(
            [0], [0],
            marker="D",
            color="w",
            markerfacecolor=SAFETY_CUSHION_COLOR,
            markeredgecolor=SAFETY_CUSHION_COLOR,
            markersize=8,
            label="Safety Cushion (q90)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        frameon=True,
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.015)
    plt.tight_layout(rect=[0, 0.02, 1, 1], h_pad=2.0)

    _finalize_figure_xlim(panel_axes, fig)

    fig.canvas.draw()
    footnote_x = axes[0, 0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0
    fig.text(
        footnote_x,
        0.01,
        "Note: Dollar values in subgroup labels (e.g., $269) represent the actual median out-of-pocket costs of that subgroup.",
        fontsize=9,
        style="italic",
        color="#555555",
        ha="left",
    )

    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)

    plt.show()


plot_quantile_subgroup_predictions(
    quantile_subgroup_df,
    quantile_reliability_labels,
    "XGBoost Quantile Regression: Predicted Cost by Reliability Subgroup (Validation)"
)

plot_quantile_subgroup_predictions(
    quantile_subgroup_df,
    quantile_fairness_labels,
    "XGBoost Quantile Regression: Predicted Cost by Fairness Subgroup (Validation)"
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Plan-Around Estimate:</strong> The model successfully separates risk across all subgroups. Plan-around estimates (q50) increase for higher-risk groups: age 18-34 (\$131 q50) → 65+ (\$560), 0 chronic conditions (\$128) → 4+ (\$724), Excellent physical health (\$253) → Poor (\$587). This confirms that the quantile model learns meaningful risk patterns.</li>
#         <li><strong>Typical Range:</strong> Typical prediction range widths increase substantially for high-risk vs. low-risk groups. Walking limitation: No (width \$802) vs. Yes (\$1,608). Physical health: Excellent (\$676) vs. Poor (\$1,357). This shows the model widens its prediction intervals for groups with high cost variance.</li>
#         <li><strong>Safety Cushion:</strong> For most demographic groups (sex, age, race), the safety cushion (q90) provides a buffer of roughly \$1,400–\$3,000 above the plan-around estimate. This represents a meaningful safety boundary for budgeting without encouraging excessive over-allocation.</li>
#         <li><strong>Chronic Conditions:</strong> Each additional condition raises both the plan-around and the safety cushion. From 0 conditions (q50=\$128, q90=\$1,530) through 4+ conditions (q50=\$724, q90=\$4,090), the plan-around scales ~5.7× while the safety cushion scales ~2.7×, reflecting higher expected costs and greater cost uncertainty.</li>
#         <li><strong>Underestimating High Costs:</strong> Even for the highest predicted cost tier, the safety cushion tops out around \$6,250, whereas the actual Very High Spend group has a median of \$10,086. The model's predicted range compresses for extreme actual spenders: high-spend and very-high-spend actual cost groups receive near-identical predictions (q50 \$570 vs. \$674), confirming that the model cannot distinguish high from extreme costs.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Final Candidate Decision</b>
#     <ul style="margin-top:8px">
#         <li><b>Selected final candidate:</b> XGBoost quantile regression with q25, q50, q75, and q90 outputs.</li>
#         <li><b>Rationale:</b> The q50 plan-around estimate meets the accuracy target; q25–q75 and q90 coverage pass validation gates; intervals are narrow enough for budgeting; subgroup diagnostics show no broad demographic calibration failure; and Winkler interval score beats the naive population interval baseline.</li>
#         <li><b>Known risks:</b> q25 is slightly conservative, the uninsured group shows typical-range undercoverage, and very high actual spenders remain difficult to distinguish from high spenders.</li>
#         <li><b>CQR decision:</b> Do not add conformalized quantile regression at this stage. Revisit if the unseen test set misses coverage targets or shows stronger endpoint bias.</li>
#         <li><b>Next step:</b> Lock this model choice and evaluate once on the unseen test set.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Final Model Evaluation</h1>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Holdout Test Set Evaluation</b> <br>
#     The model choice, calibration decision, and release gates are now locked. This section evaluates the selected XGBoost quantile regression model once on the untouched test set. These results estimate performance in terms of generalization to unseen data and should not be used to tune the model further.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Predictions</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Generate q25, q50, q75, and q90 predictions for the test set using the selected XGBoost quantile model.
# </div>

# %%
xgb_quantile_final_model = load_model("../models/xgb_quantile_model.joblib")

y_test_quantile_pred_raw = xgb_quantile_final_model.predict(X_test_preprocessed)
y_test_quantile_pred = postprocess_quantile_predictions(y_test_quantile_pred_raw)

y_test_pred_q25, y_test_pred_q50, y_test_pred_q75, y_test_pred_q90 = y_test_quantile_pred.T

print(f"Generated quantile predictions for {len(y_test_quantile_pred):,} test samples.")

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Postprocessing Impact</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Check how much postprocessing of the quantile predictions (non-negative clipping and monotonic quantile enforcement) change the final test predictions.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Assess impact of non-negative clipping.
# </div>

# %%
def summarize_negative_prediction_clipping(y_pred_raw, sample_weight=None, label="Test"):
    """Measure how often non-negative clipping changes raw quantile predictions."""
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    clipped_amount = np.maximum(-y_pred_raw, 0)
    clipped = clipped_amount > 1e-9

    if sample_weight is None:
        sample_weight = np.ones(y_pred_raw.shape[0])
    sample_weight = np.asarray(sample_weight)

    rows = []
    row_clipped = clipped.any(axis=1)
    row_max_clip = clipped_amount.max(axis=1)
    row_affected_clip = row_max_clip[row_clipped]
    row_affected_weights = sample_weight[row_clipped]

    rows.append({
        "Split": label,
        "Output": "Any quantile",
        "Rows": y_pred_raw.shape[0],
        "Rows Clipped": row_clipped.sum(),
        "Weighted Share Clipped": np.average(row_clipped, weights=sample_weight),
        "Mean Clip Amount": np.average(row_affected_clip, weights=row_affected_weights) if row_clipped.any() else 0.0,
        "Median Clip Amount": weighted_quantile(row_affected_clip, row_affected_weights, 0.50) if row_clipped.any() else 0.0,
        "Max Clip Amount": row_max_clip.max(),
    })

    for idx, output in enumerate(["q25", "q50", "q75", "q90"]):
        output_clipped = clipped[:, idx]
        output_clip_amount = clipped_amount[:, idx]
        affected_clip_amount = output_clip_amount[output_clipped]
        affected_weights = sample_weight[output_clipped]

        rows.append({
            "Split": label,
            "Output": output,
            "Rows": y_pred_raw.shape[0],
            "Rows Clipped": output_clipped.sum(),
            "Weighted Share Clipped": np.average(output_clipped, weights=sample_weight),
            "Mean Clip Amount": np.average(affected_clip_amount, weights=affected_weights) if output_clipped.any() else 0.0,
            "Median Clip Amount": weighted_quantile(affected_clip_amount, affected_weights, 0.50) if output_clipped.any() else 0.0,
            "Max Clip Amount": output_clip_amount.max(),
        })

    return pd.DataFrame(rows)


negative_clipping_impact = summarize_negative_prediction_clipping(
    y_test_quantile_pred_raw,
    sample_weight=w_test,
    label="Test",
)

display(
    negative_clipping_impact.style
    .pipe(add_table_caption, "Impact of Non-Negative Clipping")
    .format({
        "Weighted Share Clipped": "{:.2%}",
        "Mean Clip Amount": "${:,.2f}",
        "Median Clip Amount": "${:,.2f}",
        "Max Clip Amount": "${:,.2f}",
    })
    .hide()
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Non-negative clipping affects 5.7% of the weighted test population (89 of 1,477 test rows). The clipped amounts are tiny (max \$0.41). For q50, clipping affects 1.1% of the population (18 rows), with max adjustment \$0.37. This confirms clipping is negligible for user-facing dollar predictions.
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Assess impact of monotonic quantile enforcement.
# </div>

# %%
# Monotonic quantile enforcement can affect later quantiles when a lower quantile has a higher predicted value.
# Verify that q50 changes are rare and small enough for SHAP to explain the postprocessed q50.
def summarize_monotonic_quantile_adjustment(y_pred_raw, sample_weight=None, label="Test"):
    """
    Measure how often monotonic quantile enforcement changes each quantile.

    For example, q50 is affected when raw q25 exceeds raw q50 after non-negative clipping.
    In that case, postprocessed q50 becomes raw q25 rather than raw q50.
    """
    y_pred_nonnegative = np.maximum(np.asarray(y_pred_raw, dtype=float), 0)
    y_pred_postprocessed = postprocess_quantile_predictions(y_pred_raw)
    monotonic_adjustment = y_pred_postprocessed - y_pred_nonnegative

    if sample_weight is None:
        sample_weight = np.ones(len(y_pred_nonnegative))
    sample_weight = np.asarray(sample_weight)

    rows = []
    for idx, q in enumerate(quantiles):
        quantile_label = f"q{int(q * 100)}"
        quantile_adjustment = monotonic_adjustment[:, idx]
        quantile_adjusted = quantile_adjustment > 1e-9

        affected_rate = np.average(quantile_adjusted, weights=sample_weight)
        affected_deltas = quantile_adjustment[quantile_adjusted]
        affected_weights = sample_weight[quantile_adjusted]

        if len(affected_deltas) == 0:
            mean_delta = 0.0
            median_delta = 0.0
            max_delta = 0.0
        else:
            mean_delta = np.average(affected_deltas, weights=affected_weights)
            median_delta = weighted_quantile(affected_deltas, affected_weights, 0.50)
            max_delta = affected_deltas.max()

        rows.append({
            "Split": label,
            "Quantile": quantile_label,
            "Rows": len(quantile_adjusted),
            "Rows Adjusted": quantile_adjusted.sum(),
            "Weighted Share Adjusted": affected_rate,
            "Mean Adjustment": mean_delta,
            "Median Adjustment": median_delta,
            "Max Adjustment": max_delta,
        })

    return pd.DataFrame(rows)


monotonic_quantile_adjustment = summarize_monotonic_quantile_adjustment(
    y_test_quantile_pred_raw,
    sample_weight=w_test,
    label="Test",
)

display(
    monotonic_quantile_adjustment.style
    .pipe(add_table_caption, "Impact of Monotonic Quantile Enforcement")
    .format({
        "Weighted Share Adjusted": "{:.2%}",
        "Mean Adjustment": "${:,.2f}",
        "Median Adjustment": "${:,.2f}",
        "Max Adjustment": "${:,.2f}",
    })
    .hide()
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Monotonic quantile enforcement has negligible impact on test predictions. The q50 plan-around estimate (used for SHAP) changed for only 0.24% of the weighted test population (3 of 1,477 test rows), with max adjustment $0.01. No monotonic adjustment was needed for q25, q75, or q90. This confirms postprocessed q50 is appropriate for SHAP.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Pinball Loss & Skill Score</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate final model on test set using pinball loss and quantile skill score (against the naive population-quantile baseline).
# </div>

# %%
test_pinball_results = []

for idx, q in enumerate(quantiles):
    quantile_label = f"q{int(q * 100)}"
    y_test_pred_q = y_test_quantile_pred[:, idx]

    naive_quantile_value = weighted_quantile(y_train, w_train, q)
    y_test_naive_q = np.full_like(y_test, fill_value=naive_quantile_value)

    test_pinball_loss = mean_pinball_loss(
        y_test,
        y_test_pred_q,
        alpha=q,
        sample_weight=w_test,
    )
    test_naive_pinball_loss = mean_pinball_loss(
        y_test,
        y_test_naive_q,
        alpha=q,
        sample_weight=w_test,
    )
    test_quantile_skill_score = 1.0 - (test_pinball_loss / test_naive_pinball_loss)

    test_pinball_results.append({
        "Quantile": quantile_label,
        "Model Pinball Loss": test_pinball_loss,
        "Naive Pinball Loss": test_naive_pinball_loss,
        "Skill Score": test_quantile_skill_score,
    })

test_pinball_df = pd.DataFrame(test_pinball_results)

display(
    test_pinball_df.style
    .hide()
    .pipe(add_table_caption, "Final Model: Pinball Loss & Skill Scores (Test)")
    .format(
        lambda value: f"{DOLLAR}{value:,.2f}",
        subset=["Model Pinball Loss", "Naive Pinball Loss"],
    )
    .format("{:.2%}", subset=["Skill Score"])
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>All Quantiles Beat Naive:</strong> Every test-set quantile has lower pinball loss than the naive population-quantile baseline. This confirms that the model adds value across the full q25-q90 range, not only for the median plan-around estimate.</li>
#         <li><strong>Skill Improves Toward Higher Costs:</strong> Quantile skill score rises from 4.9% at q25 to 15.6% at q90. This means the model is most useful where personalization matters most for budgeting: identifying users who need a larger safety cushion.</li>
#         <li><strong>Generalization Looks Good:</strong> Skill scores are slightly higher on the test than validation set. q90 skill remains below training skill, so the earlier tail-overfitting concern still exists, but it does not translate into a test-set failure.</li>
#         <li><strong>Decision Role:</strong> Pinball loss and skill score remain purely diagnostic, they do not have product release gates. They support the final decision by showing that the quantile model improves on a simple baseline while the product metrics decide whether the model is useful enough to ship.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Quantile Calibration</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Check whether each predicted quantile remains calibrated on the untouched test set.
# </div>

# %%
test_quantile_metric_bootstrap_samples = generate_quantile_metric_bootstrap_samples(
    y_test,
    y_test_quantile_pred,
    w_test,
    quantiles,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
)

test_quantile_coverage_results = []

for idx, q in enumerate(quantiles):
    quantile_label = f"q{int(q * 100)}"
    test_coverage = np.average(y_test <= y_test_quantile_pred[:, idx], weights=w_test)
    ci_lower, ci_upper = get_bootstrap_ci(test_quantile_metric_bootstrap_samples[f"{quantile_label}_coverage"])
    calibration_error = test_coverage - q

    test_quantile_coverage_results.append({
        "Quantile": quantile_label,
        "Nominal Level": q,
        "Coverage (95% CI)": f"{test_coverage:.1%} [{ci_lower:.1%}, {ci_upper:.1%}]",
        "Calibration Error": calibration_error,
        "Status": "Pass" if abs(calibration_error) <= 0.05 else "Review",
    })

test_quantile_coverage_df = pd.DataFrame(test_quantile_coverage_results)

display(
    test_quantile_coverage_df.style
    .hide()
    .pipe(add_table_caption, "Final Model: Quantile Calibration (Test)")
    .format("{:.1%}", subset=["Nominal Level"])
    .format("{:+.1%}", subset=["Calibration Error"])
    .map(style_status_cells, subset=["Status"])
)


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul>
#         <li><strong>Calibration Passed:</strong> All four predicted quantiles pass the final test calibration tolerance. q50, q75, and q90 are close to their nominal levels, confirming that the plan-around estimate, upper typical-range bound, and safety cushion remain well calibrated on unseen data.</li>
#         <li><strong>q25 Remains Slightly Conservative:</strong> q25 covers 29.9% of test outcomes instead of the nominal 25.0%, matching the validation pattern. This means the lower endpoint of the typical range is still somewhat high, but the +4.9% calibration error stays within the ±5% calibration tolerance.</li>
#         <li><strong>Strong Safety Cushion:</strong> q90 coverage is 91.0% with a 95% CI of [89.2%, 92.6%], comfortably within the 85%–95% product tolerance. This supports using q90 as the budget-safe estimate.</li>
#         <li><strong>No Test-Set Surprise:</strong> Test calibration is consistent with validation. With 1,477 rows in both data sets, a <3% difference in coverage between validation and test is normal. We are estimating population-level coverage from two finite samples, and healthcare cost outcomes are noisy, weighted, zero-inflated, and heavy-tailed. It would actually be suspicious if every quantile matched almost exactly.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Product Metrics</h2>
# </div> 
#
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate final model q50 accuracy, product coverage, interval width, and Winkler interval score on the untouched test set.
# </div>

# %%
def get_test_metric_status(metric, value):
    """Return a compact release status for final test metrics."""
    if metric == "Plan Around MdAE (q50)":
        return "Pass" if value < 500 else "Review"
    if metric == "Typical Range Coverage (q25–q75)":
        return "Pass" if 0.45 <= value <= 0.55 else "Review"
    if metric == "Safety Cushion Coverage (q90)":
        return "Pass" if 0.85 <= value <= 0.95 else "Review"
    if metric == "Typical Range Width":
        return "Pass" if value < 1500 else "Review"
    if metric == "Safety Cushion Width":
        return "Pass" if value < 3500 else "Review"
    return "Diagnostic"


test_product_metric_specs = [
    {
        "Metric": "Plan Around MdAE (q50)",
        "Test": weighted_median_absolute_error(y_test, y_test_pred_q50, sample_weight=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q50_mdae"],
        "Format": "currency_2",
    },
    {
        "Metric": "Plan Around MAE (q50)",
        "Test": mean_absolute_error(y_test, y_test_pred_q50, sample_weight=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q50_mae"],
        "Format": "currency_2",
    },
    {
        "Metric": "Plan Around R² (q50)",
        "Test": r2_score(y_test, y_test_pred_q50, sample_weight=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q50_r2"],
        "Format": "decimal",
    },
    {
        "Metric": "Typical Range Coverage (q25–q75)",
        "Test": np.average((y_test >= y_test_pred_q25) & (y_test <= y_test_pred_q75), weights=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q25_q75_coverage"],
        "Format": "percent",
    },
    {
        "Metric": "Safety Cushion Coverage (q90)",
        "Test": np.average(y_test <= y_test_pred_q90, weights=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q90_coverage"],
        "Format": "percent",
    },
    {
        "Metric": "Typical Range Width",
        "Test": np.average(y_test_pred_q75 - y_test_pred_q25, weights=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q25_q75_width"],
        "Format": "currency_0",
    },
    {
        "Metric": "Safety Cushion Width",
        "Test": np.average(y_test_pred_q90 - y_test_pred_q50, weights=w_test),
        "Samples": test_quantile_metric_bootstrap_samples["q50_q90_width"],
        "Format": "currency_0",
    },
]

test_product_metrics_display = []

for metric_spec in test_product_metric_specs:
    metric_format = metric_spec["Format"]
    test_display = (
        format_metric_value(metric_spec["Test"], metric_format)
        if metric_spec["Samples"] is None
        else format_metric_with_ci(metric_spec["Test"], metric_spec["Samples"], metric_format)
    )

    test_product_metrics_display.append({
        "Metric": metric_spec["Metric"],
        "Estimate (95% CI)": test_display,
        "Release Gate": PRODUCT_METRIC_GATES_AND_TARGETS[metric_spec["Metric"]]["Release Gate"],
        "Product Target": PRODUCT_METRIC_GATES_AND_TARGETS[metric_spec["Metric"]]["Product Target"],
        "Status": get_test_metric_status(metric_spec["Metric"], metric_spec["Test"]),
    })

test_product_metrics_df = pd.DataFrame(test_product_metrics_display).set_index("Metric")
test_product_metrics_df.index.name = None

display(
    test_product_metrics_df.style
    .pipe(add_table_caption, "Final Model: Product Metrics (Test)")
    .map(style_status_cells, subset=["Status"])
)

# %%
test_interval_score_bootstrap_samples = generate_interval_score_bootstrap_samples(
    y_test,
    y_test_pred_q25,
    y_test_pred_q75,
    w_test,
    naive_q25,
    naive_q75,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
)

test_model_interval_score = interval_score(
    y_test,
    y_test_pred_q25,
    y_test_pred_q75,
    alpha=0.50,
    sample_weight=w_test,
)
test_naive_interval_score = interval_score(
    y_test,
    np.full_like(y_test, naive_q25, dtype=float),
    np.full_like(y_test, naive_q75, dtype=float),
    alpha=0.50,
    sample_weight=w_test,
)
test_interval_skill_score = 1 - (test_model_interval_score / test_naive_interval_score)

test_interval_score_specs = [
    {
        "Metric": "XGBoost Winkler Interval Score",
        "Test": test_model_interval_score,
        "Samples": test_interval_score_bootstrap_samples["model_interval_score"],
        "Format": "currency_0",
    },
    {
        "Metric": "Naive Winkler Interval Score",
        "Test": test_naive_interval_score,
        "Samples": test_interval_score_bootstrap_samples["naive_interval_score"],
        "Format": "currency_0",
    },
    {
        "Metric": "Interval Skill Score",
        "Test": test_interval_skill_score,
        "Samples": test_interval_score_bootstrap_samples["interval_skill_score"],
        "Format": "percent",
    },
]

test_interval_score_display = []

for metric_spec in test_interval_score_specs:
    test_interval_score_display.append({
        "Metric": metric_spec["Metric"],
        "Estimate (95% CI)": format_metric_with_ci(
            metric_spec["Test"],
            metric_spec["Samples"],
            metric_spec["Format"],
        ),
    })

test_interval_score_df = pd.DataFrame(test_interval_score_display).set_index("Metric")
test_interval_score_df.index.name = None

display(
    test_interval_score_df.style
    .pipe(add_table_caption, "Typical Range: Winkler Interval Score (Test)")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> The final model passes all product-facing release gates on the holdout test set. 
#     <ul>
#         <li><b>Plan Around:</b> The q50 estimate achieves MdAE = &#36;239.5, comfortably below both the release gate and product target.</li>
#         <li><b>Typical Range and Safety Cushion:</b> Coverage remains within release gates. Interval widths meet the product targets. This suggests the model is calibrated without making the ranges overly broad.</li>
#         <li><b>Usefulness vs Simple Baseline:</b> The trained model improves on naive population baselines for all user-facing outputs. Plan-around q50 skill = 9.75%, typical-range interval skill = 11.2%, and safety-cushion q90 skill = 15.63%. This shows that feature-based model predictions add value compared with giving every user the same population median, generic q25-q75 range, or population q90 estimate.</li>
#         <li><b>Remaining Risk:</b> The model struggles with the rare high-cost cases, reflected by the large MAE-vs-MdAE gap and near-zero R².</li>
#     </ul>
#     <p style="margin-top:8px"><em>Note:</em> Skill scores for plan-around (q50) and safety-cushion (q90) are quantile skill scores based on pinball loss. Typical-range (q25-q75) skill score is an interval skill score based on Winkler score. Both compare the trained model against a naive population baseline, but based on different evaluation metrics.</p>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Stratified Error Analysis</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px; margin-bottom:12px;">
#     ℹ️ <b>Subgroup Reliability and Fairness Audit (Test)</b> <br>
#     Audit the final XGBoost quantile regression model on the untouched test set. Reuse the same stratification columns from validation. Because these subgroup definitions were chosen before test-set evaluation, this section is a final reporting check rather than another model-selection loop.
#     <ul style="margin-top:8px">
#         <li><b>Reliability groups:</b> actual cost tier, predicted plan-around cost tier (q50), predicted safety-cushion cost tier (q90), physical health, insurance status, and chronic condition count.</li>
#         <li><b>Fairness audit groups:</b> sex, age group, race/ethnicity, mental health, family income, education, region, and walking limitation.</li>
#         <li><b>Metrics by group:</b> plan-around MdAE (q50), typical range coverage/width (q25-q75), and safety cushion coverage/width (q90 and q50-q90).</li>
#         <li><b>Subgroup review bands:</b> Release gates apply to overall test coverage (45-55% for typical range, 85-95% for safety cushion). Subgroup review bands are diagnostic ranges, not additional pass/fail gates. For subgroups with sufficient sample size (n ≥ 30), typical range coverage should generally stay within 40-60%, and safety cushion coverage should generally stay within 80-97%. Small subgroups (n &lt; 30) are labeled review-only because their estimates are less stable.</li>
#         <li><b>Diagnostic flags:</b> Flag high plan-around error for subgroups with q50 MdAE above 3× the overall q50 MdAE. For low-cost subgroups (median actual cost ≤ overall median), flag typical range or safety cushion widths above the overall average as potentially too broad for practical budgeting. These flags point to calibration, usefulness, or sampling issues that need review; they do not by themselves imply model unfairness or a release failure.</li>
#         <li><b>Interpretation:</b> Use diagnostic flags to qualify reporting, document caveats, and inform product planning notices or scope disclaimers. Because this is the locked test set, flagged subgroups should not drive another tuning loop; follow-up changes should be evaluated in a new validation cycle.</li>
#     </ul>
# </div>

# %%
# --- Prepare Features ---
print("Recovering raw test features...")
df_raw_test, y_test_audit, w_test_audit = prepare_human_readable_split_data(TEST_MODEL_READY_DATA_PATH, "test")

# Align test-set quantile predictions to the raw test rows
y_test_pred_q25_audit, y_test_pred_q50_audit, y_test_pred_q75_audit, y_test_pred_q90_audit = [
    pd.Series(values, index=y_test_audit.index)
    for values in y_test_quantile_pred.T
]

df_raw_test = add_quantile_stratification_columns(
    df_raw_test,
    y_test_audit,
    y_test_pred_q50_audit,
    y_test_pred_q90_audit,
)
test_quantile_subgroup_df = create_quantile_subgroup_df(
    df_raw_test,
    y_test_audit,
    w_test_audit,
    (y_test_pred_q25_audit, y_test_pred_q50_audit, y_test_pred_q75_audit, y_test_pred_q90_audit),
    quantile_stratified_configs,
)

display(
    test_quantile_subgroup_df[display_columns]
    .style
    .hide()
    .pipe(add_table_caption, "XGBoost Quantile Regression: Stratified Error Analysis (Test)")
    .format("{:,}", subset=["Sample Size"])
    .format("${:,.2f}", subset=["Median Actual Cost", "Plan Around MdAE (q50)", "Typical Range Width (q25–q75)", "Safety Cushion Width (q50–q90)"])
    .format("{:.1%}", subset=["Typical Range Coverage (q25–q75)", "Safety Cushion Coverage (q90)"])
    .apply(lambda row: ["background-color: #fff3cd" if row["Reliability Flags"] != "None" else "" for _ in row], axis=1)
)

# %%
plot_quantile_subgroup_performance(
    test_quantile_subgroup_df,
    quantile_reliability_labels,
    "XGBoost Quantile Regression: Subgroup Reliability (Test)",
    save_to_file="../figures/evaluation/xgb_quantile_test_subgroup_reliability.png"
)

plot_quantile_subgroup_performance(
    test_quantile_subgroup_df,
    quantile_fairness_labels,
    "XGBoost Quantile Regression: Subgroup Fairness (Test)",
    save_to_file="../figures/evaluation/xgb_quantile_test_subgroup_fairness.png"
)

plot_quantile_subgroup_predictions(
    test_quantile_subgroup_df,
    quantile_reliability_labels,
    "XGBoost Quantile Regression: Predicted Cost by Reliability Subgroup (Test)"
)

plot_quantile_subgroup_predictions(
    test_quantile_subgroup_df,
    quantile_fairness_labels,
    "XGBoost Quantile Regression: Predicted Cost by Fairness Subgroup (Test)"
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> The final subgroup audit supports launch. There is no broad demographic fairness failure and predicted-risk tiers remain usable for deployment. There are some caveats, the main one being unreliable predictions of rare actual tail costs, which are only visible after the year is observed.
#     <ul>
#         <li><strong>Actual Cost Tiers:</strong> The biggest caveat is predictability of high actual costs. Actual High spenders have 12.4% typical-range coverage and 59.0% safety-cushion coverage; actual Very High spenders have 0.0% and 6.7%, respectively. Zero- and low-cost actual groups are heavily overprotected by the safety cushion (100.0% and 99.9%). This is an expected limitation of zero-inflated, heavy-tailed medical costs.</li>
#         <li><strong>Predicted Cost Tiers:</strong> The deployable risk tiers are much better behaved because they are known at prediction time. Predicted plan-around cost tiers keep typical-range coverage inside the subgroup review band (42.6%-54.6%), and predicted safety-cushion tiers keep q90 coverage inside the subgroup review band (85.6%-93.4%). Widths also increase monotonically with predicted risk, from a \$1,125 safety-cushion width in the predicted q90 Low tier to \$5,582 in the predicted q90 Very High tier, which supports using predicted-risk bands to communicate prediction uncertainty to the user.</li>
#         <li><strong>Subgroup Reliability:</strong> Physical health, chronic conditions, private insurance, and public insurance groups remain within subgroup review bands. Uninsured users are the main coverage watchlist group: typical-range coverage is low at 34.7%, while safety-cushion coverage is conservative at 96.3%. The safety cushion is not under-warning, but the typical range may be too narrow or shifted for uninsured users, so the app should show an uninsured uncertainty note.</li>
#         <li><strong>Subgroup Fairness:</strong> Sex, age, race/ethnicity, region, and walking limitation groups do not show systematic undercoverage. Subgroups to watch are poor mental health (30.1% typical-range coverage), low income (39.2%), doctorate degree (34.7%), and near poor income (97.7% safety-cushion overcoverage). These should qualify reporting and future MEPS validation, not trigger test-set-driven model retuning.</li>
#         <li><strong>Prediction Usefulness:</strong> Several low-cost groups have wide prediction intervals despite in-band coverage, including good physical health, good mental health, Asian respondents, and the West region. This is a practical-budgeting caveat rather than a safety failure, because the model often protects users in these subgroups from underestimation, but some low-cost users may receive ranges that are less specific than desired.</li>
#     </ul>
#     <p style="margin-top:12px"><strong>Caveats and Recommended Actions</strong> <br> 
#         The main user-facing response to typical-range undercoverage is a planning note that tells users to treat the plan-around amount and typical range as starting points and plan closer to the safety cushion. Use this same generic note for all affected profiles:</p>
#     <p style="margin-top:8px; padding:10px; background-color:#f4fbf4; border-left:4px solid #6aa56a;">
#         "Costs for profiles like yours can vary a lot from year to year. The plan-around amount and typical range are useful starting points, but for budgeting decisions, plan closer to the safety cushion."
#     </p>
#     <p style="margin-top:8px">
#         Add additional detail for high predicted costs and uninsured profiles because those reasons are directly useful for budgeting. Do not name low income, poor mental health, or doctorate degree to avoid stigmatization.</p>
#     <table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.92em;">
#         <thead>
#             <tr style="background-color:#e0f0e0;">
#                 <th style="padding:8px; border:1px solid #d6e8d6; text-align:left;">Finding</th>
#                 <th style="padding:8px; border:1px solid #d6e8d6; text-align:left;">Affected Profiles</th>
#                 <th style="padding:8px; border:1px solid #d6e8d6; text-align:left;">Planning Notice Policy</th>
#             </tr>
#         </thead>
#         <tbody>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Rare actual high-cost years remain hard to predict.</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Actual high and very-high spenders, which are not knowable at prediction time.</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Keep the rare-high-cost limitation in the always-on disclaimer. Also show planning note for <code>HIGH_PREDICTED_UNCERTAINTY</code> when predicted q90 is in the top 20%: "This estimate falls in a higher-cost range, where out-of-pocket costs can be harder to predict."</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Typical range undercoverage means users may see actual costs above the displayed range more often than intended.</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Uninsured users (34.7% typical-range coverage), poor mental health (30.1%), low income (39.2%), and doctorate degree (34.7%).</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Show the generic planning note for all affected subgroups. For uninsured users, also note that "Because you are uninsured, out-of-pocket costs can be harder to predict." For low income, poor mental health, and doctorate degree, do not name the subgroup.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Safety-cushion overcoverage means q90 may be more conservative than needed.</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Zero- and low-cost actual spenders, and near-poor income users (97.7% safety-cushion coverage).</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Do not trigger safety-cushion guidance. Document and monitor the pattern, but avoid telling these users to plan closer to the safety cushion unless another undercoverage trigger also applies.</td>
#             </tr>
#             <tr>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Some low-cost groups receive wide intervals despite acceptable coverage.</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Good physical health, good mental health, Asian respondents, and the West region.</td>
#                 <td style="padding:8px; border:1px solid #d6e8d6;">Do not add a separate planning notice for wide intervals. The wide range already communicates uncertainty. Treat this as a usefulness caveat and revisit in future MEPS years.</td>
#             </tr>
#         </tbody>
#     </table>
#     <i>Implementation rule: render one compact planning note panel and combine triggered messages without repeating the safety-cushion guidance.</i>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Launch Decision</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Release Gate Metrics (Test)</b>
#
# | Metric | Estimate [95% CI] | Release Gate | Product Target | Status |
# | --- | ---: | ---: | ---: | --- |
# | Plan-around MdAE (q50) | `$240` [`$215`, `$279`] | < `$500` | < `$350` | Pass |
# | Typical-range coverage (q25-q75) | 47.3% [44.0%, 50.6%] | 45%-55% | 50% | Pass |
# | Safety-cushion coverage (q90) | 91.0% [89.2%, 92.6%] | 85%-95% | 90% | Pass |
# | Typical-range width (q25-q75) | `$912` [`$875`, `$955`] | < `$1,500` | < `$1,000` | Pass |
# | Safety-cushion width (q50-q90) | `$2,032` [`$1,964`, `$2,108`] | < `$3,500` | < `$2,500` | Pass |
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     <b>Final Model: Summary and Launch Decision</b> 
#     <ul style="margin-top:8px">
#         <li><b>Decision:</b> Launch XGBoost quantile regression as the MVP model, with guardrails. The product should be framed as a budgeting aid for individual out-of-pocket cost planning, not as a bill estimate, procedure-price tool, or medical advice.</li>
#         <li><b>Evidence:</b> The model passes all product-facing release gates on the unseen test set: q50 MdAE = \$240, q25-q75 coverage = 47.3%, q90 coverage = 91.0%, q25-q75 width = \$912, and q50-q90 width = \$2,032. It also improves on naive population baselines for every user-facing output: q50 skill = 9.8%, typical-range interval skill = 11.2%, and q90 skill = 15.6%.</li>
#         <li><b>Calibration:</b> Do not add conformalized quantile regression for the MVP. Test calibration passes the predefined gates. Any calibration change should be evaluated in a new validation cycle, preferably against a later MEPS year when available.</li>
#         <li><b>Prediction Output:</b> Show q50 as the plan-around estimate, q25-q75 as the typical range, and q90 as the safety cushion. Do not present a single point estimate.</li>
#         <li><b>Launch Conditions:</b> Ship only with range-based predictions, the scope disclaimer, 2023-to-current-dollar adjustment, conditional planning notices to communicate prediction uncertainty for high predicted costs and uninsured users, and privacy-preserving aggregate monitoring.</li>
#         <li><b>Scope Disclaimer:</b> Explain that the model uses 2023 MEPS individual out-of-pocket spending. It excludes premiums, over-the-counter costs, family totals, and procedure prices. It can miss rare high-cost years, especially for users whose realized costs land in the extreme tail.</li>
#         <li><b>Monitoring:</b> Track aggregate app health, completion rate, input drift, prediction drift, missingness, q50 distribution, q25-q75 width, q90 safety cushion, and high-uncertainty flags. Broad slices such as insurance status, family income, mental health, and chronic-condition count can explain shifts, but they cannot measure calibration without observed annual costs.</li>
#         <li><b>Post-Launch Learning:</b> Do not calibrate on app user data. True calibration requires observed annual out-of-pocket costs, and collecting linked follow-up outcomes would conflict with the anonymous, zero-retention product requirement. If outcome collection becomes a product goal, treat it as a separate opt-in study with consent, retention limits, data minimization, and a privacy review.</li>
#     </ul>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     <b>Example Prediction Output</b>
#     <table style="width:100%; border-spacing:10px 0px; border-collapse:separate; margin-top:10px; font-size:0.92em;">
#         <tr>
#             <td style="width:50%; padding:0 10px; vertical-align:bottom;">
#                 <strong>Low Cost Profile</strong><br>
#                 <i>28-year-old with no chronic conditions</i>
#             </td>
#             <td style="width:50%; padding:0 10px; vertical-align:bottom;">
#                 <strong>High Cost Profile</strong><br>
#                 <i>68-year-old, uninsured, and with multiple chronic conditions</i>
#             </td>
#         </tr>
#         <tr>
#             <td style="background-color:#fcfcfc; border:1px solid #ddd; padding:15px; vertical-align:top; border-radius:4px;">
#                 <b>Your Estimated Out-of-Pocket Costs for Next Year</b><br><br>
#                 💰 <b>Plan around:</b> \$420<br>
#                 📊 <b>Typical range:</b> \$120 – \$980<br>
#                 🛡️ <b>Safety cushion:</b> budget up to \$1,850<br><br>
#                 <span style="font-size:0.85em; color:#555;">Use the plan-around number as a reasonable midpoint for budgeting. The typical range shows where about half of people with similar profiles fall. The safety cushion gives extra room for a higher-cost year.</span>
#                 <br><br>
#                 <details style="margin-bottom:8px;">
#                 <summary style="cursor:pointer;"><strong>What shaped your plan-around estimate?</strong></summary>
#                 <p style="font-size:0.85em; color:#555; margin-bottom:6px;">These five answers contributed most to your plan-around estimate:</p>
#                 <table style="width:100%; border-collapse:collapse; font-size:0.85em;">
#                     <thead><tr><th style="text-align:left; padding:4px;">Your answer</th><th style="text-align:right; padding:4px;">Contribution</th></tr></thead>
#                     <tbody>
#                         <tr><td style="padding:4px;"><strong>Age:</strong> 28</td><td style="text-align:right; padding:4px;"><strong>↓ −\$210</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>Physical health:</strong> Very Good</td><td style="text-align:right; padding:4px;"><strong>↓ −\$130</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>High blood pressure:</strong> No</td><td style="text-align:right; padding:4px;"><strong>↓ −\$100</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>High cholesterol:</strong> No</td><td style="text-align:right; padding:4px;"><strong>↓ −\$90</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>Usual source of care:</strong> Yes</td><td style="text-align:right; padding:4px;"><strong>↑ +\$60</strong></td></tr>
#                     </tbody>
#                 </table>
#                 </details>
#                 <details style="margin-bottom:8px;">
#                 <summary style="cursor:pointer;""><strong>How you compare to others</strong> <i>(click to expand)</i></summary>
#                 <span style="font-size:0.85em; color:#555;">
#                 <i>(bar chart in app)</i><br>
#                 Your plan-around estimate: \$420<br>
#                 Typical American: \$248<br>
#                 Typical for ages 18-34: \$70<br>
#                 </span>
#                 </details>
#                 <br>
#                 <b>About this estimate</b><br>
#                 <span style="font-size:0.85em; color:#555;">This is a planning estimate, not a bill estimate. It is based on 2023 national survey data and adjusted to current dollars. It does not include premiums, over-the-counter costs, family totals, or procedure prices. New diagnoses, accidents, hospitalizations, and plan-specific billing details can make actual costs higher.</span>
#             </td>
#             <td style="background-color:#fcfcfc; border:1px solid #ddd; padding:15px; vertical-align:top; border-radius:4px;">
#                 <b>Your Estimated Out-of-Pocket Costs for Next Year</b>
#                 <br><br>
#                 💰 <b>Plan around:</b> \$1,350<br>
#                 📊 <b>Typical range:</b> \$520 – \$2,400<br>
#                 🛡️ <b>Safety cushion:</b> budget up to \$5,200<br><br>
#                 <span style="font-size:0.85em; color:#555;">Use the plan-around number as a reasonable midpoint for budgeting. The typical range shows where about half of people with similar profiles fall. The safety cushion gives extra room for a higher-cost year.</span>
#                 <br><br>
#                 <b>Planning note</b><br>
#                 <span style="font-size:0.85em; color:#555;">Costs for profiles like yours can vary a lot from year to year. This estimate falls in a higher-cost range, and because you are uninsured, out-of-pocket costs can be harder to predict. The plan-around amount and typical range are useful starting points, but for budgeting decisions, plan closer to the safety cushion.</span>
#                 <br><br>
#                 <details style="margin-bottom:8px;">
#                 <summary style="cursor:pointer;"><strong>What shaped your plan-around estimate?</strong></summary>
#                 <p style="font-size:0.85em; color:#555; margin-bottom:6px;">These five answers contributed most to your plan-around estimate:</p>
#                 <table style="width:100%; border-collapse:collapse; font-size:0.85em;">
#                     <thead><tr><th style="text-align:left; padding:4px;">Your answer</th><th style="text-align:right; padding:4px;">Contribution</th></tr></thead>
#                     <tbody>
#                         <tr><td style="padding:4px;"><strong>Age:</strong> 68</td><td style="text-align:right; padding:4px;"><strong>↑ +\$480</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>Diabetes:</strong> Yes</td><td style="text-align:right; padding:4px;"><strong>↑ +\$370</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>Insurance:</strong> Uninsured</td><td style="text-align:right; padding:4px;"><strong>↑ +\$310</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>High blood pressure:</strong> Yes</td><td style="text-align:right; padding:4px;"><strong>↑ +\$180</strong></td></tr>
#                         <tr><td style="padding:4px;"><strong>Physical health:</strong> Good</td><td style="text-align:right; padding:4px;"><strong>↓ −\$90</strong></td></tr>
#                     </tbody>
#                 </table>
#                 </details>
#                 <details style="margin-bottom:8px;">
#                 <summary style="cursor:pointer;""><strong>How you compare to others</strong> <i>(click to expand)</i></summary>
#                 <span style="font-size:0.85em; color:#555;">
#                 <i>(bar chart in app)</i><br>
#                 Your plan-around estimate: \$1,350<br>
#                 Typical American: \$248<br>
#                 Typical for ages 65+: \$608<br>
#                 </span>
#                 </details>
#                 <br>
#                 <b>About this estimate</b><br>
#                 <span style="font-size:0.85em; color:#555;">This is a planning estimate, not a bill estimate. It is based on 2023 national survey data and adjusted to current dollars. It does not include premiums, over-the-counter costs, family totals, or procedure prices. New diagnoses, accidents, hospitalizations, and plan-specific billing details can make actual costs higher.</span>
#             </td>
#         </tr>
#     </table>
#     App Implementation: For "What shaped your plan-around estimate?", use a Gradio accordion (<code>gr.Accordion(open=False)</code>) containing a Markdown table.
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Model Explainability (SHAP)</h1>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ SHapley Additive exPlanations (SHAP) will be used for both model audit and user-facing prediction explanations.
#     <br><br>
#     <strong>SHAP Approach</strong><br>
#     SHAP will be used in two ways:
#     <ol>
#         <li><strong>Feature Importance Analysis for Model Audit:</strong> Rank the features by mean absolute SHAP value (<code>mean(|SHAP|)</code>) across many predictions (survey-weighted) to identify which features have the largest dollar impact on predicted median costs (postprocessed q50). Compare this SHAP feature importance with native XGBoost importance to audit whether the model relies on plausible cost drivers.</li>
#         <li><strong>User-Facing Explanations as a Product Feature:</strong> Use individual SHAP values to show which inputs moved that user's plan-around estimate above or below the SHAP baseline. This is a product feature, not just a diagnostic. It makes the prediction more useful for financial planning by explaining the main cost drivers. Use cautious wording ("moved the estimate") and avoid causal language.</li>
#     </ol>
#     <strong>Explained Output</strong><br>
#     SHAP explanations focus on the q50 plan-around estimate (predicted median cost). The q25, q75, and q90 outputs define the typical range and safety cushion, but should not be mixed into the q50 explanation.
#     <br><br>
#     <strong>SHAP Prediction Function</strong><br>
#     SHAP can explain a callable prediction function, not only a raw model object. The saved <code>xgb_quantile_model</code> is a <code>TransformedTargetRegressor</code> wrapping the inner XGBoost estimator, but the explainer calls <code>predict_median_cost</code> instead. This callable converts 27 preprocessor input features into 40 model-ready features, predicts all four quantiles, applies the inverse target transformation, enforces non-negative and monotonic quantiles (<code>q25 ≤ q50 ≤ q75 ≤ q90</code>), and selects q50. Medical-cost inflation remains outside the callable and is applied only during API/UI output formatting.
#     <br><br>
#     Postprocessed q50 is appropriate as the explained output because it matches the plan-around estimate and the impact of postprocessing on q50 is negligible. On the test set, non-negative clipping affects 1.1% of the weighted test population with a maximum adjustment of \$0.37, and monotonic quantile enforcement affects 0.2% with a maximum adjustment of \$0.01.
#     <br><br>
#     <strong>Why Not TreeExplainer?</strong><br>
#     TreeExplainer is faster for raw tree models, but it would explain only the inner XGBoost estimator operating on the 40 model-ready features. It would not include the fitted preprocessor, inverse target transformation, non-negative and monotonic quantile postprocessing, or q50 selection. Use <code>shap.Explainer(..., algorithm="permutation")</code> with the <code>predict_median_cost</code> prediction function so SHAP explains the q50 estimate based on the 27 preprocessor input features. These preprocessor input features are interpretable, semantically meaningful features.
#     <br><br>
#     <strong>Feature Importance</strong><br>
#     SHAP feature importance is based on the preprocessor input features produced by several MEPS data preparation steps, but before the preprocessor performs imputation, medical feature engineering, scaling, and one-hot encoding. The callable follows the complete prediction path: <code>27 preprocessor input features → preprocessor → 40 model-ready features → quantile model prediction → inverse target transformation → quantile postprocessing → q50</code>.
#     <br><br>
#     XGBoost native feature importance is based on all quantiles (q25, q50, q75, q90) and the 40 model-ready features. These are the scaled numerical features, one-hot encoded nominal features, passed-through binary features, and derived medical features (e.g., chronic condition count) that the trees use for splits. Unlike SHAP, native importance describes each feature's role in reducing the training objective rather than its dollar contribution to q50.
#     <br><br>
#     Treat SHAP and XGBoost native importance as complementary rather than interchangeable. SHAP explains the postprocessed q50 prediction over 27 interpretable preprocessor input features and reports dollar impacts. Native <code>total_gain</code> summarizes how the joint four-quantile estimator used 40 model-ready features during training.
#     <br><br>
#     <strong>Background Data</strong><br>
#     The background data is a 200–500 row sample from the training data (with preprocessor input features), drawn using weighted sampling with replacement. SHAP treats background rows as equal-weight, so sampling with MEPS person weights (<code>PERWT23F</code>) makes the background approximate the U.S. adult population distribution. High-weight respondents may appear more than once. That is expected because duplicates represent their larger population share. The SHAP baseline is the mean postprocessed q50 prediction across those background rows.
#     <br><br>
#     Background data validation: Compare the background sample's SHAP baseline against the full weighted training baseline. Accept the sample if the absolute relative difference is at most 10%. If it exceeds 10%, increase the background size before creating app artifacts. This is an initial validation. Consider stronger validation criteria.
#     <br><br>
#     <strong>Core SHAP Explanation Latency</strong><br>
#     A normal prediction scores one user row. A SHAP explanation scores many masked versions of that row against the background. With 27 preprocessor input features, one complete permutation round evaluates up to <code>2 × 27 + 1 = 55</code> masked feature combinations: one initial fully masked state, 27 forward steps that add features, and 27 backward steps that remove them. The default <code>max_evals=500</code> permits 9 complete rounds, or 495 masks. With a 300-row background this is at most about 148,500 synthetic predictions. SHAP predicts in batches, but this computation is the main expected contributor to prediction request latency.
#     <br><br>
#     Select the best SHAP configuration for production empirically. Benchmark permutation rounds and background size together: permutation rounds control the number of mask evaluations, background size controls baseline stability, and latency scales roughly as <code>mask evaluations × background rows</code>. Compare candidate configurations with a larger reference configuration, then choose the smallest one that keeps the top cost drivers, their signs, and dollar impacts stable while supporting the NFR-04 prediction request latency target (P95 &lt;1 second server-side under subsequent-call conditions on the target hardware). Stable top drivers and signs matter more than exact dollar impacts for low-ranked features.
#     <br><br>
#     <strong>Communicating SHAP Values</strong>
#     <ul>
#         <li><strong>End users (cost-driver overview):</strong> "These factors show which of your answers moved your estimate up or down the most."</li>
#         <li><strong>End users (single cost driver):</strong> "Your insurance answer, <code>Public Only</code>, lowered this estimate by about \$99."</li>
#         <li><strong>Non-technical stakeholders:</strong> "The estimate starts from an average predicted cost for a representative sample of U.S. adults. Each person's inputs then move the estimate up or down from that starting point. Because each feature is evaluated in the context of that person's other features, the same answer can have a different dollar impact for different people. For example, a <code>Public Only</code> insurance answer can affect the estimate differently for someone with several chronic conditions than for someone with none."</li>
#     </ul>
#     <strong>SHAP Limitations</strong>
#     <ul>
#         <li><strong>Not Causal:</strong> SHAP explains how each answer contributed to the model's estimate relative to the SHAP background. It does not explain what caused the person's actual costs. The explanation inherits any errors or missing relationships in the model. More fundamentally, the model learned associations from observational survey data, not the effects of changing someone's circumstances. For example, if smoking moved the estimate up by \$200, that does not mean quitting smoking would cause costs to fall by $200. Changing the smoking answer may lower the estimate, but this does not show what quitting would do to the person’s actual costs.</li>
#         <li><strong>Background-Dependent:</strong> SHAP explains a prediction relative to the selected background data. Changing that reference group can change the baseline and the contribution assigned to each answer. For example, age 70 may move an estimate up substantially when the person is compared with U.S. adults of all ages, but it may contribute much less when the reference group contains only adults aged 65 and older. The person's prediction does not change, but the explanation of how it differs from the reference does.</li>
#         <li><strong>Correlated Features:</strong> EDA shows notable correlations between ADL/IADL help (Spearman's ρ=0.60) and between arthritis/joint pain (ρ=0.56). When features are correlated, SHAP can distribute credit unevenly between them. For example, arthritis and joint pain both signal similar health burden, but one prediction may assign more dollar impact to arthritis and another to joint pain. Interpret correlated features as a group and do not treat small ranking differences between them as meaningful.</li>
#         <li><strong>Predicted, Not Actual Costs:</strong> SHAP explains the model's prediction, not the true drivers of real-world medical costs. If the model overstates, understates, or misses a relationship, the SHAP values reflect that error. SHAP does not fix model limitations: if the model underpredicts high-cost cases, reflects noisy survey data, or lacks important predictors, SHAP explains those imperfect predictions.</li>
#     </ul>
#     <strong>SHAP Metadata</strong><br>
#     Store metadata alongside the SHAP background artifact for developer and prediction-service auditability. It records the model and preprocessor artifacts, explainer configuration, explained output, background sampling method, and background validation results (see the <a href="../docs/specs/technical_specifications.md#shap-metadata-contract">SHAP Metadata Artifact Contract</a> in the technical specifications).
#     <br><br>
#     <strong>App/API Implementation Plan</strong>
#     <ol>
#         <li><strong>Create Artifacts:</strong> Use <code>scripts/preprocess.py</code> to create <code>data/training_data_preprocessor_input.parquet</code> and <code>models/preprocessor.joblib</code>. Use <code>scripts/train_xgboost_quantile.py</code> to create <code>models/xgb_quantile_model.joblib</code>. Use <code>scripts/build_app_artifacts.py</code> to create <code>app/data/shap_background.parquet</code> and <code>app/data/shap_metadata.json</code>.</li>
#         <li><strong>During Application Startup:</strong> Load the preprocessor, quantile model, and SHAP background. Build the explainer once.</li>
#         <li><strong>At Inference Time:</strong> Map the user inputs to the preprocessor inputs. Predict all quantiles using the fitted preprocessor and model, postprocess them, and compute permutation SHAP for q50. Rank contributions by absolute value and select the five largest. Apply medical-cost inflation to predictions, comparison benchmarks, and the selected SHAP contributions.</li>
#     </ol>
#     API response and privacy requirements are defined in the <a href="../docs/specs/technical_specifications.md#api-contract">API Contract</a> and <a href="../docs/specs/technical_specifications.md#privacy-preserving-monitoring">Privacy-Preserving Monitoring</a> sections of the technical specifications.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">SHAP Explainer Setup</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Set up the SHAP explainer with the preprocessor, model, background data (derived from training data), and prediction function.
# </div>

# %%
# 1. Prepare SHAP inputs and load the data (with preprocessor input features), preprocessing pipeline and model artifacts
SHAP_INPUT_FEATURES = (
    PIPELINE_NUMERICAL_FEATURES
    + PIPELINE_NOMINAL_FEATURES
    + PIPELINE_BINARY_FEATURES
)

X_train_preprocessor_input = pd.read_parquet(
    TRAIN_PREPROCESSOR_INPUT_DATA_PATH,
    columns=SHAP_INPUT_FEATURES,
)
X_val_preprocessor_input = pd.read_parquet(
    VAL_PREPROCESSOR_INPUT_DATA_PATH,
    columns=SHAP_INPUT_FEATURES,
)
X_test_preprocessor_input = pd.read_parquet(
    TEST_PREPROCESSOR_INPUT_DATA_PATH,
    columns=SHAP_INPUT_FEATURES,
)

preprocessor = load_model("../models/preprocessor.joblib", verbose=False)
xgb_quantile_model = load_model("../models/xgb_quantile_model.joblib", verbose=False)


# %%
# 2. Define the prediction callable that SHAP will explain
def predict_median_cost(X):
    """Predict postprocessed q50 cost from preprocessor input features."""
    if isinstance(X, pd.DataFrame):
        X_preprocessor_input = X.loc[:, SHAP_INPUT_FEATURES]
    else:
        # SHAP arrays follow the column order defined by the background data.
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(SHAP_INPUT_FEATURES):
            raise ValueError(
                "SHAP input must have shape "
                f"(n_rows, {len(SHAP_INPUT_FEATURES)})."
            )
        X_preprocessor_input = pd.DataFrame(X, columns=SHAP_INPUT_FEATURES)

    X_model_ready = preprocessor.transform(X_preprocessor_input)
    quantile_predictions = xgb_quantile_model.predict(X_model_ready)
    return postprocess_quantile_predictions(quantile_predictions)[:, 1]

# %%
# Artifact consistency check: confirm that the saved preprocessor inputs and preprocessor reproduce the model-ready training features
X_train_reprocessed = preprocessor.transform(X_train_preprocessor_input)
pd.testing.assert_frame_equal(
    X_train_reprocessed,
    X_train_preprocessed,
    check_exact=False,
    rtol=0,
    atol=1e-12,
)
del X_train_reprocessed

# %%
# 3. Create and validate the background sample using the selected SHAP configuration
SHAP_BACKGROUND_N = 225
SHAP_PERMUTATION_ROUNDS = 1
SHAP_MASKS_PER_ROUND = 2 * len(SHAP_INPUT_FEATURES) + 1
SHAP_MAX_EVALS = SHAP_PERMUTATION_ROUNDS * SHAP_MASKS_PER_ROUND
SHAP_BASELINE_REL_DIFF_MAX = 0.10

shap_background = X_train_preprocessor_input.sample(
    n=SHAP_BACKGROUND_N,
    weights=w_train,
    replace=True,
    random_state=RANDOM_STATE,
)

background_baseline = predict_median_cost(shap_background).mean()
training_baseline = np.average(
    predict_median_cost(X_train_preprocessor_input),
    weights=w_train,
)
baseline_absolute_relative_difference = abs(background_baseline / training_baseline - 1)

print(f"SHAP background baseline: ${background_baseline:,.2f}")
print(f"Full training baseline:   ${training_baseline:,.2f}")
print(f"Absolute relative difference: {baseline_absolute_relative_difference:.1%}")

if baseline_absolute_relative_difference > SHAP_BASELINE_REL_DIFF_MAX:
    raise ValueError(
        "SHAP background baseline differs from the weighted training baseline by "
        f"{baseline_absolute_relative_difference:.1%}, which exceeds the "
        f"{SHAP_BASELINE_REL_DIFF_MAX:.0%} acceptance threshold. "
        "Resample the background data or increase SHAP_BACKGROUND_N."
    )

# %%
# 4. Build the explainer and define the explanation function
shap_masker = shap.maskers.Independent(
    shap_background,
    max_samples=SHAP_BACKGROUND_N,
)
explainer = shap.Explainer(
    predict_median_cost,
    shap_masker,
    algorithm="permutation",
    seed=RANDOM_STATE,
)


def calculate_shap_explanation(X):
    """Calculate a 2023-dollar SHAP explanation for one or more input rows."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("SHAP inputs must be provided as a pandas DataFrame.")

    missing_features = [
        feature for feature in SHAP_INPUT_FEATURES if feature not in X.columns
    ]
    if missing_features:
        raise ValueError(
            "SHAP input is missing preprocessor input features: "
            f"{missing_features}"
        )

    return explainer(
        X.loc[:, SHAP_INPUT_FEATURES],
        max_evals=SHAP_MAX_EVALS,
        silent=True,
    )

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Local Prediction Explanation</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Calculate a SHAP explanation for one example test row.
# </div>

# %%
example_idx = 0
X_test_example = X_test_preprocessor_input.iloc[[example_idx]]
shap_explanation = calculate_shap_explanation(X_test_example)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Explanation summary (for single row).
# </div>

# %%
baseline = shap_explanation.base_values[0]
example_prediction = predict_median_cost(X_test_example)[0]
example_actual = y_test.loc[X_test_example.index[0]]
example_shap_sum = shap_explanation.values[0].sum()

example_shap_result = pd.DataFrame({
    "Metric": [
        "Baseline",
        "Feature contribution sum",
        "Predicted median cost",
        "Actual cost",
        "Baseline + SHAP sum",
        "Additivity error",
    ],
    "Value": [
        baseline,
        example_shap_sum,
        example_prediction,
        example_actual,
        baseline + example_shap_sum,
        abs(example_prediction - (baseline + example_shap_sum)),
    ],
})

display(
    example_shap_result.style
    .pipe(add_table_caption, f"SHAP Explanation Summary (Test Row {example_idx})")
    .format({"Value": "${:,.2f}"})
    .hide()
)
# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Interpretation:</b> 
#     <ul>
#         <li><strong>For non-technical stakeholders:</strong> The explanation uses \$333 as its comparison point. This is the average of the model’s predicted median costs across a representative group of U.S. adults. This person’s inputs moved the model estimate down by \$251, resulting in a predicted median cost of \$82.</li>
#         <li><strong>For end users:</strong> Your estimate is \$82, which is \$251 below the Medical Cost Planner&rsquo;s average estimate of \$333 for U.S. adults. <br><small>Note: This comparison amount is based on estimates for a representative group of U.S. adults.</small></li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Contribution breakdown by feature (for single row).
# </div>

# %%
def format_shap_input(feature, value):
    """Return one preprocessor input value in a readable format."""
    if pd.isna(value):
        return "Missing"

    category_labels = CATEGORY_LABELS_EDA.get(feature)
    if category_labels is not None:
        try:
            category_key = int(value)
        except (TypeError, ValueError):
            category_key = value
        value = category_labels.get(category_key, value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isclose(value, round(value)):
            return f"{value:,.0f}"
        return f"{value:,.1f}"
    return value


example_row = X_test_example.iloc[0]
example_shap_feature_values = pd.DataFrame({
    "Feature": [
        DISPLAY_LABELS.get(feature, feature)
        for feature in SHAP_INPUT_FEATURES
    ],
    "Input Value": [
        format_shap_input(feature, example_row[feature])
        for feature in SHAP_INPUT_FEATURES
    ],
    "SHAP Contribution (2023 USD)": shap_explanation.values[0],
}).sort_values(
    "SHAP Contribution (2023 USD)",
    key=lambda values: values.abs(),
    ascending=False,
)

display(
    example_shap_feature_values.style
    .pipe(
        add_table_caption,
        f"SHAP Contribution Breakdown by Feature (Test Row {example_idx})",
    )
    .format({
        "SHAP Contribution (2023 USD)": lambda value: f"{'-' if value < 0 else ''}${abs(value):,.1f}",
    })
    .hide()
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Interpretation:</b>
#     <ul>
#         <li><strong>Education:</strong> The answer &ldquo;No Degree&rdquo; moved the plan-around estimate down by about \$132.</li>
#         <li><strong>Insurance:</strong> The answer &ldquo;Public Only&rdquo; moved the estimate down by about \$104</li>
#         <li><strong>Usual Source of Care:</strong> The answer &ldquo;No&rdquo; moved the estimate down by about \$80.</li>
#         <li><strong>Age:</strong> The entered age of 70 moved the estimate up by about \$48.</li>
#         <li><strong>Joint Pain:</strong> The answer &ldquo;No&rdquo; moved the estimate down by about \$47.</li>
#     </ul>
#     <em>Note: These are local contributions relative to the SHAP background and depend on the person's other answers. They are not comparisons with specific alternative answers. They explain predicted, not actual, costs and should not be interpreted causally. For example, they do not show how changing public to private insurance or stopping to smoke would change a person's costs.</em>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">SHAP Benchmarking</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <strong>Benchmarking Plan</strong><br>
#     <strong>Goal:</strong> Identify the least computationally expensive combination of permutation rounds and background size that produces stable explanations while supporting the prediction request latency requirement.
#     <br><br>
#     <strong>Implementation:</strong> The notebook documents the evaluation plan and reviews the results. The single source of truth for the benchmarking code implementation is the executable <a href="../scripts/benchmark_shap.py"><code>scripts/benchmark_shap.py</code></a>. See the technical specification for the complete <a href="../docs/specs/technical_specifications.md#latency-definitions-and-measurement">latency definitions and measurement boundaries</a>.
#     <ul>
#         <li><strong>Latency requirement:</strong> For requests that include a SHAP explanation, P95 prediction request latency (server-side) must be less than one second under subsequent-call conditions on the target hardware. Measure first-call latency separately. Target end-to-end latency (user-perceived) is approximately three seconds.</li>
#         <li><strong>Core SHAP explanation latency:</strong> The benchmark measures one <code>explainer(...)</code> call for one validation or test row at a time. This includes the repeated masked predictions through the complete q50 callable: preprocessing, quantile prediction, inverse target transformation, quantile postprocessing, and q50 selection. It excludes the other server work, network transfer, and interface rendering.</li>
#         <li><strong>First-call and subsequent-call SHAP latency:</strong> For each candidate, build the explainer outside the timer and measure its first call separately. This is the first call for that explainer, not a full application cold start. Then measure the remaining rows individually and calculate p50, p90, and p95 from those subsequent calls.</li>
#         <li><strong>Background data validation:</strong> Compare the baseline (mean postprocessed q50) of each candidate background sample against the full weighted training baseline. Accept a candidate only if the absolute relative difference is at most 10%.</li>
#         <li><strong>Candidate grid:</strong> Benchmark background sizes <code>[225, 250, 275, 300]</code> and SHAP evaluation budgets (<code>max_evals</code>) <code>[55, 110, 165]</code>, equal to 1, 2, and 3 permutation rounds. With 27 preprocessor input features, one permutation round uses <code>2 * 27 + 1 = 55</code> masks because SHAP evaluates one forward and one backward pass through a feature ordering plus the baseline mask. Note: This grid refines an initial broader screen of background sizes [50, 100, 200, 300] and 3, 6, and 12 rounds, which showed that the smaller backgrounds failed the initial representativeness gate and additional permutation rounds increased latency without meaningful stability gains.</li>
#         <li><strong>Reference:</strong> Compare candidates against a reference configuration with a larger background size (<code>500</code>) and higher evaluation budget (<code>max_evals=1,320</code>, or 24 permutation rounds).</li>
#         <li><strong>Explanation stability:</strong>
#             <ul>
#                 <li><strong>Top-five overlap (primary metric):</strong> For at least 90% of evaluation rows, require at least four of the five drivers to match the reference.</li>
#                 <li><strong>Direction agreement:</strong> A matched reference contribution of at least \$25 must not change from increasing to decreasing the estimate, or vice versa.</li>
#                 <li><strong>Dollar difference:</strong> Use an initial tolerance of \$25 in 2023 dollars for the median absolute difference among matched top-five contributions.</li>
#             </ul>
#         </li>
#         <li><strong>Correctness checks:</strong> Require background validation to pass and additivity error to remain near zero.</li>
#         <li><strong>Stage 1 - Candidate Screening:</strong> Evaluate all 12 candidates on the same 20 validation rows. Mark candidates that fail background validation, are clearly too slow, or produce unstable explanations as unsuitable for Stage 2.</li>
#         <li><strong>Stage 2 - Shortlist Evaluation:</strong> Evaluate the three most promising candidates on the same 100 validation rows. Keep these rows separate from the Stage 1 and first-call rows.</li>
#         <li><strong>Final Test-Set Evaluation:</strong> Freeze the winning configuration from Stage 2. Evaluate it once on 100 test rows plus one separate first-call row. Use this final evaluation to confirm that explanation stability and correctness generalize. Afterward, measure prediction request latency (server-side) on the intended Hugging Face hardware.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 After running the Stage 1 candidate screening with <code>scripts/benchmark_shap.py stage1</code>, load the generated <code>.csv</code> results and display a decision table to shortlist Stage 2 candidates.
# </div>

# %%
shap_stage_1_results = pd.read_csv(
    "../models/shap_benchmark_stage1_results.csv"
)

shap_stage_1_decision_table = (
    shap_stage_1_results
    .sort_values(
        ["explanation_stability_passed", "p95_subsequent_call_latency_s"],
        ascending=[False, True],
    )
    .assign(
        background_validation=lambda df: np.where(
            df["background_baseline_validation_passed"],
            "Pass",
            "Fail",
        ),
        explanation_stability_gate=lambda df: np.where(
            df["explanation_stability_passed"],
            "Pass",
            "Fail",
        ),
        p95_latency_screen=lambda df: np.where(
            df["p95_subsequent_call_latency_s"] < 1.0,
            "Pass",
            "Review",
        ),
    )
    .rename(columns={
        "background_size": "Background Rows",
        "permutation_rounds": "Permutation Rounds",
        "background_baseline_absolute_relative_difference": (
            "Background vs. Training Difference"
        ),
        "first_call_latency_s": "First-Call Latency",
        "p50_subsequent_call_latency_s": "P50 Latency",
        "p95_subsequent_call_latency_s": "P95 Latency",
        "share_rows_with_at_least_4_of_5_matches": "Top-5 Match Rate",
        "material_direction_reversal_count": "Direction Reversals",
        "median_matched_top_5_abs_delta_2023_usd": (
            "Median Contribution Difference"
        ),
        "background_validation": "Background Validation",
        "explanation_stability_gate": "Explanation Stability",
        "p95_latency_screen": "P95 < 1 s",
    })
    [[
        "Background Rows",
        "Permutation Rounds",
        "Background vs. Training Difference",
        "Background Validation",
        "First-Call Latency",
        "P50 Latency",
        "P95 Latency",
        "Top-5 Match Rate",
        "Direction Reversals",
        "Median Contribution Difference",
        "Explanation Stability",
        "P95 < 1 s",
    ]]
)

display(
    shap_stage_1_decision_table.style
    .pipe(add_table_caption, "SHAP Benchmarking Stage 1: Candidate Configurations (20 Validation Rows)")
    .format({
        "Background vs. Training Difference": "{:.1%}",
        "First-Call Latency": "{:.2f} s",
        "P50 Latency": "{:.2f} s",
        "P95 Latency": "{:.2f} s",
        "Top-5 Match Rate": "{:.0%}",
        "Direction Reversals": "{:.0f}",
        "Median Contribution Difference": DOLLAR + "{:,.2f}",
    })
    .background_gradient(
        cmap="RdYlGn_r",
        subset=["P95 Latency"],
    )
    .map(
        style_status_cells,
        subset=[
            "Background Validation",
            "Explanation Stability",
            "P95 < 1 s",
        ],
    )
    .hide()
)

# %% [markdown]
# <em>Note: P50 and P95 summarize subsequent SHAP explanation calls after the separately measured first call. They exclude other prediction-request work. All candidates passed the additivity check: the SHAP baseline plus all 27 feature contributions reproduced q50 to floating-point precision.</em>

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <strong>Insights:</strong>
#     <ul>
#         <li>All 12 candidate SHAP configurations passed background validation, explanation stability, and the additivity check.</li>
#         <li>One permutation round produced a 95% top-five match rate, which means 19 of the 20 validation rows passed. Two and three rounds produced 100%. This sample is too small to know whether the extra round provides a consistent improvement.</li>
#         <li><strong>Stage 2 Shortlist:</strong>
#             <ul>
#                 <li><code>225 background rows, 1 round</code>: the fastest configuration.</li>
#                 <li><code>250 background rows, 1 round</code>: a slightly larger background with almost the same latency.</li>
#                 <li><code>225 background rows, 2 rounds</code>: tests whether a second round improves explanation stability enough to justify the added latency.</li>
#             </ul>
#         </li>
#         <li>Stage 2 will compare these configurations on 100 separate validation rows. After Stage 2, confirm the selected configuration once on test data and measure complete P95 prediction request latency on the target Hugging Face hardware.</li>
#     </ul>
# </div>
#

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 After running the Stage 2 shortlist evaluation with <code>scripts/benchmark_shap.py stage2</code>, load the results and display a decision table to select the production configuration.
# </div>

# %%
shap_stage_2_results = pd.read_csv(
    "../models/shap_benchmark_stage2_results.csv"
)

shap_stage_2_decision_table = (
    shap_stage_2_results
    .sort_values(
        [
            "explanation_stability_passed",
            "p95_subsequent_call_latency_s",
        ],
        ascending=[False, True],
    )
    .assign(
        background_validation=lambda df: np.where(
            df["background_baseline_validation_passed"],
            "Pass",
            "Fail",
        ),
        explanation_stability_gate=lambda df: np.where(
            df["explanation_stability_passed"],
            "Pass",
            "Fail",
        ),
        p95_latency_screen=lambda df: np.where(
            df["p95_subsequent_call_latency_s"] < 1.0,
            "Pass",
            "Review",
        ),
    )
    .rename(columns={
        "background_size": "Background Rows",
        "permutation_rounds": "Permutation Rounds",
        "background_baseline_absolute_relative_difference": (
            "Background vs. Training Difference"
        ),
        "first_call_latency_s": "First-Call Latency",
        "p50_subsequent_call_latency_s": "P50 Latency",
        "p95_subsequent_call_latency_s": "P95 Latency",
        "share_rows_with_at_least_4_of_5_matches": "Top-5 Match Rate",
        "material_direction_reversal_count": "Direction Reversals",
        "median_matched_top_5_abs_delta_2023_usd": (
            "Median Contribution Difference"
        ),
        "background_validation": "Background Validation",
        "explanation_stability_gate": "Explanation Stability",
        "p95_latency_screen": "P95 < 1 s",
    })
    [[
        "Background Rows",
        "Permutation Rounds",
        "Background vs. Training Difference",
        "Background Validation",
        "First-Call Latency",
        "P50 Latency",
        "P95 Latency",
        "Top-5 Match Rate",
        "Direction Reversals",
        "Median Contribution Difference",
        "Explanation Stability",
        "P95 < 1 s",
    ]]
)

display(
    shap_stage_2_decision_table.style
    .pipe(add_table_caption, "SHAP Benchmarking Stage 2: Shortlisted Configurations (100 Validation Rows)")
    .format({
        "Background vs. Training Difference": "{:.1%}",
        "First-Call Latency": "{:.2f} s",
        "P50 Latency": "{:.2f} s",
        "P95 Latency": "{:.2f} s",
        "Top-5 Match Rate": "{:.0%}",
        "Direction Reversals": "{:.0f}",
        "Median Contribution Difference": DOLLAR + "{:,.2f}",
    })
    .background_gradient(
        cmap="RdYlGn_r",
        subset=["P95 Latency"],
    )
    .map(
        style_status_cells,
        subset=[
            "Background Validation",
            "Explanation Stability",
            "P95 < 1 s",
        ],
    )
    .hide()
)

# %% [markdown]
# <em>Note: P50 and P95 summarize subsequent SHAP explanation calls after the separately measured first call. They exclude other prediction-request work. All candidates passed the additivity check: the SHAP baseline plus all 27 feature contributions reproduced q50 to floating-point precision.</em>

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <strong>Insights:</strong>
#     <ul>
#         <li>All three shortlisted SHAP configurations passed background validation, explanation stability, and the additivity check.</li>
#         <li><strong>Decision:</strong> <code>225 background rows, 1 permutation round</code> (<code>max_evals=55</code>).</li>
#         <li><strong>Justification:</strong> It is the fastest configuration and still matched at least four of the reference top-five drivers for 99 of 100 validation rows, with no material direction reversals and a median contribution difference of \$6.36.</li>
#         <li>Increasing the background to 250 rows produced the same 99% top-five match rate and improved the median contribution difference by only \$0.62, while increasing P95 core SHAP latency from 0.38 to 0.42 seconds.</li>
#         <li>Using two rounds improved the match rate from 99% to 100% and the median contribution difference by \$1.59, but more than doubled P95 core SHAP latency from 0.38 to 0.80 seconds.</li>
#         <li>The selected configuration is the fastest candidate that passes background validation and explanation stability in line with the predetermined decision rule.</li>
#         <li>Before deployment, confirm this configuration once on held-out test data and measure complete P95 prediction request latency on the target Hugging Face hardware. Report first-call latency separately.</li>
#     </ul>
# </div>
#
# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 After running the final test-set evaluation with <code>scripts/benchmark_shap.py test</code>, load and display the result for the fixed production configuration (225 background rows and 1 permutation round).
# </div>

# %%
shap_test_results = pd.read_csv(
    "../models/shap_benchmark_test_results.csv"
)

shap_test_evaluation_table = (
    shap_test_results
    .assign(
        background_validation=lambda df: np.where(
            df["background_baseline_validation_passed"],
            "Pass",
            "Fail",
        ),
        explanation_stability=lambda df: np.where(
            df["explanation_stability_passed"],
            "Pass",
            "Fail",
        ),
        additivity_check=lambda df: np.where(
            df["p95_additivity_abs_error_2023_usd"] <= 0.01,
            "Pass",
            "Fail",
        ),
        final_evaluation=lambda df: np.where(
            df["background_baseline_validation_passed"]
            & df["explanation_stability_passed"]
            & (df["p95_additivity_abs_error_2023_usd"] <= 0.01),
            "Pass",
            "Fail",
        ),
    )
    .rename(columns={
        "background_size": "Background Rows",
        "permutation_rounds": "Permutation Rounds",
        "background_baseline_absolute_relative_difference": (
            "Background vs. Training Difference"
        ),
        "first_call_latency_s": "First-Call Latency",
        "p50_subsequent_call_latency_s": "P50 Latency",
        "p95_subsequent_call_latency_s": "P95 Latency",
        "share_rows_with_at_least_4_of_5_matches": "Top-5 Match Rate",
        "material_direction_reversal_count": "Direction Reversals",
        "median_matched_top_5_abs_delta_2023_usd": (
            "Median Contribution Difference"
        ),
        "background_validation": "Background Validation",
        "explanation_stability": "Explanation Stability",
        "additivity_check": "Additivity Check",
        "final_evaluation": "Final Evaluation",
    })
    [[
        "Background Rows",
        "Permutation Rounds",
        "Background vs. Training Difference",
        "Background Validation",
        "First-Call Latency",
        "P50 Latency",
        "P95 Latency",
        "Top-5 Match Rate",
        "Direction Reversals",
        "Median Contribution Difference",
        "Explanation Stability",
        "Additivity Check",
        "Final Evaluation",
    ]]
)

display(
    shap_test_evaluation_table.style
    .pipe(add_table_caption, "SHAP Benchmarking: Final Configuration Evaluation (100 Test Rows)")
    .format({
        "Background vs. Training Difference": "{:.1%}",
        "First-Call Latency": "{:.2f} s",
        "P50 Latency": "{:.2f} s",
        "P95 Latency": "{:.2f} s",
        "Top-5 Match Rate": "{:.0%}",
        "Direction Reversals": "{:.0f}",
        "Median Contribution Difference": DOLLAR + "{:,.2f}",
    })
    .map(
        style_status_cells,
        subset=[
            "Background Validation",
            "Explanation Stability",
            "Additivity Check",
            "Final Evaluation",
        ],
    )
    .hide()
)

# %% [markdown]
# <em>Note: P50 and P95 summarize subsequent SHAP explanation calls after the separately measured first call. They exclude other prediction-request work and do not replace complete request-latency measurement on the target Hugging Face hardware.</em>

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <strong>Insights:</strong>
#     <ul>
#         <li>The final configuration passed background validation, explanation stability, and the additivity check on 100 test rows.</li>
#         <li>It matched at least four of the reference top-five drivers for 98 of 100 rows, had no material direction reversals, and had a median contribution difference of \$6.32.</li>
#         <li>Core SHAP explanation latency was 0.31 seconds at P50 and 0.65 seconds at P95. The higher P95 on test than validation (0.38 seconds) reinforces the need to measure complete prediction-request latency on the target Hugging Face hardware.</li>
#     </ul>
# </div>
#
# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Feature Importance</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Feature importance will be evaluated with two complementary approaches:
#     <ul>
#         <li><strong>SHAP feature importance</strong> ranks the 27 preprocessor input features by their mean absolute SHAP contribution across predictions (survey-weighted). It shows which interpretable inputs have the largest average impact on the postprocessed q50 prediction, measured in 2023 dollars.</li>
#         <li><strong>XGBoost native feature importance</strong> ranks the 40 model-ready features primarily by their share of <code>total_gain</code>. It shows how much each feature reduced the training objective across the joint q25, q50, q75, and q90 estimator. It is not q50-specific and is not measured in dollars.</li>
#     </ul>
#     The rankings are not directly comparable because they use different features, outputs, units, and data stages. Neither approach is causal or measures a feature's true effect on medical costs. Instead, they provide complementary views: SHAP shows what drives the model's q50 predictions, while native importance shows what the model used during training. Agreement strengthens confidence that the model relies on plausible signals. Differences identify features, transformations, derived features, or correlations that need closer review.
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">SHAP</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ SHAP feature importance is examined with four complementary plots:
#     <ul>
#         <li><strong>Overall Feature Importance (Bar Plot):</strong> Shows which features matter most overall by ranking their mean absolute SHAP contributions (survey-weighted). It shows typical contribution size, not whether a feature usually moves estimates up or down.</li>
#         <li><strong>Contribution Distributions (Beeswarm Plot):</strong> Shows whether features move estimates up or down and how much their contributions vary across people. For unordered categorical features, it doesn't show which category produced each contribution.</li>
#         <li><strong>Contributions by Category (Interval Plot):</strong> Shows the median and percentile ranges for each category of the unordered categorical features (survey-weighted). This reveals which categories tend to move estimates up or down and how much contributions vary within each category.</li>
#         <li><strong>Contributions Across Ordered Values (Dependence and Interval Plots):</strong> Zooms in on Age, Family Income, and Family Size, the three highest-ranked numerical or ordinal features. It shows how contributions change across their values, revealing gradients, nonlinear patterns, and plateaus that the broader beeswarm plot cannot show precisely.</li>
#     </ul>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 After running <code>scripts/audit_shap_feature_importance.py</code>, load the SHAP feature importances from the test set. Display a table of all 27 features and a bar plot of the top 15 features.
# </div>

# %%
# Load SHAP feature importances from .csv into a DataFrame
shap_feature_importance_test = pd.read_csv(
    "../models/shap_feature_importance_test.csv"
)
shap_feature_importance_test = (
    shap_feature_importance_test
    .rename(columns={
        "rank": "Rank",
        "feature": "Feature Code",
        "feature_label": "Feature",
        "mean_absolute_contribution_2023_usd": (
            "Mean Absolute Contribution"
        ),
        "share_of_total_importance": "Share of Total Importance",
    })
    [[
        "Rank",
        "Feature Code",
        "Feature",
        "Mean Absolute Contribution",
        "Share of Total Importance",
    ]]
)

# Display table of SHAP feature importance for all 27 preprocessor input features
display(
    shap_feature_importance_test
    .drop(columns="Feature Code")
    .style
    .pipe(
        add_table_caption,
        "SHAP Feature Importance (Test Set)",
    )
    .format({
        "Mean Absolute Contribution": "${:,.2f}",
        "Share of Total Importance": "{:.1%}",
    })
    .hide()
)

# %% [markdown]
# <em>Note: Mean absolute contributions are averaged across test rows using MEPS survey weights and shown in 2023 USD.</em>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Overall Feature Importance (Bar Plot)</strong><br>
#     📌 Create bar chart of the top 15 features on the test set.
# </div>

# %%
shap_top_15 = shap_feature_importance_test.head(15).sort_values(
    "Mean Absolute Contribution"
)
top_15_importance_share = shap_top_15[
    "Share of Total Importance"
].sum()

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    shap_top_15["Feature"],
    shap_top_15["Mean Absolute Contribution"],
    color=POP_COLOR,
)
ax.bar_label(
    bars,
    labels=[
        f"${contribution:,.0f} ({share:.1%})"
        for contribution, share in zip(
            shap_top_15["Mean Absolute Contribution"],
            shap_top_15["Share of Total Importance"],
        )
    ],
    padding=4,
    fontsize=9,
)
ax.set_xlim(
    0,
    shap_top_15["Mean Absolute Contribution"].max() * 1.30,
)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f"${value:,.0f}")
)
ax.set_title(
    "SHAP Feature Importance: Top 15 Features (Test Set)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.text(
    0.5,
    0.99,
    f"Top 15 share of total importance: {top_15_importance_share:.1%}",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    fontweight="normal",
)
ax.set_xlabel("Mean Absolute Contribution")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.20)
ax.set_axisbelow(True)
sns.despine(ax=ax, left=True)

fig.text(
    0.01,
    0.01,
    (
        "Note: Test-set mean absolute contributions use MEPS survey weights "
        "and 2023 USD. Percentages show each feature's share of total "
        "importance across all 27 features."
    ),
    ha="left",
    va="bottom",
    fontsize=9,
    style="italic",
    color="#4A4A4A",
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(
    "../figures/evaluation/shap_feature_importance.png",
    bbox_inches="tight",
    dpi=200,
)
plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b>
#     <ul style="margin-top:10px; margin-bottom:8px">
#         <li><strong>The model draws on four types of information:</strong>
#             <ul>
#                 <li><strong>Medical need (7 features; 30.6% of total importance):</strong> Joint Pain, High Cholesterol, Walking Limitation, High Blood Pressure, Arthritis, Cancer, and Asthma.</li>
#                 <li><strong>Demographic (4 features; 25.9% of total importance):</strong> Family Size, Sex, Age, and Marital Status.</li>
#                 <li><strong>Socioeconomic (2 features; 17.8% of total importance):</strong> Family Income and Education.</li>
#                 <li><strong>Healthcare access (2 features; 18.0% of total importance):</strong> Insurance and Usual Source of Care.</li>
#             </ul>
#         </li>
#         <li><strong>Importance is concentrated but not dominated by one feature:</strong> The top five features account for 47% of total importance, while the top 15 account for 92%. Insurance ranks first at 12.2%, followed by Family Income at 10.8%, but the model still distributes importance across many inputs.</li>
#         <li><strong>Age remains important but ranks sixth:</strong> This analysis explains median (q50) out-of-pocket costs after accounting for health conditions, limitations, insurance, and other features that capture part of the age-related signal. MEPS also top-codes age at 85, so the model cannot distinguish among people aged 85 and older. Age importance could differ for mean costs or upper quantiles, where rare high-cost cases have more influence.</li>
#         <li><strong>SHAP broadly agrees with EDA but adds model context:</strong>
#             <ul>
#                 <li>EDA correlations describe one-feature-at-a-time relationships with actual out-of-pocket costs. SHAP importance describes how the fitted model combines features to predict median out-of-pocket costs. Agreement supports model plausibility, while differences highlight nonlinear effects, interactions, shared information, or model limitations.</li>
#                 <li>Across the 20 features included in both analyses, absolute correlation strength and SHAP importance show strong rank agreement (Spearman's ρ = 0.83). Family Income, Joint Pain, Usual Source of Care, and Walking Limitation have similar relative importance.</li>
#                 <li>Family Size and Sex become more important in the model, while Arthritis becomes less important, suggesting nonlinear effects or information shared with other features.</li>
#                 <li>Mental Health, ADL Help, IADL Help, Stroke, and Smoking show weak relationships in both analyses. Their low SHAP importance therefore mirrors weak observed relationships with actual out-of-pocket costs.</li>
#             </ul>
#         </li>
#         <li><strong>SHAP bar plot shows size, not direction:</strong> The bars show the average size of each feature's contribution, not whether it moves estimates up or down. The contribution distribution plot below shows direction and variation across people.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Contribution Distributions (Beeswarm Plot)</strong><br>
#     📌 Create a contribution distribution plot for the top 15 features. Because the beeswarm plot does not accept survey weights, use weighted bootstrap sampling with replacement so the displayed dot density approximates the U.S. adult population represented by MEPS.
# </div>

# %%
# Create a sample of the test set that is representative for the population by using the same weighted bootstrap approach as in the EDA notebook.
SHAP_BEESWARM_SAMPLE_N = 5000
shap_test_contributions = pd.read_parquet(
    "../models/shap_test_contributions.parquet"
)
shap_beeswarm_sample = shap_test_contributions.sample(
    n=SHAP_BEESWARM_SAMPLE_N,
    weights=WEIGHT_COLUMN,
    replace=True,
    random_state=RANDOM_STATE,
)

# Keep the exact top 15 features from the global SHAP importance table.
shap_top_feature_names = (
    shap_feature_importance_test.head(15)["Feature Code"].tolist()
)
shap_top_feature_labels = (
    shap_feature_importance_test.head(15)["Feature"].tolist()
)

# Align the corresponding test inputs used only to color the plotted dots.
shap_beeswarm_feature_values = X_test_preprocessor_input.loc[
    shap_beeswarm_sample.index,
    shap_top_feature_names,
].copy()

# Keep unordered categories neutral instead of implying a low-to-high order.
shap_beeswarm_neutral_features = PIPELINE_NOMINAL_FEATURES + [
    "EMPST31_GRP"
]
for feature in shap_beeswarm_neutral_features:
    if feature in shap_beeswarm_feature_values:
        shap_beeswarm_feature_values[feature] = (
            shap_beeswarm_feature_values[feature].astype("category")
        )

# Plot-only mapping: blue for Male and red for Female.
if "SEX" in shap_beeswarm_feature_values:
    shap_beeswarm_feature_values["SEX"] = (
        1 - shap_beeswarm_feature_values["SEX"]
    )

shap_beeswarm_explanation = shap.Explanation(
    values=shap_beeswarm_sample[shap_top_feature_names].to_numpy(),
    data=shap_beeswarm_feature_values,
    feature_names=shap_top_feature_labels,
)

# Create beeswarm plot
fig, ax = plt.subplots(figsize=(10, 7))

# Make SHAP's dot jitter reproducible without changing NumPy's global random state.
numpy_random_state = np.random.get_state()
np.random.seed(RANDOM_STATE)
try:
    shap.plots.beeswarm(
        shap_beeswarm_explanation,
        max_display=15,
        order=np.arange(len(shap_top_feature_names)),
        color=shap.plots.colors.red_blue,
        alpha=0.45,
        s=12,
        color_bar=False,
        group_remaining_features=False,
        plot_size=None,
        ax=ax,
        show=False,
    )
finally:
    np.random.set_state(numpy_random_state)


ax.set_title(
    "SHAP Contribution Distributions: Top 15 Features (Test Set)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("SHAP Contribution to Predicted Median Cost", labelpad=10)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, _: (
            f"−${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"
        )
    )
)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)

fig.text(
    0.01,
    0.01,
    (
        "Note: Dots are test rows sampled with replacement using MEPS survey weights. Contributions are in 2023 USD. Blue to red means lower to higher,\n"
        "No to Yes, and Male to Female. Unordered categories and missing inputs are gray."
    ),
    ha="left",
    va="bottom",
    fontsize=9,
    style="italic",
    color="#4A4A4A",
)
fig.tight_layout(rect=(0, 0.06, 1, 0.99))
fig.savefig(
    "../figures/evaluation/shap_contribution_distributions.png",
    bbox_inches="tight",
    dpi=200,
)
plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b>
#     <ul style="margin-top:10px; margin-bottom:8px">
#         <li><strong>Contribution direction and size vary across people:</strong> The same feature can move predicted median costs up for some people and down for others. The wide spreads show that a feature's contribution can also be much larger for some people than for others, depending on the feature value and the other features.</li>
#         <li><strong>Directional patterns by feature group:</strong>
#             <ul>
#                 <li><strong>Medical need:</strong> Conditions and limitations generally move estimates up when present and down when absent.</li>
#                 <li><strong>Demographic:</strong> Older ages, Female, and smaller families move estimates up; younger ages, Male, and larger families move them down.</li>
#                 <li><strong>Socioeconomic:</strong> Higher family income moves estimates up and lower family income moves them down.</li>
#                 <li><strong>Healthcare access:</strong> Having a usual source of care moves estimates up, while not having one moves them down.</li>
#             </ul>
#         </li>
#         <li><strong>Unordered categories need a closer look:</strong> Insurance, Education, and Marital Status are gray, so the plot shows their contribution distributions but not which categories drive each direction.</li>
#         <li><strong>Limitations:</strong> These contributions explain q50 only. They may differ for q90 and do not show that changing an input would cause actual costs to change.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Contributions by Category (Interval Plot)</strong><br>
#     📌 Compare SHAP contributions for Insurance, Education, and Marital Status. Show the median and percentile ranges for each category (survey-weighted) so contribution direction and variation are easy to compare.
# </div>

# %%
# Summarize categorical SHAP contribution distributions with exact survey weights.
SHAP_CATEGORICAL_FEATURES = ["INSCOV23", "HIDEG", "MARRY31X_GRP"]
SHAP_CATEGORICAL_QUANTILES = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
shap_total_test_weight = shap_test_contributions[WEIGHT_COLUMN].sum()

shap_categorical_summary_rows = []
for feature in SHAP_CATEGORICAL_FEATURES:
    category_labels = (
        X_test_preprocessor_input[feature]
        .map(CATEGORY_LABELS_EDA[feature])
        .fillna("Missing")
    )
    category_order = [
        category
        for category in CATEGORY_LABELS_EDA[feature].values()
        if category_labels.eq(category).any()
    ]
    if category_labels.eq("Missing").any():
        category_order.append("Missing")

    for category in category_order:
        category_mask = category_labels.eq(category)
        contribution_quantiles = weighted_quantile(
            shap_test_contributions.loc[category_mask, feature],
            shap_test_contributions.loc[category_mask, WEIGHT_COLUMN],
            SHAP_CATEGORICAL_QUANTILES,
        )
        shap_categorical_summary_rows.append({
            "Feature": feature,
            "Category": category,
            "Population Share": (
                shap_test_contributions.loc[
                    category_mask,
                    WEIGHT_COLUMN,
                ].sum()
                / shap_total_test_weight
            ),
            "P10": contribution_quantiles[0],
            "P25": contribution_quantiles[1],
            "Median": contribution_quantiles[2],
            "P75": contribution_quantiles[3],
            "P90": contribution_quantiles[4],
        })

shap_categorical_summary = pd.DataFrame(shap_categorical_summary_rows)

category_counts = [
    shap_categorical_summary["Feature"].eq(feature).sum()
    for feature in SHAP_CATEGORICAL_FEATURES
]
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10, 8),
    sharex=True,
    gridspec_kw={"height_ratios": category_counts},
)
shap_currency_formatter = plt.FuncFormatter(
    lambda value, _: (
        f"−${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"
    )
)

for ax, feature in zip(axes, SHAP_CATEGORICAL_FEATURES):
    feature_summary = (
        shap_categorical_summary
        .loc[shap_categorical_summary["Feature"].eq(feature)]
        .reset_index(drop=True)
    )
    category_positions = np.arange(len(feature_summary))

    ax.hlines(
        category_positions,
        feature_summary["P10"],
        feature_summary["P90"],
        color="#AEB8C2",
        linewidth=2,
    )
    ax.hlines(
        category_positions,
        feature_summary["P25"],
        feature_summary["P75"],
        color=POP_COLOR,
        linewidth=6,
    )
    ax.scatter(
        feature_summary["Median"],
        category_positions,
        color=POP_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=45,
        zorder=3,
    )
    ax.axvline(0, color="#4A4A4A", linewidth=1)
    category_tick_labels = [
        f"{category} ({population_share:.0%})"
        for category, population_share in zip(
            feature_summary["Category"],
            feature_summary["Population Share"],
        )
    ]
    ax.set_yticks(category_positions, category_tick_labels)
    ax.set_ylim(len(feature_summary) - 0.5, -0.5)
    ax.set_title(
        DISPLAY_LABELS[feature],
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelbottom=True)
    ax.xaxis.set_major_formatter(shap_currency_formatter)

axes[-1].set_xlabel(
    "SHAP Contribution to Predicted Median Cost",
    labelpad=10,
)
fig.suptitle(
    "SHAP Contributions by Category (Test Set)",
    fontsize=13,
    fontweight="bold",
    x=0.5,
    y=0.99,
)
fig.text(
    0.01,
    0.01,
    (
        "Note: Dots show survey-weighted medians, blue bars the 25th–75th percentiles, and gray lines the 10th–90th. Values are in 2023 USD.\n"
        "Percentages show weighted population shares."
    ),
    ha="left",
    va="bottom",
    fontsize=9,
    style="italic",
    color="#4A4A4A",
)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.8)
fig.savefig(
    "../figures/evaluation/shap_categorical_contributions.png",
    bbox_inches="tight",
    dpi=200,
)
plt.show()
# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b>
#     <ul style="margin-top:10px; margin-bottom:8px">
#         <li><strong>Insurance moves estimates the most:</strong> Private insurance moves estimates up by about \$70 (weighted median), while public-only insurance moves them down by about \$142 and being uninsured by about \$85. In this test-set analysis, every insurance contribution was positive for private insurance and negative for public-only insurance or no insurance.</li>
#         <li><strong>Higher education generally moves estimates up:</strong> No Degree, GED, and HS Diploma generally move estimates down, while Bachelor's, Master's, and Doctorate generally move them up. The pattern is not perfectly ordered, with Master's showing the largest positive median contribution.</li>
#         <li><strong>Marital Status plays a smaller role:</strong> Married, Widowed, and Divorced move estimates slightly up, Never Married moves them down, and Separated is centered near zero.</li>
#         <li><strong>Access and utilization may explain the pattern:</strong> Higher family income, higher educational attainment, private insurance, and having a usual source of care generally move q50 estimates up. This matches the EDA and may reflect greater healthcare access and use and, for private insurance, deductibles and other cost sharing rather than greater medical need alone. These analyses identify the pattern but do not establish its cause.</li>
#         <li><strong>Contribution size varies within categories:</strong> The percentile ranges show that a category's contribution depends on the person's other features.</li>
#         <li><strong>Limitations:</strong> Interpret sparse categories such as Separated (1%), Doctorate (3%), and GED (4%) cautiously. These are model attributions, not causal effects, and do not show what would happen if someone changed categories.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Contributions Across Ordered Values (Dependence and Interval Plots)</strong><br>
#     📌 Examine Age, Family Income, and Family Size in more detail. Use a dependence plot for Age and interval plots for Family Income and Family Size to show how SHAP contributions change across ordered values.
# </div>

# %%
# Calculate a five-year rolling weighted median for the Age dependence plot.
age_values = X_test_preprocessor_input["AGE23X"]
age_trend_rows = []
for age in range(int(age_values.min()), int(age_values.max()) + 1):
    age_window = age_values.between(age - 2, age + 2)
    age_trend_rows.append({
        "Age": age,
        "Median": weighted_quantile(
            shap_test_contributions.loc[age_window, "AGE23X"],
            shap_test_contributions.loc[age_window, WEIGHT_COLUMN],
            0.50,
        ),
    })
age_trend = pd.DataFrame(age_trend_rows)

# Summarize the ordered Family Income and Family Size contributions.
shap_ordered_summary_rows = []
for income_value, income_label in CATEGORY_LABELS_EDA["POVCAT23"].items():
    income_mask = X_test_preprocessor_input["POVCAT23"].eq(income_value)
    income_quantiles = weighted_quantile(
        shap_test_contributions.loc[income_mask, "POVCAT23"],
        shap_test_contributions.loc[income_mask, WEIGHT_COLUMN],
        SHAP_CATEGORICAL_QUANTILES,
    )
    shap_ordered_summary_rows.append({
        "Feature": "POVCAT23",
        "Value": income_label,
        "Population Share": (
            shap_test_contributions.loc[
                income_mask,
                WEIGHT_COLUMN,
            ].sum()
            / shap_total_test_weight
        ),
        "P10": income_quantiles[0],
        "P25": income_quantiles[1],
        "Median": income_quantiles[2],
        "P75": income_quantiles[3],
        "P90": income_quantiles[4],
    })

family_size_groups = [
    (str(family_size), X_test_preprocessor_input["FAMSZE23"].eq(family_size))
    for family_size in range(1, 6)
]
family_size_groups.append(("6+", X_test_preprocessor_input["FAMSZE23"].ge(6)))
for family_size_label, family_size_mask in family_size_groups:
    family_size_quantiles = weighted_quantile(
        shap_test_contributions.loc[family_size_mask, "FAMSZE23"],
        shap_test_contributions.loc[family_size_mask, WEIGHT_COLUMN],
        SHAP_CATEGORICAL_QUANTILES,
    )
    shap_ordered_summary_rows.append({
        "Feature": "FAMSZE23",
        "Value": family_size_label,
        "Population Share": (
            shap_test_contributions.loc[
                family_size_mask,
                WEIGHT_COLUMN,
            ].sum()
            / shap_total_test_weight
        ),
        "P10": family_size_quantiles[0],
        "P25": family_size_quantiles[1],
        "Median": family_size_quantiles[2],
        "P75": family_size_quantiles[3],
        "P90": family_size_quantiles[4],
    })

shap_ordered_summary = pd.DataFrame(shap_ordered_summary_rows)

# Create one figure with a wide Age panel above two compact interval plots.
fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[1.2, 1],
    hspace=0.30,
    wspace=0.20,
)
age_ax = fig.add_subplot(grid[0, :])
income_ax = fig.add_subplot(grid[1, 0], sharey=age_ax)
family_size_ax = fig.add_subplot(grid[1, 1], sharey=age_ax)

# Plot weighted-bootstrap test rows for Age without changing NumPy's global state.
age_plot_rng = np.random.default_rng(RANDOM_STATE)
age_plot_values = X_test_preprocessor_input.loc[
    shap_beeswarm_sample.index,
    "AGE23X",
].to_numpy()
age_plot_jitter = age_plot_rng.uniform(-0.20, 0.20, len(age_plot_values))
age_ax.scatter(
    age_plot_values + age_plot_jitter,
    shap_beeswarm_sample["AGE23X"],
    color=POP_COLOR,
    alpha=0.15,
    edgecolors="none",
    s=12,
    rasterized=True,
)
age_ax.plot(
    age_trend["Age"],
    age_trend["Median"],
    color=POP_COLOR,
    linewidth=2.5,
)
age_ax.set_xlim(age_values.min() - 1, age_values.max() + 1)
age_ticks = [18, 25, 35, 45, 55, 65, 75, 85]
age_ax.set_xticks(age_ticks, [*map(str, age_ticks[:-1]), "85+"])

age_ax.set_title("Age", loc="left", fontsize=11, fontweight="bold", pad=10)


def plot_ordered_shap_intervals(ax, feature, title):
    """Plot weighted SHAP medians and percentile intervals by ordered value."""
    feature_summary = (
        shap_ordered_summary
        .loc[shap_ordered_summary["Feature"].eq(feature)]
        .reset_index(drop=True)
    )
    positions = np.arange(len(feature_summary))
    ax.vlines(
        positions,
        feature_summary["P10"],
        feature_summary["P90"],
        color="#AEB8C2",
        linewidth=2,
    )
    ax.vlines(
        positions,
        feature_summary["P25"],
        feature_summary["P75"],
        color=POP_COLOR,
        linewidth=6,
    )
    ax.plot(
        positions,
        feature_summary["Median"],
        color=POP_COLOR,
        linewidth=1.5,
        alpha=0.65,
    )
    ax.scatter(
        positions,
        feature_summary["Median"],
        color=POP_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=45,
        zorder=3,
    )
    value_labels = feature_summary["Value"].astype(str).tolist()
    if feature == "FAMSZE23":
        value_labels = [
            f"{value} person" if value == "1" else f"{value} people"
            for value in value_labels
        ]
    tick_labels = [
        f"{value}\n({population_share:.0%})"
        for value, population_share in zip(
            value_labels,
            feature_summary["Population Share"],
        )
    ]
    ax.set_xticks(positions, tick_labels)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)


plot_ordered_shap_intervals(income_ax, "POVCAT23", "Family Income")
plot_ordered_shap_intervals(family_size_ax, "FAMSZE23", "Family Size")
income_ax.tick_params(axis="x", labelrotation=20)
for label in income_ax.get_xticklabels():
    label.set_horizontalalignment("right")



plotted_contribution_min = min(
    shap_test_contributions["AGE23X"].min(),
    shap_ordered_summary["P10"].min(),
)
plotted_contribution_max = max(
    shap_test_contributions["AGE23X"].max(),
    shap_ordered_summary["P90"].max(),
)
y_min = np.floor(plotted_contribution_min / 50) * 50
y_max = np.ceil(plotted_contribution_max / 50) * 50
for ax in (age_ax, income_ax, family_size_ax):
    ax.axhline(0, color="#4A4A4A", linewidth=1)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(shap_currency_formatter)

fig.supylabel(
    "SHAP Contribution to Predicted Median Cost",
    x=0.01,
)
fig.suptitle(
    "SHAP Contributions Across Ordered Values (Test Set)",
    fontsize=13,
    fontweight="bold",
    x=0.5,
    y=0.99,
)
fig.text(
    0.01,
    0.01,
    (
        "Note: Age dots are test rows sampled with replacement using MEPS survey weights, and the blue line shows the five-year rolling weighted median. "
        "Family Income and Family Size dots\n"
        "show weighted medians, blue bars show the 25th–75th percentiles, and gray lines the 10th–90th. Values are in 2023 USD. Percentages show weighted population shares."
    ),
    ha="left",
    va="bottom",
    fontsize=9,
    style="italic",
    color="#4A4A4A",
)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.92)
fig.savefig(
    "../figures/evaluation/shap_ordered_feature_contributions.png",
    bbox_inches="tight",
    dpi=200,
)
plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b>
#     <ul style="margin-top:10px; margin-bottom:8px">
#         <li><strong>Age contributions change most after the mid-50s:</strong> The Age contribution stays fairly stable and negative through early and middle adulthood, rises from around age 55 to 75, and then levels off. Contributions also vary much more among older adults.</li>
#         <li><strong>Only High Income moves estimates up:</strong> Poor through Middle Income generally move estimates down, while High Income generally moves them up.</li>
#         <li><strong>Family Size changes direction after two people:</strong> Contributions generally move estimates up for family sizes one and two and down for sizes three or more, with little additional change beyond five people.</li>
#     </ul>
# </div>
# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">XGBoost</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <strong>How Native XGBoost Feature Importance Works</strong><br>
#     Native XGBoost feature importance summarizes how the 40 model-ready features were used across the fitted trees. This model has 400 boosting rounds. Each round adds one tree for each of the four quantiles, producing 400 trees per quantile and 1,600 trees in total.
#     <br><br>
#     The training objective combines the MEPS-weighted pinball loss for q25, q50, q75, and q90 on log-transformed costs, plus penalties that limit tree complexity. Pinball loss weighs underprediction and overprediction differently for each quantile. Each quantile contributes its own loss component, and each quantile tree is built to reduce that component.
#     <br><br>
#     Each internal tree split uses one feature and has a gain, which measures how much the split improved its quantile's part of the training objective. Each split belongs to one quantile tree. XGBoost calculates feature importance by aggregating these split statistics across all 1,600 trees and therefore across q25, q50, q75, and q90.
#     <br><br>
#     XGBoost provides several native importance measures:
#     <ul>
#         <li><code>total_gain</code> is the total training-objective improvement from all splits using a feature. It is the primary ranking because it combines how often the feature was used with how valuable those splits were.</li>
#         <li><code>gain</code> is the average objective improvement per split. It identifies features that produced valuable splits when used.</li>
#         <li><code>weight</code> is the number of splits using the feature. It shows how frequently the model relied on that feature.</li>
#     </ul>
#     Gain and weight are supporting diagnostics that explain whether high total gain came from frequent modest improvements or fewer high-value splits. For example, Mental Health was used in 797 splits, slightly more often than Uninsured at 733 splits. However, its average gain was only 1.0 compared with 12.8 for Uninsured. Mental Health therefore accounts for 0.7% of total gain, while Uninsured accounts for 8.0%. This shows why split count alone can give a misleading importance ranking.
#     <br><br>
#     These measures are based on training data, are not q50-specific, and are not measured in dollars or shares of predictive accuracy.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Extract <code>total_gain</code>, <code>gain</code>, and <code>weight</code> from the fitted XGBoost. Use each feature's share of total gain for the primary ranking. Display a table of all 40 model-ready features and a bar plot of the top 15 features.
# </div>

# %%
def format_model_ready_feature(feature):
    """Return a readable label for a model-ready feature."""
    for source_feature in PIPELINE_NOMINAL_FEATURES:
        prefix = f"{source_feature}_"
        if feature.startswith(prefix):
            category = feature.removeprefix(prefix)
            return f"{DISPLAY_LABELS[source_feature]}: {category}"
    return DISPLAY_LABELS.get(
        feature,
        feature.replace("_", " ").title(),
    )


xgb_booster = xgb_quantile_model.regressor_.get_booster()
model_ready_features = X_train_preprocessed.columns.tolist()
if xgb_booster.feature_names != model_ready_features:
    raise ValueError(
        "XGBoost booster features do not match the model-ready training features."
    )

native_total_gain = xgb_booster.get_score(importance_type="total_gain")
native_gain = xgb_booster.get_score(importance_type="gain")
native_weight = xgb_booster.get_score(importance_type="weight")

# get_score omits unused features, so explicitly retain all model inputs with zeros.
xgb_native_feature_importance = pd.DataFrame({
    "Feature Code": model_ready_features,
    "Feature": [
        format_model_ready_feature(feature)
        for feature in model_ready_features
    ],
    "Total Gain": [
        native_total_gain.get(feature, 0.0)
        for feature in model_ready_features
    ],
    "Average Gain per Split": [
        native_gain.get(feature, 0.0)
        for feature in model_ready_features
    ],
    "Split Count (Weight)": [
        native_weight.get(feature, 0.0)
        for feature in model_ready_features
    ],
})

total_native_gain = xgb_native_feature_importance["Total Gain"].sum()
if total_native_gain <= 0:
    raise ValueError("XGBoost total gain must be greater than zero.")

xgb_native_feature_importance["Share of Total Gain"] = (
    xgb_native_feature_importance["Total Gain"]
    / total_native_gain
)
xgb_native_feature_importance = (
    xgb_native_feature_importance
    .sort_values("Total Gain", ascending=False)
    .reset_index(drop=True)
)
xgb_native_feature_importance.insert(
    0,
    "Rank",
    np.arange(1, len(xgb_native_feature_importance) + 1),
)

display(
    xgb_native_feature_importance[[
        "Rank",
        "Feature",
        "Total Gain",
        "Share of Total Gain",
        "Average Gain per Split",
        "Split Count (Weight)",
    ]]
    .style
    .pipe(
        add_table_caption,
        "XGBoost Native Feature Importance (Training)",
    )
    .format({
        "Total Gain": "{:,.0f}",
        "Share of Total Gain": "{:.1%}",
        "Average Gain per Split": "{:,.1f}",
        "Split Count (Weight)": "{:,.0f}",
    })
    .hide()
)

# %% [markdown]
# <em>Note: Feature importance measures are aggregated across the q25, q50, q75, and q90 trees. Total gain and average gain are improvements in the weighted quantile training objective on log-transformed costs, not dollar amounts.</em>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>XGBoost Feature Importance (Bar Plot)</strong><br>
#     📌 Plot the top 15 model-ready features by share of total gain.
# </div>

# %%
xgb_native_top_15 = (
    xgb_native_feature_importance
    .head(15)
    .sort_values("Share of Total Gain")
)
xgb_native_top_15_share = xgb_native_top_15[
    "Share of Total Gain"
].sum()

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    xgb_native_top_15["Feature"],
    xgb_native_top_15["Share of Total Gain"],
    color=POP_COLOR,
)
ax.bar_label(
    bars,
    labels=[
        f"{share:.1%}"
        for share in xgb_native_top_15["Share of Total Gain"]
    ],
    padding=4,
    fontsize=9,
)
ax.set_xlim(
    0,
    xgb_native_top_15["Share of Total Gain"].max() * 1.14,
)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f"{value:.0%}")
)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.set_title(
    "XGBoost Quantile Feature Importance (Training)",
    fontsize=13,
    fontweight="bold",
    pad=16,
)
ax.text(
    0.5,
    0.99,
    (
        "Top 15 model-ready features account for "
        f"{xgb_native_top_15_share:.1%} of total gain"
    ),
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    fontweight="normal",
)
ax.set_xlabel("Share of Total Gain")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.20)
ax.set_axisbelow(True)
sns.despine(ax=ax, left=True)

fig.text(
    0.01,
    0.01,
    (
        "Note: Percentages show each feature's share of total gain across 40 model-ready features. Total gain aggregates improvements in the weighted quantile\n"
        "training objective across the q25, q50, q75, and q90 trees. They describe training split improvements, not improvements in prediction accuracy."
    ),
    ha="left",
    va="bottom",
    fontsize=9,
    style="italic",
    color="#4A4A4A",
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(
    "../figures/evaluation/xgb_quantile_feature_importance.png",
    bbox_inches="tight",
    dpi=200,
)
plt.show()
