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
#         Last updated: April 2026
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
import matplotlib.ticker as mtick  # to format axis ticks
import seaborn as sns
import math  # to calculate n_rows in subplot matrix

# Target Preprocessing
from sklearn.compose import TransformedTargetRegressor

# Models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Model selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ParameterSampler

# Model evaluation
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score
)
import time  # to measure training time

# Local imports
from src.modeling import (
    train_and_evaluate,
    get_baseline_models,
    weighted_median_absolute_error,
    save_model,
    load_model,
    save_metrics,
    load_metrics,
    get_core_model_params
)
from src.data import create_stratification_bins
from src.params import (
    EN_PARAM_DISTRIBUTIONS,
    RF_PARAM_DISTRIBUTIONS, 
    XGB_PARAM_DISTRIBUTIONS 
)
from src.constants import (
    ID_COLUMN,
    WEIGHT_COLUMN,
    TARGET_COLUMN,
    RANDOM_STATE
)
from src.display import (
    DISPLAY_LABELS, 
    METRIC_LABELS,
    MODEL_DISPLAY_LABELS,
    CATEGORY_LABELS_EDA,
    POP_COLOR,
    add_table_caption
)

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Data Loading</h1>
# </div>
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Load the preprocessed data from the <code>.parquet</code> files into Pandas DataFrames.
# </div>

# %%
df_train_preprocessed = pd.read_parquet("../data/training_data_preprocessed.parquet")
df_val_preprocessed = pd.read_parquet("../data/validation_data_preprocessed.parquet")
df_test_preprocessed = pd.read_parquet("../data/test_data_preprocessed.parquet")

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
#         <li>Store each fitted model as an individual <code>.joblib</code> file, all evaluation metrics collectively as a <code>.json</code> file and all predictions collectively as a <code>.joblib</code> file.</li>
#     </ul>  
#     For more details, see <a href="../src/modeling.py">src/modeling.py</a>.
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
#     Setup
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
VAL_DATA_PATH = "../data/validation_data_preprocessed.parquet"

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
def prepare_human_readable_validation_data():
    """
    Recover human-readable feature values for the validation set.

    The saved parquet contains scaled/encoded features (after StandardScaler and
    OneHotEncoder). This function reloads the raw MEPS SAS file, applies the same
    cleaning steps 1-7 as preprocess.py (but NOT the sklearn pipeline), then filters
    to only the validation set rows by matching DUPERSID indices.

    Returns:
        tuple: (df_raw_val, y_val, w_val) where df_raw_val has human-readable
               feature values, y_val is the target, and w_val are sample weights.
               All aligned by DUPERSID index in parquet row order.
    """
    # Load preprocessed validation data to get row IDs, target, and weights
    df_val = pd.read_parquet(VAL_DATA_PATH)
    val_ids = set(df_val.index.astype(str))
    y_val = df_val[TARGET_COLUMN]
    w_val = df_val[WEIGHT_COLUMN]

    # --- Data Preparation (mirrors preprocess.py steps 1-7) ---
    # Step 1: Load raw MEPS data
    print("  Loading raw MEPS SAS data...")
    df = pd.read_sas(RAW_DATA_PATH, format="sas7bdat", encoding="latin1")

    # Step 2: Variable selection
    print("  Selecting variables...")
    df = df[RAW_COLUMNS_TO_KEEP]

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

    # Filter to validation set rows and align to preprocessed data row order
    print("  Filtering rows to match preprocessed validation data...")
    df_raw_val = df.loc[df.index.isin(val_ids)].reindex(y_val.index)
    n_matched = df_raw_val.index.isin(val_ids).sum()
    n_complete = df_raw_val.notna().all(axis=1).sum()
    print(f"  Matched {n_matched:,} of {len(val_ids):,} rows of the preprocessed validation data ({n_complete:,} complete, {n_matched - n_complete:,} with missing values)")

    return df_raw_val, y_val, w_val


# Example usage: Prepare validation data for LLM benchmarking
# df_raw_val, y_val, w_val = prepare_human_readable_validation_data()

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

# Display metric comparison table
display(
    pd.DataFrame(all_metrics).T[["val_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x).replace(" (Baseline)", ""))
    .style
    .pipe(add_table_caption, "General LLM vs. Specialized ML Models")
    .format("{:.2f}")
    .highlight_min(subset=["MdAE", "MAE"], color="#d4edda")
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
#     <b>Tuning Framework (Shared Logic):</b>
#     <ul>
#         <li><b>Search Strategy:</b> Manual loop with <code>ParameterSampler</code> to avoid <code>sample_weight</code> routing issues.</li>
#         <li><b>Target Transform:</b> <code>TransformedTargetRegressor(log1p)</code> to handle skewness and optimize in log-space.</li>
#         <li><b>Sample Weights:</b> Normalized weights (mean=1.0) for training; raw survey weights for evaluation.</li>
#         <li><b>Scoring:</b> Weighted Median Absolute Error (MdAE) on raw-dollar predictions.</li>
#         <li><b>Iterations:</b> Small number (2-5) in notebook for prototyping. Scale to 50-100 in production scripts (e.g., <code>scripts/tune_random_forest.py</code>).</li>
#     </ul>
#     <b>Why not <code>RandomizedSearchCV</code>?</b><br>
#     Avoids <code>sample_weight</code> routing complexities and ensures transparent weighted scoring on a fixed holdout set.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Elastic Net</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Tune <code>ElasticNet</code> hyperparameters.
#     <br><br>
#     For hyperparameter details, refer to the official <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html" target="_blank">ElasticNet documentation</a>. <br> 
#     For hyperparamter search space and rationale, refer to <a href="../src/params.py">src/params.py</a>.
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
#     <ul>
#         <li><b>Objective:</b> <code>criterion="absolute_error"</code> to minimize L1 loss on log-costs.</li>
#         <li><b>Key Params:</b> Control variance via <code>min_samples_leaf</code> and <code>max_features</code>.</li>
#     </ul>
#     For hyperparameter details, refer to the official scikit-learn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">RandomForestRegressor documentation</a> <br> 
#     For hyperparamter search space and rationale, refer to <a href="../src/params.py">src/params.py</a>.
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
#     <ul>
#         <li><b>Objective:</b> <code>objective="reg:absoluteerror"</code> to minimize L1 loss on log-costs.</li>
#         <li><b>Speed:</b> Uses <code>tree_method="hist"</code> for efficient histogram-based splitting.</li>
#     </ul>
#     For hyperparameter details, refer to the official <a href="https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor" target="_blank">XGBoost documentation</a>. <br> 
#     For hyperparamter search space and rationale, refer to <a href="../src/params.py">src/params.py</a>.
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
#         <li>Metrics Comparison Tables</li>
#         <li>Overfitting Analysis</li>
#         <li>Stratified Error Analysis</li>
#         <li>Heteroscedasticity (Residuals vs. Predicted)</li> 
#         <li>(optionally) Feature Dependencies (Residuals vs. Features)</li> 
#     </ul>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Metric Comparison Table</strong><br>
#     📌 Compare evaluation metrics of LLM benchmark, baseline models, and tuned models on the validation data.
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

# Display metric comparison table
display(
    pd.DataFrame(all_metrics).T[["val_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x))
    .style
    .pipe(add_table_caption, "Tuned Model Metrics (Validation Data)")
    .format("{:.2f}")
    .highlight_min(subset=["MdAE", "MAE"], color="#d4edda")
    .highlight_max(subset=["R²"], color="#d4edda")
)

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

# Filter combined metrics for only the finalists
final_metrics = {k: all_metrics[k] for k in finalists if k in all_metrics}

# Display curated comparison table
display(
    pd.DataFrame(final_metrics).T[["val_mdae", "val_mae", "val_r2"]]
    .rename(columns=lambda x: METRIC_LABELS.get(x, x).replace(" (Val)", ""))
    .rename(index=lambda x: MODEL_DISPLAY_LABELS.get(x, x))
    .reset_index().rename(columns={"index": "Model"})
    .style
    .hide()
    .set_properties(subset=["Model"], **{"font-weight": "bold"})
    .pipe(add_table_caption, "Tuned Model Metrics (Validation Data)")
    .format({"MdAE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.2f}"})
    .highlight_min(subset=["MdAE", "MAE"], color="#d4edda")
    .highlight_max(subset=["R²"], color="#d4edda")
)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Overfitting Analysis</strong><br>
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
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Stratified Error Analysis</strong><br>
#     📌 Compare model performance across different population segments or groups.
# </div> 
# %%
# --- Stratified Error Analysis: Table ---
# Recover raw features of validation data for stratification
# We need the original categorical codes (before pipeline one-hot encoding) to group the data.
print("Recovering raw validation features...")
df_raw_val, y_val_true, w_val_weights = prepare_human_readable_validation_data()

# Load predictions (on validation data) of tuned XGBoost
y_val_pred_xgb = load_model("../models/xgb_tuned_predictions.joblib", verbose=False)

# Create medical cost strata (consistent with preprocessing/EDA)
COST_BIN_LABELS = {
    0: "Zero Costs",
    1: "Low Spend (0-50%)",
    2: "Moderate  (50-80%)",
    3: "High Spend (80-95%)",
    4: "Very High (95-99%)",
    5: "Massively High (99-99.9%)",
    6: "Super Spenders (Top 0.1%)"
}
df_raw_val["ACTUAL_COSTS"] = create_stratification_bins(y_val_true)

# Create predicted cost strata (measures reliability of model predictions for high vs. low cost ranges)
# Note: y_val_pred_xgb is a numpy array, so we convert to Series to use the helper's index-based logic
y_val_pred_series = pd.Series(y_val_pred_xgb, index=y_val_true.index)
df_raw_val["PREDICTED_COSTS"] = create_stratification_bins(y_val_pred_series)

# Create chronic conditions count feature 
chronic_cols = list(CHRONIC_CONDITIONS.keys())
df_raw_val["CHRONIC_COUNT"] = df_raw_val[chronic_cols].sum(axis=1).astype(int)
df_raw_val["CHRONIC_COUNT_GRP"] = df_raw_val["CHRONIC_COUNT"].apply(lambda x: str(x) if x < 4 else "4+")  # Group the tail (4+) for better statistical stability

# Create age groups for a more stable and interpretable Fairness Audit
age_bins = [18, 35, 50, 65, 120]
age_labels = ["18-34", "35-49", "50-64", "65+"]
df_raw_val["AGE_GRP"] = pd.cut(df_raw_val["AGE23X"], bins=age_bins, labels=age_labels, right=False)

# Define configurations for two distinct audits
# 1. Model Reliability: Performance across cost levels and medical complexity
reliability_configs = [
    {"col": "ACTUAL_COSTS", "label": "Medical Costs (Actual)", "category_map": COST_BIN_LABELS},
    {"col": "PREDICTED_COSTS", "label": "Medical Costs (Predicted)", "category_map": COST_BIN_LABELS},
    {"col": "INSCOV23", "label": DISPLAY_LABELS["INSCOV23"], "category_map": CATEGORY_LABELS_EDA["INSCOV23"]},
    {"col": "CHRONIC_COUNT_GRP", "label": DISPLAY_LABELS["CHRONIC_COUNT"], "category_map": None}
]

# 2. Demographic Fairness: Performance across protected social attributes
fairness_configs = [
    {"col": "SEX", "label": DISPLAY_LABELS["SEX"], "category_map": CATEGORY_LABELS_EDA["SEX"]},
    {"col": "AGE_GRP", "label": "Age Group", "category_map": None},
    {"col": "POVCAT23", "label": DISPLAY_LABELS["POVCAT23"], "category_map": CATEGORY_LABELS_EDA["POVCAT23"]},
]

# Combine for the calculation loop
stratified_error_configs = reliability_configs + fairness_configs

# Calculate weighted MdAE for each column
stratified_error_results = []
for config in stratified_error_configs:
    col = config["col"]
    label = config["label"]
    category_map = config["category_map"]
    
    # Calculate weighted MdAE for each group
    groups = sorted(df_raw_val[col].dropna().unique())
    for group in groups:
        mask = (df_raw_val[col] == group)
            
        group_mdae = weighted_median_absolute_error(
            y_val_true[mask],      # Aligns via Index
            y_val_pred_xgb[mask],  # Aligns via Position (df_raw_val was reindexed to match in prepare_human_readable_validation_data)
            sample_weight=w_val_weights[mask]  # Aligns via Index
        )
        
        # Determine group name (map code to label if available, otherwise use raw value/code)
        group_name = category_map.get(int(group), group) if category_map else group

        stratified_error_results.append({
            "Column": label,
            "Group": group_name,
            "MdAE": group_mdae,
            "Sample Size": mask.sum()
        })

# Display results table
stratified_error_df = pd.DataFrame(stratified_error_results)
display(
    stratified_error_df.set_index(["Column", "Group"])
    .style
    .pipe(add_table_caption, "Tuned XGBoost: Stratified Error Analysis")
    .format({"MdAE": "${:.2f}", "Sample Size": "{:,}"})
    .background_gradient(subset=["MdAE"], cmap="Reds")
)

# --- Stratified Analysis: Visualization ---

# 1. Model Reliability Audit
# Focuses on performance across cost levels and medical complexity
reliability_labels = [c["label"] for c in reliability_configs]
plot_df_rel = plot_df[plot_df["Column"].isin(reliability_labels)].copy()

g_rel = sns.catplot(
    data=plot_df_rel,
    kind="bar",
    x="MdAE",
    y="Group",
    col="Column",
    col_wrap=2,
    height=4,
    aspect=1.5,
    sharex=False,  # Independent x-axes to handle different error scales
    sharey=False,  # Independent y-axes to show only relevant groups per column
    color=POP_COLOR,
    legend=False
)

# Customize bar plot grid
g_rel.set_titles("{col_name}", weight="bold", size=14)
g_rel.fig.suptitle("Tuned XGBoost: Model Reliability Audit", fontsize=18, weight="bold")
plt.subplots_adjust(top=0.90, hspace=0.3)  # Increases spacing between subplots to prevent overlap

# Iterate and customize subplots
for ax in g_rel.axes.flat:
    ax.set_ylabel("")
    ax.set_xlabel("Weighted MdAE")
    ax.set_xticklabels([])
    for c in ax.containers:
        value_labels = [f"${v:,.0f}" for v in c.datavalues]  # Formats value labels with thousand separator and no decimals
        ax.bar_label(c, labels=value_labels, padding=3, fontsize=9)
    ax.margins(x=0.15)

plt.show()

# 2. Demographic Fairness Audit
# Focuses on performance across protected/vulnerable social groups
fairness_labels = [c["label"] for c in fairness_configs]
plot_df_fair = plot_df[plot_df["Column"].isin(fairness_labels)].copy()

g_fair = sns.catplot(
    data=plot_df_fair,
    kind="bar",
    x="MdAE",
    y="Group",
    col="Column",
    col_wrap=2,
    height=4,
    aspect=1.5,
    sharex=False,
    sharey=False,
    color=POP_COLOR,
    legend=False
)

g_fair.set_titles("{col_name}", weight="bold", size=14)
g_fair.fig.suptitle("Tuned XGBoost: Demographic Fairness Audit", fontsize=18, weight="bold")
plt.subplots_adjust(top=0.88, hspace=0.3)

for ax in g_fair.axes.flat:
    ax.set_ylabel("")
    ax.set_xlabel("Weighted MdAE")
    ax.set_xticklabels([])
    for c in ax.containers:
        labels = [f"${v:,.0f}" for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=3, fontsize=9)
    ax.margins(x=0.15)

plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px; margin-bottom:16px;">
#     💡 <strong>Insights from Stratified Error Analysis:</strong>
#     <ul style="margin-top:8px; margin-bottom:8px">
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#f0f7ff; padding:15px; border:3px solid #cfe2ff; border-radius:6px; margin-bottom:16px;">
#     <strong>⚖️ Regulatory Compliance & Ethical AI</strong> <br>
#     This project is designed for the US Market (NIST, FTC) with a roadmap for EU Expansion (AI Act, GDPR). To ensure responsible and compliant deployment, performed <b>stratified error analysis</b> to detect algorithmic bias and use <b>survey weights</b> to ensure population representativeness.
#     <p style="margin-top:10px;">
#         For a details on regulatory compliance and ethical AI, refer to: <a href="../docs/research/regulatory_compliance.md">docs/research/regulatory_compliance.md</a>
#     </p>
# </div>
#
