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

# Preprocessing (scikit-learn)
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Model selection
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform  # for random hyperparameter values

# MLOps
import time  # to measure model training time

# Models
from sklearn.dummy import DummyRegressor  # for median baseline prediction
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Model evaluation
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score
)

# Model persistence
import joblib

# Local imports
from src.modeling import train_and_evaluate
from src.constants import (
    ID_COLUMN,
    WEIGHT_COLUMN,
    TARGET_COLUMN,
    DISPLAY_LABELS, 
    METRIC_LABELS,
    RANDOM_STATE
)
from src.utils import (
    add_table_caption,
    weighted_median_absolute_error,
    save_model,
    load_model
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
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Train 6 baseline models on the full feature set (27 raw, 40 preprocessed) with distribution-aware baseline hyperparameters.  
#     <ul>
#         <li>Linear Regression (lr)</li>
#         <li>Elastic Net Regression (en)</li>
#         <li>Decision Tree Regressor (tree)</li>
#         <li>Random Forest Regressor (rf)</li>
#         <li>XGBoost Regressor (xgb)</li>
#         <li>Support Vector Regressor (svr)</li>
#     </ul>
#     <hr style="height: 2px; border: none; background-color: #d0e7fa; margin: 16px 0; opacity: 0.8;">
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
#                 <li>Metrics Comparison Plots and Tables</li>
#                 <li>Error Analysis</li>
#                 <ul>
#                     <li>Heteroscedasticity (Residuals vs. Predicted)</li> 
#                     <li>Feature Dependencies (Residuals vs. Features)</li> 
#                     <li>Stratified Error Analysis</li>
#                 </ul>
#                 <li>Overfitting</li>
#             </ul>
#         </li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Training</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Train each baseline model and evaluate model performance. 
#     <ul>
#         <li>Train on preprocessed data (standardized, imputed, feature engineered, scaled, and encoded).</li>
#         <li>Use sample weights for population representativeness. Normalize weights (mean=1.0) to maintain relative importance while ensuring numerical stability during model training (especially for svr).</li>
#         <li>Apply log-transformation of target variable for all baseline models using <code>TransformedTargetRegressor</code>. Use <code>log1p</code> instead of <code>log</code> to handle zeros in target (<code>log(0)</code> is undefined).</li>
#         <li>Implement polynomial features for elastic net regression using second-degree <code>PolynomialFeatures</code> with a small <code>Pipeline</code>.</li>
#         <li>Store fitted models, predicted values, and evaluation metrics in a results dictionary and persist as a <code>.joblib</code> file.</li>
#     </ul>  
# </div> 

# %%
# Define baseline models
baseline_models = {
    "Median Prediction": DummyRegressor(strategy="median"),  # Always predict median as a baseline
    "Linear Regression": TransformedTargetRegressor(      
        regressor=LinearRegression(),  # Simple, linear predictions as an interpretable baseline
        func=np.log1p,
        inverse_func=np.expm1
    ),
    "Elastic Net": TransformedTargetRegressor(
        regressor=Pipeline([
            ("polynomials", PolynomialFeatures(degree=2, include_bias=False)),  # Intercept (bias) handled by model 
            ("model", ElasticNet())
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    ),
    "Decision Tree": TransformedTargetRegressor(
        regressor=DecisionTreeRegressor(
            criterion="absolute_error",  # Optimize for MAE of log-costs; corresponds to predicting the Median of raw costs
            max_depth=12,          # Limits tree depth to prevent overfitting (Default: None)
            min_samples_split=100, # Prevents splitting on tiny, noisy groups of patients (Default: 2)
            min_samples_leaf=50,   # Ensures each leaf's cost prediction has enough support (Default: 1)
            max_features="sqrt",   # Uses random feature subset for each split to prevent overfitting on individual features (Default: None)
            random_state=RANDOM_STATE  
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ),
    "Random Forest": TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            criterion="absolute_error",  # Optimize for MAE of log-costs; corresponds to predicting the Median of raw costs
            n_estimators=200,      # More trees for more stable estimates (Default: 100)
            max_depth=16,          # Limits tree depth to prevent overfitting (Default: None)
            min_samples_split=50,  # Prevents splitting on tiny, noisy groups of patients (Default: 2)
            min_samples_leaf=25,   # Ensures each leaf's cost prediction has enough support (Default: 1)
            max_features="sqrt",   # Uses random feature subset for each split to prevent overfitting on individual features (Default: 1.0)
            n_jobs=-1,             # Use all CPU cores to speed up training
            random_state=RANDOM_STATE 
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ),
    "XGBoost": TransformedTargetRegressor(
        regressor=XGBRegressor(
            objective="reg:absoluteerror", # Optimized for MAE of log-costs; corresponds to predicting the Median raw costs (Default: "reg:squarederror")
            n_estimators=600,      # More rounds with lower learning rate for smooth fitting (Default: 100)
            learning_rate=0.05,    # Smaller steps to prevent overshooting noisy targets (Default: 0.3)
            min_child_weight=10,   # Requires more "support" per node to split (Default: 1)
            subsample=0.8,         # Uses random row subset (80%) to prevent overfitting (Default: 1)
            colsample_bytree=0.5,  # Uses random feature subset (50%) for each split to prevent overfitting on individual features (Default: 1)
            reg_lambda=2.0,        # L2 regularization on leaf weights to avoid sharp peaks (Default: 1)
            tree_method="hist",    # Uses histogram algorithm for significantly faster training (Default: auto)
            n_jobs=-1,             # Use all CPU cores to speed up training
            random_state=RANDOM_STATE  
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ),
    "Support Vector Machine": TransformedTargetRegressor(
        regressor=SVR(
            kernel="rbf",    # Handles non-linearity and feature interactions (Default: same)
            C=10.0,          # Slightly higher regularization strength for log-costs (Default: 1.0)
            cache_size=1000  # Increase memory budget (1GB) to speed up training time (Default: 200)
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )
}

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
    # Iterate over all models
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        result = train_and_evaluate(model, X_train, y_train, X_val, y_val, w_train, w_val)
        results[model_name] = result
        print(f"  {model_name} trained in {round(result['training_time'], 2)} sec")
    print("\n✅ Baseline model training and evaluation complete.")    
    return results

    
# Train and evaluate all baseline models
# baseline_results = train_and_evaluate_all_models(baseline_models, X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val)

# Save baseline model results to file
# save_model(baseline_results, "../models/baseline.joblib")

# Load baseline model results from file
baseline_results = load_model("../models/baseline.joblib")


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     <strong>Evaluation</strong><br>
#     📌 Compare evaluation metrics of all baseline models on the validation data.
# </div> 

# %%
# Extract metrics from baseline model results
baseline_metrics = pd.DataFrame(baseline_results).T[["mdae", "mae", "r2", "training_time"]]

# Display metric comparison table
display(
    baseline_metrics
    .rename(columns=METRIC_LABELS)
    .style
    .pipe(add_table_caption, "Baseline Model Metrics")
    .format("{:.2f}")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <strong>Note on Negative $r^2$ Scores</strong>
#     <br>
#     Indicates the model performs worse than always predicting the mean (as seen e.g. in Linear Regression). This is common in medical cost prediction for several reasons:
#     <ul>
#         <li><strong>Sensitivity to Outliers</strong>: $r^2$ uses squared errors. Since medical costs (MEPS) are extremely heavy-tailed, even a few large mispredictions on high-cost individuals can cause the squared error to explode.</li>
#         <li><strong>Log-Transformation Impact</strong>: While log-transforming handles skewness, small errors in "log-space" become exponential errors when converted back to raw dollars.</li>
#         <li><strong>Sample Weights</strong>: Weighted $r^2$ penalizes errors more heavily on observations that represent larger portions of the US population.</li>
#     </ul>
#     Observation: The relatively small MdAE (~\$200) vs. large MAE (~\$1000) confirms that the baseline models predict typical costs well, but fail on high-cost outliers.
# </div>

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     🔍 <strong>Diagnostic: Log-Scale Evaluation</strong><br>
#     📌 Recalculate metrics in log-space to assess model learning without the "explosion" effect of high-cost outliers on the dollar scale.
# </div> 

# %%
# Log-transform true values
y_val_log = np.log1p(y_val)

log_metrics = {}
for model_name, result in baseline_results.items():
    # Log-transform predictions (they were inverse-transformed to dollars by TransformedTargetRegressor)
    y_val_pred_log = np.log1p(result["y_val_pred"])
    
    # Calculate weighted metrics in log-space
    log_metrics[model_name] = {
        "MdAE (Log)": weighted_median_absolute_error(y_val_log, y_val_pred_log, sample_weight=w_val),
        "MAE (Log)": mean_absolute_error(y_val_log, y_val_pred_log, sample_weight=w_val),
        "R² (Log)": r2_score(y_val_log, y_val_pred_log, sample_weight=w_val)
    }

# Display log-scale comparison
df_log_metrics = pd.DataFrame(log_metrics).T
display(
    df_log_metrics
    .style
    .pipe(add_table_caption, "Baseline Model Metrics (Log-Scale)")
    .format("{:.2f}")
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px; margin-bottom:16px;">
#     💡 <strong>Insights & Key Findings:</strong>
#     <ul style="margin-top:8px; margin-bottom:8px">
#         <li><strong>The Log-Scale "North Star":</strong> While R² on the raw dollar scale is near zero (or negative), the <b>Log-Scale R² is ~0.30</b> across all top models. This confirms the features have strong predictive signal for healthcare utilization and that the negative raw R² is simply a scaling artifact caused by rare high-cost "black swan" events.</li>
#         <li><strong>MdAE Priority:</strong> For our typical app user, <b>MdAE is the most meaningful success metric</b>. The data confirms that predicting the "typical experience" is statistically distinct from predicting the catastrophic extreme costs.</li>
#         <li><strong>Mean vs. Median Trade-off:</strong> Objectives like <em>reg:tweedie</em> fix the dollar-scale $R^2$ but hurt the MdAE because they are biased toward the high-expenditure tail. For a budgeting app, sticking to <b>Log-Absolute-Error</b> models seems the better strategy.</li>
#     </ul>
#     <hr style="height: 1px; border: none; background-color: #e0f0e0; margin: 12px 0;">
#     🎯 <strong>Selected Models for Hyperparameter Tuning:</strong>
#     <ol style="margin-top:8px; margin-bottom:0px">
#         <li><strong>Elastic Net:</strong> The current "Champion" (MdAE 163). Its combination of second-degree polynomial features and L1/L2 regularization handles the correlated medical inputs well.</li>
#         <li><strong>XGBoost:</strong> Displays the deepest predictive "signal" (Log R² 0.30). Its histogram-based gradient boosting captures non-linear health interactions that simpler models miss. Tune it to beat the Elastic Net performance.</li>
#         <li><strong>Random Forest:</strong> A highly stable alternate learner that currently leads on MAE (958). It provides an essential check against the boosting-bias of XGBoost.</li>
#     </ol>
# </div>
