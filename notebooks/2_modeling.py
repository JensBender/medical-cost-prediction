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
#         Last updated: March 2026
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
import mlflow
import time  # to measure model training time

# Models
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor 

# Model evaluation
from sklearn.metrics import (
    median_absolute_error,
    mean_absolute_error, 
    r2_score
)

# Model persistence
import joblib

# Local imports
from src.constants import (
    DISPLAY_LABELS, 
    CATEGORY_LABELS_EDA,
    METRIC_LABELS,
    RANDOM_STATE,
    POP_COLOR,
    SAMPLE_COLOR
)
from src.utils import (
    add_table_caption,
    weighted_median_absolute_error,
    save_model,
    load_model
)

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     <strong>MLflow Settings</strong>
# </div>

# %%
# Set the tracking URI to point to your running MLflow UI server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

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
            - str: Status icon for target variable presence (✅ if "TOTSLF23" exists, ❌ otherwise).
            - str: Status icon for weight variable presence (✅ if "PERWT23F" exists, ❌ otherwise).
    """
    shape = df.shape
    no_missings = "✅" if not df.isna().any().any() else "❌"
    all_numerical = "✅" if df.select_dtypes(exclude=[np.number]).empty else "❌"
    no_infinites = "✅" if not np.isinf(df.select_dtypes(include=[np.number])).any().any() else "❌"
    no_constants = "✅" if (df.nunique(dropna=False) > 1).all() else "❌"
    unique_id = "✅" if df.index.is_unique else "❌"
    target_present = "✅" if "TOTSLF23" in df.columns else "❌"
    weights_present = "✅" if "PERWT23F" in df.columns else "❌"

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
X_train_preprocessed = df_train_preprocessed.drop(["TOTSLF23", "PERWT23F"], axis=1)
y_train = df_train_preprocessed["TOTSLF23"]
w_train = df_train_preprocessed["PERWT23F"]

X_val_preprocessed = df_val_preprocessed.drop(["TOTSLF23", "PERWT23F"], axis=1)
y_val = df_val_preprocessed["TOTSLF23"]
w_val = df_val_preprocessed["PERWT23F"]

X_test_preprocessed = df_test_preprocessed.drop(["TOTSLF23", "PERWT23F"], axis=1)
y_test = df_test_preprocessed["TOTSLF23"]
w_test = df_test_preprocessed["PERWT23F"]

# Delete redundant DataFrames to free up memory
del df_train_preprocessed, df_val_preprocessed, df_test_preprocessed

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Baseline Models</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Train 7 baseline models on the full feature set (27 raw, 40 preprocessed) with mostly default hyperparameter values.  
#     <ul>
#         <li>Linear Regression (lr)</li>
#         <li>Elastic Net Regression (en)</li>
#         <li>Decision Tree Regressor (tree)</li>
#         <li>Random Forest Regressor (rf)</li>
#         <li>XGBoost Regressor (xgb)</li>
#         <li>Support Vector Regressor (svr)</li>
#         <li>Multi-Layer Perceptron Regressor (mlp)</li>
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
#         <li>Apply log-transformation of target variable for designated models (lr, en, mlp, svr) using <code>TransformedTargetRegressor</code>. Use <code>log1p</code> instead of <code>log</code> to handle zeros in target (<code>log(0)</code> is undefined).</li>
#         <li>Implement polynomial features for elastic net regression using second-degree <code>PolynomialFeatures</code> with a small <code>Pipeline</code>.</li>
#         <li>Store fitted models, predicted values, and evaluation metrics in a results dictionary and persist as a <code>.joblib</code> file.</li>
#     </ul>  
# </div> 

# %%
# Define baseline models
baseline_models = {
    "Linear Regression": TransformedTargetRegressor(
        regressor=LinearRegression(),
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
    "Decision Tree": DecisionTreeRegressor(criterion="absolute_error", random_state=RANDOM_STATE),  # absolute_error uses MAE loss function for training
    "Random Forest": RandomForestRegressor(criterion="absolute_error", n_jobs=-1, random_state=RANDOM_STATE), # n_jobs=-1 uses all CPU cores to speed up training
    "XGBoost": XGBRegressor(objective="reg:tweedie", n_jobs=-1, random_state=RANDOM_STATE), # Tweedie handles zero-inflation and heavy tail
    "Support Vector Machine": TransformedTargetRegressor(
        regressor=SVR(cache_size=1000), # Increasing cache_size uses more RAM to speed up training
        func=np.log1p,
        inverse_func=np.expm1
    )
}


def evaluate_model(model, X_train, y_train, X_val, y_val, w_train=None, w_val=None, model_name="model", log_model=False):
    """
    Train and evaluate a single machine learning model with MLflow experiment tracking.

    Args:
        model (estimator): The Scikit-learn estimator or pipeline to be trained.
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Target variable for training data.
        X_val (pd.DataFrame): Preprocessed validation features.
        y_val (pd.Series): Target variable for validation data.
        w_train (pd.Series, optional): Sample weights for training data. Defaults to None.
        w_val (pd.Series, optional): Sample weights for validation data. Defaults to None.
        model_name (str, optional): Display name of the model for MLflow experiment tracking. Defaults to "model".
        log_model (bool, optional): Whether to log the fitted model as an artifact to MLflow. Defaults to False.

    Returns:
        dict: A dictionary containing the evaluation results:
            - "mdae" (float): Median Absolute Error (unweighted).
            - "mae" (float): Mean Absolute Error (weighted).
            - "r2" (float): Coefficient of Determination (weighted).
            - "training_time" (float): Training time in seconds.
            - "fitted_model" (estimator): The trained model object.
            - "y_val_pred" (np.ndarray): The predicted values on the validation set.
    """
    # Track model run
    with mlflow.start_run(run_name=model_name):
        # Tag raw data source 
        mlflow.set_tag("data_source", "h251.sas7bdat")

        # Log data lineage
        data_train = mlflow.data.from_pandas(
            X_train, 
            targets=y_train,
            source="../data/training_data_preprocessed.parquet", 
            name="training_data"
        )
        data_val = mlflow.data.from_pandas(
            X_val, 
            targets=y_val,
            source="../data/validation_data_preprocessed.parquet", 
            name="validation_data"
        )
        mlflow.log_input(data_train, context="training")
        mlflow.log_input(data_val, context="validation")

        # Log model hyperparameters
        mlflow.log_params(model.get_params())

        # Ensure weights are passed to model even when wrapped in a Pipeline (for Elastic Net)
        fit_params = {}
        if w_train is not None:
            # Normalize weights so mean is 1.0 (prevents numerical instability in algorithms like SVR)
            w_train_norm = w_train / w_train.mean()
            
            reg = getattr(model, "regressor", model)
            key = f"{reg.steps[-1][0]}__sample_weight" if isinstance(reg, Pipeline) else "sample_weight"
            fit_params[key] = w_train_norm

        # Fit model on training data
        start_time = time.time()  # Measure training time
        model.fit(X_train, y_train, **fit_params)
        end_time = time.time()

        # Log fitted model
        if log_model:
            mlflow.sklearn.log_model(model, "model")

        # Predict on validation data
        y_val_pred = model.predict(X_val)

        # Calculate evaluation metrics
        mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)  
        mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)

        # Log metrics
        mlflow.log_metric("mdae", mdae)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Return results dictionary
        return {
            "mdae": mdae,
            "mae": mae,
            "r2": r2,
            "training_time": end_time - start_time,
            "fitted_model": model,
            "y_val_pred": y_val_pred,
        }


# Example usage: Train and evaluate linear regression model
# lr_results = evaluate_model(baseline_models["Linear Regression"], X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val, model_name="Linear Regression")
# lr_metrics = pd.DataFrame([lr_results])[["mdae", "mae", "r2", "training_time"]]
# display(
#     lr_metrics
#     .rename(columns=METRIC_LABELS)
#     .style
#     .pipe(add_table_caption, "Linear Regression: Metrics")
#     .format("{:.2f}")
#     .hide()  # hides index
# ) 


def evaluate_all_models(models, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
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
              values are the dictionaries returned by the `evaluate_model` function.
    """
    # Iterate over all models
    results = {}
    for model_name, model in baseline_models.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model, X_train, y_train, X_val, y_val, w_train, w_val, model_name)
        results[model_name] = result
        print(f"Training Time: {round(result['training_time'], 2)} sec")
    return results

    
# Train and evaluate all baseline models
mlflow.set_experiment("Baseline Models") 
baseline_results = evaluate_all_models(baseline_models, X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val)

# Save baseline model results to file
save_model(baseline_results, "../models/baseline.joblib")

# Load baseline model results from file
# baseline_results = load_model("../models/baseline.joblib")


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

