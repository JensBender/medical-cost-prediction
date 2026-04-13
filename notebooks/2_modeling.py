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
from sklearn.ensemble import RandomForestRegressor

# Model selection
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform  # for random hyperparameter values

# Model evaluation
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score
)
import time  # to measure training time

# Local imports
from src.modeling import (
    train_and_evaluate,
    get_baseline_models
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
)
from src.params import RF_PARAM_DISTRIBUTIONS
from src.utils import (
    add_table_caption,
    weighted_median_absolute_error,
    save_model,
    load_model,
    save_metrics,
    load_metrics
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
#                 <li>Error Analysis</li>
#                 <ul>
#                     <li>Heteroscedasticity (Residuals vs. Predicted)</li> 
#                     <li>Feature Dependencies (Residuals vs. Features)</li> 
#                     <li>Stratified Error Analysis</li>
#                 </ul>
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
# Load baseline model metrics from JSON file
baseline_metrics = load_metrics("../models/baseline_metrics.json")

# Display metric comparison table
display(
    pd.DataFrame(baseline_metrics).T
    [["val_mdae", "val_mae", "val_r2"]]
    .rename(columns=METRIC_LABELS)
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
# Extract train and val mdae
overfitting_data = []
for model_name, metrics in baseline_metrics.items():
        overfitting_data.append({
            "Model": model_name,
            "MdAE (Train)": metrics["train_mdae"],
            "MdAE (Val)": metrics["val_mdae"],
            "Delta": metrics["val_mdae"] - metrics["train_mdae"],
            "Delta %": ((metrics["val_mdae"] - metrics["train_mdae"]) / metrics["train_mdae"]) * 100
        })

# Display overfitting table
display(
    pd.DataFrame(overfitting_data)
    .set_index("Model")
    .style
    .pipe(add_table_caption, "Baseline Models: Overfitting Analysis (MdAE)")
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

# Load predicted values from .joblib file
predictions = load_model("../models/baseline_predictions.joblib")

# Evaluate all models on log-scale
log_metrics = {}
for model_name, y_val_pred in predictions.items():
    # Log-transform predictions (they were inverse-transformed to dollars by TransformedTargetRegressor)
    y_val_pred_log = np.log1p(y_val_pred)
    
    # Calculate weighted metrics in log-space
    log_metrics[model_name] = {
        "MdAE (Log)": weighted_median_absolute_error(y_val_log, y_val_pred_log, sample_weight=w_val),
        "MAE (Log)": mean_absolute_error(y_val_log, y_val_pred_log, sample_weight=w_val),
        "R² (Log)": r2_score(y_val_log, y_val_pred_log, sample_weight=w_val)
    }

# Display log-scale comparison table
display(
    pd.DataFrame(log_metrics).T
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
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Hyperparameter Tuning</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Tune the hyperparameters of Random Forest, XGBoost, and Elastic Net using randomized search on the fixed holdout validation set. 
#     <br><br>
#     <b>Tuning Framework (Shared Logic):</b>
#     <ul>
#         <li><b>Search Strategy:</b> Manual loop with <code>ParameterSampler</code> to avoid <code>sample_weight</code> routing issues.</li>
#         <li><b>Target Transform:</b> <code>TransformedTargetRegressor(log1p)</code> to handle skewness and optimize in log-space.</li>
#         <li><b>Sample Weights:</b> Normalized weights (mean=1.0) for training; raw survey weights for evaluation.</li>
#         <li><b>Scoring:</b> Weighted Median Absolute Error (MdAE) on raw-dollar predictions.</li>
#         <li><b>Iterations:</b> 10 in notebook for prototyping. Scale to 100 in <code>scripts/tune_random_forest.py</code> for production.</li>
#     </ul>
#     <b>Why not <code>RandomizedSearchCV</code>?</b><br>
#     Avoids <code>sample_weight</code> routing complexities and ensures transparent weighted scoring on a fixed holdout set.
#     <br><br>
#     <b>Evaluate Model Performance:</b>
#     <ul>
#         <li>Metrics Comparison Tables</li>
#         <li>Overfitting Analysis</li>
#         <li>Error Analysis</li>
#         <ul>
#             <li>Heteroscedasticity (Residuals vs. Predicted)</li> 
#             <li>Stratified Error Analysis</li>
#             <li>(optionally) Feature Dependencies (Residuals vs. Features)</li> 
#         </ul>
#     </ul>
# </div>

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
#     For hyperparameter details, refer to the official scikit-learn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">RandomForestRegressor documentation</a>.
#
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
#     📌 Perform randomized search and store results (best model as <code>.joblib</code>, metrics of all model runs as <code>.json</code>, predictions of best model as <code>.joblib</code>).
# </div>

# %%
# Normalize training weights (mean=1.0) for numerical stability during model fitting
w_train_norm = w_train / w_train.mean()

# Run randomized search
print("Tuning random forest...")    
rf_tuning_metrics = []

for i, params in enumerate(rf_param_list):
    # Build model: RandomForest wrapped in log-transform
    rf_model = TransformedTargetRegressor(
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
    start_time = time.time()  # Measure training time
    rf_model.fit(X_train_preprocessed, y_train, sample_weight=w_train_norm)
    training_time = time.time() - start_time
    
    # Predict on training and validation set (predictions are in raw dollars due to inverse_func)
    y_train_pred = rf_model.predict(X_train_preprocessed)
    y_val_pred = rf_model.predict(X_val_preprocessed)
    
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
        "val_r2": val_r2
    })
    
    print(f"  [{i+1:3d}/{N_ITER}] MdAE: {val_mdae:8.2f} | trees={params['n_estimators']}, depth={params['max_depth']}, leaf={params['min_samples_leaf']}, feats={params['max_features']}, samples={params['max_samples']:.2f}, split={params['min_samples_split']} | training: {training_time:5.1f} s")

# Retrain best model 
best_rf_params = sorted(rf_tuning_metrics, key=lambda x: x["val_mdae"])[0]["params"]
best_rf_model = TransformedTargetRegressor(
    regressor=RandomForestRegressor(
        criterion="absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **best_rf_params
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
best_rf_results = train_and_evaluate(best_rf_model, X_train_preprocessed, y_train, X_val_preprocessed, y_val, w_train, w_val)

# Persist results
save_metrics(rf_tuning_metrics, "../models/tuned_rf_metrics.json", verbose=False)
print("  Saved tuned random forest metrics to 'models/tuned_rf_metrics.json'")
save_model(best_rf_results["fitted_model"], "../models/random_forest_tuned.joblib", verbose=False)
print("  Saved best model to 'models/random_forest_tuned.joblib'")
save_model(best_rf_results["y_val_pred"], "../models/tuned_rf_predictions.joblib", verbose=False)
print("  Saved predicted values of best model to 'models/tuned_rf_predictions.joblib'")

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Evaluate tuning results. 
# </div>

# %%
# Load tuned random forest metrics from JSON file
rf_tuning_metrics = load_metrics("../models/tuned_rf_metrics.json")

# Display metric comparison table  
rf_tuning_metrics_df = pd.DataFrame(rf_tuning_metrics)
rf_tuning_metrics_df = rf_tuning_metrics_df.sort_values("val_mdae")  # sort by primary metric
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
