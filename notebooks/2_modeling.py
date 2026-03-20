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

# Model selection
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform  # for random hyperparameter values

# Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor 

# Model evaluation
from sklearn.metrics import (
    median_absolute_error,
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score
)

# Model persistence
import joblib

# Local imports
from src.constants import (
    DISPLAY_LABELS, 
    CATEGORY_LABELS_EDA,
    RANDOM_STATE,
    POP_COLOR,
    SAMPLE_COLOR
)
from src.pipeline import (
    create_preprocessing_pipeline, 
    create_missing_value_handling_pipeline
)
from src.utils import add_table_caption

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
data_inspection = pd.DataFrame(
    {
        "Train": [
            df_train_preprocessed.shape,
            "✅" if not df_train_preprocessed.isna().any().any() else "❌",
            "✅" if (df_train_preprocessed.select_dtypes(include=[np.number]).shape[1] == df_train_preprocessed.shape[1]) else "❌",
        ],
        "Val": [
            df_val_preprocessed.shape,
            "✅" if df_val_preprocessed.isna().any().any() == 0 else "❌",
            "✅" if (df_val_preprocessed.select_dtypes(include=[np.number]).shape[1] == df_val_preprocessed.shape[1]) else "❌",
        ],
        "Test": [
            df_test_preprocessed.shape,
            "✅" if df_test_preprocessed.isna().any().any() == 0 else "❌",
            "✅" if (df_test_preprocessed.select_dtypes(include=[np.number]).shape[1] == df_test_preprocessed.shape[1]) else "❌",
        ],
    },
    index=["Shape", "No Missing Values", "All Numerical"],
)
display(data_inspection.style.pipe(add_table_caption, "Data Inspection"))
