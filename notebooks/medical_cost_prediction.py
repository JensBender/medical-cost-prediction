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
# <div style="text-align:center; font-size:36px; font-weight:bold; color:#4A4A4A; background-color:#fff6e4; padding:10px; border:3px solid #f5ecda; border-radius:6px">
#     Medical Cost Prediction
#     <p style="text-align:center; font-size:14px; font-weight:normal; color:#4A4A4A; margin-top:12px;">
#         Author: Jens Bender <br> 
#         Created: December 2025<br>
#         Last updated: March 2026
#     </p>
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform  # for random hyperparameter values

# Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor 

# Model evaluation
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score
)

# Local imports
from src.constants import (
    DISPLAY_LABELS, 
    CATEGORY_LABELS_EDA
)
from src.transformers import (
    MedicalFeatureDeriver,
    OutlierRemover3SD,
    OutlierRemoverIQR
)
from src.pipeline import (
    create_preprocessing_pipeline, 
    create_missing_value_handling_pipeline
)

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     <strong>Constants & Helper Functions</strong>
# </div>

# %%
# Configuration
RANDOM_STATE = 42

# Plotting aesthetics
POP_COLOR = "#084594"    # deep navy for population
SAMPLE_COLOR = "#14b8a6" # vibrant teal for sample

def add_caption(styler, caption, font_size="14px", font_weight="bold", text_align="left"):
    """Adds a styled caption to a Pandas Styler object."""
    return styler.set_caption(caption).set_table_styles([{
        "selector": "caption", 
        "props": [("font-size", font_size), ("font-weight", font_weight), ("text-align", text_align)]
    }])

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Data Loading and Inspection</h1>
# </div>
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Load the MEPS-HC 2023 data from the <code>h251.sas7bdat</code> file (SAS V9 format) into a Pandas DataFrame.
# </div>

# %%
try:
    # Load data using 'latin1' encoding because MEPS SAS files don't store text as UTF-8 and instead use Western European (ISO-8859-1), also known as latin1.
    df = pd.read_sas("../data/h251.sas7bdat", format="sas7bdat", encoding="latin1")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file content could not be parsed.")
except PermissionError:
    print("Error: Permission denied when accessing the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> 📌 Initial data inspection to understand the structure of the dataset and detect obvious issues.</p>

# %%
# Show DataFrame info to check the number of rows and columns, data types and missing values
df.info()

# %%
# Show top five rows of the data
df.head()

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Data Preparation</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <strong>Note:</strong> Kept column names in ALL CAPS to ensure consistency with official <b><a href="../docs/references/h251doc.pdf">MEPS documentation</a></b>, <b><a href="../docs/references/h251cb.pdf">codebook</a></b>, and <b><a href="../docs/references/data_dictionary.md">data dictionary</a></b>.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Handling Duplicates</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Identify duplicates based on:
# <ul>
#     <li><strong>All columns</strong>: To detect exactly identical rows.</li>
#     <li><strong>ID column only</strong>: To ensure that no two people share the same ID.</li>
#     <li><strong>All columns except ID</strong>: To catch "hidden" duplicates where the same respondent may have been recorded twice under different IDs.</li>
# </ul>
# </div>

# %%
# Identify duplicates based on all columns
df.duplicated().value_counts()

# %%
# Identify duplicates based on the ID column
df.duplicated(["DUPERSID"]).value_counts()

# %% [markdown]
# <p style="background-color:#f7fff8; padding:15px; border-width:3px; border-color:#e0f0e0; border-style:solid; border-radius:6px"> ✅ No duplicates were found based on all columns or the ID column.</p>

# %%
# Identify duplicates based on all columns except ID columns
id_columns = ["DUPERSID", "DUID", "PID", "PANEL"]
duplicates_without_id = df.duplicated(subset=df.columns.drop(id_columns), keep=False)
duplicates_without_id.value_counts()

# %%
# Show duplicates
df[duplicates_without_id]

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Detected 3 pairs (6 rows) of  duplicates based on all columns except ID columns. 
#     <p style="margin-top:10px; margin-bottom:0px">
#         These respondents have identical values across all 1,300+ columns except for their IDs. They appear to be young siblings (ages 1 and 5) from the same household with identical parent-reported health data, sample weights, and costs. Analysis suggests they are valid respondents rather than "ghost" records. Regardless, they will be excluded when filtering for the adult target population.
#     </p>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Variable Selection</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Filter the following 29 columns (out of 1,374):
#     <ul style="margin-bottom:0px">
#         <li><b>ID:</b> Unique identifier for each respondent (<code>DUPERSID</code>).</li>
#         <li><b>Sample Weights:</b> Ensures population representativeness (<code>PERWT23F</code>).</li>
#         <li><b>Candidate Features:</b> 26 variables selected for their consumer accessibility, beginning-of-year measurement, and predictive power.</li> 
#         <li><b>Target Variable:</b> Total out-of-pocket health care costs (<code>TOTSLF23</code>).</li>
#     </ul>
#     <br>
#     <b>Rationale:</b> For a detailed breakdown of the target variable selection and feature selection criteria, see the <b><a href="../docs/specs/technical_specifications.md">Technical Specifications</a></b> and <b><a href="../docs/research/candidate_features.md">Candidate Features Research</a></b>.
# </div>

# %%
# List of columns to keep 
columns_to_keep = [
    # 1. ID
    "DUPERSID",
    
    # 2. Sample Weights
    "PERWT23F", 

    # 3 Candidate Features.
    # 3.1 Demographics
    "AGE23X", "SEX", "REGION23", "MARRY31X",
    
    # 3.2 Socioeconomic
    "POVCAT23", "FAMSZE23", "HIDEG", "EMPST31",
    
    # 3.3 Insurance & Access
    "INSCOV23", "HAVEUS42",
    
    # 3.4 Perceived Health & Lifestyle
    "RTHLTH31", "MNHLTH31", "ADSMOK42",
    
    # 3.5 Limitations & Symptoms
    "ADLHLP31", "IADLHP31", "WLKLIM31", "COGLIM31", "JTPAIN31_M18",
    
    # 3.6 Chronic Conditions
    "HIBPDX", "CHOLDX", "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX", 
    
    # 4. Healthcare Expenditure (Target)
    "TOTSLF23"
]

# Drop all other columns (keeping 29 out of 1,374)
df = df[columns_to_keep]

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Target Population Filtering</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Filter rows to match the target population (target audience for app) based on the following criteria:
#     <ul style="margin-bottom:0px">
#         <li><b>Positive person weight (<code>PERWT23F > 0</code>):</b> Drop respondents with a person weight of zero (i.e., 456 respondents). These individuals are considered "out-of-scope" for the full-year population (e.g., they joined the military, were institutionalized, or moved abroad).</li>
#         <li><b>Adults (<code>AGE23X >= 18</code>):</b> Drop respondents under age 18 (i.e., 3796 respondents), as the medical cost planner app targets adults.</li>
#     </ul>
#     <br>
#     <b>Note:</b> Keeps 14,768 out of 18,919 respondents.
# </div>

# %%
# Filter DataFrame 
df = df[(df["PERWT23F"] > 0) & (df["AGE23X"] >= 18)].copy() 

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Handling Data Types</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Identify and convert incorrect storage data types.
#     <ul>
#         <li><b>ID:</b> <code>DUPERSID</code> is an identifier, not a quantity. Converting them to <code>string</code> prevents unintended math.</li>
#         <li><b>Sample Weights:</b> <code>PERWT23F</code> contains decimal precision critical for population-level estimates. Must remain <code>float</code>.</li>
#         <li><b>Candidate Features:</b> The SAS loader stored all 26 features as floats by default. Although many features are categorical and represent integer codes (e.g., 1=Male, 2=Female), they are maintained as <code>float</code> for three practical reasons:
#             <ul>
#                 <li>Missing Value Compatibility: In standard Pandas, <code>np.nan</code> is a floating-point object. Assigning it to an integer column automatically casts back to <code>float64</code>.</li>
#                 <li>Data Preprocessing Consistency: scikit-learn transformers (e.g., <code>SimpleImputer</code>, <code>StandardScaler</code>) internally use floats and automatically convert numerical inputs to <code>float</code>, even when using <code>set_config(transform_output="pandas")</code>. Keeping them as floats avoids redundant type casting.</li>
#                 <li>Model Consistency: Most machine learning models (e.g., XGBoost, Linear Regression) internally use floats and automatically convert numerical inputs to <code>float</code> during training and inference. Keeping them as floats avoids redundant type casting.</li>
#             </ul>
#         </li>
#         <li><b>Target Variable:</b> <code>TOTSLF23</code> was stored as <code>float</code> by the SAS loader. Although it is rounded to whole dollars in the MEPS data, it is kept as <code>float</code> for data preprocessing and model consistency and to avoid redundant type casting, as ML models deliver <code>float</code> predictions during training and inference.</li>
#     </ul>
# </div>

# %%
# Identify storage data types (defaulted to float/object by SAS loader)
df.dtypes

# %%
# Convert ID to string and set as index
df["DUPERSID"] = df["DUPERSID"].astype(str)
df.set_index("DUPERSID", inplace=True)

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> 
#     📌 Define semantic data types of features (numerical, binary, nominal, ordinal) for downstream tasks like EDA, further preprocessing steps, and machine learning. Distinguishes between <code>raw_</code> lists for discovery EDA and <code>input_</code> lists for validation of engineered features before they enter the preprocessing pipeline.
# </div> 

# %%
# Define semantic data types (raw)
raw_numerical_features = ["AGE23X", "FAMSZE23", "RTHLTH31", "MNHLTH31"]
raw_binary_features = [
    "SEX", "HAVEUS42", "ADSMOK42", "ADLHLP31", "IADLHP31", 
    "WLKLIM31", "COGLIM31", "JTPAIN31_M18", "HIBPDX", "CHOLDX", 
    "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"
]
raw_nominal_features = ["REGION23", "MARRY31X", "EMPST31", "INSCOV23", "HIDEG"]
raw_ordinal_features = ["POVCAT23"]

# Combined raw feature sets
raw_categorical_features = raw_nominal_features + raw_ordinal_features + raw_binary_features
raw_all_features = raw_numerical_features + raw_categorical_features

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Standardizing Missing Values</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Pandas Missing Value Types</b>:
#     <ul style="margin-bottom:0px">
#         <li><b>np.nan:</b> Standard missing value indicator (technically a float); often the default in Pandas for numerical data.</li>
#         <li><b>pd.NA:</b> Unified missing value indicator for modern nullable data types (mostly integer and boolean).</li>
#         <li><b>None:</b> Python's native type; often used for object and string data.</li>
#         <li><b>pd.NaT:</b> For datetime and timedelta data types.</li>
#     </ul>
#     <br>
#     ℹ️ <b>MEPS Missing Value Codes:</b>
#     <ul style="margin-bottom:0px">
#         <li><b>-1 INAPPLICABLE:</b> Variable does not apply (structural skip).</li>
#         <li><b>-7 REFUSED:</b> Person refused to answer.</li>
#         <li><b>-8 DON'T KNOW:</b> Person did not know the answer.</li>
#         <li><b>-9 NOT ASCERTAINED:</b> Administrative or technical error in collection.</li>
#         <li><b>-15 CANNOT BE COMPUTED:</b> Incomplete data for a constructed variable.</li>
#     </ul>
# </div>


# %%
# Identify Pandas missing values
df.isnull().sum()

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 <b>Handle MEPS-Specific Missing Values and Skip Patterns</b>
#     <br><br>
#     <b>Understanding Skip Patterns</b><br>
#     Skip patterns (routing logic) are used in surveys to ensure respondents only answer questions relevant to them, reducing burden and improving data quality. For analysis, this creates "structural" missingness (coded as -1 Inapplicable) can often be recovered by looking at the respondent's path through the survey.
#     <br><br>
#     <b>Recovering Implied Values:</b>  
#     <ul>
#         <li><b>Smoker (<code>ADSMOK42</code>):</b> This question is only asked if the respondent already confirmed smoking 100+ cigarettes in their life. Those who said "No" skip this and are coded -1. In this project, these "Never Smokers" are mapped to "No" (2).</li>
#         <li><b>Joint Pain (<code>JTPAIN31_M18</code>):</b> Respondents who already reported an arthritis diagnosis (<code>ARTHDX = 1</code>) earlier in the interview skip this question and are coded -1. Since arthritis inherently involves joint symptoms, these values are mapped to "Yes" (1).</li>
#     </ul>
# </div>

# %%
# Identify MEPS missing values 
missing_codes = [-1, -7, -8, -9, -15]
missing_frequency_df = pd.DataFrame({code: (df == code).sum() for code in missing_codes})
missing_frequency_df["TOTAL"] = missing_frequency_df.sum(axis=1)
missing_frequency_df["PERCENTAGE"] = (missing_frequency_df["TOTAL"] / len(df) * 100).round(2)
missing_frequency_df.sort_values("TOTAL", ascending=False) 

# %%
# Recover implied values
# Smoker: Convert -1 (Never Smoker) to 2 (No)
df.loc[df["ADSMOK42"] == -1, "ADSMOK42"] = 2

# Joint Pain: Convert -1 to 1 (Yes) only if they have Arthritis 
df.loc[(df["JTPAIN31_M18"] == -1) & (df["ARTHDX"] == 1), "JTPAIN31_M18"] = 1

# Convert remaining MEPS missing codes to np.nan
df.replace(missing_codes, np.nan, inplace=True)

# Verify results
df.isnull().sum().sort_values(ascending=False)

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Standardizing Binary Features</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Standardize all binary features to 0/1 encoding. This ensures interpretability for regression coefficients and compatibility with all models.
# </div>

# %%
# Standardize all binary features to 0/1 encoding. In MEPS, binary features typically use 1 (Yes) and 2 (No).
# Mapping 2 to 0 aligns with the standard format (1 = presence, 0 = absence).
df[raw_binary_features] = df[raw_binary_features].replace({2: 0})

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Exploratory Data Analysis (EDA)</h1>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Analyze univariate distributions using descriptive statistics and visualizations. Focus on understanding raw feature distributions to identify data quality issues (e.g., sparse categories) and inform decisions on subsequent data preprocessing and feature engineering. After each preprocessing or engineering step, conduct EDA again to confirm that transformations (like category collapsing) were successful and that the resulting distributions are robust for pipeline input.
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Univariate EDA</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Analyze the distribution of a single column using descriptive statistics and visualizations.
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Sample Weights</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Examine descriptive statistics and visualize the distribution of the sample weights. 
# </div>

# %%
# Descriptive statistics of sample weights
df["PERWT23F"].describe()

# %%
# Sum of sample weights
df["PERWT23F"].sum()

# %%
# Histogram of sample weights
sns.histplot(df["PERWT23F"])
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Using sample weights in machine learning models is essential to correct for oversampling and ensure that predictions are representative of the U.S. civilian noninstitutionalized adult population. 
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Sum:</b> The sum of all weights is approximately 260 million, representing the estimated U.S. adult population in 2023.</li>
#         <li><b>Median:</b> A typical respondent represents roughly 14,600 people.</li>
#         <li><b>Right-Skewed Distribution:</b> The mean (17,584) is higher than the median, confirming that a small number of respondents represent a disproportionately large share of the population.</li>
#         <li><b>Sampling Strategy:</b> Weights range from 502 to 131,657. This reflects MEPS's strategy of oversampling specific subgroups to ensure reliable estimates for minority or high-need populations.</li>
#         <li><b>Bias Correction:</b> Because weights vary significantly (std ≈ 12,334), unweighted models or averages would be biased. Using <code>sample_weight</code> during training ensures the model's loss function prioritizes population representativeness.</li>
#     </ul>
# </div>

# %% [markdown]
# <a id="target-variable" name="target-variable"></a>
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Target Variable</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Examine descriptive statistics and visualize the distribution of total annual out-of-pocket healthcare costs.
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Descriptive Statistics</strong> <br>
#     📌 Examine descriptive statistics of out-of-pocket healthcare costs, both on sample-level and population-level. 
# </div>

# %%
# Helper function to calculate weighted quantiles (using numpy; not available in pandas)
def weighted_quantile(variable, weights, quantile):
    sorter = np.argsort(variable)
    values = variable.iloc[sorter]
    weights = weights.iloc[sorter]
    cumulative_weight = np.cumsum(weights) - 0.5 * weights
    cumulative_weight /= np.sum(weights)
    return np.interp(quantile, cumulative_weight, values)

# Helper function to calculate weighted standard deviation (using numpy; not available in pandas)
def weighted_std(variable, weights):
    weighted_mean = np.average(variable, weights=weights)
    weighted_variance = np.average((variable - weighted_mean)**2, weights=weights)
    return np.sqrt(weighted_variance)


# Calculate population statistics (weighted)
pop_mean = np.average(df["TOTSLF23"], weights=df["PERWT23F"])
pop_std = weighted_std(df["TOTSLF23"], weights=df["PERWT23F"])
pop_p25 = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], 0.25)
pop_median = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], 0.5)
pop_p75 = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], 0.75)
pop_p95 = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], 0.95)
pop_p99 = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], 0.99)
pop_p999 = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], 0.999)

# Calculate sample quantiles (unweighted)
sample_p95 = df["TOTSLF23"].quantile(0.95)
sample_p99 = df["TOTSLF23"].quantile(0.99)
sample_p999 = df["TOTSLF23"].quantile(0.999)

# Calculate total costs (sum)
sample_total_costs = df["TOTSLF23"].sum()
df["pop_costs"] = df["TOTSLF23"] * df["PERWT23F"]  
pop_total_costs = df["pop_costs"].sum()

# Create comparison table: Sample vs. population statistics 
sample_vs_population_stats = pd.DataFrame({
    "Sample (Unweighted)": [
        len(df),
        df["TOTSLF23"].mean(),
        df["TOTSLF23"].std(),
        df["TOTSLF23"].min(),
        df["TOTSLF23"].quantile(0.25),
        df["TOTSLF23"].median(),
        df["TOTSLF23"].quantile(0.75),
        sample_p95,
        sample_p99,
        sample_p999,
        df["TOTSLF23"].max(),
        sample_total_costs
    ],
    "Population (Weighted)": [
        df["PERWT23F"].sum(),  # sum of weights = count of target population
        pop_mean,
        pop_std,
        df["TOTSLF23"].min(),  # min is identical
        pop_p25,
        pop_median,
        pop_p75,
        pop_p95,
        pop_p99,
        pop_p999,
        df["TOTSLF23"].max(),  # max is identical
        pop_total_costs
    ]
}, index=["count", "mean", "std", "min", "25%", "50%", "75%", "95%", "99%", "99.9%", "max", "sum"])

# Display table
# Formatting: Comma thousand separator and rounded to zero decimals (sample sum in Millions, population sum in Billions with one decimal)
sample_vs_population_stats.style \
    .pipe(add_caption, "Descriptive Statistics") \
    .format("{:,.0f}") \
    .format(lambda x: f"${x/1e6:.1f}M", subset=("sum", "Sample (Unweighted)")) \
    .format(lambda x: f"${x/1e9:.1f}B", subset=("sum", "Population (Weighted)"))


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The descriptive statistics reveal a highly skewed and volatile cost distribution, emphasizing the importance of using sample weights for representative estimates.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Right Skewness:</b> The population mean (\$1,106) is over 4x higher than the median (\$251), indicating that a few high-cost cases disproportionately influence the average.</li>
#         <li><b>Sampling Bias Correction:</b> The mean, median and most percentiles (up to the 99th) are lower in the population compared to the sample (e.g., mean drops from \$1,160 to \$1,106), showing that the raw sample slightly over-represented most higher-cost individuals. However, the 99.9th percentile and standard deviation are actually higher in the population, revealing that the raw sample slightly under-represented the most extreme tail risk of the "super-spenders" (top 0.1%).</li>
#         <li><b>Extreme Financial Risk:</b> While 75% of the population spends less than \$1,042 out-of-pocket, the maximum reaches \$104,652, highlighting severe financial exposure for a minority.</li>
#         <li><b>High Dispersion:</b> The standard deviation (~\$3,000) is nearly triple the mean, reflecting the inherent unpredictability and high variance in health care costs.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Histogram</strong> <br> 
#     📌 Visualize the distribution of out-of-pocket healthcare costs. 
# </div>

# %%
# Histogram: Sample vs. Population (overlayed)
plt.figure(figsize=(10, 6))

# Population histogram 
sns.histplot(
    data=df, 
    x="TOTSLF23", 
    weights="PERWT23F", 
    label="Population (Weighted)",
    stat="probability", 
    bins=50, 
    color=POP_COLOR, 
    alpha=0.4,  # increased alpha for richer color
    element="bars",
    edgecolor="white",
    linewidth=0.3
)
# Sample histogram 
sns.histplot(
    data=df, 
    x="TOTSLF23", 
    label="Sample (Unweighted)",
    stat="probability", 
    bins=50, 
    color=SAMPLE_COLOR,
    alpha=1.0, 
    element="step",
    linewidth=2.0,
    linestyle="--" # dashed for comparison
)

# Add population mean and median lines for context
plt.axvline(pop_mean, color="#e63946", linestyle="--", alpha=0.8, label=f"Population Mean: ${pop_mean:,.0f}")
plt.axvline(pop_median, color="#fb8500", linestyle="--", alpha=0.8, label=f"Population Median: ${pop_median:,.0f}")

# Customize
plt.title("Distributions of Out-of-Pocket Costs", fontsize=14, fontweight="bold")
plt.xlabel("")
plt.ylabel("Share", fontsize=12)
plt.grid(True, alpha=0.3)
sns.despine()  # removes top & right spines
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))  # format X as dollars with comma thousand separator rounded to zero decimals
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # format Y as percentages
plt.legend()

# Adjust layout 
plt.tight_layout()

plt.show()

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     ℹ️ Note: Due to the zero-inflation (22% have \$0 costs) and the extremely heavy tail (max \$104k), the full distribution is heavily compressed into the first few bins.<br>
#     📌 Visualize the "typical" distribution excluding zero costs and the top 5% of spenders (zooming in).
# </div>

# %%
# Histogram of typical range (excluding zero costs and top 5%)
plot_data = df[(df["TOTSLF23"] > 0) & (df["TOTSLF23"] < pop_p95)].copy()

plt.figure(figsize=(10, 6))

# Population histogram 
sns.histplot(
    data=plot_data, 
    x="TOTSLF23", 
    weights="PERWT23F", 
    label="Population (Weighted)",
    stat="probability", 
    bins=50, 
    color=POP_COLOR, 
    alpha=0.3,  # lighter filling for the background
    element="bars",
    edgecolor="white",
    linewidth=0.5
)
# Sample histogram 
sns.histplot(
    data=plot_data, 
    x="TOTSLF23", 
    label="Sample (Unweighted)",
    stat="probability", 
    bins=50, 
    color=SAMPLE_COLOR,
    alpha=1.0,        # full opacity for the line
    element="step",
    linewidth=2.5     # thicker line to stand out
)

# Add population mean and median lines for context
plt.axvline(pop_mean, color="#e63946", linestyle="--", alpha=0.8, label=f"Population Mean: ${pop_mean:,.0f}")
plt.axvline(pop_median, color="#fb8500", linestyle="--", alpha=0.8, label=f"Population Median: ${pop_median:,.0f}")

# Customize
plt.title("Distributions of Typical Out-of-Pocket Costs (excluding zero costs and top 5%)", fontsize=14, fontweight="bold")
plt.xlabel("")
plt.ylabel("Share", fontsize=12)
plt.grid(True, alpha=0.3)
sns.despine()  # removes top & right spines
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))  # format X as dollars with thousand separator
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # format Y as percentages
plt.legend()

# Adjust layout 
plt.tight_layout()

plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The "typical" distribution (excluding zeros and the top 5%) remains heavily right-skewed with a massive share of the population still concentrated in the lowest cost bins. To get a better picture, I will conduct in-depth Zero Costs Analysis, Lorenz Curve, Cost Concentration Analysis, and Top 1% Analysis.
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Zero Costs Analysis</strong> <br>
#     📌 Deeper analysis of people with zero out-of-pocket health care costs (sample and population). 
# </div>

# %%
# Zero costs analysis
zero_costs_df = pd.DataFrame({
    "Sample (Unweighted) Count": [len(df[df["TOTSLF23"] == 0]), len(df[df["TOTSLF23"] > 0])],
    "Population (Weighted) Count": [df[df["TOTSLF23"] == 0]["PERWT23F"].sum(), df[df["TOTSLF23"] > 0]["PERWT23F"].sum()],
    "Sample (Unweighted) %": [
        (df["TOTSLF23"] == 0).mean() * 100,
        (df["TOTSLF23"] > 0).mean() * 100
    ],
    "Population (Weighted) %": [
        (df.loc[df["TOTSLF23"] == 0, "PERWT23F"].sum() / df["PERWT23F"].sum()) * 100,
        (df.loc[df["TOTSLF23"] > 0, "PERWT23F"].sum() / df["PERWT23F"].sum()) * 100
    ]
}, index=["Zero Costs", "Positive Costs"]).round(2)

zero_costs_df.style \
    .pipe(add_caption, "Zero Costs Analysis") \
    .format({
        "Sample (Unweighted) Count": "{:,.0f}",
        "Population (Weighted) Count": "{:,.0f}",
        "Sample (Unweighted) %": "{:.1f}%",
        "Population (Weighted) %": "{:.1f}%"
    })

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The large proportion of zeros confirms that the target variable is zero-inflated.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>High Zero-Cost Prevalence:</b> Over 22% of the U.S. adult population (approx. 58 million people) had zero out-of-pocket health care costs in 2023.</li>
#         <li><b>Correction of Sampling Bias:</b> The weighted population percentage (22.3%) is higher than the unweighted sample percentage (20.7%), indicating that zero-cost individuals were slightly under-represented in the raw survey data.</li>
#         <li><b>Modeling Implications:</b> The zero-inflated target variable suggests that a two-part modeling strategy (e.g., predicting the probability of any spend vs. the amount of spend) may be more effective than a single standard regression.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Lorenz Curve</strong> <br>
#     📌 Plot the Lorenz Curve. 
# </div>

# %%
# Helper Function: Plot Lorenz Curve
def plot_lorenz_curve(df, column, weights=None, save_to_file=None):
    """Plot a Lorenz Curve to visualize concentration and inequality.

    Args:
        df:            DataFrame containing the data.
        column:        Column name of the variable.
        weights:       Optional name of the weight column.
        save_to_file:  Optional file path to save the figure (e.g., 'lorenz_curve.png').
    """
    # Create DataFrame copy, sorted by column and reset index to ensure alignment
    lorenz_df = df[[column]].copy()
    if weights:
        lorenz_df[weights] = df[weights]
    
    lorenz_df = lorenz_df.sort_values(column).reset_index(drop=True)
    
    if weights:
        # Cumulative percentage and costs of population (weighted)
        cum_pct = lorenz_df[weights].cumsum() / lorenz_df[weights].sum() * 100
        pop_costs = lorenz_df[column] * lorenz_df[weights]
        cum_costs = pop_costs.cumsum() / pop_costs.sum() * 100
    else:
        # Cumulative percentage and costs of sample (unweighted)
        cum_pct = pd.Series(np.arange(1, len(lorenz_df) + 1) / len(lorenz_df) * 100)
        cum_costs = lorenz_df[column].cumsum() / lorenz_df[column].sum() * 100

    # Calculate Gini Coefficient
    def calculate_gini(pct, costs):
        return 1 - 2 * np.trapezoid(costs / 100, pct / 100)
    
    gini = calculate_gini(cum_pct, cum_costs)

    # Plotting
    plt.figure(figsize=(10, 8))  

    # Line of Equality (with label)
    plt.plot([0, 100], [0, 100], linestyle="--", color="gray", alpha=0.6)
    plt.text(50, 49, "Line of Equality", rotation=38, color="gray", 
             fontsize=10, ha="center", va="bottom")

    # Lorenz Curve
    plt.plot(cum_pct, cum_costs, color=POP_COLOR if weights else SAMPLE_COLOR, lw=3)

    # Add Gini Coefficient (in text box)
    plt.text(4, 96, f"Gini Coefficient: {gini:.2f}", fontsize=12, fontweight="bold",
             va="top", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "boxstyle": "round,pad=0.4"})

    # Highlight Pareto Point (80/20 Rule) 
    idx_80 = (cum_pct - 80).abs().idxmin()
    x_80 = cum_pct.loc[idx_80]
    y_80 = cum_costs.loc[idx_80]
    top_20_share = 100 - y_80

    plt.plot(x_80, y_80, "o", color="#fb8500", markersize=8)
    plt.annotate(f"Top 20% account for\n{top_20_share:.1f}% of costs", 
                 xy=(x_80, y_80), xytext=(x_80 - 30, y_80 + 10),
                 arrowprops={"arrowstyle": "->", "color": "black", "connectionstyle": "arc3,rad=-0.2", "alpha": 0.8},
                 fontsize=10, fontweight="bold")

    # Highlight Top 1%
    idx_99 = (cum_pct - 99).abs().idxmin()
    x_99 = cum_pct.loc[idx_99]
    y_99 = cum_costs.loc[idx_99]
    top_1_share = 100 - y_99

    plt.plot(x_99, y_99, "o", color="#fb8500", markersize=8)
    plt.annotate(f"Top 1% account for\n{top_1_share:.1f}% of costs", 
                 xy=(x_99, y_99), xytext=(x_99 - 8, y_99 - 15),
                 arrowprops={"arrowstyle": "->", "color": "black", "connectionstyle": "arc3,rad=-0.2", "alpha": 0.8},
                 fontsize=10, fontweight="bold", ha="right")

    # Highlight Zero-Cost Threshold
    zero_mask = df[column].eq(0)
    if weights:
        zero_pct = (np.sum(df.loc[zero_mask, weights]) / np.sum(df[weights])) * 100
    else:
        zero_pct = len(df[df[column] == 0]) / len(df) * 100
        
    plt.plot(zero_pct, 0, "o", color="#fb8500", markersize=8)
    plt.annotate(f"{zero_pct:.1f}% have $0 costs", 
                 xy=(zero_pct, 0), xytext=(zero_pct - 7, 10),
                 arrowprops={"arrowstyle": "->", "color": "black", "connectionstyle": "arc3,rad=-0.2", "alpha": 0.8},
                 fontsize=10, fontweight="bold")

    # Fill for emphasis (The "Inequality Gap")
    plt.fill_between(cum_pct, cum_costs, cum_pct, color=POP_COLOR if weights else SAMPLE_COLOR, alpha=0.08)

    # Customize
    plt.title(f"Lorenz Curve: Concentration of {DISPLAY_LABELS.get(column, column)}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel(f"Cumulative % of {'U.S. Population' if weights else 'Sample'} (Lowest to Highest Cost)", fontsize=12, labelpad=10)
    plt.ylabel("Cumulative % of Total Costs", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, 101, 10))
    plt.xlim(-3, 103)  # Adds space around the curve so no overlap with axis
    plt.ylim(-3, 103)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)
    plt.show()


# Plot sample Lorenz curve of out-of-pocket costs
plot_lorenz_curve(df, column="TOTSLF23")

# Plot population Lorenz curve of out-of-pocket costs
plot_lorenz_curve(df, column="TOTSLF23", weights="PERWT23F", save_to_file="../figures/eda/lorenz_curve.png")

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The Lorenz Curve reveals extreme inequality in out-of-pocket health care spending, far exceeding typical measures of economic inequality.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>The 80/20 Rule:</b> The top 20% of spenders account for <b>79.3%</b> of total costs, almost perfectly reflecting the Pareto Principle.</li>
#         <li><b>Inequality Comparison:</b> A Gini coefficient of <b>0.77</b> represents massive concentration. For context, U.S. income inequality (Gini ~0.45) is often considered high; healthcare cost inequality is significantly more than wealth.</li>
#         <li><b>The "Low-Cost" Majority:</b> The curve remains flat for the first <b>70%</b> of the population, who combined account for only about <b>12%</b> of total out-of-pocket costs.</li>
#         <li><b>Zero Costs:</b> A significant "hurdle" for modeling is the <b>22.3%</b> of U.S. adults who incur $0 in costs, requiring a model capable of handling zero-inflation.</li>
#     </ul>
# </div> 

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Cost Concentration Analysis</strong> <br>
#     📌 Examine the cost thresholds, totals, and shares for the top 1%, 5%, 10%, 20%, and 50% of spenders (sample and population). 
# </div>

# %%
# Cost Concentration Benchmarks 
percentiles = [0.99, 0.95, 0.9, 0.8, 0.5]
stats = []

sample_total_costs = df["TOTSLF23"].sum()
pop_total_costs = df["pop_costs"].sum()

for p in percentiles:
    # Calculate Thresholds
    sample_threshold = df["TOTSLF23"].quantile(p)
    pop_threshold = weighted_quantile(df["TOTSLF23"], df["PERWT23F"], p)
    
    # Sample Share (Unweighted)
    sample_share = (df[df["TOTSLF23"] >= sample_threshold]["TOTSLF23"].sum() / sample_total_costs) * 100
    
    # Population Share (Weighted)
    pop_share = (df[df["TOTSLF23"] >= pop_threshold]["pop_costs"].sum() / pop_total_costs) * 100
    
    stats.append({
        "Top X%": f"Top {(1-p)*100:.0f}%",
        "Threshold (Sample)": sample_threshold,
        "Threshold (Population)": pop_threshold,
        "Share of Total Costs (Sample)": sample_share,
        "Share of Total Costs (Population)": pop_share
    })

# Set "Top X%" as DataFrame index
cost_concentration_df = pd.DataFrame(stats).set_index("Top X%")

# Show table with formatted values
cost_concentration_df.style \
    .pipe(add_caption, "Cost Concentration Analysis") \
    .format({
        "Threshold (Sample)": "${:,.0f}",
        "Threshold (Population)": "${:,.0f}",
        "Share of Total Costs (Sample)": "{:.1f}%",
        "Share of Total Costs (Population)": "{:.1f}%"
    })

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> The cost concentration analysis quantifies the "heavy tail" of the distribution:
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Top 1% vs. Bottom 50%:</b> The top 1% of spenders account for <b>20.6%</b> of total costs. This is nearly <b>ten times</b> more than the bottom 50% of the population combined (2.4%).</li>
#         <li><b>Exponential Escalation:</b> The cost threshold more than <b>doubles</b> between the top 5% ($4,518) and the top 1% (\$12,868), indicating that financial risk increases exponentially as one moves into the tail.</li>
#         <li><b>Predictive Challenge:</b> Because the bottom half of the population contributes so little to the total sum, a model that focuses on the "average" person will be highly accurate for the majority but fail catastrophically for the "super-spenders" who drive the actual financial risk.</li>
#     </ul>
# </div> 

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Top 1% Analysis</strong> <br>
#     📌 Deeper analysis of respondents in the top 1% and top 0.1% of out-of-pocket health care costs to understand extreme tail risk. 
# </div>

# %%
# Top 1% analysis
# Sample (Unweighted)
sample_top_1_costs = df[df["TOTSLF23"] >= sample_p99]["TOTSLF23"].sum()
sample_top_01_costs = df[df["TOTSLF23"] >= sample_p999]["TOTSLF23"].sum()

# Population (Weighted)
pop_top_1_costs = df[df["TOTSLF23"] >= pop_p99]["pop_costs"].sum()
pop_top_01_costs = df[df["TOTSLF23"] >= pop_p999]["pop_costs"].sum()

# Create comparison table
top_1_df = pd.DataFrame({
    "Sample (Unweighted)": [
        sample_p99,
        sample_top_1_costs / 1e6,  # Millions
        (sample_top_1_costs / sample_total_costs) * 100,
        sample_p999,
        sample_top_01_costs / 1e6,  # Millions
        (sample_top_01_costs / sample_total_costs) * 100
    ],
    "Population (Weighted)": [
        pop_p99,
        pop_top_1_costs / 1e9,  # Billions
        (pop_top_1_costs / pop_total_costs) * 100,
        pop_p999,
        pop_top_01_costs / 1e9,  # Billions
        (pop_top_01_costs / pop_total_costs) * 100
    ]
}, index=[
    "Top 1% Threshold",
    "Top 1% Total Costs",
    "Top 1% Share of Costs",
    "Top 0.1% Threshold",
    "Top 0.1% Total Costs",
    "Top 0.1% Share of Costs"
])

# Style the table
top_1_df.style \
    .pipe(add_caption, "Top 1% Analysis") \
    .format("${:,.0f}", subset=(["Top 0.1% Threshold", "Top 1% Threshold"], slice(None))) \
    .format("${:,.1f}M", subset=(["Top 0.1% Total Costs", "Top 1% Total Costs"], "Sample (Unweighted)")) \
    .format("${:,.1f}B", subset=(["Top 0.1% Total Costs", "Top 1% Total Costs"], "Population (Weighted)")) \
    .format("{:.1f}%", subset=(["Top 0.1% Share of Costs", "Top 1% Share of Costs"], slice(None)))

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Extreme cost concentration in the tail of the distribution.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>The 1% Rule:</b> The top 1% of spenders —those spending over ~\$13k—account for roughly 20% of all out-of-pocket costs.</li>
#         <li><b>The Hyper-Tail (Top 0.1%):</b> The top 0.1% of spenders—those spending over ~\$39k—account for a disproportionate share of total costs (approx. 5%). This highlights that the tail is not just long, but extremely heavy.</li>
#         <li><b>Extreme Outliers:</b> The gap between the 99th percentile (\$12,868) and the maximum (\$104,652) is massive. These "super-spenders" represent a significant challenge for predictive modeling, as a single misprediction here could lead to very high error (RMSE).</li>
#         <li><b>Financial Risk Benchmarks:</b> These thresholds provide a clear picture of what "catastrophic" spending looks like in the U.S. adult population.</li>
#     </ul>
# </div>

# %%
# Inspecting the "Super-Spenders" (Everyone in the Top 0.1% of the population)
# This helps us identify if there are common patterns (e.g., age or chronic conditions) among extreme spenders.
chronic_cols = ["HIBPDX", "CHOLDX", "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"]
df["CHRONIC_COUNT"] = (df[chronic_cols] == 1).sum(axis=1)

super_spenders = df[df["TOTSLF23"] >= pop_p999].sort_values("TOTSLF23", ascending=False)
super_spenders[["TOTSLF23", "PERWT23F", "AGE23X", "SEX", "INSCOV23", "CHRONIC_COUNT"] + chronic_cols].style.pipe(add_caption, "Super-Spenders")

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The "Super-Spender" profiles reveal a critical dichotomy for predictive modeling.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Acute vs. Chronic Drivers:</b> We see two distinct profiles: "Multi-Morbid" elderly (e.g., 85yo with 7 conditions) and "Acute Shock" individuals (young with 0-1 conditions). This suggests that while chronic disease predicts higher costs on average, the extreme tail is often driven by unpredictable acute events (accidents, rare surgeries).</li>
#         <li><b>The "Black Swan":</b> The absolute maximum outlier is a 24-year-old with only 1 chronic condition but ~\$104k in out-of-pocket costs. With a high sample weight (~31k), this single respondent represents over 30,000 people. This is a "Black Swan" event—atypical and unpredictable—that could severely impair model performance if not handled carefully. Mathematically, because RMSE squares the error, a single \$50,000 prediction error on the top outlier adds as much to the loss function as \$1,500 errors on 1,000 different people. The model will be naturally "obsessed" with fitting these few outliers at the expense of the general population.</li>
#         <li><b>The Insurance Paradox:</b> Surprisingly, only one of the top 15 spenders is uninsured (<code>INSCOV23=3</code>). Most even have private insurance (n=11). This shows that high out-of-pocket costs in the U.S. are not just a problem for the uninsured, but often hit those with coverage who face high deductibles, co-pays, or non-covered services.</li>
#         <li><b>Strategy:</b> To prevent these outliers from dominating the training, we must consider target transformations (Log/Box-Cox) or robust regression techniques that "downweight" the influence of extreme residuals.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Numerical Features</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Examine descriptive statistics and visualize the distributions of all numerical features, both on sample-level and population-level.
# </div>

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Descriptive Statistics</strong> <br>
#     📌 Examine descriptive statistics (e.g., mean, median, standard deviation) of numerical features.
# </div>

# %%
# Sample statistics (unweighted) 
df[raw_numerical_features].describe().T.style \
    .pipe(add_caption, "Descriptive Statistics (Sample)") \
    .format({
        "count": "{:,.0f}",
        "mean": "{:.2f}",
        "std": "{:.2f}",
        "min": "{:.1f}",
        "25%": "{:.1f}",
        "50%": "{:.1f}",
        "75%": "{:.1f}",
        "max": "{:.1f}"
    })

# %%
# Population statistics (weighted) 
pop_stats = []
# Iterate over all numerical features
for feature in raw_numerical_features:
    # Drop missing values for the current feature
    valid_mask = df[feature].notna()
    values = df.loc[valid_mask, feature]
    weights = df.loc[valid_mask, "PERWT23F"]
    
    pop_stats.append({
        "count": weights.sum(),  # Sum of weights is the estimated population count
        "mean": np.average(values, weights=weights),
        "std": weighted_std(values, weights),
        "min": values.min(),
        "25%": weighted_quantile(values, weights, 0.25),
        "50%": weighted_quantile(values, weights, 0.50),
        "75%": weighted_quantile(values, weights, 0.75),
        "max": values.max()
    })

# Show table of population statistics
pop_stats_df = pd.DataFrame(pop_stats, index=raw_numerical_features)
pop_stats_df.style \
    .pipe(add_caption, "Descriptive Statistics (Population)") \
    .format({
        "count": "{:,.0f}",
        "mean": "{:.2f}",
        "std": "{:.2f}",
        "min": "{:.1f}",
        "25%": "{:.1f}",
        "50%": "{:.1f}",
        "75%": "{:.1f}",
        "max": "{:.1f}"
    })


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Visualize Distributions</strong> <br> 
#     📌 Plot a histogram matrix that shows the distributions of all numerical features. 
# </div>

# %%
# Helper Function: Plot the Distributions of Numerical Features
def plot_numerical_distributions(df, numerical_features, display_labels=None, weights=None, save_to_file=None):
    # Define subplot matrix grid
    n_plots = len(numerical_features)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create subplot matrix
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    
    # Flatten axes for easier iteration
    axes_flat = axes.flat
    
    # Iterate over all numerical features
    for i, feature in enumerate(numerical_features):
        # Get current axes
        ax = axes_flat[i]
    
        # Population
        if weights:  
            # Create histogram (weighted)
            sns.histplot(
                data=df, x=feature, weights=weights, ax=ax, discrete=True,
                kde=True if feature == "AGE23X" else False,
                edgecolor="white", alpha=0.7, color=POP_COLOR
            )
            # Calculate completion rate (weighted)
            completion_rate = df.loc[df[feature].notna(), weights].sum() / df[weights].sum() * 100
        # Sample
        else:  
            # Create histogram (unweighted)
            sns.histplot(
                data=df, x=feature, ax=ax, discrete=True,
                kde=True if feature == "AGE23X" else False,
                edgecolor="white", alpha=0.7, color=SAMPLE_COLOR
            )
            # Calculate completion rate (unweighted)
            completion_rate = df[feature].count() / len(df) * 100
    
        # Customize current histogram
        ax.set_title(display_labels.get(feature, feature), fontsize=14, fontweight="bold", pad=20)
        completion_rate_label = "100% Complete" if completion_rate >= 99.95 else f"{completion_rate:.1f}% Complete"
        ax.annotate(completion_rate_label, xy=(0.5, 1), xytext=(0, 5),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#666666")
        ax.set_xlabel("")
        
        # Format population counts in millions
        if weights:  
            ax.set_ylabel("Count (Millions)" if i % n_cols == 0 else "", fontsize=12)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}"))
        else:
            ax.set_ylabel("Count" if i % n_cols == 0 else "", fontsize=12)
            
        # Customize ticks for perceived health features
        if feature in ["RTHLTH31", "MNHLTH31"]:
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(["1) Excellent", "2) Very Good", "3) Good", "4) Fair", "5) Poor"], fontsize=9, rotation=15)
            
        ax.grid(True, axis="y", alpha=0.3)
        sns.despine(ax=ax)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    # Customize histogram matrix
    fig.suptitle(f"{'Population' if weights else 'Sample'} Distributions of Numerical Features", fontsize=16, fontweight="bold", y=1)
    
    # Adjust layout
    fig.tight_layout(h_pad=2.0)

    # Save to file
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)

    plt.show()


# Plot sample distributions (unweighted) of numerical features
plot_numerical_distributions(df, raw_numerical_features, DISPLAY_LABELS)  

# Plot population distributions (weighted) of numerical features  
plot_numerical_distributions(df, raw_numerical_features, DISPLAY_LABELS, weights="PERWT23F", save_to_file="../figures/eda/numerical_distributions.png")


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights: </b>
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Median Imputation Strategy:</b> The right-skewed nature of <code>FAMSZE23</code> and the discrete 1–5 scales of <code>RTHLTH31</code>/<code>MNHLTH31</code> make the median the most robust imputation choice. It prevents exceptionally large families or rare "Poor Health" self-ratings from biasing the central tendency for typical respondents while maintaining scale integrity (imputing observed integers like 2 rather than decimals like 2.4).</li>
#         <li><b>Life-Cycle Coverage (Age):</b> The uniform age distribution confirms the dataset adequately represents all adult life stages. The spike at age 85 is a known artifact of top-coding for privacy but represents a significant, high-cost sub-population in the population-level data.</li>
#         <li><b>The "Skewed Health" Challenge:</b> While the population is generally healthy (skewed toward 'Excellent' to 'Good'), the relative sparsity of "Fair" or "Poor" health ratings (4–5) creates a challenge. Predicting costs for this critical high-risk segment will depend on a limited number of extreme training samples.</li>
#         <li><b>Data Integrity:</b> With missingness below 0.3% across these features, the risk of imputation bias is extremely low. This high data quality allows us to proceed with standard imputation techniques without significant loss of predictive power.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Categorical Features</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Visualize the distributions of all categorical features, both on sample-level and population-level. Use one plot for nominal and ordinal features and a separate plot for binary features.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Visualize the distributions of categorical (nominal and ordinal) features for sample and population. 
# </div>

# %%
# Helper Function: Plot the Distributions of Categorical Features
def plot_categorical_distributions(df, nominal_features, ordinal_features, display_labels=None, categorical_labels=None, weights=None, save_to_file=None):
    # Define subplot matrix grid
    n_plots = len(nominal_features + ordinal_features)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create subplot matrix
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    
    # Flatten axes for easier iteration
    axes_flat = axes.flat
    
    # Iterate over all categorical features
    for i, feature in enumerate(nominal_features + ordinal_features):
        # Get current axes
        ax = axes_flat[i]
    
        # Population (weighted)
        if weights: 
            # Calculate counts 
            counts = df.groupby(feature)[weights].sum()  
            # Calculate completion rate
            completion_rate = df.loc[df[feature].notna(), weights].sum() / df[weights].sum() * 100
        # Sample (unweighted)
        else:   
            counts = df[feature].value_counts()  
            completion_rate = df[feature].count() / len(df) * 100

        # Sort categories
        if feature in ordinal_features:   
            counts = counts.sort_index()  # Retains inherent order of categories for ordinal features
        else:
            counts = counts.sort_values(ascending=False)  # Sorts by frequency for nominal features
        
        # Calculate percentages
        percentages = counts / counts.sum() * 100
        
        # Map integer to string labels for current feature
        feature_label_map = categorical_labels.get(feature, {})  
        string_labels = [feature_label_map.get(label, label) for label in percentages.index]  # Fallback to int if no str mapped
         
        # Create bar plot of current feature
        sns.barplot(
            x=percentages.values,
            y=string_labels,
            ax=ax,
            color=POP_COLOR if weights else SAMPLE_COLOR,
            alpha=0.7
        )
    
        # Add value labels on bars
        if weights:
            value_labels = [f"{pct:.1f}%\n({count/1e6:.1f}M)" for pct, count in zip(percentages, counts)]
        else:
            value_labels = [f"{pct:.1f}%\n({count:,})" for pct, count in zip(percentages, counts)]
        for container in ax.containers:
            ax.bar_label(container, labels=value_labels, padding=3, fontsize=9 if len(value_labels) < 7 else 8, alpha=0.9)  # fontsize 9 for 1-6 categories; 8 for 7+
    
        # Customize current bar plot
        ax.set_title(display_labels.get(feature, feature), fontsize=14, fontweight="bold", pad=20)
        completion_rate_label = "100% Complete" if completion_rate >= 99.95 else f"{completion_rate:.1f}% Complete"
        ax.annotate(  # annotates completion rate under title
            completion_rate_label, xy=(0.5, 1), xytext=(0, 5), xycoords="axes fraction", textcoords="offset points", 
            ha="center", va="bottom", fontsize=9, color="#666666"
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])  # Removes x-axis tick marks and labels
        sns.despine(ax=ax, left=True, bottom=True)  # Removes all 4 borders

    # Customize bar plot matrix
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")  # hides unused subplots
    fig.suptitle(f"{'Population' if weights else 'Sample'} Distributions of Categorical Features", fontsize=16, fontweight="bold", y=1)
    
    # Adjust layout
    fig.tight_layout(h_pad=2.0, w_pad=4.0)
    
    # Save to file
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)
    
    # Show bar plot matrix
    plt.show()


# Plot sample distributions (unweighted) of categorical features 
plot_categorical_distributions(df, raw_nominal_features, raw_ordinal_features, DISPLAY_LABELS, CATEGORY_LABELS_EDA) 

# Plot population distributions (weighted) of categorical features
plot_categorical_distributions(df, raw_nominal_features, raw_ordinal_features, DISPLAY_LABELS, CATEGORY_LABELS_EDA, weights="PERWT23F", save_to_file="../figures/eda/categorical_distributions.png")


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Visualize the distributions of binary features for sample and population. 
# </div>

# %%
# Helper Function: Plot the Distributions of Binary Features
def plot_binary_distributions(df, binary_features, display_labels=None, categorical_labels=None, weights=None, save_to_file=None):
    # Define subplot matrix grid
    n_plots = len(binary_features)
    n_cols = 4
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create subplot matrix
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 1.8))
    
    # Flatten axes for easier iteration
    axes_flat = axes.flat
    
    # Iterate over all categorical features
    for i, feature in enumerate(binary_features):
        # Get current axes
        ax = axes_flat[i]
    
        # Population (weighted)
        if weights: 
            # Calculate counts 
            counts = df.groupby(feature)[weights].sum()  
            # Calculate completion rate
            completion_rate = df.loc[df[feature].notna(), weights].sum() / df[weights].sum() * 100
        # Sample (unweighted)
        else:   
            counts = df[feature].value_counts()  
            completion_rate = df[feature].count() / len(df) * 100

        # Sort categories
        counts = counts.sort_index(ascending=False)  # sorts "No" (2) before "Yes" (1)
        
        # Calculate percentages
        percentages = counts / counts.sum() * 100
        
        # Map integer to string labels for current feature
        feature_label_map = categorical_labels.get(feature, {})  
        string_labels = [feature_label_map.get(label, label) for label in percentages.index]  # Fallback to int if no str mapped
         
        # Create bar plot of current feature
        sns.barplot(
            x=percentages.values,
            y=string_labels,
            ax=ax,
            color=POP_COLOR if weights else SAMPLE_COLOR,
            alpha=0.7
        )
    
        # Add value labels on bars
        if weights:
            value_labels = [f"{pct:.1f}%\n({count/1e6:.1f}M)" for pct, count in zip(percentages, counts)]
        else:
            value_labels = [f"{pct:.1f}%\n({count:,})" for pct, count in zip(percentages, counts)]
        for container in ax.containers:
            ax.bar_label(container, labels=value_labels, padding=3, fontsize=8, alpha=0.9)
    
        # Customize current bar plot
        ax.set_title(display_labels.get(feature, feature), fontsize=12, fontweight="bold", pad=18)
        completion_rate_label = "100% Complete" if completion_rate >= 99.95 else f"{completion_rate:.1f}% Complete"
        ax.annotate(  # annotates completion rate under title
            completion_rate_label, xy=(0.5, 1), xytext=(0, 4), xycoords="axes fraction", textcoords="offset points", 
            ha="center", va="bottom", fontsize=8, color="#666666"
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])  # Removes x-axis tick marks and labels
        sns.despine(ax=ax, left=True, bottom=True)  # Removes all 4 borders

    # Hides unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")  

    # Customize bar plot matrix
    fig.suptitle(f"{'Population' if weights else 'Sample'} Distributions of Binary Features", fontsize=16, fontweight="bold", y=1)
    
    # Adjust layout
    fig.tight_layout(h_pad=3.0, w_pad=8.0)
    
    # Save to file
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)

    # Show bar plot matrix
    plt.show()


# Plot sample distributions (unweighted) of binary features 
plot_binary_distributions(df, raw_binary_features, DISPLAY_LABELS, CATEGORY_LABELS_EDA)  

# Plot population distributions (weighted) of binary features
plot_binary_distributions(df, raw_binary_features, DISPLAY_LABELS, CATEGORY_LABELS_EDA, weights="PERWT23F", save_to_file="../figures/eda/binary_distributions.png")


# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> Categorical distributions reveal key socio-economic drivers and the critical impact of survey weighting.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Oversampling of Low SES:</b> Comparing sample vs. population distributions confirms that the raw sample over-samples lower socio-economic segments. Applying survey weights "uplifts" employment (57% → 63%) and private insurance (60% → 66%) while "down-weighting" the proportions of those in poverty (14% → 10%) and without degrees (13% → 10%).</li>
#         <li><b>Oversampling of Healthy Individuals:</b> Comparing sample vs. population data reveals that the raw survey over-samples individuals with chronic conditions. Survey weights act as a "health correction," reflecting lower population-wide rates for Hypertension (32% vs. 38%), Cholesterol (31% vs. 36%), and Walking Limitations (12% vs. 15%).</li>
#         <li><b>Insurance as a Financial Buffer:</b> With 66% of the population holding private insurance and only 7.3% remaining uninsured, the target variable (<code>TOTSLF23</code>) will be largely driven by deductibles and plan-specific cost-sharing rather than the total cost of care.</li>
#         <li><b>Socio-Economic Concentration:</b> The population is heavily concentrated in "High/Middle Income" (75%) and "Married" (51%) categories. The relative sparsity of the "Poor" category (10%) suggests the model will have less "training experience" in predicting cost patterns for the most economically vulnerable.</li>
#         <li><b>Regional Dominance:</b> The South (39%) and West (25%) account for nearly two-thirds of the population. Regional variations in healthcare pricing, state-level mandates, and provider density in these areas will likely exert the strongest geographic influence on cost predictions.</li>
#         <li><b>High Prevalence of Joint Pain:</b> A substantial 45.0% of the population reports Joint Pain, with 24.5% diagnosed with Arthritis. These high-frequency features likely serve as primary drivers for recurring healthcare utilization and "maintenance" spending.</li>
#         <li><b>Care Continuity:</b> Over 70% of respondents maintain a "Usual Source of Care." This widespread engagement implies that cost drivers are often rooted in established medical relationships rather than isolated acute shocks.</li>
#         <li><b>Sparse High-Severity Risks:</b> Severe conditions such as Stroke (3.8%) and Coronary Heart Disease (4.7%) are relatively sparse. This scarcity reinforces the necessity of our distribution-informed stratified split to ensure these high-impact tail events are adequately represented for model evaluation.</li>
#     </ul>
# </div>
# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <strong>Implications for Feature Engineering</strong><br>
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><strong>Recent Life Transition:</strong> Sparse "In Round" transitions regarding marrital status (e.g., recent divorce) and employment status (e.g., recent job loss) signal major "life shocks" between survey interviews. These stressors often precede health volatility and high out-of-pocket costs. To leverage this signal robustly, I will create a <code>RECENT_LIFE_TRANSITION</code> flag and collapse these categories into their parent categories (e.g., "Divorced in Round" → "Divorced").</li>
#     </ul>
# </div>
#
# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Bivariate EDA</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Analyze the relationship between two columns using correlations or group-wise statistics and visualize the relationships using scatter plots, bar plots, or grouped box plots.
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Correlations</h3>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Spearman Rank Correlation</b> is used instead of Pearson or Pearson on log-transformed costs for three reasons:
#     <ul style="margin-bottom:0px">
#         <li><b>Zero-inflation:</b> ~22% of respondents have $0 costs. <code>log(0)</code> is undefined, making log-transformation require either dropping zeros (biasing the population) or using <code>log(x+1)</code> (an arbitrary shift that distorts low-cost relationships).</li>
#         <li><b>Skew-robustness:</b> Spearman measures monotonic rank association, making it immune to the heavy right-skew of <code>TOTSLF23</code> without requiring any transformation.</li>
#         <li><b>Unified measure:</b> Spearman works identically for numerical, ordinal, and binary (0/1) features, allowing all feature types to appear in one consistent heatmap.</li>
#     </ul>
#     <br>
#     <b>Interpretation Rule of Thumb:</b>
#     <table style="margin-top:8px; margin-bottom:0px; margin-left: 0; margin-right: auto;">
#         <tr>
#             <th style="background-color:#f5ecda; padding:4px 10px;">|ρ|</th>
#             <th style="background-color:#f5ecda; padding:4px 10px;">Strength</th>
#         </tr>
#         <tr><td style="padding:3px 10px;">&lt; 0.1</td><td style="padding:3px 10px;">Negligible</td></tr>
#         <tr><td style="padding:3px 10px;">0.1 – 0.3</td><td style="padding:3px 10px;">Weak</td></tr>
#         <tr><td style="padding:3px 10px;">0.3 – 0.5</td><td style="padding:3px 10px;">Moderate</td></tr>
#         <tr><td style="padding:3px 10px;">&gt; 0.5</td><td style="padding:3px 10px;">Strong</td></tr>
#     </table>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Correlation heatmap of the target variable and all numerical and binary features. 
# </div>

# %%
# Helper Function: Calculate Population-Weighted Correlation Matrix (using numpy; not available in pandas)
def calculate_weighted_correlations(df, columns, weights, method="pearson"):
    """Calculate a population-weighted correlation matrix (Pearson or Spearman).

    This function computes correlations that account for survey weights, ensuring
    estimates are representative of the target population.

    Methodology:
    - Weighted Pearson: Calculates the weighted covariance matrix using weighted means
      and centered data.
    - Weighted Spearman: Transforms variables into weighted ranks (sum of weights of
      smaller values + 0.5 * current weight) and averages ranks for tied values,
      then applies the weighted Pearson calculation to these ranks.

    Args:
        df:       DataFrame containing the columns to correlate and the weights.
        columns:  List of numerical, ordinal, or binary column names to include.
        weights:  Name of the column containing population weights.
        method:   Correlation method to use ('pearson' or 'spearman').

    Returns:
        pd.DataFrame: Population-weighted correlation matrix.
    """
    w = df[weights].values
    
    if method == "pearson":
        X = df[columns].values
    elif method == "spearman":
        # Calculate weighted ranks for each column
        X = np.zeros((len(df), len(columns)))
        for i, col in enumerate(columns):
            x = df[col].values
            # Sort x and get indices
            idx = np.argsort(x)
            x_sorted = x[idx]
            w_sorted = w[idx]
            
            # Cumulative weights
            cum_w = np.cumsum(w_sorted)
            # Weighted ranks: sum of weights before + 0.5 * current weight
            ranks_sorted = cum_w - 0.5 * w_sorted
            
            # Handle ties: average ranks for tied values
            unique_vals, first_indices, counts = np.unique(x_sorted, return_index=True, return_counts=True)
            if len(unique_vals) < len(x_sorted):
                for start_idx, count in zip(first_indices, counts):
                    if count > 1:
                        avg_rank = np.mean(ranks_sorted[start_idx : start_idx + count])
                        ranks_sorted[start_idx : start_idx + count] = avg_rank
            
            # Invert sorting to match original order
            inv_idx = np.empty_like(idx)
            inv_idx[idx] = np.arange(len(idx))
            X[:, i] = ranks_sorted[inv_idx]
    else:
        raise ValueError("Only 'pearson' and 'spearman' are supported for weighted correlation.")

    # Calculate weighted Pearson correlation on the data (or weighted ranks)
    # 1. Calculate weighted mean
    m = np.average(X, axis=0, weights=w)
    
    # 2. Center the data
    X_c = X - m
    
    # 3. Calculate weighted covariance matrix
    # Cov = (X_c.T * w) @ X_c / sum(w)
    cov = (X_c.T * w) @ X_c / np.sum(w)
    
    # 4. Calculate weighted correlation matrix
    # Corr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    return pd.DataFrame(corr, index=columns, columns=columns)


# Helper Function: Plot Correlation Heatmap 
def plot_correlation_heatmap(df, columns, method, weights=None, save_to_file=None):
    """Plot a correlation heatmap with annotated values.

    This function visualizes the relationship between columns using a masked
    heatmap in the lower-triangle. It supports both standard sample correlations 
    and population-weighted correlations for survey data.

    Args:
        df:            DataFrame containing the columns to correlate.
        columns:       Ordered list of column names to include. Features must be
                       numerical, ordinal, or binary (0/1). Nominal columns must
                       be excluded as their numeric codes lack rank meaning.
        method:        Correlation method: 'pearson', 'spearman', or 'kendall'.
        weights:       Optional name of the weight column for population-weighted
                       correlations.
        save_to_file:  Optional file path to save the figure (e.g., '.png').
    """
    # Create correlation matrix and round to 2 decimals
    if weights:
        correlation_matrix = calculate_weighted_correlations(df, columns, weights, method=method)
    else:
        correlation_matrix = df[columns].corr(method=method)
    
    # Round to 2 decimals
    correlation_matrix = round(correlation_matrix, 2)
    
    # Mask upper triangle (k=0 removes diagonal with self-correlations)
    mask = np.triu(np.ones(correlation_matrix.shape), k=0).astype(bool) 
    correlation_matrix[mask] = np.nan

    # Create display labels for readability
    display_labels = [DISPLAY_LABELS.get(col, col) for col in correlation_matrix.columns]
    display_labels = [label.replace("Coronary Heart Disease", "Heart Disease") for label in display_labels]

    # Set the figure size
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color NaN cells (upper triangle) light grey so the triangle shape reads cleanly
    ax.set_facecolor("#f0f0f0")

    # Create a custom color palette: Navy (Negative) -> White -> Orange (Positive)
    custom_cmap = sns.diverging_palette(250, 25, s=90, l=50, as_cmap=True)

    # Create heatmap
    sns.heatmap(
        correlation_matrix, 
        cmap=custom_cmap, 
        vmin=-1, vmax=1,
        center=0,
        annot=True,  # Annotate correlation values
        annot_kws={"size": 8},  # Format font size of values 
        fmt=".2f",  # Format values with 2 decimals
        linewidth=0.5,  # Thin white lines between cells
        square=True,  # Force square cell shape for a cleaner look
        cbar=False,  # Remove colorbar
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax
    )

    # Customize 
    method_titles = {
        "pearson": "Pearson",
        "spearman": "Spearman Rank",
        "kendall": "Kendall Rank"
    }
    title = f"{'Population' if weights else 'Sample'} {method_titles.get(method, method.title())} Correlations"
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)  
    plt.yticks(fontsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save to file
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)
        
    # Show heatmap
    plt.show()

    
# Plot the sample Spearman rank correlations for the target variable and all numerical, ordinal, and binary features
plot_correlation_heatmap(df, columns=["TOTSLF23"] + raw_numerical_features + raw_ordinal_features + raw_binary_features, method="spearman")

# Plot the population Spearman rank correlations for the target variable and all numerical, ordinal, and binary features
plot_correlation_heatmap(
    df, 
    columns=["TOTSLF23"] + raw_numerical_features + raw_ordinal_features + raw_binary_features, 
    method="spearman",
    weights="PERWT23F",
    save_to_file="../figures/eda/correlation_heatmap.png"
)

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insights:</b> 
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Primary Drivers:</b> Age (0.30) is the strongest demographic predictor, capturing the natural accumulation of healthcare needs. Poverty Category (0.26) shows a positive correlation, suggesting that higher income levels are associated with higher out-of-pocket spending, likely due to increased financial access and utilization of services.</li>
#         <li><b>The Chronic Engine:</b> Arthritis (0.23), High Cholesterol (0.22), and High Blood Pressure (0.19) are the most significant medical predictors. Their consistent correlation suggests they act as "steady drivers" of costs through frequent prescription fills and specialist office visits.</li>
#         <li><b>Family Size Paradox:</b> The negative correlation with out-of-pocket costs (-0.22) likely reflects a demographic shift—larger households often include more children, who generally incur significantly lower healthcare expenses than adults, pulling down the per-person average.</li>
#         <li><b>Feature Redundancy:</b> High correlations between ADL/IADL Help (0.60) and Arthritis/Joint Pain (0.56) indicate strong multicollinearity. While tree-based models handle this well, these features may provide redundant signals for regression-based models during training.</li>
#         <li><b>Objective vs. Subjective Health:</b> Clinical diagnoses (chronic conditions) are more strongly correlated with financial outcomes than subjective ratings (Physical/Mental Health). </li>
#     </ul>
# </div>
# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Numerical Features vs. Target</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Visualize the pairwise relationships between the target variable and each numerical feature. 
#     <br><br>
#     Note: Apply log-transformation to out-of-pocket costs to handle the extremely right-skewed and zero-inflated distribution. This "stretches" the low-cost range and "squeezes" the extreme tail, making the relationship with numerical features (like Age and Physical Health) more visible. 
# </div>

# %%
# Helper Function: Plot Numerical Feature-Target Relationships
def plot_numerical_feature_target_relationships(df, features, target, log_scale=False, weights=None, save_to_file=None):
    """Visualize the bivariate relationships between numerical features and the target using scatterplots.
    
    Supports optional log-transformation of the target to handle heavy-tailed distributions.
    Supports weights to apply larger size of data points for larger weights.
    """
    n_plots = len(features)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes_flat = axes.flatten()
    
    plot_df = df.reset_index(drop=True)
    y_col = target
    
    if log_scale:
        y_col = f"{target}_LOG"
        plot_df[y_col] = np.log1p(plot_df[target])
        y_label = "Out-of-Pocket Costs (Log-Scaled)"
    else:
        y_label = "Out-of-Pocket Costs"

    for i, feature in enumerate(features):
        ax = axes_flat[i]
        
        # Create scatterplot between current feature and target 
        sns.scatterplot(
            data=plot_df,
            x=feature,
            y=y_col,
            ax=ax,
            alpha=0.1,  # handle density with alpha
            color=POP_COLOR if weights else SAMPLE_COLOR,
            size=weights if weights else None,
            sizes=(2, 50) if weights else None,
            s=10 if not weights else None,
            legend=False
        )
        
        # Calculate Spearman rank correlation (for title)
        if weights:
            corr = calculate_weighted_correlations(df, [feature, target], weights, method="spearman").iloc[0, 1]
            corr_label = f"(ρ={corr:.2f})"
        else:
            corr = df[[feature, target]].corr(method="spearman").iloc[0, 1]
            corr_label = f"(ρ={corr:.2f})"

        # Customize
        ax.set_title(f"{DISPLAY_LABELS.get(feature, feature)} {corr_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(y_label if i % n_cols == 0 else "", fontsize=12)
        sns.despine(ax=ax)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    # Add super title 
    fig.suptitle(f"{'Population' if weights else 'Sample'} Numerical Feature-Target Relationships", fontsize=16, fontweight="bold", y=1.0)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save to file
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches="tight", dpi=200)
    
    plt.show()

# Visualize sample numerical feature-target relationships
plot_numerical_feature_target_relationships(
    df, 
    features=raw_numerical_features + raw_ordinal_features, 
    target="TOTSLF23", 
    log_scale=True
)

# Visualize population numerical feature-target relationships
plot_numerical_feature_target_relationships(
    df, 
    features= raw_numerical_features + raw_ordinal_features, 
    target="TOTSLF23", 
    log_scale=True,
    weights="PERWT23F", 
    save_to_file="../figures/eda/numerical_feature_target_relationships.png"
)

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Categorical Features vs. Target</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Visualize the pairwise relationships between the target variable and each categorical feature using grouped box plots. 
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Feature Engineering (Stateless)</h1>
# </div> 
# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Feature Refinement</h2>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Refine features to enhance the predictive signal and improve model generalization. This involves category collapsing to fix sparse categories and ensure the model learns from stable, well-populated demographics.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Recent Life Transition</strong><br>
#     📌 Capture critical life transitions related to marital and employment status (e.g., recent divorce, job loss) into a unified <code>RECENT_LIFE_TRANSITION</code> indicator. This preserves the predictive signal of major "life shocks" while ensuring model robustness by mapping sparse status categories back to stable, well-populated categories.
#     <br><br>
#     Note on Web App Implementation: In web form, add a question <em>"In the last 12 months, have you experienced a change in marital or employment status?"</em> with response options "Yes" and "No".
# </div>

# %%
# Identify all "in round" transitions in marital and employment status
# MARRY31X: 7-10 represent transitions (Married/Widowed/Divorced/Separated in Round)
# EMPST31: 2-3 represent transitions (Job to Return To, Job in Ref Period)
marital_transitions = [7, 8, 9, 10]
employment_transitions = [2, 3]

# Create unified RECENT_LIFE_TRANSITION flag
df["RECENT_LIFE_TRANSITION"] = (df["MARRY31X"].isin(marital_transitions) | df["EMPST31"].isin(employment_transitions)).astype(float)
df.loc[df["MARRY31X"].isna() & df["EMPST31"].isna(), "RECENT_LIFE_TRANSITION"] = np.nan

# Collapse categories into their stable counterparts 
# For Marital Status: Map 7->1 (Married), 8->2 (Widowed), 9->3 (Divorced), 10->4 (Separated)
marital_map = {7: 1, 8: 2, 9: 3, 10: 4}
df["MARRY31X_GRP"] = df["MARRY31X"].replace(marital_map)

# For Employment Status: Map 2, 3, 4 to 0 (Not Employed). 1 remains 1 (Employed).
# EMPST31: 1=Employed, 4=Not Employed. 2 and 3 are effectively "Not working now but attached". 
# Transitioner signal is preserved in RECENT_LIFE_TRANSITION.
employment_map = {2: 0, 3: 0, 4: 0} 
df["EMPST31_GRP"] = df["EMPST31"].replace(employment_map)

# %% [markdown]
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <code>CHRONIC_COUNT</code> (total number of diagnosed chronic conditions) and <code>LIMITATION_COUNT</code> (total number of functional limitations) are derived using the <code>MedicalFeatureDeriver</code> transformer inside the preprocessing pipeline after handling missing value. This architectural choice ensures that counts are calculated from clean, imputed data.
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Feature Validation</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Define the lists of features that enter the pipeline.
# </div>

# %%
# Pipeline Input Features
input_numerical_features = raw_numerical_features.copy() + raw_ordinal_features.copy()  # combine raw numerical and ordinal features for pipeline to apply median imputation and Z-score scaling 
input_binary_features = raw_binary_features + ["RECENT_LIFE_TRANSITION", "EMPST31_GRP"]  # employment is binary after collapsing categories
input_nominal_features = ["REGION23", "MARRY31X_GRP", "INSCOV23", "HIDEG"] 

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Verify feature engineering results.
# </div>

# %%
plot_categorical_distributions(df, input_nominal_features, ["POVCAT23"], DISPLAY_LABELS, CATEGORY_LABELS_EDA, weights="PERWT23F")
plot_binary_distributions(df, input_binary_features, DISPLAY_LABELS, CATEGORY_LABELS_EDA, weights="PERWT23F")

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Train-Validation-Test Split</h1>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Split the data into 80% for training, 10% for validation, and 10% for testing.
#     <br><br>
#     <b>Why not a simple random split?</b><br>
#     Healthcare costs exhibit a zero-inflated, heavily right-skewed distribution (see <a href="#target-variable"><b>Target Variable EDA</b></a>). The primary risk of a random split in healthcare cost data is the "luck-of-the-draw" misdistribution of super-spenders. Since the extreme tail of the distribution (the top 1%) accounts for a massive share of total population spending, omitting even a few of these individuals from the test set—or over-representing them in the train set—creates catastrophic "metric drift." This makes performance indicators (like R² or MSE) highly volatile and unreliable for predicting real-world financial risk. 
#     <br><br>
#     <b>Why not a quartile or quintile split?</b><br>
#     Standard quartile or quintile bins are too coarse to capture the extreme tail. Because healthcare costs are so concentrated, a quintile-based split (top 20%) would treat a respondent at the 81st percentile (e.g., \$2,000) the same as a "super-spender" at the 99.9th percentile (e.g., \$100,000). Only high-resolution, non-linear strata at the 95th and 99th percentiles can guarantee that these "Black Swan" cases are balanced across all subsets.
#     <br><br>
#     <b>Distribution-Informed Stratified Split</b><br>
#     To ensure the model is evaluated on a representative mirror of the population, I use a distribution-informed stratified split:
#     <ul>
#         <li><b>Mitigating 'Black Swan' Risks (Primary):</b> Uses high-resolution non-linear bins (80, 95, and 99th percentiles) to force the inclusion of extreme high-cost individuals in all sets, preventing unstable performance metric fluctuations.</li>
#         <li><b>Preserving the Zero-Hurdle (Secondary):</b> Guarantees that the 22.3% of zero-cost individuals remain identical across all sets, ensuring consistent evaluation of the model's ability to predict cost occurrence.</li>
#         <li><b>Capturing the Pareto Distribution:</b> Prevents evaluation bias by ensuring the 20% of spenders who drive ~80% of the total economic burden are proportionally represented in the test set.</li>
#     </ul>
#     <b>Strata Distribution</b>
#     <table style="margin-left:0; margin-top:20px; margin-bottom:20px">
#         <tr>
#             <th style="background-color:#f5ecda;">Bin</th>
#             <th style="background-color:#f5ecda;">Category</th>
#             <th style="background-color:#f5ecda;">Percentile (of Positives)</th>
#             <th style="background-color:#f5ecda;">Train (80%)</th>
#             <th style="background-color:#f5ecda;">Val (10%)</th>
#             <th style="background-color:#f5ecda;">Test (10%)</th>
#         </tr>
#         <tr>
#             <td style="background-color:#fff6e4; text-align:center;"><b>0</b></td>
#             <td style="background-color:#fff6e4;">Zero Costs</td>
#             <td style="background-color:#fff6e4;">N/A (Hurdle)</td>
#             <td style="background-color:#fff6e4; text-align:center;">2,640</td>
#             <td style="background-color:#fff6e4; text-align:center;">330</td>
#             <td style="background-color:#fff6e4; text-align:center;">330</td>
#         </tr>
#         <tr>
#             <td style="background-color:#f5ecda; text-align:center;"><b>1</b></td>
#             <td style="background-color:#f5ecda;">Low Spend</td>
#             <td style="background-color:#f5ecda;">0 - 50%</td>
#             <td style="background-color:#f5ecda; text-align:center;">4,587</td>
#             <td style="background-color:#f5ecda; text-align:center;">573</td>
#             <td style="background-color:#f5ecda; text-align:center;">574</td>
#         </tr>
#         <tr>
#             <td style="background-color:#fff6e4; text-align:center;"><b>2</b></td>
#             <td style="background-color:#fff6e4;">Moderate</td>
#             <td style="background-color:#fff6e4;">50 - 80%</td>
#             <td style="background-color:#fff6e4; text-align:center;">2,752</td>
#             <td style="background-color:#fff6e4; text-align:center;">344</td>
#             <td style="background-color:#fff6e4; text-align:center;">344</td>
#         </tr>
#         <tr>
#             <td style="background-color:#f5ecda; text-align:center;"><b>3</b></td>
#             <td style="background-color:#f5ecda;">High Spend</td>
#             <td style="background-color:#f5ecda;">80 - 95%</td>
#             <td style="background-color:#f5ecda; text-align:center;">1,376</td>
#             <td style="background-color:#f5ecda; text-align:center;">172</td>
#             <td style="background-color:#f5ecda; text-align:center;">172</td>
#         </tr>
#         <tr>
#             <td style="background-color:#fff6e4; text-align:center;"><b>4</b></td>
#             <td style="background-color:#fff6e4;">Very High</td>
#             <td style="background-color:#fff6e4;">95 - 99%</td>
#             <td style="background-color:#fff6e4; text-align:center;">367</td>
#             <td style="background-color:#fff6e4; text-align:center;">46</td>
#             <td style="background-color:#fff6e4; text-align:center;">46</td>
#         </tr>
#         <tr>
#             <td style="background-color:#f5ecda; text-align:center;"><b>5</b></td>
#             <td style="background-color:#f5ecda;">Massively High</td>
#             <td style="background-color:#f5ecda;">99 - 99.9%</td>
#             <td style="background-color:#f5ecda; text-align:center;">83</td>
#             <td style="background-color:#f5ecda; text-align:center;">10</td>
#             <td style="background-color:#f5ecda; text-align:center;">10</td>
#         </tr>
#         <tr>
#             <td style="background-color:#fff6e4; text-align:center;"><b>6</b></td>
#             <td style="background-color:#fff6e4;">Super Spenders</td>
#             <td style="background-color:#fff6e4;">Top 0.1%</td>
#             <td style="background-color:#fff6e4; text-align:center;">12</td>
#             <td style="background-color:#fff6e4; text-align:center;">1</td>
#             <td style="background-color:#fff6e4; text-align:center;">2</td>
#         </tr>
#     </table>
# </div>

# %%
# Split the data into X features and y target
X = df.drop("TOTSLF23", axis=1)
y = df["TOTSLF23"]

# Helper function for distribution-informed stratification
def create_stratification_bins(y):
    # Initialize strata series 
    strata = pd.Series(index=y.index, dtype=int)
    
    # Bin 0: Zero Costs (Handle the hurdle separately)
    is_zero = (y == 0)
    strata[is_zero] = 0
    
    # Custom non-linear quantiles for positive values to capture the tail
    positive_y = y[~is_zero]
    bins = [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0]
    
    # Assign positive spenders to bins 1 through 5 
    # note: labels=False returns the bin indices (0-4) instead of Interval objects (e.g., [0, 150.5]). We add 1 to shift the indices to 1-5 with 0 being reserved for the zero cost bin.
    # note: duplicates="drop" avoids errors if multiple quantiles (e.g., 0.95 and 0.99) result in the same cost value.
    strata[~is_zero] = pd.qcut(positive_y, q=bins, labels=False, duplicates="drop") + 1
    return strata

# Generate the stratification column 
y_strata = create_stratification_bins(y)

# Perform the first distribution-informed stratified split (80% Train, 20% Temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y_strata
)

# Re-calculate strata for the temporary set to ensure the 50/50 split is also representative
temp_strata = create_stratification_bins(y_temp)

# Perform the second stratified split (resulting in 10% Val, 10% Test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_strata
)

# Helper function to verify the stratification splits
def verify_split(y_subset, y_subset_strata, name):
    # Calculate relative frequencies of the bins
    strata_freq = y_subset_strata.value_counts(normalize=True).sort_index() * 100
    
    # Calculate key distribution metrics
    stats = {
        "Samples": len(y_subset),
        "Mean Cost": y_subset.mean(),
        "Median Cost": y_subset.median(),
        "Max Cost": y_subset.max()
    }
    
    # Merge metrics and strata proportions 
    for i, freq in strata_freq.items():
        stats[f"Bin {int(i)}"] = freq
        
    return pd.Series(stats, name=name)

# Create DataFrame of split verification statistics
split_verification_df = pd.concat([
    verify_split(y, y_strata, "Total Dataset"),
    verify_split(y_train, y_strata.loc[y_train.index], "Train (80%)"),
    verify_split(y_val, y_strata.loc[y_val.index], "Validation (10%)"),
    verify_split(y_test, y_strata.loc[y_test.index], "Test (10%)")
], axis=1).T

# Delete temporary variables to free up memory
del X_temp, y_temp, temp_strata, y_strata

# Display the verification DataFrame (format for readability) 
split_verification_df.style \
    .pipe(add_caption, "Stratified Split Verification") \
    .format("{:,.1f}") \
    .format("{:,.0f}", subset=["Samples", "Max Cost"]) \
    .format("${:,.0f}", subset=["Mean Cost", "Median Cost", "Max Cost"]) \
    .format("{:.2f}%", subset=["Bin 0", "Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"])

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The distribution-informed stratification successfully created representative and robust data splits.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Structural Precision:</b> Relative frequencies of all strata (Bins 0–6) are near-perfectly preserved across all splits, ensuring that the zero-inflation and the Pareto-style concentration are balanced.</li>
#         <li><b>Mitigation of "Metric Drift":</b> By forcing the inclusion of extreme high-cost individuals (99.9th percentile) in the Test set, we prevent "luck-of-the-draw" bias and ensure that performance metrics (e.g., R²) are robust against catastrophic outliers.</li>
#         <li><b>Representative Benchmarking:</b> The stability of central tendencies (Median Cost) confirms that the typical patient profile is identical in each set, allowing for reliable and generalizable model evaluation.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Data Preprocessing (Stateful)</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Stateful Preprocessing:</b> Perform data preprocessing and feature engineering steps that are <i>stateful</i>, meaning they learn parameters from the data (e.g., medians for imputation, cutoff values for missing values). 
#     <br><br>
#     <b>⚠️ Prevent Data Leakage:</b> Data leakage happens when information from the test set accidentally leaks into the training process, creating a model that looks incredibly accurate in your notebook but fails completely in the real world. To prevent data leakage, you must always <code>.fit()</code> your transformers on the training data only, then apply those learned parameters to <code>.transform()</code> the validation and test data. 
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Handling Missing Values</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Identification</strong> <br>
#     📌 Identify missing values.
# </div>

# %%
# Create a summary table for missing values
missing_value_df = pd.DataFrame({
    "Training": X_train.isnull().sum(),
    "Validation": X_val.isnull().sum(),
    "Test": X_test.isnull().sum(),
})
# Add target variable missings
missing_value_df.loc["TOTSLF23"] = [
    y_train.isnull().sum(),
    y_val.isnull().sum(),
    y_test.isnull().sum(),
]
# Display table (sorted and with percentages)
missing_value_df.sort_values("Training", ascending=False).style \
    .pipe(add_caption, "Missing Value Analysis") \
    .format({
        "Training": lambda x: f"{x} ({x / len(X_train) * 100:.1f}%)",
        "Validation": lambda x: f"{x} ({x / len(X_val) * 100:.1f}%)",
        "Test": lambda x: f"{x} ({x / len(X_test) * 100:.1f}%)"
    })

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Missing value analysis reveals a high level of data integrity and consistency across all splits.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>High Data Quality:</b> Missingness is quite low (Maximum ~3.8% for Usual Source of Care), with most features well below 1%, minimizing the risk of imputation bias.</li>
#         <li><b>Consistent Splits:</b> Missing value frequencies are near-identical across Training, Validation, and Test sets, suggesting the stratification did not introduce feature bias.</li>
#         <li><b>Key Variable Completeness:</b> Expected cost drivers such as Age, Sex, Region, Poverty Status, Insurance, and the Target Variable are 100% complete.</li>
#         <li><b>Implication for App Design:</b> The 100% completeness of demographics justifies making them required in the app, while high completeness allows us to safely treat unchecked boxes as an explicit "No" (0) rather than a missing value.</li>
#     </ul>
# </div>


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Imputation</strong> <br>
#     📌 Impute missing values. Use the median for numerical features and the mode (most frequent value) for categorical features.
# </div>

# %%
# Define required vs. optional features
required_features = [
    "AGE23X",    # Primary driver of medical utilization and costs
    "SEX",       # Key driver of utilization frequency and spending disparities documented in healthcare literature  
    "INSCOV23",  # Critical for out-of-pocket cost prediction
    "REGION23",  # Captures geographic variance in healthcare pricing
    "RTHLTH31"   # Self-reported physical health is a powerful proxy for healthcare demand
]

optional_features = [
    "MARRY31X_GRP", "FAMSZE23", "POVCAT23", "HIDEG", "EMPST31_GRP", "RECENT_LIFE_TRANSITION",
    "HAVEUS42", "MNHLTH31", "ADSMOK42",
    "ADLHLP31", "IADLHP31", "WLKLIM31", "COGLIM31", "JTPAIN31_M18",
    "HIBPDX", "CHOLDX", "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"
]

# Create missing value handling pipeline
missing_value_handling_pipeline = create_missing_value_handling_pipeline(
    required_features, 
    optional_features, 
    input_numerical_features, 
    input_nominal_features,
    input_binary_features,
    strict=False
)

# Handle missing values in training, validation, and test data
X_train_preprocessed = missing_value_handling_pipeline.fit_transform(X_train)
X_val_preprocessed = missing_value_handling_pipeline.transform(X_val)
X_test_preprocessed = missing_value_handling_pipeline.transform(X_test)

# Verify results: Missing value counts of raw vs. preprocessed data
pd.DataFrame({
    "Training": X_train[input_numerical_features + input_nominal_features + input_binary_features].isnull().sum(),
    "Training (Preprocessed)": X_train_preprocessed[input_numerical_features + input_nominal_features + input_binary_features].isnull().sum(),
    "Validation": X_val[input_numerical_features + input_nominal_features + input_binary_features].isnull().sum(),
    "Validation (Preprocessed)": X_val_preprocessed[input_numerical_features + input_nominal_features + input_binary_features].isnull().sum(),
    "Test": X_test[input_numerical_features + input_nominal_features + input_binary_features].isnull().sum(),
    "Test (Preprocessed)": X_test_preprocessed[input_numerical_features + input_nominal_features + input_binary_features].isnull().sum(),
}).style.pipe(add_caption, "Verification of Missing Value Imputation")

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Derive Medical Features</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Derive medical features from raw indicators based on medical domain knowledge. Calculate aggregate counts of chronic conditions (<code>CHRONIC_COUNT</code>) and functional limitations (<code>LIMITATION_COUNT</code>).
#     <br><br>
#     Note: Even though this is a stateless feature engineering step, it is placed AFTER imputation to ensure a deterministic derivation from complete data.
# </div>

# %%
# Initialize  
medical_feature_deriver = MedicalFeatureDeriver()

# Fit on training data
medical_feature_deriver.fit(X_train_preprocessed)

# Transform on training, validation, and test data
X_train_preprocessed = medical_feature_deriver.transform(X_train_preprocessed)
X_val_preprocessed = medical_feature_deriver.transform(X_val_preprocessed)
X_test_preprocessed = medical_feature_deriver.transform(X_test_preprocessed)

# Derived feature List
derived_numerical_features = MedicalFeatureDeriver.OUTPUT_FEATURES

# Inspect distributions of new derived features
X_train_inspect  = X_train_preprocessed.assign(
    PERWT23F=X_train.loc[X_train_preprocessed.index, "PERWT23F"]  # Temporarily attach weights for population analysis
) 
plot_numerical_distributions(X_train_inspect, derived_numerical_features, DISPLAY_LABELS, weights="PERWT23F")

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Handling Outliers</h2>
# </div>
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>General Outlier Handling Strategy</b><br>
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Outlier Detection:</b> Use univariate methods (e.g., 3SD, 1.5 IQR) for individual features and multivariate methods (e.g., Isolation Forest) to detect anomalies driven by complex feature interactions.</li>
#         <li><b>Outlier Profiling:</b> Compare feature and target distributions between inliers and outliers to identify the "driver" of the anomaly and to decide if a point is a data error or a valid extreme.</li>
#         <li><b>Outlier Treatment:</b> Decision depends on the nature of the outlier:
#             <ul style="margin-top:5px; margin-bottom:0px">
#                 <li><b>Remove:</b> For <em>Measurement Errors</em> (e.g., negative age).</li>
#                 <li><b>Keep (As-Is):</b> For <em>Representative Extremes</em> (e.g., family size of 10) when using robust models like XGBoost or Huber Regression.</li>
#                 <li><b>Winsorize (Cap):</b> For <em>Valid but Disruptive Extremes</em> (e.g., a \$1M cost) when using sensitive models like OLS to prevent them from shifting the global mean.</li>
#                 <li><b>Transform (Log):</b> For <em>Naturally Skewed Populations</em> (e.g., income or healthcare costs) to stabilize variance and normalize the distribution.</li>
#             </ul>
#         </li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">3SD Method</h3>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> 
#     📌 Identify univariate outliers in numerical columns by applying the 3 standard deviation rule (3SD). A data point is considered an outlier if it falls more than 3 standard deviations above or below the mean of the column.
# </div> 

# %%
# Initialize outlier remover 
outlier_remover_3sd = OutlierRemover3SD()

# Fit outlier remover to training data
outlier_remover_3sd.fit(X_train_preprocessed, input_numerical_features + derived_numerical_features)

# Show outliers in training data
summary_3sd = f"Training data contains {outlier_remover_3sd.outliers_} rows ({outlier_remover_3sd.outliers_ / len(outlier_remover_3sd.final_mask_) * 100:.1f}%) with outliers. Outliers by column below."
outlier_remover_3sd.stats_.style \
    .pipe(add_caption, f"<b>Outliers (3SD Method)</b> <br><span style='font-size:12px'>{summary_3sd}</span>", font_weight="normal") \
    .format({
        "mean": "{:.2f}",
        "std": "{:.2f}",
        "lower_cutoff": "{:.2f}",
        "upper_cutoff": "{:.2f}",
        "n_outliers": "{:.0f}",
        "pct_outliers": "{:.1f}%"
    })

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">1.5 IQR Method </h3>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> 
#     📌 Identify univariate outliers in numerical columns using the 1.5 interquartile range (IQR) rule. A data point is considered an outlier if it falls more than 1.5 interquartile ranges above the third quartile (Q3) or below the first quartile (Q1) of the column.
# </div> 

# %%
# Initialize outlier remover 
outlier_remover_iqr = OutlierRemoverIQR()

# Fit outlier remover to training data
outlier_remover_iqr.fit(X_train_preprocessed, input_numerical_features + derived_numerical_features)

# Show outliers by column for training data
summary_iqr = f"Training data contains {outlier_remover_iqr.outliers_} rows ({outlier_remover_iqr.outliers_ / len(outlier_remover_iqr.final_mask_) * 100:.1f}%) with outliers. Outliers by column below."
outlier_remover_iqr.stats_.style \
    .pipe(add_caption, f"<b>Outliers (1.5 IQR)</b> <br><span style='font-size:12px'>{summary_iqr}</span>", font_weight="normal") \
    .format({
        "Q1": "{:.1f}",
        "Q3": "{:.1f}",
        "IQR": "{:.1f}",
        "lower_cutoff": "{:.1f}",
        "upper_cutoff": "{:.1f}",
        "n_outliers": "{:.0f}",
        "pct_outliers": "{:.1f}%"
    })

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Comparison of Outlier Handling Methods:</b> Use the 3SD method for this dataset. 
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>1.5 IQR Method:</b> Is overly aggressive for discrete features with concentrated distributions.
#             <ul style="margin-top:5px">
#                 <li>For <code>RTHLTH31</code>, it yields a cutoff of 4.5, flagging all respondents that self-report "5 - Poor" health as outliers (3.6% of data). </li>
#                 <li>For <code>LIMITATION_COUNT</code>, it yields a cutoff of 2.5, flagging all respondents with 3 or more limitations as outliers (5.7%).</li>
#             </ul>
#             These are not data errors, but legitimate responses representing high-risk people. Quartiles are often not meaningful for discrete features with narrow value ranges.
#         </li>
#         <li><b>3SD Method:</b> Is more appropriate because it uses the full distribution (mean ± 3 standard deviations). Its cutoffs are naturally wider for these features (e.g., flagging only 2.3% for <code>LIMITATION_COUNT</code>), better preserving valid extreme cases that drive healthcare costs.</li>
#     </ul>
# </div>

# %%
# Outlier Profiling: Compare out-of-pocket costs for outliers vs. inliers  
# Create outlier profiling DataFrame from training data
outlier_df = X_train_preprocessed.assign(
    TOTSLF23=y_train, 
    outlier=outlier_remover_3sd.final_mask_,
    PERWT23F=X_train.loc[X_train_preprocessed.index, "PERWT23F"] # Pass sample weights for population-level stats
)

# Outlier Profiling: Cost Concentration
# This analysis shows what percentage of each group (Inliers vs. Outliers) falls into the top population-wide spending brackets.
outlier_in_df = outlier_df[outlier_df["outlier"] == True]
outlier_out_df = outlier_df[outlier_df["outlier"] == False]

percentiles = [0.999, 0.99, 0.95, 0.9, 0.8, 0.5]
benchmarks = []

# Pre-calculate group populations (weighted) for efficiency
outlier_in_pop = outlier_in_df["PERWT23F"].sum()
outlier_out_pop = outlier_out_df["PERWT23F"].sum()

for p in percentiles:
    # Calculate global (population-wide) threshold
    threshold = weighted_quantile(outlier_df["TOTSLF23"], outlier_df["PERWT23F"], p)
    
    # Calculate representation within each subgroup
    outlier_in_rep = (outlier_in_df.loc[outlier_in_df["TOTSLF23"] >= threshold, "PERWT23F"].sum() / outlier_in_pop) * 100
    outlier_out_rep = (outlier_out_df.loc[outlier_out_df["TOTSLF23"] >= threshold, "PERWT23F"].sum() / outlier_out_pop) * 100
    
    benchmarks.append({
        "Benchmark": f"Top {(1-p)*100:.0f}% (>= ${threshold:,.0f})" if p != 0.999 else f"Top 0.1% (>= ${threshold:,.0f})",
        "Inliers": outlier_in_rep,
        "Outliers": outlier_out_rep,
        "Outlier/Inlier Ratio": outlier_out_rep / outlier_in_rep if outlier_in_rep > 0 else np.nan
    })

# Create comparison table and display
benchmark_df = pd.DataFrame(benchmarks).set_index("Benchmark")
benchmark_df.style \
    .pipe(add_caption, "Univariate Outlier Profiling: Cost Concentration") \
    .format("{:.1f}%", subset=["Inliers", "Outliers"]) \
    .format("{:.1f}x", subset="Outlier/Inlier Ratio")

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> Do not remove univariate outliers. They represent a critical high-risk demographic.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Outlier Profile:</b> The 412 outliers are primarily driven by extreme values in limitations count (4+), family size (8+), and chronic conditions count (7+).</li>
#         <li><b>Cost Concentration:</b> Outliers are significantly over-represented in the high-cost tail. They are 4.8x more likely to fall into the Top 1% of spenders and 2.8x more likely to be in the Top 0.1% compared to inliers.</li>
#         <li><b>Decision:</b> Outliers reflect valid demographic and insurance dynamics, not noise. Removing them would "sanitize" the data of critical high-cost cases, leading to a biased model that fails to predict financial catastrophes.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Isolation Forest</h3>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ An unsupervised learning algorithm that detects anomalies by "isolating" them. It builds an ensemble of random decision trees where outliers require fewer splits to isolate, resulting in shorter path lengths than normal data points.
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 Identify multivariate outliers using an isolation forest.
# </div> 

# %%
# Define features to be used for multivariate outlier detection
# Note: Include numerical and binary features from both pipeline input and derived, exclude ordinal features (POVCAT23)
mutlivariate_outlier_features = [feat for feat in input_numerical_features if feat != "POVCAT23"] + input_binary_features + derived_numerical_features

# Initialize isolation forest
isolation_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)

# Fit isolation forest on training data 
isolation_forest.fit(X_train_preprocessed[mutlivariate_outlier_features])

# Predict outliers on training data
X_train_preprocessed["outlier"] = isolation_forest.predict(X_train_preprocessed[mutlivariate_outlier_features])
X_train_preprocessed["outlier"] = X_train_preprocessed["outlier"].map({1: 0, -1: 1})  # recode to 0/1 (outlier no/yes)
X_train_preprocessed["outlier_score"] = isolation_forest.decision_function(X_train_preprocessed[mutlivariate_outlier_features])

# Show number of outliers
n_outliers_train = X_train_preprocessed["outlier"].value_counts()[1]
contamination_train = n_outliers_train / X_train_preprocessed["outlier"].value_counts().sum()
print(f"Training Data: Identified {n_outliers_train} rows ({100 * contamination_train:.1f}%) as multivariate outliers.")

# %% [markdown]
# <div style="background-color:#4e8ac8; color:white; padding:10px; border-radius:6px;">
#     <h3 style="margin:0px">Outlier Profiling</h3>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ Compare the feature and target distributions between outliers and inliers identified with the isolation forest. To ensure findings are representative of the U.S. adult population, all statistics (medians, IQRs) and visualizations in this section are calculated using population-level weighted estimates.
# </div>

# %%
# Create outlier profiling DataFrame from training data
outlier_df = X_train_preprocessed.assign(
    TOTSLF23=y_train, 
    TOTSLF23_LOG=lambda df: np.log1p(df["TOTSLF23"]),  # Log-scale out-of-pocket costs for plotting
    outlier_display=lambda df: df["outlier"].map({0: "Inliers", 1: "Outliers"}),
    PERWT23F=X_train.loc[X_train_preprocessed.index, "PERWT23F"] # Pass sample weights for population-level stats
)

# Create outliers and inliers DataFrames 
outlier_in_df = outlier_df[outlier_df["outlier"] == 0]
outlier_out_df = outlier_df[outlier_df["outlier"] == 1]


# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 <strong>Outlier Profile for Target Variable</strong>
# </div> 

# %%
# Outlier Profiling: Overlapping Lorenz Curves 
def get_lorenz_metrics(subset_df):
    """Calculates Lorenz curve coordinates and Gini coefficient."""
    # Create local copy and calculate population costs
    l_df = subset_df[["TOTSLF23", "PERWT23F"]].sort_values("TOTSLF23").copy()
    l_df["pop_costs"] = l_df["TOTSLF23"] * l_df["PERWT23F"]
    
    # Cumulative percentages (weighted)
    cum_pop_pct = l_df["PERWT23F"].cumsum() / l_df["PERWT23F"].sum() * 100
    cum_pop_costs = l_df["pop_costs"].cumsum() / l_df["pop_costs"].sum() * 100
    
    # Prepend origin
    pct = np.insert(cum_pop_pct.values, 0, 0)
    costs = np.insert(cum_pop_costs.values, 0, 0)
    
    # Gini Coefficient
    gini = 1 - 2 * np.trapezoid(costs / 100, pct / 100)
    
    # Identify 80/20 point (Pareto)
    idx_80 = (cum_pop_pct - 80).abs().idxmin()
    x_80 = cum_pop_pct.loc[idx_80]
    y_80 = cum_pop_costs.loc[idx_80]
    
    return pct, costs, gini, (x_80, y_80)

# Calculate metrics for each group
outlier_in_lorenz = get_lorenz_metrics(outlier_in_df)
outlier_out_lorenz = get_lorenz_metrics(outlier_out_df)

# Plotting
plt.figure(figsize=(10, 8))

# Line of Equality
plt.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Line of Equality", alpha=0.6)

# Colors and Groups
# Inliers: #4F81BD (blue), Outliers: #D32F2F (red)
groups = [
    {"data": outlier_in_lorenz, "name": "Inliers", "color": "#4F81BD", "alpha_fill": 0.08},
    {"data": outlier_out_lorenz, "name": "Outliers", "color": "#D32F2F", "alpha_fill": 0.05}
]

for g in groups:
    pct, costs, gini, pareto = g["data"]
    
    # Lorenz Curve
    plt.plot(pct, costs, label=f"{g['name']} (Gini: {gini:.2f})", color=g["color"], lw=3)
    plt.fill_between(pct, costs, pct, color=g["color"], alpha=g["alpha_fill"])
    
    # Pareto Point
    plt.plot(pareto[0], pareto[1], 'o', color=g["color"], markersize=8)
    
    # Annotation for Pareto
    top_20_share = 100 - pareto[1]
    offset = 12 if g["name"] == "Inliers" else 0
    plt.annotate(f"Top 20% of {g['name']} account\n for {top_20_share:.1f}% of costs", 
                 xy=pareto, xytext=(pareto[0] - 35, pareto[1] + offset),
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.5),
                 fontsize=9, fontweight="bold", color=g["color"])

    # Highlight Zero-Cost Threshold for this group
    group_df = outlier_out_df if g["name"] == "Outliers" else outlier_in_df
    group_zero_pct = (group_df.loc[group_df["TOTSLF23"] == 0, "PERWT23F"].sum() / group_df["PERWT23F"].sum()) * 100
    
    plt.plot(group_zero_pct, 0, 'o', color=g["color"], markersize=8)
    
    # Annotation for Zero Costs
    y_offset = 13 if g["name"] == "Outliers" else 26
    plt.annotate(f"{group_zero_pct:.1f}% of {g['name']}\nhave $0 costs", 
                 xy=(group_zero_pct, 0), xytext=(group_zero_pct - 10, y_offset),
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.5),
                 fontsize=9, fontweight="bold", color=g["color"])

# Customize
plt.title("Outlier Profiling: Lorenz Curves for Out-of-Pocket Costs", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Cumulative % of Population (Sorted from Lowest to Highest Cost)", fontsize=11)
plt.ylabel("Cumulative % of Total Costs", fontsize=11)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, alpha=0.2)
plt.xticks(range(0, 101, 10))
plt.yticks(range(0, 101, 10))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 1, 1])

# Add footnote
plt.figtext(0.01, 0.01, "Note: Population-weighted estimates.", ha="left", fontsize=9, style="italic", color="#555555")

plt.savefig("../figures/outliers/outlier_lorenz_curve.png", bbox_inches="tight", dpi=200)
plt.show()

# %%
# Outlier Profiling: Cost Concentration
# This analysis shows what percentage of each group (Inliers vs. Outliers) falls into the top population-wide spending brackets.
percentiles = [0.999, 0.99, 0.95, 0.9, 0.8, 0.5]
benchmarks = []

# Pre-calculate group populations (weighted) for efficiency
outlier_in_pop = outlier_in_df["PERWT23F"].sum()
outlier_out_pop = outlier_out_df["PERWT23F"].sum()

for p in percentiles:
    # Calculate global (population-wide) threshold
    threshold = weighted_quantile(outlier_df["TOTSLF23"], outlier_df["PERWT23F"], p)
    
    # Calculate representation within each subgroup
    outlier_in_rep = (outlier_in_df.loc[outlier_in_df["TOTSLF23"] >= threshold, "PERWT23F"].sum() / outlier_in_pop) * 100
    outlier_out_rep = (outlier_out_df.loc[outlier_out_df["TOTSLF23"] >= threshold, "PERWT23F"].sum() / outlier_out_pop) * 100
    
    benchmarks.append({
        "Benchmark": f"Top {(1-p)*100:.0f}% (>= ${threshold:,.0f})" if p != 0.999 else f"Top 0.1% (>= ${threshold:,.0f})",
        "Inliers": outlier_in_rep,
        "Outliers": outlier_out_rep,
        "Outlier/Inlier Ratio": outlier_out_rep / outlier_in_rep
    })

# Create comparison table directly
benchmark_df = pd.DataFrame(benchmarks).set_index("Benchmark")

# Display table
benchmark_df.style \
    .pipe(add_caption, "Outlier Profiling: Cost Concentration") \
    .format("{:.1f}%", subset=["Inliers", "Outliers"]) \
    .format("{:.1f}x", subset="Outlier/Inlier Ratio")
# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 <strong>Outlier Profile for Numerical Features</strong>
# </div> 

# %%
# Outlier Numeric Profile: Median Differences (Population)
outlier_num_cols = [feat for feat in input_numerical_features if feat != "POVCAT23"] + derived_numerical_features + ["TOTSLF23"]

# Calculate population medians (weighted) for each feature/target
outlier_stats_num = pd.DataFrame(index=outlier_num_cols, columns=["Inliers", "Outliers"])
for col in outlier_num_cols:
    outlier_stats_num.loc[col, "Inliers"] = weighted_quantile(outlier_in_df[col], outlier_in_df["PERWT23F"], 0.5)
    outlier_stats_num.loc[col, "Outliers"] = weighted_quantile(outlier_out_df[col], outlier_out_df["PERWT23F"], 0.5)

# Calculate median difference
outlier_stats_num["Difference"] = (outlier_stats_num["Outliers"] - outlier_stats_num["Inliers"]).astype(float)

# Calculate population interquartile range (IQR) of inliers for standardization
inlier_iqrs = pd.Series(index=outlier_num_cols, dtype=float)
for col in outlier_num_cols:
    q1 = weighted_quantile(outlier_in_df[col], outlier_in_df["PERWT23F"], 0.25)
    q3 = weighted_quantile(outlier_in_df[col], outlier_in_df["PERWT23F"], 0.75)
    inlier_iqrs[col] = q3 - q1

# Calculate how many IQRs the outlier median is different from the inlier median
outlier_stats_num["IQR Difference"] = outlier_stats_num["Difference"] / inlier_iqrs

# Get all numerical columns sorted by absolute impact (IQR Difference) for profiling
outlier_num_cols_sorted = outlier_stats_num["IQR Difference"].abs().sort_values(ascending=False).index.tolist()

# Identify the top 4 numerical drivers (for pairwise plot)
outlier_num_drivers = outlier_num_cols_sorted[:4]

# Create lists for visualization (mapping TOTSLF23 to its log-scaled version)
outlier_num_cols_viz = [col if col != "TOTSLF23" else "TOTSLF23_LOG" for col in outlier_num_cols_sorted]
outlier_num_drivers_viz = [col if col != "TOTSLF23" else "TOTSLF23_LOG" for col in outlier_num_drivers] 

# Display table (Renaming the index only for the view to keep the DF's raw IDs intact)
outlier_stats_num.rename(index=DISPLAY_LABELS).sort_values(by="IQR Difference", ascending=False).style \
    .pipe(add_caption, "Outlier Numeric Profile: Median Differences") \
    .format("{:.1f}") \
    .set_properties(**{"font-weight": "bold"}, subset=["Difference", "IQR Difference"])

# %%
# Outlier Numeric Profile: Overlapping Histograms of Numerical Features and Target
n_features = len(outlier_num_cols_viz)
n_cols = 2
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes_flat = axes.flatten()

colors = {"Inliers": "#4F81BD", "Outliers": "#D32F2F"}

# Iterate over each numerical column
for i, numeric_column in enumerate(outlier_num_cols_viz):
    ax = axes_flat[i]

    # Create histogram for current numerical column
    sns.histplot(
        data=outlier_df, 
        x=numeric_column, 
        hue="outlier_display", 
        hue_order=["Inliers", "Outliers"],
        weights="PERWT23F", # Use sample weights for population-level density
        ax=ax,
        palette=colors,
        stat="density",  # Changes y-axis to density
        common_norm=False,  # Normalizes each group
        kde=True, 
        bins=20, 
        element="step",  # Shows outlines of bars only (shape like steps)
        discrete=True if numeric_column in ["RTHLTH31", "MNHLTH31", "FAMSZE23", "CHRONIC_COUNT", "LIMITATION_COUNT"] else False
    )

    # Calculate population-level (weighted) medians and differences 
    median_inliers = weighted_quantile(outlier_in_df[numeric_column], outlier_in_df["PERWT23F"], 0.5)
    median_outliers = weighted_quantile(outlier_out_df[numeric_column], outlier_out_df["PERWT23F"], 0.5)
    median_diff = median_outliers - median_inliers
    
    # Calculate standardized difference (using IQR of inliers)
    q1_in = weighted_quantile(outlier_in_df[numeric_column], outlier_in_df["PERWT23F"], 0.25)
    q3_in = weighted_quantile(outlier_in_df[numeric_column], outlier_in_df["PERWT23F"], 0.75)
    iqr_in = q3_in - q1_in
    iqr_text = f" ({median_diff/iqr_in:+.1f} IQR)" if iqr_in > 0 else ""

    # Add vertical median lines
    ax.axvline(median_inliers, color=colors["Inliers"], linestyle="--", lw=1.5, alpha=0.8)
    ax.axvline(median_outliers, color=colors["Outliers"], linestyle="--", lw=1.5, alpha=0.8)

    # Add median labels 
    ylim = ax.get_ylim()[1]
    ax.text(median_inliers, ylim * 0.7, f"M={median_inliers:.1f}", color=colors["Inliers"], 
            fontweight="bold", ha="right", va="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    ax.text(median_outliers, ylim * 0.7, f"M={median_outliers:.1f}", color=colors["Outliers"], 
            fontweight="bold", ha="left", va="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    # Add title
    ax.set_title(DISPLAY_LABELS.get(numeric_column, numeric_column), fontsize=12, fontweight="bold", pad=20)
    
    # Add subtitle (median difference)
    ax.text(0.5, 1.03, fr"$\Delta$ Median: {median_diff:+.1f}{iqr_text}", 
            transform=ax.transAxes, fontsize=10, ha="center", fontweight="normal")

    # Customize axis labels and legend
    ax.set_xlabel("")
    ax.set_ylabel("")     
    if ax.get_legend():
        ax.get_legend().set_title(None)  # Removes legend title

# Hide unused subplots
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis("off")

# Customize histogram matrix
fig.suptitle("Outlier Profiling: Numerical Features and Target", fontsize=14, fontweight="bold", y=0.98)

# Adjust layout 
fig.tight_layout(rect=[0, 0.02, 1, 0.99], h_pad=1.5, w_pad=2.0)

# Add footnote
fig.text(0.01, 0.01, "Note: Population-weighted estimates.", ha="left", fontsize=9, style="italic", color="#555555")

plt.savefig("../figures/outliers/outlier_numeric_profile.png", bbox_inches="tight", dpi=200)
plt.show()

# %%
# Outlier Numeric Profile: Overlapping KDE Plots of Numerical Features and Target
n_features = len(outlier_num_cols_viz)
n_cols = 2
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes_flat = axes.flatten()

colors = {"Inliers": "#4F81BD", "Outliers": "#D32F2F"}

for i, numeric_column in enumerate(outlier_num_cols_viz):
    ax = axes_flat[i]

    # Create KDE plot for current numerical column
    sns.kdeplot(
        data=outlier_df, 
        x=numeric_column, 
        hue="outlier_display", 
        hue_order=["Inliers", "Outliers"],
        weights="PERWT23F", # Use sample weights for population-level density
        fill=True, 
        common_norm=False, 
        ax=ax,
        palette=colors,
        cut=0  # Truncates curve on min and max
    )

    # Calculate population-level (weighted) medians and differences 
    median_inliers = weighted_quantile(outlier_in_df[numeric_column], outlier_in_df["PERWT23F"], 0.5)
    median_outliers = weighted_quantile(outlier_out_df[numeric_column], outlier_out_df["PERWT23F"], 0.5)
    median_diff = median_outliers - median_inliers
    
    # Calculate standardized difference using weighted IQR of inliers
    q1_in = weighted_quantile(outlier_in_df[numeric_column], outlier_in_df["PERWT23F"], 0.25)
    q3_in = weighted_quantile(outlier_in_df[numeric_column], outlier_in_df["PERWT23F"], 0.75)
    iqr_in = q3_in - q1_in
    iqr_text = f" ({median_diff/iqr_in:+.1f} IQR)" if iqr_in > 0 else ""

    # Add vertical median lines
    ax.axvline(median_inliers, color=colors["Inliers"], linestyle="--", lw=1.5, alpha=0.8)
    ax.axvline(median_outliers, color=colors["Outliers"], linestyle="--", lw=1.5, alpha=0.8)

    # Add median labels 
    ylim = ax.get_ylim()[1]
    ax.text(median_inliers, ylim * 0.7, f"M={median_inliers:.1f}", color=colors["Inliers"], 
            fontweight="bold", ha="right", va="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    ax.text(median_outliers, ylim * 0.7, f"M={median_outliers:.1f}", color=colors["Outliers"], 
            fontweight="bold", ha="left", va="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    # Add title
    ax.set_title(DISPLAY_LABELS.get(numeric_column, numeric_column), fontsize=12, fontweight="bold", pad=20)
    
    # Add subtitle (median difference)
    ax.text(0.5, 1.03, fr"$\Delta$ Median: {median_diff:+.1f}{iqr_text}", 
            transform=ax.transAxes, fontsize=10, ha="center", fontweight="normal")

    # Customize axis labels and legend
    ax.set_xlabel("")
    ax.set_ylabel("")     
    if ax.get_legend():
        ax.get_legend().set_title(None)  # Removes legend title

# Hide unused subplots
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis("off")

# Customize KDE plot matrix
fig.suptitle("Outlier Profiling: Numerical Features and Target", fontsize=14, fontweight="bold", y=0.99)

# Adjust layout
fig.tight_layout(rect=[0, 0.02, 1, 1], h_pad=1.5, w_pad=2.0)

# Add footnote
fig.text(0.01, 0.01, "Note: Population-weighted estimates.", ha="left", fontsize=9, style="italic", color="#555555")

plt.show()

# %%
# Outlier Numeric Profile: Pairwise Plot of Top Numerical Drivers 
# Create a population-representative subsample using weighted bootstrap resampling. 
# Since sns.pairplot does not natively support weights, this approach 'bakes' the survey weights directly into the subsample's density. 
# Setting replace=True ensures that high-weight respondents are proportionally represented as multiple 'virtual' individuals in the plot.
outlier_subsample = outlier_df[outlier_num_drivers_viz + ["outlier_display", "PERWT23F"]].sample(
    n=5000, 
    weights="PERWT23F", 
    replace=True, 
    random_state=RANDOM_STATE
).drop(columns="PERWT23F")

# Create pair plot matrix
grid = sns.pairplot(
    outlier_subsample.rename(columns=DISPLAY_LABELS), 
    hue="outlier_display", 
    palette={"Inliers": "#4F81BD", "Outliers": "#D32F2F"}, 
    plot_kws={"alpha":0.4, "s":30}, 
    diag_kws={"common_norm": False}
)

# Remove legend title and position it at the top center
sns.move_legend(
    grid, "lower center",
    bbox_to_anchor=(0.5, 0.97), 
    ncol=2, 
    title=None 
)

# Add title
grid.fig.suptitle("Outlier Profiling: Top Numerical Drivers", fontsize=14, fontweight="bold", y=1.03)

# Adjust layout
grid.fig.tight_layout(rect=[0, 0.03, 1, 1])

# Add footnote
grid.fig.text(0.01, 0.01, "Note: Population-weighted estimates.", ha="left", fontsize=9, style="italic", color="#555555")

plt.show() 

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 <strong>Outlier Profile for Binary Features</strong>
# </div> 

# %%
# Outlier Profile: Binary Features (Population)
outlier_stats_bin = pd.DataFrame(index=input_binary_features, columns=["Inliers", "Outliers"])

for group, mask in [("Inliers", outlier_df["outlier"] == 0), ("Outliers", outlier_df["outlier"] == 1)]:
    subset = outlier_df[mask]
    for col in input_binary_features:
        # Weighted avg of 0/1 columns = population prevalence
        outlier_stats_bin.loc[col, group] = np.average(subset[col], weights=subset["PERWT23F"])

outlier_stats_bin = outlier_stats_bin.astype(float)

# Map to display labels
# Since prevalence of a 0/1 variable reflects the '1' category, names should reflect that category.
binary_plot_remap = {"Sex": "Male", "Employment": "Employed"}
outlier_stats_bin.index = outlier_stats_bin.index.map(lambda x: DISPLAY_LABELS.get(x, x))
outlier_stats_bin.index = outlier_stats_bin.index.map(lambda x: binary_plot_remap.get(x, x))

outlier_stats_bin["Difference"] = outlier_stats_bin["Outliers"] - outlier_stats_bin["Inliers"]

# Sort by difference for better visualization
outlier_stats_bin = outlier_stats_bin.sort_values(by="Difference", ascending=False)

# Prepare data for plotting
plot_df = outlier_stats_bin.reset_index().melt(
    id_vars="index", value_vars=["Inliers", "Outliers"], var_name="Group", value_name="Prevalence"
)

# Visualize: Horizontal Grouped Bar Plot
plt.figure(figsize=(10, 12))
sns.barplot(
    data=plot_df, 
    y="index", 
    x="Prevalence", 
    hue="Group", 
    hue_order=["Outliers", "Inliers"],
    palette={"Outliers": "#D32F2F", "Inliers": "#4F81BD"},
    alpha=0.8
)

# Annotate bars with percentages
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt="{:.0%}", padding=3, fontsize=9)

# Customize
plt.title("Outlier Profiling: Binary Features", fontsize=14, fontweight="bold", pad=30)
plt.xlabel("Population Prevalence", fontsize=12)
plt.ylabel("")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
plt.grid(True, axis="x", alpha=0.3)
plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
sns.despine(left=True)

# Adjust layout 
plt.tight_layout(rect=[0, 0.02, 1, 0.99])

# Add footnote
plt.figtext(0.01, 0.01, "Note: Population-weighted estimates.", ha="left", fontsize=9, style="italic", color="#555555")

plt.savefig("../figures/outliers/outlier_binary_profile.png", bbox_inches="tight", dpi=200)
plt.show()

# %% [markdown]
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     📌 <strong>Outlier Profile for Categorical Features</strong>
# </div> 

# %%
# Outlier Profile: Categorical Features (Population)
n_features = len(input_nominal_features + ["POVCAT23"])
n_cols = 2
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes_flat = axes.flatten()

for i, feature in enumerate(input_nominal_features + ["POVCAT23"]):
    ax = axes_flat[i]
    
    # Calculate weighted distribution for each group
    # Pivot table handles sum of weights across categories
    ct = outlier_df.pivot_table(
        index=feature, 
        columns="outlier_display", 
        values="PERWT23F", 
        aggfunc="sum"
    )
    # Normalize within each group to get percentages
    ct = (ct / ct.sum()) * 100
    
    # Prepare for plotting
    plot_data = ct.reset_index().melt(id_vars=feature, var_name="Group", value_name="Percentage")
    
    # Ordering logic: Ordinals maintain index order; Nominals sort by Inlier prevalence
    if feature == "POVCAT23":
        plot_data = plot_data.sort_values(by=feature)
    else:
        sort_order = ct["Inliers"].sort_values(ascending=False).index
        plot_data[feature] = pd.Categorical(plot_data[feature], categories=sort_order, ordered=True)
        plot_data = plot_data.sort_values(by=feature)

    # Map integer codes to human-readable labels
    label_map = CATEGORY_LABELS_EDA.get(feature, {})
    plot_data["display_label"] = plot_data[feature].map(lambda x: label_map.get(x, x))
    
    # Grouped Bar Plot
    sns.barplot(
        data=plot_data, 
        y="display_label", 
        x="Percentage", 
        hue="Group", 
        hue_order=["Outliers", "Inliers"], 
        palette={"Outliers": "#D32F2F", "Inliers": "#4F81BD"},
        ax=ax, 
        alpha=0.8
    )
    
    # Customize Subplot
    ax.set_title(DISPLAY_LABELS.get(feature, feature), fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend().set_visible(False)
    ax.grid(True, axis="x", alpha=0.2)
    sns.despine(ax=ax, left=True)
    
    # Add annotations
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", padding=3, fontsize=8)

# Hide unused subplots
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis("off")

# Global Title and Legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98), frameon=False)
fig.suptitle("Outlier Profiling: Categorical Features", fontsize=16, fontweight="bold", y=1.0)

# Adjust layout
fig.tight_layout(rect=[0, 0.02, 1, 1], h_pad=2.0, w_pad=3.0)

# Add footnote
fig.text(0.01, 0.01, "Note: Population-weighted estimates.", ha="left", fontsize=9, style="italic", color="#555555")

plt.savefig("../figures/outliers/outlier_categorical_profile.png", bbox_inches="tight", dpi=200)
plt.show()

# %% [markdown]
# <div style="background-color:#f7fff8; padding:15px; border:3px solid #e0f0e0; border-radius:6px;">
#     💡 <b>Insight:</b> The Isolation Forest identifies outliers that represent a clinically vulnerable, high-cost population essential for modeling tail-risk.
#     <ul style="margin-top:10px; margin-bottom:0px">
#         <li><b>Clinical Profile:</b> Outliers are characterized by functional limitations and high comorbidity. They typically have 3 limitations compared with 0 for inliers and 4x more medical conditions (median 4 vs. 1). This "High-Needs" profile, often involving seniors (median age 66 vs. 47), is the primary driver of their outlier status.</li>
#         <li><b>Risk Escalation:</b> While only 1.2x more likely to exceed the median spend, outliers are 3.9x more likely to be in the Top 1% of spenders. They are also 2.5x more likely to incur some cost (zero-spenders: 9.2% vs 22.9%).</li>
#         <li><b>Structural Equality:</b> Both groups show near-identical Gini coefficients (~0.78). This confirms that cost concentration is a fundamental property of healthcare data that persists even within high-risk subgroups.</li>
#         <li><b>Decision: Keep Outliers.</b> These are valid tail-risk cases, not noise. Keeping them ensures the model learns the drivers of high-impact outcomes and correctly identifies the "High Comorbidity" spenders.</li>
#     </ul>
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Pipeline</h2>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Use the complete data preprocessing pipeline to create preprocessed data sets from raw data.
# </div>

# %%
# Create data preprocessing pipeline
preprocessor = create_preprocessing_pipeline(
    required_features, 
    optional_features, 
    input_numerical_features, 
    input_nominal_features,
    input_binary_features,
    strict=False
)

# Preprocess training, validation, and test data 
# Note: Overwrite the preprocessed DataFrames created during earlier steps
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# %%
# --- Verify results ---
# Verify absence of missing values and numeric types only in preprocessed data, and matching row counts between raw and processed data
datasets = {
    "Train": (X_train, X_train_preprocessed),
    "Val": (X_val, X_val_preprocessed),
    "Test": (X_test, X_test_preprocessed)
}
print("--- Preprocessing Pipeline Sanity Checks ---")
for name, (raw, processed) in datasets.items():
    # Check row counts (should match since no outlier removal)
    rows_match = "✅" if len(raw) == len(processed) else "❌"
    
    # Check for any remaining missing values
    null_count = processed.isnull().sum().sum()
    nulls_status = f"✅" if null_count == 0 else f"❌ ({null_count})"
    
    # Check if all columns are now numeric (floats/ints)
    all_numeric = "✅" if processed.apply(pd.api.types.is_numeric_dtype).all() else "❌"
    
    print(f"{name:5}: Rows Match: {rows_match} | No Missing Values: {nulls_status} | All Numeric: {all_numeric} | Raw Shape: {raw.shape} | Processed Shape: {processed.shape}")

# Verify feature scaling
output_numerical_features = preprocessor.named_steps["feature_scaler_encoder"].named_transformers_["numerical_scaler"].get_feature_names_out()
preprocessor_verify_scaling = X_train_preprocessed[output_numerical_features].describe().loc[["mean", "std", "min", "max"]]
preprocessor_verify_scaling.style \
    .pipe(add_caption, "Verification of Feature Scaling") \
    .format("{:.2f}")
# %%
# Check feature names of pipeline output to verify human-readable encoding
# Note: Should show names like 'REGION23_South' instead of 'REGION23_3.0'
nominal_feature_names = preprocessor.named_steps["categorical_label_standardizer"].nominal_features
encoded_feature_names = preprocessor.named_steps["feature_scaler_encoder"].named_transformers_["nominal_encoder"].get_feature_names_out(nominal_feature_names)
print(f"--- Nominal Feature Names (Pipeline Input) ---")
print(nominal_feature_names)
print(f"\n--- Encoded Nominal Feature Names (Pipeline Output) ---")
print(encoded_feature_names)

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Summary</h1>
# </div> 
#
# - **Data Loading:** Imported MEPS-HC 2023 SAS data using `pandas` `read_sas`.
# - **Data Preparation:**
#     - **Handling Duplicates:** Verified the absence of duplicates based on the ID column, complete rows, and all columns except ID.
#     - **Variable Selection:** Filtered 29 essential columns (target variable, candidate features, ID, sample weights) from the original 1,374 columns.
#     - **Target Population Filtering:** Filtered rows for adults with positive person weights (14,768 out of 18,919 respondents).
#     - **Handling Data Types:** Converted ID to string and maintained features and target as floats to ensure compatibility with scikit-learn transformers and models. Defined raw semantic data types for all features (numerical, binary, nominal, ordinal).
#     - **Standardizing Missing Values:** Recovered values from survey skip patterns and converted MEPS-specific missing codes to `np.nan`.
# - **Exploratory Data Analysis (EDA):** Analyzed raw distributions to inform data preprocessing and feature engineering decisions.
#     - **Sample Weights:** Verified survey weights represent ~260M adults and confirmed weighting is essential for population-level representativeness.
#     - **Target Variable:** Identified a zero-inflated (22%) and extremely right-skewed distribution where the top 1% of spenders drive ~21% of costs.
#     - **Numerical Features:** Analyzed distributions for age, family size, and self-reported health to inform robust median-based imputation.
#     - **Categorical Features:** Identified oversampling of low socio-economic status and chronic conditions, requiring survey weight adjustments for modeling.
# - **Feature Engineering (Stateless):**
#     - **Standardizing Binary Features:** Standardized binary features to 0/1 encoding.
#     - **Feature Refinement:** Created a recent life transition flag and collapsed sparse categories (e.g., recent divorce, job loss) into stable parent categories.
#     - **Feature Validation:** Defined pipeline input feature lists and verified feature engineering results.
# - **Train-Validation-Test Split:** Split data into training (80%), validation (10%), and test (10%) sets using a distribution-informed stratified split to balance zero-inflation and the extreme tail of the target variable.
# - **Data Preprocessing (Stateful):**
#     - **Handling Missing Values:** Imputed missing values using the median for numerical and mode for categorical features, calculated from the training data. 
#     - **Handling Outliers:** Detected univariate outliers with 3SD and 1.5 IQR methods and identified multivariate outliers with an isolation forest. Profiled outliers by comparing out-of-pocket costs and feature distributions between inliers and outliers. Confirmed that outliers represent legitimate high risk profiles rather than data errors, and retained all outliers to preserve the model's ability to predict extreme out-of-pocket costs.
