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
#         Last updated: January 2026
#     </p>
# </div>

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h1 style="margin:0px">Imports</h1>
# </div>

# %%
# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing (Scikit-learn)
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    OrdinalEncoder
)

# Model selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score
)

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
#     <h1 style="margin:0px">Data Preprocessing</h1>
# </div> 
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     <strong>Note</strong>: Kept column names in ALL CAPS in this project to ensure consistency with official <b><a href="../docs/references/h251doc.pdf">MEPS documentation</a></b>, <b><a href="../docs/references/h251cb.pdf">codebook</a></b>, and <b><a href="../docs/references/data_dictionary.md">data dictionary</a></b>.
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
# <div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px">
#     💡 There are 3 pairs (or 6 rows) of duplicates that have identical values across all 1,300+ columns except for their IDs. Analysis shows these are young siblings (ages 1 and 5) living in the same household with identical parent-reported health data, identical sample weights, and identical costs. Thus, they appear to be valid respondents rather than "ghost records". However, they will be naturally excluded when filtering for the adult target population.
# </div>

# %% [markdown]
# <div style="background-color:#3d7ab3; color:white; padding:12px; border-radius:6px;">
#     <h2 style="margin:0px">Variable Selection</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Filter the following 29 columns (out of 1,374):
#     <ul style="margin-bottom:0px">
#         <li><b>ID</b>: Unique identifier for each respondent (<code>DUPERSID</code>).</li>
#         <li><b>Sample Weights</b>: Ensures population representativeness (<code>PERWT23F</code>).</li>
#         <li><b>Candidate Features</b>: 26 variables selected for their consumer accessibility, beginning-of-year measurement, and predictive power.</li> 
#         <li><b>Target Variable</b>: Total out-of-pocket health care costs (<code>TOTSLF23</code>).</li>
#     </ul>
#     <br>
#     <b>Rationale</b>: For a detailed breakdown of the target variable selection and feature selection criteria, see the <b><a href="../docs/specs/technical_specifications.md">Technical Specifications</a></b> and <b><a href="../docs/research/candidate_features.md">Candidate Features Research</a></b>.
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
#     <h2 style="margin:0px">Filtering Target Population</h2>
# </div>
#
# <div style="background-color:#fff6e4; padding:15px; border:3px solid #f5ecda; border-radius:6px;">
#     📌 Filter rows to match the target population based on the following criteria:
#     <ul style="margin-bottom:0px">
#         <li><b>Positive person weight</b> (<code>PERWT23F > 0</code>): Drop respondents with a person weight of zero (i.e., 456 respondents). These individuals are considered "out-of-scope" for the full-year population (e.g., they joined the military, were institutionalized, or moved abroad).</li>
#         <li><b>Adults</b> (<code>AGE23X >= 18</code>): Drop respondents under age 18 (i.e., 3796 respondents), as the medical cost planner app targets adults.</li>
#     </ul>
#     <br>
#     <b>Note</b>: Keeps 14,768 out of 18,919 respondents.
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
#         <li><b>ID</b>: <code>DUPERSID</code> is an identifier, not a quantity. Converting them to <code>string</code> prevents unintended math.</li>
#         <li><b>Sample Weights</b>: <code>PERWT23F</code> contains decimal precision critical for population-level estimates. Must remain <code>float</code>.</li>
#         <li><b>Candidate Features</b>: The SAS loader stored all 26 features as floats by default. Although many features are categorical and represent integer codes (e.g., 1=Male, 2=Female), they are maintained as <code>float</code> for three practical reasons:
#             <ul>
#                 <li>Missing Value Compatibility: In standard Pandas, <code>np.nan</code> is a floating-point object. Assigning it to an integer column triggers an automatic cast back to <code>float64</code>.</li>
#                 <li>Pipeline Consistency: scikit-learn transformers (e.g., <code>SimpleImputer</code>, <code>StandardScaler</code>) internally use floats and automatically convert numerical inputs to <code>float</code>, even when using <code>set_config(transform_output="pandas")</code>. Keeping them as floats avoids redundant type casting.</li>
#                 <li>Model Consistency: Most machine learning models (e.g., XGBoost, Linear Regression) internally use floats and automatically convert numerical inputs to <code>float</code> during training and inference.</li>
#             </ul>
#         </li>
#         <li><b>Target Variable</b>: <code>TOTSLF23</code> is rounded to whole dollars in the raw MEPS data. It is kept as <code>float</code> for Model Consistency and to avoid redundant type casting, as regression models deliver <code>float</code> predictions during training and inference.</li>
#     </ul>
# </div>

# %%
# Identify storage data types (defaulted to float/object by SAS loader)
df.dtypes

# %%
# Convert ID to string
df["DUPERSID"] = df["DUPERSID"].astype(str)

# Features and Target are kept as float to accommodate np.nan and ML requirements
# No manual conversion to int is needed as sklearn pipelines will operate on floats anyway.


# Verify the changes
df.info()

# %% [markdown]
# <div style="background-color:#2c699d; color:white; padding:15px; border-radius:6px;">
#     <h2 style="margin:0px">Standardizing Missing Values</h1>
# </div> 
#
# <div style="background-color:#e8f4fd; padding:15px; border:3px solid #d0e7fa; border-radius:6px;">
#     ℹ️ <b>Pandas Missing Value Types</b>:
#     <ul style="margin-bottom:0px">
#         <li><b>np.nan</b>: Standard missing value indicator (technically a float); often the default in Pandas for numerical data.</li>
#         <li><b>pd.NA</b>: Unified missing value indicator for modern nullable data types (mostly integer and boolean).</li>
#         <li><b>None</b>: Python's native type; often used for object and string data.</li>
#         <li><b>pd.NaT</b>: For datetime and timedelta data types.</li>
#     </ul>
#     <br>
#     ℹ️ <b>MEPS Missing Value Codes</b>:
#     <ul style="margin-bottom:0px">
#         <li><b>-1 INAPPLICABLE</b>: Variable does not apply (structural skip).</li>
#         <li><b>-7 REFUSED</b>: Person refused to answer.</li>
#         <li><b>-8 DON'T KNOW</b>: Person did not know the answer.</li>
#         <li><b>-9 NOT ASCERTAINED</b>: Administrative or technical error in collection.</li>
#         <li><b>-15 CANNOT BE COMPUTED</b>: Incomplete data for a constructed variable.</li>
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
#         <li><b>Smoker</b> (<code>ADSMOK42</code>): This question is only asked if the respondent already confirmed smoking 100+ cigarettes in their life. Those who said "No" skip this and are coded -1. In this project, these "Never Smokers" are mapped to "No" (2).</li>
#         <li><b>Joint Pain</b> (<code>JTPAIN31_M18</code>): Respondents who already reported an arthritis diagnosis (<code>ARTHDX = 1</code>) earlier in the interview skip this question and are coded -1. Since arthritis inherently involves joint symptoms, these values are mapped to "Yes" (1).</li>
#     </ul>
# </div>

# %%
# Recover implied values
# Smoker: Convert -1 (Never Smoker) to 2 (No)
df.loc[df["ADSMOK42"] == -1, "ADSMOK42"] = 2

# Joint Pain: Convert -1 to 1 (Yes) only if they have Arthritis 
df.loc[(df["JTPAIN31_M18"] == -1) & (df["ARTHDX"] == 1), "JTPAIN31_M18"] = 1

# %%
# Identify MEPS missing values 
missing_codes = [-1, -7, -8, -9, -15]
missing_frequency_df = pd.DataFrame({code: (df == code).sum() for code in missing_codes})
missing_frequency_df["TOTAL"] = missing_frequency_df.sum(axis=1)
missing_frequency_df["PERCENTAGE"] = (missing_frequency_df["TOTAL"] / len(df) * 100).round(2)
missing_frequency_df.sort_values("TOTAL", ascending=False) 

# %%
# Convert all remaining MEPS missing codes to np.nan
df = df.replace(missing_codes, np.nan)
