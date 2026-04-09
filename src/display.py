# =========================
# Display Constants
# =========================
# This module contains constants used exclusively for display, visualization,
# and UI purposes (notebooks, app frontend).
# These do NOT affect pipeline logic and are intentionally kept separate from
# src/constants.py to avoid triggering unnecessary DVC pipeline reruns.

from src.constants import CATEGORY_LABELS_PIPELINE


# Mapping MEPS variable names to human-readable display labels (for notebook EDA and app UI)
DISPLAY_LABELS = {
    # Target
    "TOTSLF23": "Out-of-Pocket Costs",
    "TOTSLF23_LOG": "Out-of-Pocket Costs (Log)",
    
    # Demographics
    "AGE23X": "Age",
    "SEX": "Sex",
    "REGION23": "Region",
    "MARRY31X": "Marital Status",
    "FAMSZE23": "Family Size",
    
    # Socioeconomic
    "POVCAT23": "Poverty Category",
    "HIDEG": "Education",
    "EMPST31": "Employment",
    
    # Insurance & Access
    "INSCOV23": "Insurance",
    "HAVEUS42": "Usual Source of Care",
    
    # Perceived Health & Lifestyle
    "RTHLTH31": "Physical Health",
    "MNHLTH31": "Mental Health",
    "ADSMOK42": "Smoker",
    
    # Limitations
    "ADLHLP31": "ADL Help",
    "IADLHP31": "IADL Help",
    "WLKLIM31": "Walking Limitation",
    "COGLIM31": "Cognitive Limitation",
    "JTPAIN31_M18": "Joint Pain",
    
    # Chronic Conditions
    "HIBPDX": "High Blood Pressure",
    "CHOLDX": "High Cholesterol",
    "DIABDX_M18": "Diabetes",
    "CHDDX": "Coronary Heart Disease",
    "STRKDX": "Stroke",
    "CANCERDX": "Cancer",
    "ARTHDX": "Arthritis",
    "ASTHDX": "Asthma",

    # Feature Engineered
    "RECENT_LIFE_TRANSITION": "Recent Life Transition",
    "MARRY31X_GRP": "Marital Status",
    "EMPST31_GRP": "Employment",
    "CHRONIC_COUNT": "Chronic Conditions Count",
    "LIMITATION_COUNT": "Limitations Count",
    "condition_count": "Medical Conditions"
}

# Mapping metrics to human-readable display labels (for notebook model evaluation)
METRIC_LABELS = {
    "model": "Model",
    "val_mdae": "MdAE (Val)",
    "val_mae": "MAE (Val)",
    "val_r2": "R² (Val)",
    "train_mdae": "MdAE (Train)",
    "train_mae": "MAE (Train)",
    "train_r2": "R² (Train)",
    "mdae": "MdAE",
    "mae": "MAE",
    "r2": "R²",
    "training_time": "Training Time (s)"
}

# Mapping categorical labels from numbers to strings (for notebook EDA)
# Uses CATEGORY_LABELS_PIPELINE constant used by pipeline and adds the raw MEPS category codes (for EDA of raw features)
CATEGORY_LABELS_EDA = {
    # Demographics
    "SEX": {**CATEGORY_LABELS_PIPELINE["SEX"], 2: "Female"},
    "REGION23": CATEGORY_LABELS_PIPELINE["REGION23"],
    "MARRY31X": {
        1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married",
        7: "Married in Round", 8: "Widowed in Round", 9: "Divorced in Round", 10: "Separated in Round"
    },
    "MARRY31X_GRP": CATEGORY_LABELS_PIPELINE["MARRY31X_GRP"],

    # Socioeconomic
    "POVCAT23": CATEGORY_LABELS_PIPELINE["POVCAT23"],
    "HIDEG": CATEGORY_LABELS_PIPELINE["HIDEG"],
    "EMPST31": {1: "Employed", 2: "Job to Return To", 3: "Job in Ref Period", 4: "Not Employed"},
    "EMPST31_GRP": CATEGORY_LABELS_PIPELINE["EMPST31_GRP"],

    # Insurance & Access
    "INSCOV23": CATEGORY_LABELS_PIPELINE["INSCOV23"],
    "HAVEUS42": {**CATEGORY_LABELS_PIPELINE["HAVEUS42"], 2: "No"},

    # Perceived Health & Lifestyle
    "ADSMOK42": {**CATEGORY_LABELS_PIPELINE["ADSMOK42"], 2: "No"},

    # Limitations
    "ADLHLP31": {**CATEGORY_LABELS_PIPELINE["ADLHLP31"], 2: "No"},
    "IADLHP31": {**CATEGORY_LABELS_PIPELINE["IADLHP31"], 2: "No"},
    "WLKLIM31": {**CATEGORY_LABELS_PIPELINE["WLKLIM31"], 2: "No"},
    "COGLIM31": {**CATEGORY_LABELS_PIPELINE["COGLIM31"], 2: "No"},
    "JTPAIN31_M18": {**CATEGORY_LABELS_PIPELINE["JTPAIN31_M18"], 2: "No"},

    # Chronic Conditions
    "HIBPDX": {**CATEGORY_LABELS_PIPELINE["HIBPDX"], 2: "No"},
    "CHOLDX": {**CATEGORY_LABELS_PIPELINE["CHOLDX"], 2: "No"},
    "DIABDX_M18": {**CATEGORY_LABELS_PIPELINE["DIABDX_M18"], 2: "No"},
    "CHDDX": {**CATEGORY_LABELS_PIPELINE["CHDDX"], 2: "No"},
    "STRKDX": {**CATEGORY_LABELS_PIPELINE["STRKDX"], 2: "No"},
    "CANCERDX": {**CATEGORY_LABELS_PIPELINE["CANCERDX"], 2: "No"},
    "ARTHDX": {**CATEGORY_LABELS_PIPELINE["ARTHDX"], 2: "No"},
    "ASTHDX": {**CATEGORY_LABELS_PIPELINE["ASTHDX"], 2: "No"},

    # Feature Engineered
    "RECENT_LIFE_TRANSITION": CATEGORY_LABELS_PIPELINE["RECENT_LIFE_TRANSITION"]
}

# Visualization Colors (for notebook EDA)
POP_COLOR = "#084594"    # deep navy for population
SAMPLE_COLOR = "#14b8a6" # vibrant teal for sample
