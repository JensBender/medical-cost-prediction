# Mapping MEPS variable names to human-readable display labels (for EDA and UI)
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

# Mapping metric keys to human-readable labels for display in notebook tables
METRIC_LABELS = {
    "model": "Model",
    "mdae": "MdAE",
    "mae": "MAE",
    "r2": "R-squared",
    "training_time": "Training Time (s)"
}

# --- Categorical Label Mappings ---
# Mapping categorical labels from numbers to strings (for pipeline)
CATEGORY_LABELS_PIPELINE = {
    # Demographics
    "SEX": {1: "Male", 0: "Female"},
    "MARRY31X_GRP": {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married"},
    "REGION23": {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    
    # Socioeconomic
    "POVCAT23": {1: "Poor/Negative", 2: "Near Poor", 3: "Low Income", 4: "Middle Income", 5: "High Income"},
    "HIDEG": {1: "No Degree", 2: "GED", 3: "HS Diploma", 4: "Bachelor's", 5: "Master's", 6: "Doctorate", 7: "Other"},
    "EMPST31_GRP": {1: "Employed", 4: "Not Employed", 0: "Not Employed"},
   
    # Insurance & Access
    "INSCOV23": {1: "Any Private", 2: "Public Only", 3: "Uninsured"},
    "HAVEUS42": {1: "Yes", 0: "No"},
    
    # Perceived Health & Lifestyle
    "ADSMOK42": {1: "Yes", 0: "No"},
    
    # Limitations
    "ADLHLP31": {1: "Yes", 0: "No"},
    "IADLHP31": {1: "Yes", 0: "No"},
    "WLKLIM31": {1: "Yes", 0: "No"},
    "COGLIM31": {1: "Yes", 0: "No"},
    "JTPAIN31_M18": {1: "Yes", 0: "No"},
    
    # Chronic Conditions
    "HIBPDX": {1: "Yes", 0: "No"},
    "CHOLDX": {1: "Yes", 0: "No"},
    "DIABDX_M18": {1: "Yes", 0: "No"},
    "CHDDX": {1: "Yes", 0: "No"},
    "STRKDX": {1: "Yes", 0: "No"},
    "CANCERDX": {1: "Yes", 0: "No"},
    "ARTHDX": {1: "Yes", 0: "No"},
    "ASTHDX": {1: "Yes", 0: "No"},

    # Feature Engineered
    "RECENT_LIFE_TRANSITION": {1: "Yes", 0: "No"},
}

# Add the MEPS category codes (for notebook EDA of raw features)
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


# List of Required Features
# Note: Features selected based on subject matter expertise (known from prior healthcare research). 
# Refine based on data (feature importance scores) after model training.
REQUIRED_FEATURES = [
    "AGE23X",    # Primary driver of medical utilization and costs
    "SEX",       # Key driver of utilization frequency and spending disparities documented in healthcare literature
    "INSCOV23",  # Critical for out-of-pocket cost prediction
    "REGION23",  # Captures geographic variance in healthcare pricing
    "RTHLTH31"   # Self-reported health is a powerful proxy for healthcare demand
]

# List of Optional Features (can be imputed if missing)
OPTIONAL_FEATURES = [
    "MARRY31X", "FAMSZE23", "POVCAT23", "HIDEG", "EMPST31",
    "HAVEUS42", "MNHLTH31", "ADSMOK42",
    "ADLHLP31", "IADLHP31", "WLKLIM31", "COGLIM31", "JTPAIN31_M18",
    "HIBPDX", "CHOLDX", "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"
]

# Define the categories for the nominal features (for OneHotEncoder)
NOMINAL_CATEGORIES = [
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["REGION23"].values())),     # 1st
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["MARRY31X_GRP"].values())), # 2nd
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["INSCOV23"].values())),     # 3rd
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["HIDEG"].values()))         # 4th
]

# Define the nominal categories to drop for one-hot encoder (must be in same order as NOMINAL_CATEGORIES)
# Rationale: (1) Allow meaningful comparisons to other categories and (2) often the most frequent category (thus more robust baseline than sparse categories)
NOMINAL_DROP_CATEGORIES = ["South", "Married", "Any Private", "HS Diploma"]

# Configuration
RANDOM_STATE = 42

# Plotting Aesthetics
POP_COLOR = "#084594"    # deep navy for population
SAMPLE_COLOR = "#14b8a6" # vibrant teal for sample