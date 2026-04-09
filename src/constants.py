# =========================
# Feature Lists 
# =========================

# Raw MEPS Features 
RAW_NUMERICAL_FEATURES = ["AGE23X", "FAMSZE23", "RTHLTH31", "MNHLTH31"]
RAW_BINARY_FEATURES = [
    "SEX", "HAVEUS42", "ADSMOK42", "ADLHLP31", "IADLHP31", 
    "WLKLIM31", "COGLIM31", "JTPAIN31_M18", "HIBPDX", "CHOLDX", 
    "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"
]
RAW_NOMINAL_FEATURES = ["REGION23", "MARRY31X", "EMPST31", "INSCOV23", "HIDEG"]
RAW_ORDINAL_FEATURES = ["POVCAT23"]

ID_COLUMN = "DUPERSID"
WEIGHT_COLUMN = "PERWT23F"
TARGET_COLUMN = "TOTSLF23"

#  Variable Selection List (for initial data preparation)
RAW_COLUMNS_TO_KEEP = (
    [ID_COLUMN, WEIGHT_COLUMN, TARGET_COLUMN] + 
    RAW_NUMERICAL_FEATURES + RAW_BINARY_FEATURES + 
    RAW_NOMINAL_FEATURES + RAW_ORDINAL_FEATURES
)

# Pipeline Input Features (after initial data preparation and feature engineering)
PIPELINE_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + RAW_ORDINAL_FEATURES
PIPELINE_BINARY_FEATURES = RAW_BINARY_FEATURES + ["RECENT_LIFE_TRANSITION", "EMPST31_GRP"]
PIPELINE_NOMINAL_FEATURES = ["REGION23", "MARRY31X_GRP", "INSCOV23", "HIDEG"]

# Required vs. Optional Features (for MissingValueChecker in Pipeline)
PIPELINE_REQUIRED_FEATURES = ["AGE23X", "SEX", "INSCOV23", "REGION23", "RTHLTH31"]
PIPELINE_OPTIONAL_FEATURES = [
    "MARRY31X_GRP", "FAMSZE23", "POVCAT23", "HIDEG", "EMPST31_GRP", 
    "RECENT_LIFE_TRANSITION", "HAVEUS42", "MNHLTH31", "ADSMOK42",
    "ADLHLP31", "IADLHP31", "WLKLIM31", "COGLIM31", "JTPAIN31_M18",
    "HIBPDX", "CHOLDX", "DIABDX_M18", "CHDDX", "STRKDX", "CANCERDX", "ARTHDX", "ASTHDX"
]


# =========================
# Label Mappings 
# =========================

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


# =========================
# Categorical Label Lists 
# =========================

# Nominal Feature Categories (for OneHotEncoder in Pipeline)
NOMINAL_CATEGORIES = [
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["REGION23"].values())),     # 1st
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["MARRY31X_GRP"].values())), # 2nd
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["INSCOV23"].values())),     # 3rd
    list(dict.fromkeys(CATEGORY_LABELS_PIPELINE["HIDEG"].values()))         # 4th
]

# Designated Baseline Categories of Nominal Features (to drop in OneHotEncoder in Pipeline) 
# Must be in same order as NOMINAL_CATEGORIES
# Rationale: (1) Allow meaningful comparisons to other categories and (2) often the most frequent category (thus more robust baseline than sparse categories)
NOMINAL_DROP_CATEGORIES = ["South", "Married", "Any Private", "HS Diploma"]


# =========================
# Data Preparation
# =========================

# MEPS Missing Value Codes 
# Used to standardize survey-specific markers (-1 to -15) to NaN
MEPS_MISSING_CODES = [-1, -7, -8, -9, -15]


# =========================
# Feature Engineering 
# =========================

# Life Transition Codes (used to derive the 'RECENT_LIFE_TRANSITION' feature)
MARRY31X_TRANSITION_CODES = [7, 8, 9, 10]  # represent marrital transitions (Married/Widowed/Divorced/Separated in Round)
EMPST31_TRANSITION_CODES = [2, 3]  # represent employment transitions (Job to Return To, Job in Ref Period)

# Category Collapsing Maps (used to group sparse categories into stable counterparts)
MARRY31X_COLLAPSE_MAP = {7: 1, 8: 2, 9: 3, 10: 4}
EMPST31_COLLAPSE_MAP = {2: 0, 3: 0, 4: 0}


# =========================
# Modeling 
# =========================

# Configuration (for reproducible training runs and data splits)
RANDOM_STATE = 42
