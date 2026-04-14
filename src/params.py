"""
Hyperparameter tuning parameter configurations.

This module centralizes the distributions used for hyperparameter tuning with 
randomized search. It ensures consistency between exploratory notebooks 
(notebooks/2_modeling.ipynb) and production scripts (e.g., scripts/tune_random_forest.py)
and avoids unnecessary DVC pipeline reruns triggered by changes to src/constants.py.
"""

from scipy.stats import randint, uniform, loguniform

# =========================
# Random Forest
# =========================

# Hyperparameter search space for RandomForestRegressor
# Note: Used in scripts/tune_random_forest.py
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 500],          # More trees = more stable but slower
    "max_depth": randint(8, 25),              # Baseline: 16. Search around it.
    "min_samples_split": randint(20, 150),    # Baseline: 50. Explore wider.
    "min_samples_leaf": randint(10, 80),      # Baseline: 25. Explore wider. Most impactful for MdAE.
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],  # Baseline: "sqrt". Explore random subset with 30%, 50% and 70% of features.
    "max_samples": uniform(0.6, 0.4),         # Baseline: None (100%). Explore random subsample of 60% to 100%. 
}

# Number of hyperparameter combinations for tuning in production script 
RF_N_ITER = 100


# =========================
# Elastic Net
# =========================

# Hyperparameter search space for ElasticNet
# Note 1: Prefixed with step names (polynomials__, model__) to be compatible with Pipeline used in scripts/tune_elastic_net.py
# Note 2: model__alpha uses loguniform for equal sampling across orders of magnitude (e.g., 0.01-0.1 has same chance as 0.1-1.0)
EN_PARAM_DISTRIBUTIONS = {
    "polynomials__interaction_only": [True, False],  # True avoids redundant squared binary features; False captures non-linearities (e.g. Age^2).
    "model__alpha": loguniform(0.01, 1.0),           # Regularization strength. Log-scale: 0.01 (low regularization) to 1.0 (high for z-standardized features).
    "model__l1_ratio": uniform(0.0, 1.0),            # Penalty mix. 0=Ridge (L2) keeps correlated features; 1=Lasso (L1) for automated feature selection.
}

# Number of hyperparameter combinations for tuning in production script 
EN_N_ITER = 50
