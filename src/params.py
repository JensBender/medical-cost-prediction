"""
Hyperparameter tuning parameter configurations.

This module centralizes the distributions used for hyperparameter tuning with 
randomized search. It ensures consistency between exploratory notebooks 
(notebooks/2_modeling.ipynb) and production scripts (e.g., scripts/tune_random_forest.py).

Rationale: The chosen search spaces prioritize robustness to the zero-inflated, heavy tail 
distribution of medical costs.
"""

from scipy.stats import randint, uniform, loguniform

# =========================
# Elastic Net
# =========================

# Hyperparameter search space for ElasticNet
# Note: Prefixed with step names (polynomials__, model__) to be compatible with Pipeline used in "scripts/tune_elastic_net.py"
EN_PARAM_DISTRIBUTIONS = {
    # False captures non-linear, squared effects like Age^2, essential for U-shaped cost curves.
    "polynomials__interaction_only": [True, False],  
    
    # Regularization strength. "loguniform" performs equal sampling across 
    # orders of magnitude (e.g., 0.01-0.1 has same chance as 0.1-1.0)
    "model__alpha": loguniform(0.01, 1.0),           
    
    # Penalty mix. 0=Ridge (L2) to handle multicollinearity; 1=Lasso (L1) which 
    # aggressively forces coefficients to zero (sparsity) for automated feature selection.
    "model__l1_ratio": uniform(0.0, 1.0),            
}

# Number of hyperparameter combinations for tuning in production script 
EN_N_ITER = 50


# =========================
# Random Forest
# =========================

# Hyperparameter search space for RandomForestRegressor (for "scripts/tune_random_forest.py")
RF_PARAM_DISTRIBUTIONS = {
    # More trees (200-500) increase stability and reduce the high 
    # variance of individual tree predictions on skewed cost data.
    "n_estimators": [200, 300, 400],          
    
    # Baseline: 16. Moderate depth (8-25) balances the need to capture complex 
    # interactions (e.g., Age + Condition + Insurance) while preventing memorization.
    "max_depth": randint(8, 25),              
    
    # Baseline: 50. High split thresholds (20-150) ensure nodes are meaningful 
    # and not just fitting noise in low-density regions.
    "min_samples_split": randint(20, 150),    
    
    # Baseline: 25. High leaf requirements (10-80) are critical for MdAE optimization, 
    # as they force the leaf-median to be based on a robust sample of patient profiles.
    "min_samples_leaf": randint(10, 80),      
    
    # Baseline: "sqrt". Exploring feature subsets (30%-70%) reduces inter-tree correlation, 
    # which is vital when many features (e.g., medical conditions) are sparse.
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],  
    
    # Baseline: None (100%). Random subset of 60%-100% of samples introduces diversity 
    # and prevents the ensemble from being dominated by extreme "super-utilizer" cases.
    "max_samples": uniform(0.6, 0.4),         
}

# Number of hyperparameter combinations for tuning
RF_N_ITER = 50


# =========================
# XGBoost
# =========================

# Hyperparameter search space for XGBRegressor (for "scripts/tune_xgboost.py")
XGB_PARAM_DISTRIBUTIONS = {
    # More rounds (400-800) with a lower learning rate allow the gradient 
    # booster to converge slowly and smoothly on the long tail of medical costs.
    "n_estimators": [400, 600, 800],          
    
    # Shallow trees (3-10) are more robust to noise and help prevent the 
    # model from fitting deep, spurious patterns in small patient cohorts.
    "max_depth": randint(3, 10),              
    
    # Log-uniform learning rate (0.01-0.2) explores both small and big 
    # update steps for the gradient descent optimization.
    "learning_rate": loguniform(0.01, 0.2),   
    
    # Higher weights (1-20) act as a conservative split criterion, preventing 
    # partitions that only cover a few noisy or extreme cost individuals.
    "min_child_weight": randint(1, 20),       
    
    # Row and column subsampling (50%-100%) provide strong regularization 
    # against overfitting to certain features or outliers.
    "subsample": uniform(0.6, 0.4),   # random subset of 60-100% of rows        
    "colsample_bytree": uniform(0.5, 0.5),  # random subset of 50-100% of features 
    
    # L1 (alpha) and L2 (lambda) penalties further constrain model complexity 
    # and encourage parameter sparsity for better generalization.
    "reg_lambda": uniform(0, 5),              
    "reg_alpha": uniform(0, 5),               
}

# Number of hyperparameter combinations for tuning 
XGB_N_ITER = 50
