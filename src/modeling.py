# Standard library imports
import contextlib  # to train model without MLflow tracking using a null context
import time

# Third-party library imports
import mlflow
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor  # for median baseline prediction
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.utils import weighted_median_absolute_error
from src.constants import RANDOM_STATE


def train_and_evaluate(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    w_train=None, w_val=None, 
    track_mlflow=False,
    model_name="model", 
    log_model=False
):
    """
    Train and evaluate a single machine learning model with optional MLflow experiment tracking.

    Args:
        model (estimator): The Scikit-learn estimator or pipeline to be trained.
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Target variable for training data.
        X_val (pd.DataFrame): Preprocessed validation features.
        y_val (pd.Series): Target variable for validation data.
        w_train (pd.Series, optional): Sample weights for training data. Defaults to None.
        w_val (pd.Series, optional): Sample weights for validation data. Defaults to None.
        track_mlflow (bool, optional): Whether to track experiment with MLflow. Defaults to False. 
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
    # Ensure weights are passed to model even when wrapped in a Pipeline (for Elastic Net)
    fit_params = {}
    if w_train is not None:
        # Normalize weights so mean is 1.0 (prevents numerical instability in algorithms like SVR)
        w_train_norm = w_train / w_train.mean()
        
        reg = getattr(model, "regressor", model)
        key = f"{reg.steps[-1][0]}__sample_weight" if isinstance(reg, Pipeline) else "sample_weight"
        fit_params[key] = w_train_norm

    # Use a real MLflow run or a no-op context depending on track_mlflow
    run_context = (
        mlflow.start_run(run_name=model_name)
        if track_mlflow
        else contextlib.nullcontext()
    )

    with run_context:
        if track_mlflow:
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

        # Fit model on training data
        start_time = time.time()  # Measure training time
        model.fit(X_train, y_train, **fit_params)
        end_time = time.time()

        # Predict on validation data
        y_val_pred = model.predict(X_val)

        # Calculate evaluation metrics
        mdae = weighted_median_absolute_error(y_val, y_val_pred, sample_weight=w_val)  
        mae = mean_absolute_error(y_val, y_val_pred, sample_weight=w_val)
        r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)

        if track_mlflow:
            # Log metrics
            mlflow.log_metric("mdae", mdae)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

           # Log fitted model artifact
            if log_model:
                mlflow.sklearn.log_model(model, "model")

    # Return results dictionary
    return {
        "mdae": mdae,
        "mae": mae,
        "r2": r2,
        "training_time": end_time - start_time,
        "fitted_model": model,
        "y_val_pred": y_val_pred,
    }


def get_baseline_models():
    """
    Define baseline machine learning models to predict out-of-pocket medical costs with distribution-aware 
    hyperparameters.

    Returns:
        dict: A dictionary mapping model names (str) to Scikit-learn estimators or pipelines.
    """
    baseline_models = {
        "Median Prediction": DummyRegressor(strategy="median"),  # Always predict median as a benchmark
        "Linear Regression": TransformedTargetRegressor(      
            regressor=LinearRegression(),  # Simple, linear predictions as an interpretable baseline
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
        "Decision Tree": TransformedTargetRegressor(
            regressor=DecisionTreeRegressor(
                criterion="absolute_error",  # Optimize for MAE of log-costs; corresponds to predicting the Median of raw costs
                max_depth=12,          # Limits tree depth to prevent overfitting (Default: None)
                min_samples_split=100, # Prevents splitting on tiny, noisy groups of patients (Default: 2)
                min_samples_leaf=50,   # Ensures each leaf's cost prediction has enough support (Default: 1)
                max_features="sqrt",   # Uses random feature subset for each split to prevent overfitting on individual features (Default: None)
                random_state=RANDOM_STATE  
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "Random Forest": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                criterion="absolute_error",  # Optimize for MAE of log-costs; corresponds to predicting the Median of raw costs
                n_estimators=200,      # More trees for more stable estimates (Default: 100)
                max_depth=16,          # Limits tree depth to prevent overfitting (Default: None)
                min_samples_split=50,  # Prevents splitting on tiny, noisy groups of patients (Default: 2)
                min_samples_leaf=25,   # Ensures each leaf's cost prediction has enough support (Default: 1)
                max_features="sqrt",   # Uses random feature subset for each split to prevent overfitting on individual features (Default: 1.0)
                n_jobs=-1,             # Use all CPU cores to speed up training
                random_state=RANDOM_STATE 
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "XGBoost": TransformedTargetRegressor(
            regressor=XGBRegressor(
                objective="reg:absoluteerror", # Optimized for MAE of log-costs; corresponds to predicting the Median raw costs (Default: "reg:squarederror")
                n_estimators=600,      # More rounds with lower learning rate for smooth fitting (Default: 100)
                learning_rate=0.05,    # Smaller steps to prevent overshooting noisy targets (Default: 0.3)
                min_child_weight=10,   # Requires more "support" per node to split (Default: 1)
                subsample=0.8,         # Uses random row subset (80%) to prevent overfitting (Default: 1)
                colsample_bytree=0.5,  # Uses random feature subset (50%) for each split to prevent overfitting on individual features (Default: 1)
                reg_lambda=2.0,        # L2 regularization on leaf weights to avoid sharp peaks (Default: 1)
                tree_method="hist",    # Uses histogram algorithm for significantly faster training (Default: auto)
                n_jobs=-1,             # Use all CPU cores to speed up training
                random_state=RANDOM_STATE  
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        "Support Vector Machine": TransformedTargetRegressor(
            regressor=SVR(
                kernel="rbf",    # Handles non-linearity and feature interactions (Default: same)
                C=10.0,          # Slightly higher regularization strength for log-costs (Default: 1.0)
                cache_size=1000  # Increase memory budget (1GB) to speed up training time (Default: 200)
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )
    }
    
    return baseline_models
