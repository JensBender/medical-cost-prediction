# Standard library imports
import contextlib
import time

# Third-party library imports
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.utils import weighted_median_absolute_error


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
