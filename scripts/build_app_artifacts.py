"""Build cost benchmarks and prediction metadata artifacts for app deployment.

Run after XGBoost quantile regression model training:
    .venv-train/Scripts/python scripts/build_app_artifacts.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    ID_COLUMN,
    MEPS_MISSING_CODES,
    RANDOM_STATE,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)
from src.modeling import RAW_DATA_PATH, VAL_DATA_PATH
from src.stats import create_stratification_bins, weighted_quantile

APP_DATA_DIR = Path("app/data")
COST_BENCHMARKS_PATH = APP_DATA_DIR / "cost_benchmarks.json"
PREDICTION_METADATA_PATH = APP_DATA_DIR / "prediction_metadata.json"
QUANTILE_PREDICTIONS_PATH = Path("models/xgb_quantile_predictions.joblib")

AGE_BENCHMARK_BINS = [18, 35, 50, 65, 86]
AGE_BENCHMARK_LABELS = ["18-34", "35-49", "50-64", "65+"]
HIGH_PREDICTED_UNCERTAINTY_QUANTILE_LEVEL = 0.80


def write_json(path, payload):
    """Write a UTF-8 JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def build_cost_benchmarks(df_train):
    """Create weighted national and age-group median cost benchmarks from training data."""
    required_columns = [WEIGHT_COLUMN, TARGET_COLUMN, "AGE23X"]
    df_benchmarks = df_train[required_columns].copy()
    if df_benchmarks.isna().any().any():
        raise ValueError("Training data for cost benchmarking contains missing values.")

    df_benchmarks["AGE_BENCHMARK_GROUP"] = pd.cut(
        df_benchmarks["AGE23X"],
        bins=AGE_BENCHMARK_BINS,
        labels=AGE_BENCHMARK_LABELS,
        right=False,
    )

    national_median_cost = weighted_quantile(
        df_benchmarks[TARGET_COLUMN],
        df_benchmarks[WEIGHT_COLUMN],
        0.5,
    )
    age_groups = []
    for age_group, df_group in df_benchmarks.groupby(
        "AGE_BENCHMARK_GROUP", observed=True
    ):
        median_cost = weighted_quantile(
            df_group[TARGET_COLUMN],
            df_group[WEIGHT_COLUMN],
            0.5,
        )
        age_groups.append(
            {
                "label": str(age_group),
                "median_cost": int(round(median_cost)),
            }
        )

    return {
        "schema_version": 1,
        "data_source": "MEPS 2023 (HC-251), training split",
        "currency_year": 2023,
        "national": {
            "label": "Typical American",
            "median_cost": int(round(national_median_cost)),
        },
        "age_groups": age_groups,
    }


def build_prediction_metadata(validation_weights, quantile_predictions):
    """Create the high-predicted-uncertainty cutoff from validation predictions."""
    if quantile_predictions.ndim != 2 or quantile_predictions.shape[1] != 4:
        raise ValueError("Expected validation predictions for q25, q50, q75, and q90.")
    if len(validation_weights) != len(quantile_predictions):
        raise ValueError("Validation weights and quantile predictions are misaligned.")

    q90_predictions = quantile_predictions[:, 3]  # 3 is index of q90 predictions
    cutoff = float(
        weighted_quantile(
            q90_predictions,
            validation_weights,
            HIGH_PREDICTED_UNCERTAINTY_QUANTILE_LEVEL,
        )
    )
    return {
        "schema_version": 1,
        "model_artifact": "models/xgb_quantile_model.joblib",
        "data_source": "MEPS 2023 (HC-251), validation split",
        "currency_year": 2023,
        "high_predicted_uncertainty": {
            "prediction_quantile": "q90",
            "weighted_quantile_level": HIGH_PREDICTED_UNCERTAINTY_QUANTILE_LEVEL,
            "cutoff_2023_dollars": cutoff,
        },
    }


def recreate_training_data():
    """Recreate the training split with unscaled age values from the raw MEPS 2023 (HC-251) data."""
    df = pd.read_sas(RAW_DATA_PATH, format="sas7bdat", encoding="latin1")
    df = df[[ID_COLUMN, WEIGHT_COLUMN, TARGET_COLUMN, "AGE23X"]]
    df = df[(df[WEIGHT_COLUMN] > 0) & (df["AGE23X"] >= 18)].copy()
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df = df.set_index(ID_COLUMN)
    df = df.replace(MEPS_MISSING_CODES, np.nan)

    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]
    y_strata = create_stratification_bins(y)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_strata,
    )
    return X_train[[WEIGHT_COLUMN, "AGE23X"]].assign(**{TARGET_COLUMN: y_train})


def main():
    df_train = recreate_training_data()
    validation_weights = pd.read_parquet(
        VAL_DATA_PATH,
        columns=[WEIGHT_COLUMN],
    )[WEIGHT_COLUMN].to_numpy()
    quantile_predictions = joblib.load(QUANTILE_PREDICTIONS_PATH)

    cost_benchmarks = build_cost_benchmarks(df_train)
    prediction_metadata = build_prediction_metadata(
        validation_weights,
        quantile_predictions,
    )
    write_json(COST_BENCHMARKS_PATH, cost_benchmarks)
    write_json(PREDICTION_METADATA_PATH, prediction_metadata)

    national_benchmark = cost_benchmarks["national"]
    cutoff = prediction_metadata["high_predicted_uncertainty"]["cutoff_2023_dollars"]
    print(
        f"Created '{COST_BENCHMARKS_PATH}' and "
        f"'{PREDICTION_METADATA_PATH}'."
    )
    print(
        f"{national_benchmark['label']} benchmark: "
        f"${national_benchmark['median_cost']:,.0f}"
    )
    for age_group in cost_benchmarks["age_groups"]:
        print(f"Ages {age_group['label']} benchmark: ${age_group['median_cost']:,.0f}")
    print(f"Weighted q90 cutoff for top 20%: ${cutoff:,.2f}")


if __name__ == "__main__":
    main()
