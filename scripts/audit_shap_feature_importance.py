"""Calculate global SHAP feature importance on the held-out test set.

Use the selected SHAP configuration: 225 background rows and one 
permutation round (max_evals=55). Explain test rows one at a time, 
then calculate survey-weighted mean absolute SHAP contributions 
and each feature's share of total importance.

Modes:
    smoke:
        Explain two test rows and print a short validation summary without
        saving artifacts.

    full:
        Explain the complete test set and save:

            models/shap_test_contributions.parquet
            models/shap_feature_importance_test.csv

        The Parquet file retains row-level contributions for later SHAP
        distribution plots. The CSV contains the aggregated feature-importance
        results loaded by the modeling notebook.

Usage:
    .venv-train/Scripts/python scripts/audit_shap_feature_importance.py smoke
    .venv-train/Scripts/python scripts/audit_shap_feature_importance.py full
"""

from argparse import ArgumentParser
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import shap

from src.constants import (
    PIPELINE_BINARY_FEATURES,
    PIPELINE_NOMINAL_FEATURES,
    PIPELINE_NUMERICAL_FEATURES,
    RANDOM_STATE,
    WEIGHT_COLUMN,
)
from src.display import DISPLAY_LABELS
from src.modeling import (
    TEST_PREPROCESSOR_INPUT_DATA_PATH,
    TRAIN_PREPROCESSOR_INPUT_DATA_PATH,
    load_model,
    postprocess_quantile_predictions,
)


PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
MODEL_PATH = Path("models/xgb_quantile_model.joblib")
CONTRIBUTIONS_PATH = Path("models/shap_test_contributions.parquet")
IMPORTANCE_PATH = Path("models/shap_feature_importance_test.csv")

SHAP_INPUT_FEATURES = (
    PIPELINE_NUMERICAL_FEATURES
    + PIPELINE_NOMINAL_FEATURES
    + PIPELINE_BINARY_FEATURES
)
SHAP_BACKGROUND_SIZE = 225
SHAP_PERMUTATION_ROUNDS = 1
SHAP_MASKS_PER_ROUND = 2 * len(SHAP_INPUT_FEATURES) + 1
SHAP_MAX_EVALS = SHAP_PERMUTATION_ROUNDS * SHAP_MASKS_PER_ROUND
SHAP_BASELINE_REL_DIFF_MAX = 0.10
SHAP_SMOKE_ROWS = 2

preprocessor = None
xgb_quantile_model = None


def parse_args():
    """Parse the requested audit mode."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full"))
    return parser.parse_args()


def predict_median_cost(X):
    """Predict postprocessed q50 cost from preprocessor input features."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=SHAP_INPUT_FEATURES)
    else:
        missing_features = [
            feature
            for feature in SHAP_INPUT_FEATURES
            if feature not in X.columns
        ]
        if missing_features:
            raise ValueError(
                "SHAP input is missing preprocessor input features: "
                f"{missing_features}"
            )
        X = X.loc[:, SHAP_INPUT_FEATURES]

    X_model_ready = preprocessor.transform(X)
    quantile_predictions = xgb_quantile_model.predict(X_model_ready)
    return postprocess_quantile_predictions(quantile_predictions)[:, 1]


def create_and_validate_background(X_train, w_train):
    """Create the background data and validate its baseline."""
    background = X_train.sample(
        n=SHAP_BACKGROUND_SIZE,
        weights=w_train,
        replace=True,
        random_state=RANDOM_STATE,
    )
    training_baseline = np.average(
        predict_median_cost(X_train),
        weights=w_train,
    )
    background_baseline = predict_median_cost(background).mean()
    baseline_difference = abs(background_baseline / training_baseline - 1)

    print(f"Background rows:                  {SHAP_BACKGROUND_SIZE}")
    print(f"Permutation rounds:              {SHAP_PERMUTATION_ROUNDS}")
    print(f"Background vs. training:         {baseline_difference:.1%}")

    if baseline_difference > SHAP_BASELINE_REL_DIFF_MAX:
        raise ValueError(
            "SHAP background baseline differs from the weighted training "
            f"baseline by {baseline_difference:.1%}, which exceeds the "
            f"{SHAP_BASELINE_REL_DIFF_MAX:.0%} acceptance threshold."
        )
    return background


def build_explainer(background):
    """Build the permutation SHAP explainer."""
    masker = shap.maskers.Independent(
        background,
        max_samples=len(background),
    )
    return shap.Explainer(
        predict_median_cost,
        masker,
        algorithm="permutation",
        seed=RANDOM_STATE,
    )


def calculate_contributions(explainer, X):
    """Explain rows individually and return contributions and validation data."""
    contributions = np.empty((len(X), len(SHAP_INPUT_FEATURES)))
    base_values = np.empty(len(X))
    predictions = predict_median_cost(X)
    start_time = perf_counter()
    progress_interval = 1 if len(X) <= SHAP_SMOKE_ROWS else 25

    for row_position in range(len(X)):
        explanation = explainer(
            X.iloc[[row_position]],
            max_evals=SHAP_MAX_EVALS,
            silent=True,
        )
        contributions[row_position] = explanation.values[0]
        base_values[row_position] = explanation.base_values[0]

        completed_rows = row_position + 1
        if (
            completed_rows % progress_interval == 0
            or completed_rows == len(X)
        ):
            elapsed_seconds = perf_counter() - start_time
            seconds_per_row = elapsed_seconds / completed_rows
            remaining_seconds = seconds_per_row * (len(X) - completed_rows)
            print(
                f"Explained {completed_rows}/{len(X)} rows "
                f"({elapsed_seconds / 60:.1f} min elapsed, "
                f"about {remaining_seconds / 60:.1f} min remaining)",
                flush=True,
            )

    additivity_error = np.abs(
        base_values + contributions.sum(axis=1) - predictions
    )
    return contributions, base_values, predictions, additivity_error


def summarize_shap_feature_importance(contributions, weights) -> pd.DataFrame:
    """Return ranked survey-weighted SHAP feature importance as a DataFrame."""
    mean_absolute_contribution = np.average(
        np.abs(contributions),
        axis=0,
        weights=weights,
    )
    importance_df = pd.DataFrame({
        "feature": SHAP_INPUT_FEATURES,
        "feature_label": [
            DISPLAY_LABELS.get(feature, feature)
            for feature in SHAP_INPUT_FEATURES
        ],
        "mean_absolute_contribution_2023_usd": (
            mean_absolute_contribution
        ),
    })
    importance_df["share_of_total_importance"] = (
        importance_df["mean_absolute_contribution_2023_usd"]
        / importance_df["mean_absolute_contribution_2023_usd"].sum()
    )
    importance_df = importance_df.sort_values(
        "mean_absolute_contribution_2023_usd",
        ascending=False,
    ).reset_index(drop=True)
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    return importance_df


def create_shap_contribution_dataframe(
    X,
    weights,
    contributions,
    base_values,
    predictions,
    additivity_error,
) -> pd.DataFrame:
    """Return row-level SHAP contributions and audit fields as a DataFrame."""
    contributions_df = pd.DataFrame(
        contributions,
        index=X.index,
        columns=SHAP_INPUT_FEATURES,
    )
    contributions_df.insert(0, WEIGHT_COLUMN, weights.to_numpy())
    contributions_df.insert(1, "shap_baseline_2023_usd", base_values)
    contributions_df.insert(2, "predicted_q50_2023_usd", predictions)
    contributions_df.insert(3, "additivity_error_2023_usd", additivity_error)
    return contributions_df


def main():
    """Run the requested SHAP feature-importance audit mode."""
    global preprocessor
    global xgb_quantile_model

    args = parse_args()

    # SHAP sends expected missing values through the fitted preprocessor.
    # Suppress those expected warnings only in this offline audit process.
    logging.getLogger("src.transformers").setLevel(logging.ERROR)

    if len(SHAP_INPUT_FEATURES) != len(set(SHAP_INPUT_FEATURES)):
        raise ValueError("SHAP_INPUT_FEATURES contains duplicate names.")

    print("Loading preprocessor-input data (train & test) and fitted artifacts (preprocessor & model)...")
    df_train = pd.read_parquet(
        TRAIN_PREPROCESSOR_INPUT_DATA_PATH,
        columns=SHAP_INPUT_FEATURES + [WEIGHT_COLUMN],
    )
    df_test = pd.read_parquet(
        TEST_PREPROCESSOR_INPUT_DATA_PATH,
        columns=SHAP_INPUT_FEATURES + [WEIGHT_COLUMN],
    )
    X_train = df_train.loc[:, SHAP_INPUT_FEATURES]
    w_train = df_train[WEIGHT_COLUMN]
    X_test = df_test.loc[:, SHAP_INPUT_FEATURES]
    w_test = df_test[WEIGHT_COLUMN]

    if args.mode == "smoke":
        smoke_index = X_test.sample(
            n=SHAP_SMOKE_ROWS,
            random_state=RANDOM_STATE,
        ).index
        X_evaluation = X_test.loc[smoke_index]
        w_evaluation = w_test.loc[smoke_index]
    else:
        X_evaluation = X_test
        w_evaluation = w_test

    preprocessor = load_model(PREPROCESSOR_PATH, verbose=False)
    xgb_quantile_model = load_model(MODEL_PATH, verbose=False)

    background = create_and_validate_background(X_train, w_train)
    explainer = build_explainer(background)

    print(f"Explaining {len(X_evaluation)} test rows...")
    (
        contributions,
        base_values,
        predictions,
        additivity_error,
    ) = calculate_contributions(explainer, X_evaluation)
    importance_df = summarize_shap_feature_importance(
        contributions,
        w_evaluation.to_numpy(),
    )

    print(
        "Maximum additivity error:         "
        f"${additivity_error.max():,.10f}"
    )

    if args.mode == "smoke":
        print("\nTop SHAP features in smoke test:")
        print(
            importance_df.head(5).to_string(
                index=False,
                formatters={
                    "mean_absolute_contribution_2023_usd": (
                        lambda value: f"${value:,.2f}"
                    ),
                    "share_of_total_importance": (
                        lambda value: f"{value:.1%}"
                    ),
                },
            )
        )
        print("\nSHAP feature-importance smoke test passed.")
        return

    contributions_df = create_shap_contribution_dataframe(
        X_evaluation,
        w_evaluation,
        contributions,
        base_values,
        predictions,
        additivity_error,
    )
    CONTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    contributions_df.to_parquet(CONTRIBUTIONS_PATH)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    print("\nTop 15 SHAP features:")
    print(
        importance_df.head(15).to_string(
            index=False,
            formatters={
                "mean_absolute_contribution_2023_usd": (
                    lambda value: f"${value:,.2f}"
                ),
                "share_of_total_importance": (
                    lambda value: f"{value:.1%}"
                ),
            },
        )
    )
    print(f"\nSaved row-level contributions to {CONTRIBUTIONS_PATH}")
    print(f"Saved feature importance to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
