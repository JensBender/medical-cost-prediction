"""Calculate global SHAP feature importance on the held-out test set.

Load the production SHAP background and selected explainer configuration
created by `benchmark_shap.py`. Explain test rows one at a time, then
calculate survey-weighted mean absolute SHAP contributions and each feature's
share of total importance.

Use the shared prediction and explanation functions, resetting the permutation
seed for every row so its explanation does not depend on earlier audit rows.

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
    load_model,
    load_metrics,
)
from src.prediction import CostPredictor
from src.explainability import (
    build_shap_explainer,
    calculate_max_evals,
    calculate_shap_explanation,
)


PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
MODEL_PATH = Path("models/xgb_quantile_model.joblib")
SHAP_BACKGROUND_PATH = Path("app/data/shap_background.joblib")
SHAP_METADATA_PATH = Path("app/data/shap_metadata.json")
CONTRIBUTIONS_PATH = Path("models/shap_test_contributions.parquet")
IMPORTANCE_PATH = Path("models/shap_feature_importance_test.csv")

SHAP_INPUT_FEATURES = (
    PIPELINE_NUMERICAL_FEATURES
    + PIPELINE_NOMINAL_FEATURES
    + PIPELINE_BINARY_FEATURES
)
SHAP_SMOKE_ROWS = 2

predictor = None


def parse_args():
    """Parse the requested audit mode."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full"))
    return parser.parse_args()


def calculate_contributions(explainer, X, max_evals):
    """Explain rows individually and return contributions and validation data."""
    contributions = np.empty((len(X), len(SHAP_INPUT_FEATURES)))
    base_values = np.empty(len(X))
    predictions = predictor.predict_median_cost(X)
    start_time = perf_counter()
    progress_interval = 1 if len(X) <= SHAP_SMOKE_ROWS else 25

    for row_position in range(len(X)):
        explanation = calculate_shap_explanation(
            explainer,
            X.iloc[[row_position]],
            max_evals=max_evals,
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
    global predictor

    args = parse_args()

    # SHAP sends expected missing values through the fitted preprocessor.
    # Suppress those expected warnings only in this offline audit process.
    logging.getLogger("src.transformers").setLevel(logging.ERROR)

    if len(SHAP_INPUT_FEATURES) != len(set(SHAP_INPUT_FEATURES)):
        raise ValueError("SHAP_INPUT_FEATURES contains duplicate names.")

    print("Loading test data and production SHAP artifacts...")
    df_test = pd.read_parquet(
        TEST_PREPROCESSOR_INPUT_DATA_PATH,
        columns=SHAP_INPUT_FEATURES + [WEIGHT_COLUMN],
    )
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
    background = load_model(SHAP_BACKGROUND_PATH, verbose=False)
    shap_metadata = load_metrics(SHAP_METADATA_PATH, verbose=False)
    predictor = CostPredictor(preprocessor, xgb_quantile_model, SHAP_INPUT_FEATURES)

    expected_rows = shap_metadata["background_sample"]["rows"]
    expected_feature_count = shap_metadata["background_sample"]["feature_count"]
    if list(background.columns) != SHAP_INPUT_FEATURES:
        raise ValueError(
            "Production SHAP background columns do not match SHAP_INPUT_FEATURES."
        )
    if background.shape[1] != expected_feature_count:
        raise ValueError(
            "Production SHAP background feature count does not match its metadata."
        )
    if len(background) != expected_rows:
        raise ValueError(
            "Production SHAP background row count does not match its metadata."
        )
    if not shap_metadata["final_test_evaluation"]["passed"]:
        raise ValueError("Production SHAP background did not pass final evaluation.")

    permutation_rounds = shap_metadata["explainer_contract"]["permutation_rounds"]
    max_evals = calculate_max_evals(
        permutation_rounds,
        len(SHAP_INPUT_FEATURES),
    )
    if max_evals != shap_metadata["explainer_contract"]["max_evals"]:
        raise ValueError("SHAP max_evals does not match its metadata.")

    background_baseline = predictor.predict_median_cost(background).mean()
    if not np.isclose(
        background_baseline,
        shap_metadata["background_validation"]["background_baseline_2023_usd"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "Production SHAP background baseline does not match its metadata."
        )

    print(f"Background rows:                  {len(background)}")
    print(f"Permutation rounds:              {permutation_rounds}")
    explainer = build_shap_explainer(predictor, background)

    print(f"Explaining {len(X_evaluation)} test rows...")
    (
        contributions,
        base_values,
        predictions,
        additivity_error,
    ) = calculate_contributions(explainer, X_evaluation, max_evals)
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
