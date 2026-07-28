"""Benchmark permutation SHAP configurations.

Compare candidate background sizes and permutation budgets with a larger
reference configuration on fixed validation rows. Evaluate background
representativeness, top-five explanation stability, contribution direction
and size, additivity, and warmed single-row latency.

Modes:
    smoke:
        Check the complete benchmark path on two rows. Print a compact
        diagnostic summary without saving results.

    stage1:
        Screen the full candidate grid on 20 validation rows. Save candidate
        results and reference timings in the models directory.

    stage2:
        Evaluate three shortlisted configurations on 100 validation rows that
        are separate from the Stage 1 rows. Save candidate results and
        reference timings in the models directory.

For the detailed benchmarking rationale and selection criteria, see the
"SHAP Benchmarking" section in notebooks/2_modeling.py.

Usage:
    .venv-train/Scripts/python scripts/benchmark_shap.py smoke
    .venv-train/Scripts/python scripts/benchmark_shap.py stage1
    .venv-train/Scripts/python scripts/benchmark_shap.py stage2
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
from src.modeling import (
    TRAIN_PREPROCESSOR_INPUT_DATA_PATH,
    VAL_PREPROCESSOR_INPUT_DATA_PATH,
    load_model,
    postprocess_quantile_predictions,
)


PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
MODEL_PATH = Path("models/xgb_quantile_model.joblib")
RESULTS_DIR = Path("models")

SHAP_INPUT_FEATURES = (
    PIPELINE_NUMERICAL_FEATURES
    + PIPELINE_NOMINAL_FEATURES
    + PIPELINE_BINARY_FEATURES
)
SHAP_MASKS_PER_ROUND = 2 * len(SHAP_INPUT_FEATURES) + 1
SHAP_BASELINE_REL_DIFF_MAX = 0.10

SHAP_TOP_K = 5
SHAP_MIN_TOP_5_MATCHES = 4
SHAP_MIN_TOP_5_MATCH_ROW_SHARE = 0.90
SHAP_MATERIAL_CONTRIBUTION_MIN_2023_USD = 25.0
SHAP_MEDIAN_TOP_5_ABS_DELTA_MAX_2023_USD = 25.0

SHAP_STAGE_1_ROWS = 20
SHAP_STAGE_2_ROWS = 100
SHAP_STAGE_1_CANDIDATES = [
    (background_size, rounds * SHAP_MASKS_PER_ROUND)
    for background_size in [225, 250, 275, 300]
    for rounds in [1, 2, 3]
]

# Fill this list with exactly three candidates after reviewing Stage 1.
SHAP_STAGE_2_CANDIDATES = [
    # (background_size, max_evals),
]

SHAP_REFERENCE_BACKGROUND_SIZE = 500
SHAP_REFERENCE_MAX_EVALS = 24 * SHAP_MASKS_PER_ROUND

# The smoke test checks the complete code path without running Stage 1.
SHAP_SMOKE_ROWS = 2
SHAP_SMOKE_CANDIDATES = [(300, SHAP_MASKS_PER_ROUND)]
SHAP_SMOKE_REFERENCE_BACKGROUND_SIZE = 500
SHAP_SMOKE_REFERENCE_MAX_EVALS = 2 * SHAP_MASKS_PER_ROUND

# Loaded once in main and used by the SHAP prediction function.
preprocessor = None
xgb_quantile_model = None
X_train_preprocessor_input = None
w_train = None
training_baseline = None
X_shap_first_explanation = None


def predict_median_cost(X):
    """Predict postprocessed q50 cost from ordered preprocessor input features."""
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


def calculate_shap_permutation_budget(max_evals):
    """Calculate complete permutation rounds and planned mask evaluations."""
    permutation_rounds = max_evals // SHAP_MASKS_PER_ROUND
    planned_mask_evaluations = permutation_rounds * SHAP_MASKS_PER_ROUND
    return permutation_rounds, planned_mask_evaluations


def create_and_validate_shap_background(background_size):
    """Create one weighted background and validate its q50 baseline."""
    background = X_train_preprocessor_input.sample(
        n=background_size,
        weights=w_train,
        replace=True,
        random_state=RANDOM_STATE,
    )
    background_baseline = predict_median_cost(background).mean()
    baseline_absolute_relative_difference = abs(
        background_baseline / training_baseline - 1
    )
    return {
        "background": background,
        "baseline_2023_usd": background_baseline,
        "baseline_absolute_relative_difference": (
            baseline_absolute_relative_difference
        ),
        "baseline_validation_passed": (
            baseline_absolute_relative_difference
            <= SHAP_BASELINE_REL_DIFF_MAX
        ),
    }


def build_shap_explainer(background):
    """Build a permutation SHAP explainer for one background sample."""
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


def measure_first_explanation_latency(explainer, max_evals):
    """Return seconds required for the first single-row SHAP explanation."""
    start_time = perf_counter()
    explainer(
        X_shap_first_explanation,
        max_evals=max_evals,
        silent=True,
    )
    return perf_counter() - start_time


def explain_and_time_rows(explainer, X_rows, max_evals):
    """Return 2023-dollar SHAP outputs and per-row explanation latency."""
    shap_values = []
    shap_base_values = []
    predicted_median_costs = []
    explanation_latencies_s = []

    for row_position in range(len(X_rows)):
        row_frame = X_rows.iloc[[row_position]]
        start_time = perf_counter()
        explanation = explainer(
            row_frame,
            max_evals=max_evals,
            silent=True,
        )
        explanation_latencies_s.append(perf_counter() - start_time)
        shap_values.append(explanation.values[0])
        shap_base_values.append(
            np.asarray(explanation.base_values).reshape(-1)[0]
        )
        predicted_median_costs.append(predict_median_cost(row_frame)[0])

    return (
        np.vstack(shap_values),
        np.asarray(shap_base_values),
        np.asarray(predicted_median_costs),
        np.asarray(explanation_latencies_s),
    )


def calculate_top_5_stability(shap_values, reference_shap_values):
    """Evaluate the user-facing top-five drivers against the reference."""
    overlap_counts = []
    matched_contribution_abs_differences = []
    material_direction_comparisons = 0
    material_direction_reversals = 0

    for row_position in range(reference_shap_values.shape[0]):
        reference_top_indices = np.argsort(
            np.abs(reference_shap_values[row_position])
        )[::-1][:SHAP_TOP_K]
        candidate_top_indices = np.argsort(
            np.abs(shap_values[row_position])
        )[::-1][:SHAP_TOP_K]
        matched_top_indices = np.intersect1d(
            reference_top_indices,
            candidate_top_indices,
        )

        overlap_counts.append(len(matched_top_indices))
        matched_contribution_abs_differences.extend(
            np.abs(
                shap_values[row_position, matched_top_indices]
                - reference_shap_values[
                    row_position,
                    matched_top_indices,
                ]
            )
        )

        material_matched_indices = matched_top_indices[
            np.abs(
                reference_shap_values[
                    row_position,
                    matched_top_indices,
                ]
            ) >= SHAP_MATERIAL_CONTRIBUTION_MIN_2023_USD
        ]
        material_direction_comparisons += len(material_matched_indices)
        material_direction_reversals += np.count_nonzero(
            np.sign(shap_values[row_position, material_matched_indices])
            != np.sign(
                reference_shap_values[
                    row_position,
                    material_matched_indices,
                ]
            )
        )

    overlap_counts = np.asarray(overlap_counts)
    match_row_share = np.mean(overlap_counts >= SHAP_MIN_TOP_5_MATCHES)
    median_abs_difference = (
        np.median(matched_contribution_abs_differences)
        if matched_contribution_abs_differences
        else np.nan
    )
    overlap_passed = match_row_share >= SHAP_MIN_TOP_5_MATCH_ROW_SHARE
    direction_passed = material_direction_reversals == 0
    dollar_difference_passed = (
        not np.isnan(median_abs_difference)
        and median_abs_difference
        <= SHAP_MEDIAN_TOP_5_ABS_DELTA_MAX_2023_USD
    )

    return {
        "share_rows_with_at_least_4_of_5_matches": match_row_share,
        "top_5_overlap_passed": overlap_passed,
        "material_direction_comparison_count": (
            material_direction_comparisons
        ),
        "material_direction_reversal_count": material_direction_reversals,
        "material_direction_passed": direction_passed,
        "median_matched_top_5_abs_delta_2023_usd": median_abs_difference,
        "dollar_difference_passed": dollar_difference_passed,
        "explanation_stability_passed": (
            overlap_passed
            and direction_passed
            and dollar_difference_passed
        ),
    }


def summarize_shap_configuration(
    *,
    background_size,
    max_evals,
    background_info,
    first_explanation_latency_s,
    shap_values,
    shap_base_values,
    predicted_median_costs,
    explanation_latencies_s,
    reference_shap_values,
):
    """Summarize one candidate's latency and stability metrics."""
    permutation_rounds, planned_mask_evaluations = (
        calculate_shap_permutation_budget(max_evals)
    )
    additivity_abs_error = np.abs(
        predicted_median_costs
        - (shap_base_values + shap_values.sum(axis=1))
    )

    return {
        "background_size": background_size,
        "max_evals": max_evals,
        "permutation_rounds": permutation_rounds,
        "planned_mask_evaluations": planned_mask_evaluations,
        "estimated_synthetic_rows": (
            planned_mask_evaluations * background_size
        ),
        "background_baseline_2023_usd": (
            background_info["baseline_2023_usd"]
        ),
        "background_baseline_absolute_relative_difference": (
            background_info["baseline_absolute_relative_difference"]
        ),
        "background_baseline_validation_passed": (
            background_info["baseline_validation_passed"]
        ),
        "first_explanation_latency_s": first_explanation_latency_s,
        "timed_row_latencies_s": explanation_latencies_s.tolist(),
        "p50_latency_s": np.percentile(explanation_latencies_s, 50),
        "p90_latency_s": np.percentile(explanation_latencies_s, 90),
        "p95_latency_s": np.percentile(explanation_latencies_s, 95),
        **calculate_top_5_stability(shap_values, reference_shap_values),
        "median_additivity_abs_error_2023_usd": np.median(
            additivity_abs_error
        ),
        "p95_additivity_abs_error_2023_usd": np.percentile(
            additivity_abs_error,
            95,
        ),
    }


def failed_background_result(background_size, max_evals, background_info):
    """Return a visible result row when background validation fails."""
    permutation_rounds, planned_mask_evaluations = (
        calculate_shap_permutation_budget(max_evals)
    )
    return {
        "background_size": background_size,
        "max_evals": max_evals,
        "permutation_rounds": permutation_rounds,
        "planned_mask_evaluations": planned_mask_evaluations,
        "estimated_synthetic_rows": (
            planned_mask_evaluations * background_size
        ),
        "background_baseline_2023_usd": (
            background_info["baseline_2023_usd"]
        ),
        "background_baseline_absolute_relative_difference": (
            background_info["baseline_absolute_relative_difference"]
        ),
        "background_baseline_validation_passed": False,
        "first_explanation_latency_s": np.nan,
        "timed_row_latencies_s": [],
        "p50_latency_s": np.nan,
        "p90_latency_s": np.nan,
        "p95_latency_s": np.nan,
        "share_rows_with_at_least_4_of_5_matches": np.nan,
        "top_5_overlap_passed": False,
        "material_direction_comparison_count": np.nan,
        "material_direction_reversal_count": np.nan,
        "material_direction_passed": False,
        "median_matched_top_5_abs_delta_2023_usd": np.nan,
        "dollar_difference_passed": False,
        "explanation_stability_passed": False,
        "median_additivity_abs_error_2023_usd": np.nan,
        "p95_additivity_abs_error_2023_usd": np.nan,
    }


def run_shap_benchmark(
    X_evaluation,
    candidate_configurations,
    *,
    reference_background_size,
    reference_max_evals,
):
    """Benchmark candidate configurations against one reference."""
    show_progress = len(candidate_configurations) > 1
    background_sizes = {
        background_size
        for background_size, _ in candidate_configurations
    }
    background_sizes.add(reference_background_size)
    if show_progress:
        print(
            f"Preparing and validating {len(background_sizes)} "
            "background samples..."
        )
    backgrounds_by_size = {
        background_size: create_and_validate_shap_background(
            background_size
        )
        for background_size in sorted(background_sizes)
    }

    reference_background_info = backgrounds_by_size[
        reference_background_size
    ]
    if not reference_background_info["baseline_validation_passed"]:
        raise ValueError(
            "The reference SHAP background failed baseline validation."
        )

    reference_rounds, _ = calculate_shap_permutation_budget(
        reference_max_evals
    )
    if show_progress:
        print(
            "Running reference configuration: "
            f"background={reference_background_size}, "
            f"rounds={reference_rounds}..."
        )
    reference_start_time = perf_counter()
    reference_explainer = build_shap_explainer(
        reference_background_info["background"]
    )
    reference_first_latency_s = measure_first_explanation_latency(
        reference_explainer,
        reference_max_evals,
    )
    (
        reference_shap_values,
        _,
        _,
        reference_latencies_s,
    ) = explain_and_time_rows(
        reference_explainer,
        X_evaluation,
        reference_max_evals,
    )

    if show_progress:
        print(
            "Reference complete "
            f"({perf_counter() - reference_start_time:.1f} s)."
        )

    benchmark_results = []
    candidate_count = len(candidate_configurations)
    for candidate_number, (background_size, max_evals) in enumerate(
        candidate_configurations,
        start=1,
    ):
        permutation_rounds, _ = calculate_shap_permutation_budget(
            max_evals
        )
        if show_progress:
            print(
                f"Candidate {candidate_number}/{candidate_count}: "
                f"background={background_size}, "
                f"rounds={permutation_rounds}..."
            )
        candidate_start_time = perf_counter()
        background_info = backgrounds_by_size[background_size]
        if not background_info["baseline_validation_passed"]:
            benchmark_results.append(
                failed_background_result(
                    background_size,
                    max_evals,
                    background_info,
                )
            )
            if show_progress:
                print(
                    "  Skipped: background baseline difference "
                    f"{background_info['baseline_absolute_relative_difference']:.1%} "
                    f"exceeds {SHAP_BASELINE_REL_DIFF_MAX:.0%}."
                )
            continue

        candidate_explainer = build_shap_explainer(
            background_info["background"]
        )
        first_latency_s = measure_first_explanation_latency(
            candidate_explainer,
            max_evals,
        )
        (
            candidate_shap_values,
            candidate_shap_base_values,
            candidate_predictions,
            candidate_latencies_s,
        ) = explain_and_time_rows(
            candidate_explainer,
            X_evaluation,
            max_evals,
        )
        benchmark_results.append(
            summarize_shap_configuration(
                background_size=background_size,
                max_evals=max_evals,
                background_info=background_info,
                first_explanation_latency_s=first_latency_s,
                shap_values=candidate_shap_values,
                shap_base_values=candidate_shap_base_values,
                predicted_median_costs=candidate_predictions,
                explanation_latencies_s=candidate_latencies_s,
                reference_shap_values=reference_shap_values,
            )
        )

        if show_progress:
            print(
                "  Complete "
                f"({perf_counter() - candidate_start_time:.1f} s)."
            )

    benchmark_results = pd.DataFrame(benchmark_results).sort_values(
        [
            "background_baseline_validation_passed",
            "explanation_stability_passed",
            "p95_latency_s",
            "share_rows_with_at_least_4_of_5_matches",
        ],
        ascending=[False, False, True, False],
        na_position="last",
    )
    reference_summary = {
        "background_size": reference_background_size,
        "max_evals": reference_max_evals,
        "background_baseline_2023_usd": (
            reference_background_info["baseline_2023_usd"]
        ),
        "background_baseline_absolute_relative_difference": (
            reference_background_info[
                "baseline_absolute_relative_difference"
            ]
        ),
        "first_explanation_latency_s": reference_first_latency_s,
        "p50_latency_s": np.percentile(reference_latencies_s, 50),
        "p90_latency_s": np.percentile(reference_latencies_s, 90),
        "p95_latency_s": np.percentile(reference_latencies_s, 95),
    }
    return benchmark_results, pd.DataFrame([reference_summary])


def select_benchmark_mode(mode, X_stage_1, X_stage_2):
    """Return evaluation rows and configurations for one CLI mode."""
    if mode == "smoke":
        return (
            X_stage_1.iloc[:SHAP_SMOKE_ROWS],
            SHAP_SMOKE_CANDIDATES,
            SHAP_SMOKE_REFERENCE_BACKGROUND_SIZE,
            SHAP_SMOKE_REFERENCE_MAX_EVALS,
        )
    if mode == "stage1":
        return (
            X_stage_1,
            SHAP_STAGE_1_CANDIDATES,
            SHAP_REFERENCE_BACKGROUND_SIZE,
            SHAP_REFERENCE_MAX_EVALS,
        )
    if len(SHAP_STAGE_2_CANDIDATES) != 3:
        raise ValueError(
            "Set SHAP_STAGE_2_CANDIDATES to exactly three "
            "(background_size, max_evals) pairs after reviewing Stage 1."
        )
    return (
        X_stage_2,
        SHAP_STAGE_2_CANDIDATES,
        SHAP_REFERENCE_BACKGROUND_SIZE,
        SHAP_REFERENCE_MAX_EVALS,
    )


def parse_args():
    """Parse the requested benchmark mode."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["smoke", "stage1", "stage2"],
        help="Run a smoke test, Stage 1 screen, or Stage 2 validation.",
    )
    return parser.parse_args()


def print_smoke_results(results):
    """Print a compact smoke-test checklist."""
    candidate = results.iloc[0]
    matched_rows = round(
        candidate["share_rows_with_at_least_4_of_5_matches"]
        * SHAP_SMOKE_ROWS
    )
    timed_row_latencies_ms = [
        latency_s * 1_000
        for latency_s in candidate["timed_row_latencies_s"]
    ]

    print("\nCandidate")
    print(f"  Background rows:                 {int(candidate['background_size']):>5}")
    print(f"  Permutation rounds:              {int(candidate['permutation_rounds']):>5}")
    print(f"  Evaluation rows:                 {SHAP_SMOKE_ROWS:>5}")

    print("\nChecks")
    print(
        "  Background baseline difference: "
        f"{candidate['background_baseline_absolute_relative_difference']:>6.1%}  "
        f"{'PASS' if candidate['background_baseline_validation_passed'] else 'FAIL'}"
    )
    print(
        "  Top-five overlap:                "
        f"{matched_rows}/{SHAP_SMOKE_ROWS} rows  "
        f"{'PASS' if candidate['top_5_overlap_passed'] else 'FAIL'}"
    )
    print(
        "  Material direction reversals:    "
        f"{int(candidate['material_direction_reversal_count']):>5}  "
        f"{'PASS' if candidate['material_direction_passed'] else 'FAIL'}"
    )
    print(
        "  Median contribution difference:  "
        f"${candidate['median_matched_top_5_abs_delta_2023_usd']:,.2f}  "
        f"{'PASS' if candidate['dollar_difference_passed'] else 'FAIL'}"
    )
    print(
        "  Median additivity error:          "
        f"${candidate['median_additivity_abs_error_2023_usd']:,.2f}  "
        f"{'PASS' if candidate['median_additivity_abs_error_2023_usd'] < 0.01 else 'FAIL'}"
    )

    print("\nLatency after one warm-up call (diagnostic only)")
    print(
        "  Timed rows: "
        + ", ".join(
            f"{latency_ms:,.0f} ms"
            for latency_ms in timed_row_latencies_ms
        )
    )
    print(f"  P50:        {candidate['p50_latency_s'] * 1_000:,.0f} ms")
    print("\nSHAP benchmark smoke test passed.")


def main():
    """Load artifacts, select fixed rows, and run one benchmark mode."""
    global preprocessor
    global xgb_quantile_model
    global X_train_preprocessor_input
    global w_train
    global training_baseline
    global X_shap_first_explanation

    args = parse_args()

    # SHAP repeatedly sends expected missing values through the fitted
    # preprocessor. Suppress those warnings only in this benchmark process.
    logging.getLogger("src.transformers").setLevel(logging.ERROR)

    if len(SHAP_INPUT_FEATURES) != len(set(SHAP_INPUT_FEATURES)):
        raise ValueError("SHAP_INPUT_FEATURES contains duplicate names.")

    print("Loading preprocessor-input data and fitted artifacts...")
    df_train = pd.read_parquet(
        TRAIN_PREPROCESSOR_INPUT_DATA_PATH,
        columns=SHAP_INPUT_FEATURES + [WEIGHT_COLUMN],
    )
    df_validation = pd.read_parquet(
        VAL_PREPROCESSOR_INPUT_DATA_PATH,
        columns=SHAP_INPUT_FEATURES,
    )
    X_train_preprocessor_input = df_train.loc[:, SHAP_INPUT_FEATURES]
    w_train = df_train[WEIGHT_COLUMN]
    preprocessor = load_model(PREPROCESSOR_PATH, verbose=False)
    xgb_quantile_model = load_model(MODEL_PATH, verbose=False)

    training_baseline = np.average(
        predict_median_cost(X_train_preprocessor_input),
        weights=w_train,
    )

    required_rows = 1 + SHAP_STAGE_1_ROWS + SHAP_STAGE_2_ROWS
    if len(df_validation) < required_rows:
        raise ValueError(
            f"Validation data must contain at least {required_rows} rows."
        )
    validation_sample = df_validation.sample(
        n=required_rows,
        random_state=RANDOM_STATE,
    )
    X_shap_first_explanation = validation_sample.iloc[[0]]
    stage_2_start = 1 + SHAP_STAGE_1_ROWS
    X_stage_1 = validation_sample.iloc[1:stage_2_start]
    X_stage_2 = validation_sample.iloc[stage_2_start:]

    (
        X_evaluation,
        candidate_configurations,
        reference_background_size,
        reference_max_evals,
    ) = select_benchmark_mode(args.mode, X_stage_1, X_stage_2)

    if args.mode == "smoke":
        print("Running SHAP smoke test...")
    else:
        print(
            f"Running SHAP {args.mode} benchmark on {len(X_evaluation)} rows "
            f"and {len(candidate_configurations)} candidate(s)..."
        )
    results, reference = run_shap_benchmark(
        X_evaluation,
        candidate_configurations,
        reference_background_size=reference_background_size,
        reference_max_evals=reference_max_evals,
    )

    if args.mode == "smoke":
        if results["first_explanation_latency_s"].isna().all():
            raise RuntimeError(
                "Smoke test did not run an explanation because the candidate "
                "background failed validation."
            )
        print_smoke_results(results)
        return

    results = results.drop(columns=["timed_row_latencies_s"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"shap_benchmark_{args.mode}_results.csv"
    reference_path = (
        RESULTS_DIR / f"shap_benchmark_{args.mode}_reference.csv"
    )
    results.to_csv(results_path, index=False)
    reference.to_csv(reference_path, index=False)

    print("\nReference configuration:")
    print(reference.to_string(index=False))
    print("\nCandidate results:")
    print(results.to_string(index=False))
    print(f"\nSaved results to {results_path}")
    print(f"Saved reference timing to {reference_path}")


if __name__ == "__main__":
    main()

