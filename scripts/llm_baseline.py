"""
Evaluate prediction performance of a LLM as a baseline and comparison benchmark.

This script benchmarks a zero-shot LLM model (Gemini) as a general-purpose AI model
against task-specific ML models. It uses the preprocessed validation data and
reports the same evaluation metrics used for baseline models.

Usage:
    GEMINI_API_KEY=... ./.venv-train/Scripts/python scripts/llm_baseline.py
"""

# Standard library imports
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.constants import TARGET_COLUMN, WEIGHT_COLUMN, RANDOM_STATE
from src.utils import save_metrics, save_model, weighted_median_absolute_error


VAL_DATA_PATH = "data/validation_data_preprocessed.parquet"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_ROWS = 100
DEFAULT_WORKERS = 1
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Gemini LLM baseline on validation data.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id.")
    parser.add_argument("--n-rows", type=int, default=DEFAULT_ROWS, help="Number of validation rows to benchmark.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed for reproducible row sampling.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel request workers. Use 1 for safest free-tier pacing.",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Maximum retries per row.")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging if local MLflow server is not running.",
    )
    return parser.parse_args()


def build_prompt(features_dict):
    """Build strict-JSON prediction prompt for a single row."""
    payload_json = json.dumps(features_dict, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are a regression model for US annual out-of-pocket medical costs.\n"
        "Given one person's features from the MEPS dataset, predict their annual out-of-pocket cost in USD.\n"
        "Return JSON only with exactly one key:\n"
        '{"predicted_out_of_pocket_cost_usd": <number>}\n'
        "Do not include any explanation or additional keys.\n\n"
        f"Features:\n{payload_json}"
    )


def _extract_json_object(text):
    """Extract the first JSON object from a text string."""
    if not text:
        raise ValueError("Empty model response.")

    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    return json.loads(text[start : end + 1])


def call_gemini(api_key, model, prompt, timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    """Call Gemini generateContent REST API and return parsed numeric prediction."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request_payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    candidates = response_data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned by Gemini: {response_data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise ValueError(f"No response parts returned by Gemini: {response_data}")

    text = parts[0].get("text", "")
    obj = _extract_json_object(text)
    pred = float(obj["predicted_out_of_pocket_cost_usd"])
    return max(0.0, pred)  # OOP costs cannot be negative


def predict_with_retry(api_key, model, prompt, max_retries):
    """Predict with retry/backoff for 429 and transient errors."""
    attempt = 0
    while True:
        try:
            return call_gemini(api_key=api_key, model=model, prompt=prompt)
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", None)
            if attempt >= max_retries:
                raise
            # Backoff for quota/temporary server issues
            if status in {429, 500, 502, 503, 504}:
                sleep_seconds = min(2**attempt, 30)
                time.sleep(sleep_seconds)
                attempt += 1
                continue
            raise
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(2**attempt, 30))
            attempt += 1


def evaluate_rows(sample_df, feature_columns, api_key, model, workers, max_retries):
    """Run LLM inference on sampled rows and return predictions with failure tracking."""
    prompts = []
    for row_idx, row in sample_df.iterrows():
        feature_values = row[feature_columns].to_dict()
        prompts.append((row_idx, build_prompt(feature_values)))

    predictions = {}
    failures = {}

    def _run_one(item):
        row_idx, prompt = item
        pred = predict_with_retry(api_key=api_key, model=model, prompt=prompt, max_retries=max_retries)
        return row_idx, pred

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_one, item): item[0] for item in prompts}
        for future in as_completed(future_map):
            row_idx = future_map[future]
            try:
                idx, pred = future.result()
                predictions[idx] = pred
            except Exception as exc:
                failures[row_idx] = str(exc)

    return predictions, failures


def main():
    """Main entrypoint for LLM baseline evaluation."""
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY environment variable.")

    print("Step 1: Loading validation data...")
    df_val = pd.read_parquet(VAL_DATA_PATH)
    if args.n_rows <= 0:
        raise ValueError("--n-rows must be a positive integer.")
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    n_rows = min(args.n_rows, len(df_val))
    sample_df = df_val.sample(n=n_rows, random_state=args.seed).copy()
    feature_columns = [col for col in sample_df.columns if col not in {TARGET_COLUMN, WEIGHT_COLUMN}]
    print(f"  Sampled {n_rows:,} validation rows with seed={args.seed}.")

    print("Step 2: Running Gemini baseline inference...")
    start = time.time()
    predictions, failures = evaluate_rows(
        sample_df=sample_df,
        feature_columns=feature_columns,
        api_key=api_key,
        model=args.model,
        workers=args.workers,
        max_retries=args.max_retries,
    )
    elapsed = time.time() - start
    success_count = len(predictions)
    fail_count = len(failures)
    print(f"  Finished inference in {elapsed:.2f}s | Success: {success_count} | Failed: {fail_count}")

    if success_count == 0:
        raise RuntimeError("No successful predictions were returned. Cannot compute metrics.")

    print("Step 3: Computing metrics on successful predictions...")
    pred_series = pd.Series(predictions).sort_index()
    eval_df = sample_df.loc[pred_series.index]
    y_true = eval_df[TARGET_COLUMN].to_numpy()
    y_pred = pred_series.to_numpy()
    w_val = eval_df[WEIGHT_COLUMN].to_numpy()

    val_mdae = weighted_median_absolute_error(y_true, y_pred, sample_weight=w_val)
    val_mae = mean_absolute_error(y_true, y_pred, sample_weight=w_val)
    val_r2 = r2_score(y_true, y_pred, sample_weight=w_val)

    metrics = {
        "Gemini 3 Flash (LLM Baseline)": {
            "model_id": args.model,
            "n_rows_requested": int(n_rows),
            "n_rows_scored": int(success_count),
            "n_rows_failed": int(fail_count),
            "val_mdae": float(val_mdae),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
            "inference_time_seconds": float(elapsed),
            "workers": int(args.workers),
            "seed": int(args.seed),
        }
    }

    print(
        f"  Gemini LLM Baseline → MdAE: {val_mdae:8.2f} | MAE: {val_mae:8.2f} | "
        f"R²: {val_r2:.4f} | Time: {elapsed:.2f}s"
    )

    print("Step 4: Persisting metrics and predictions...")
    metrics_path = "models/gemini_3_flash_llm_baseline_metrics.json"
    predictions_path = "models/gemini_3_flash_llm_baseline_predictions.joblib"
    failures_path = "models/gemini_3_flash_llm_baseline_failures.json"

    save_metrics(metrics, metrics_path, verbose=False)
    save_model(pred_series.to_dict(), predictions_path, verbose=False)
    save_metrics(failures, failures_path, verbose=False)
    print(f"  Saved metrics to '{metrics_path}'")
    print(f"  Saved predictions to '{predictions_path}'")
    print(f"  Saved failures to '{failures_path}'")

    if not args.disable_mlflow:
        print("Step 5: Logging to MLflow...")
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("LLM Baselines")
            with mlflow.start_run(run_name=f"Gemini LLM Baseline ({args.model})"):
                mlflow.log_params(
                    {
                        "provider": "google_gemini",
                        "model_id": args.model,
                        "n_rows_requested": n_rows,
                        "workers": args.workers,
                        "seed": args.seed,
                        "max_retries": args.max_retries,
                    }
                )
                mlflow.log_metrics(
                    {
                        "val_mdae": float(val_mdae),
                        "val_mae": float(val_mae),
                        "val_r2": float(val_r2),
                        "n_rows_scored": float(success_count),
                        "n_rows_failed": float(fail_count),
                        "inference_time_seconds": float(elapsed),
                    }
                )
            print("  Logged run to MLflow experiment 'LLM Baselines'")
        except Exception as exc:
            print(f"  Skipped MLflow logging due to error: {exc}")

    print("\n✅ Gemini LLM baseline evaluation complete.")


if __name__ == "__main__":
    main()
