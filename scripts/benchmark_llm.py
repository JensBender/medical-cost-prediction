"""
LLM benchmark for medical cost prediction.

Compares specialized ML models against a general-purpose LLM to quantify
the added value of training a domain-specific model ("Why not just ask Gemini?").
This script uses a "High-Bar" approach: the LLM is given expert-level instructions 
and definitions to test its maximum potential, proving that a specialized model adds 
value even against an well instructed LLM.

Approach:
  1.  Data Preprocessing (Partial): Load raw MEPS SAS data, apply cleaning steps 1-7
      (mirroring preprocess.py), then filter to validation set rows by DUPERSID.
      This recovers human-readable feature values (e.g., Age=42, Region=South)
      from the already-preprocessed parquet which contains scaled/encoded values.
  2.  Profile Generation: Convert each row into a natural language description
      that a layperson would provide to an LLM (zero-shot, no training examples).
  3.  Batched LLM Inference: Send profiles in batches to the Gemini API with
      JSON-mode output for reliable parsing. Includes rate limiting and retry logic.
  4.  Metric Computation: Compute the same weighted metrics (MdAE, MAE, R²) used
      by the ML models, on the same validation rows, for an apples-to-apples comparison.
  5.  Persistence: Save metrics as JSON and predictions as joblib (same convention
      as ML model artifacts).

Prerequisites:
    pip install google-genai

Usage:
    1. Create a .env file with GEMINI_API_KEY=your_key
    2. Run: ./.venv-train/Scripts/python scripts/benchmark_llm.py
"""

# Standard library imports
import os
import re
import sys
import time
import json

# Third-party imports
import numpy as np
import pandas as pd
import mlflow
import warnings
from google import genai
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress benign MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Local imports
from src.constants import (
    ID_COLUMN, WEIGHT_COLUMN, TARGET_COLUMN,
    RAW_COLUMNS_TO_KEEP, RAW_BINARY_FEATURES,
    MEPS_MISSING_CODES,
    MARRY31X_TRANSITION_CODES, EMPST31_TRANSITION_CODES,
    MARRY31X_COLLAPSE_MAP, EMPST31_COLLAPSE_MAP,
)
from src.utils import weighted_median_absolute_error, save_metrics, save_model

# Load environment variables from .env file
load_dotenv()


# =========================
# Configuration
# =========================

LLM_MODEL = "gemini-3.1-flash-lite"
BATCH_SIZE = 25       # User profiles per API call (fits well within context window)
DELAY_SECONDS = 20    # Seconds between API calls (~3 RPM, safely under 5 RPM free-tier limit)
MAX_ATTEMPTS = 5      # Maximum times to try API call before giving up

# Paths (relative to project root)
RAW_DATA_PATH = "data/h251.sas7bdat"
VAL_DATA_PATH = "data/validation_data_preprocessed.parquet"


# =========================
# Human-Readable Label Maps
# =========================

SEX_LABELS = {1: "Male", 0: "Female"}
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
MARITAL_LABELS = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married"}
INCOME_LABELS = {1: "Poor/Negative", 2: "Near Poor", 3: "Low Income", 4: "Middle Income", 5: "High Income"}
EDUCATION_LABELS = {1: "No Degree", 2: "GED", 3: "High School Diploma", 4: "Bachelor's Degree", 5: "Master's Degree", 6: "Doctorate", 7: "Other Degree"}
INSURANCE_LABELS = {1: "Private Insurance", 2: "Public Insurance Only (Medicare/Medicaid)", 3: "Uninsured"}
EMPLOYMENT_LABELS = {1: "Employed", 0: "Not Employed"}
HEALTH_SCALE = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
YES_NO = {1: "Yes", 0: "No"}

CHRONIC_CONDITIONS = {
    "HIBPDX": "High Blood Pressure",
    "CHOLDX": "High Cholesterol",
    "DIABDX_M18": "Diabetes",
    "CHDDX": "Coronary Heart Disease",
    "STRKDX": "Stroke",
    "CANCERDX": "Cancer",
    "ARTHDX": "Arthritis",
    "ASTHDX": "Asthma",
}

FUNCTIONAL_LIMITATIONS = {
    "ADLHLP31": "Needs help with personal care (bathing, dressing)",
    "IADLHP31": "Needs help with daily tasks (bills, medications, shopping)",
    "WLKLIM31": "Difficulty walking or climbing stairs",
    "COGLIM31": "Difficulty concentrating, remembering, or making decisions",
    "JTPAIN31_M18": "Joint pain, aching, or stiffness",
}


# =========================
# System Prompt
# =========================

# Ensures LLM and the domain-specifc ML model solve the same problem by defining costs explicitly.
# This sets a higher bar compared to real LLM chatbot usage by providing expert-level clarity in prompt.
SYSTEM_PROMPT = """\
You are a healthcare cost estimation expert for the United States.

You will be given demographic and health profiles of US adults. For each profile, \
predict their total annual out-of-pocket healthcare costs for the year 2023 in US dollars.

Out-of-pocket costs include deductibles, copays, and coinsurance for: \
office visits, prescriptions, hospital stays, ER visits, dental, vision, \
home health care, and medical equipment.
Out-of-pocket costs EXCLUDE monthly insurance premiums and over-the-counter medications.

For each profile, provide your best single-number estimate (in dollars), 
returned in the requested list format."""


# =========================
# Data Preparation
# =========================

def prepare_human_readable_validation_data():
    """
    Recover human-readable feature values for the validation set.

    The saved parquet contains scaled/encoded features (after StandardScaler and
    OneHotEncoder). This function reloads the raw MEPS SAS file, applies the same
    cleaning steps 1-7 as preprocess.py (but NOT the sklearn pipeline), then filters
    to only the validation set rows by matching DUPERSID indices.

    Returns:
        tuple: (df_raw_val, y_val, w_val) where df_raw_val has human-readable
               feature values, y_val is the target, and w_val are sample weights.
               All aligned by DUPERSID index in parquet row order.
    """
    # Load preprocessed validation data to get row IDs, target, and weights
    df_val = pd.read_parquet(VAL_DATA_PATH)
    val_ids = set(df_val.index.astype(str))
    y_val = df_val[TARGET_COLUMN]
    w_val = df_val[WEIGHT_COLUMN]

    # --- Data Preparation (mirrors preprocess.py steps 1-7) ---
    print("Preparing human-readable validation data...")
    # Step 1: Load raw MEPS data
    print("  Loading raw MEPS SAS data...")
    df = pd.read_sas(RAW_DATA_PATH, format="sas7bdat", encoding="latin1")

    # Step 2: Variable selection
    print("  Selecting variables...")
    df = df[RAW_COLUMNS_TO_KEEP]

    # Step 3: Population filtering (adults with positive weights)
    print("  Filtering target population...")
    df = df[(df[WEIGHT_COLUMN] > 0) & (df["AGE23X"] >= 18)].copy()

    # Step 4: Data type handling
    print("  Handling data types...")
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df.set_index(ID_COLUMN, inplace=True)

    # Step 5: Missing value standardization
    print("  Standardizing missing values...")
    # Recover implied values from survey skip patterns
    df.loc[df["ADSMOK42"] == -1, "ADSMOK42"] = 2    # -1 "Never Smoker" → 2 "No"
    df.loc[(df["JTPAIN31_M18"] == -1) & (df["ARTHDX"] == 1), "JTPAIN31_M18"] = 1
    # Convert remaining MEPS codes to NaN
    df.replace(MEPS_MISSING_CODES, np.nan, inplace=True)

    # Step 6: Binary standardization (MEPS 1/2 → 1/0)
    print("  Standardizing binary features...")
    df[RAW_BINARY_FEATURES] = df[RAW_BINARY_FEATURES].replace({2: 0})

    # Step 7: Feature engineering (stateless)
    print("  Engineering stateless features...")
    df["RECENT_LIFE_TRANSITION"] = (
        df["MARRY31X"].isin(MARRY31X_TRANSITION_CODES) | df["EMPST31"].isin(EMPST31_TRANSITION_CODES)
    ).astype(float)
    df.loc[df["MARRY31X"].isna() & df["EMPST31"].isna(), "RECENT_LIFE_TRANSITION"] = np.nan
    df["MARRY31X_GRP"] = df["MARRY31X"].replace(MARRY31X_COLLAPSE_MAP)
    df["EMPST31_GRP"] = df["EMPST31"].replace(EMPST31_COLLAPSE_MAP)

    # Filter to validation set rows and align to preprocessed data row order
    print("  Filtering rows to match preprocessed validation data...")
    df_raw_val = df.loc[df.index.isin(val_ids)].reindex(y_val.index)
    n_matched = df_raw_val.index.isin(val_ids).sum()
    n_complete = df_raw_val.notna().all(axis=1).sum()
    print(f"  Matched {n_matched:,} out of {len(val_ids):,} rows of the preprocessed validation data ({n_complete:,} complete, {n_matched - n_complete:,} with missing values)")

    return df_raw_val, y_val, w_val


# =========================
# Profile Generation
# =========================

def row_to_profile(row):
    """
    Convert a single row of cleaned (pre-pipeline) data to a natural language profile
    that we feed as input to the LLM. Profiles use a bulleted list of explicit 
    feature names with corresponding values to maximize clarity during batch inference.

    Missing values (NaN) are intentionally omitted from the profile rather than
    imputed. This simulates a real-world "just ask an LLM" scenario where a user
    would simply not mention information they don't know or don't want to provide.
    This establishes a fair benchmark for the LLM's performance on natural,
    unstructured input compared to the app's structured and imputed results.
    """
    lines = []

    # --- Demographics ---
    if pd.notna(row.get("AGE23X")):
        lines.append(f"- Age: {int(row['AGE23X'])}")
    if pd.notna(row.get("SEX")):
        lines.append(f"- Sex: {SEX_LABELS.get(int(row['SEX']), 'Unknown')}")
    if pd.notna(row.get("REGION23")):
        lines.append(f"- U.S. Region: {REGION_LABELS.get(int(row['REGION23']), 'Unknown')}")
    if pd.notna(row.get("MARRY31X_GRP")):
        lines.append(f"- Marital Status: {MARITAL_LABELS.get(int(row['MARRY31X_GRP']), 'Unknown')}")
    if pd.notna(row.get("FAMSZE23")):
        lines.append(f"- Family Size: {int(row['FAMSZE23'])}")

    # --- Socioeconomic ---
    if pd.notna(row.get("POVCAT23")):
        lines.append(f"- Family Income: {INCOME_LABELS.get(int(row['POVCAT23']), 'Unknown')}")
    if pd.notna(row.get("HIDEG")):
        lines.append(f"- Education: {EDUCATION_LABELS.get(int(row['HIDEG']), 'Unknown')}")
    if pd.notna(row.get("EMPST31_GRP")):
        lines.append(f"- Employment: {EMPLOYMENT_LABELS.get(int(row['EMPST31_GRP']), 'Unknown')}")

    # --- Insurance & Access ---
    if pd.notna(row.get("INSCOV23")):
        lines.append(f"- Insurance: {INSURANCE_LABELS.get(int(row['INSCOV23']), 'Unknown')}")
    if pd.notna(row.get("HAVEUS42")):
        lines.append(f"- Has Usual Source of Healthcare: {YES_NO.get(int(row['HAVEUS42']), 'Unknown')}")

    # --- Health & Lifestyle ---
    if pd.notna(row.get("RTHLTH31")):
        lines.append(f"- Self-Rated Physical Health: {HEALTH_SCALE.get(int(row['RTHLTH31']), 'Unknown')}")
    if pd.notna(row.get("MNHLTH31")):
        lines.append(f"- Self-Rated Mental Health: {HEALTH_SCALE.get(int(row['MNHLTH31']), 'Unknown')}")
    if pd.notna(row.get("ADSMOK42")):
        lines.append(f"- Current Smoker: {YES_NO.get(int(row['ADSMOK42']), 'Unknown')}")

    # --- Chronic Conditions (list only diagnosed) ---
    conditions = [
        label for var, label in CHRONIC_CONDITIONS.items()
        if pd.notna(row.get(var)) and int(row[var]) == 1
    ]
    lines.append(f"- Diagnosed Chronic Conditions: {', '.join(conditions) if conditions else 'None'}")

    # --- Functional Limitations (list only present) ---
    limitations = [
        label for var, label in FUNCTIONAL_LIMITATIONS.items()
        if pd.notna(row.get(var)) and int(row[var]) == 1
    ]
    lines.append(f"- Functional Limitations: {', '.join(limitations) if limitations else 'None'}")

    return "\n".join(lines)


# =========================
# LLM API
# =========================

def build_batch_prompt(profiles, start_idx):
    """
    Build a prompt containing multiple profiles for prompt-batching.
    
    Rationale: Bundling multiple profiles into a single request maximizes 
    throughput under RPM-constrained free tier, reduces total latency by 
    minimizing round-trips, and improves token efficiency. 
    
    Trade-offs: Large batches can suffer from "lost in the middle" effects 
    (reduced attention to middle profiles) or cross-profile information 
    leakage/anchoring (e.g., first prediction influences subsequent 
    predictions). A batch size of 25 is chosen as a "sweet spot" that 
    maintains high prediction quality and reliable JSON arrays while reducing 
    total latency and improving token efficiency.
    """
    profile_texts = []
    for i, profile in enumerate(profiles):
        profile_texts.append(f"Profile {start_idx + i + 1}:\n{profile}")

    n = len(profiles)
    return (
        f"Predict the total annual out-of-pocket healthcare costs (in 2023 US dollars) "
        f"for each of the following {n} US adults.\n\n"
        + "\n\n".join(profile_texts)
        + f"\n\nReturn the {n} estimates as an ordered array."
    )


def parse_llm_response(response_text, expected_count):
    """
    Parse the LLM's response into a list of floats.

    Strictness Rationale:
    If the number of predictions does not exactly match `expected_count`, the 
    entire batch is discarded (returned as NaNs). This "Strict Approach" 
    prevents data shifting, where a single skipped profile in the LLM's 
    output would cause all subsequent predictions in the batch to be 
    misaligned with their ground-truth labels, ruining the benchmark's validity.

    Likely error scenarios handled:
      - JSON format errors (invalid syntax or missing markdown fences).
      - Count mismatch (LLM skips a profile or adds a summary value).
      - Value errors (LLM returns non-numeric strings within the array).
    """
    text = response_text.strip()

    # Extract list of predictions
    match = re.search(r"\[[\s\S]*?\]", text)  # extracts first [] with any characters within
    if match:
        try:
            predictions = json.loads(match.group())  # converts str to list
            if len(predictions) == expected_count:
                return [float(prediction) for prediction in predictions]
            
            print(f"    ⚠️  Count mismatch: Expected {expected_count}, got {len(predictions)}. Discarding batch to avoid alignment errors.")
            return [np.nan] * expected_count

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"    ⚠️  Parse error: {e}. Discarding batch.")
            return [np.nan] * expected_count

    # Fallback: extract numbers in case of missing list brackets []
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    if len(numbers) == expected_count:  # only if number of predictions is as expected
        try:
            return [float(n.replace(",", "")) for n in numbers]
        except ValueError:  # handles potential stray comma (which would match re)
            pass

    print(f"    ❌ Unparseable or mismatched response: {text[:200]}...")
    return [np.nan] * expected_count


def query_llm_batch(client, profiles, start_idx, batch_num):
    """Send a batch of profiles in a single prompt to the LLM API with retry logic."""
    batch_prompt = build_batch_prompt(profiles, start_idx)

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=batch_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,  # Almost deterministic model outputs (except for tiny variations due to floating-point math)
                    # Use structured JSON output (array of numbers, or list of floats in python)
                    response_mime_type="application/json",
                    response_schema={
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"}
                    },
                ),
            )
            return parse_llm_response(response.text, len(profiles))

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_ATTEMPTS - 1:  
                wait_time = DELAY_SECONDS * (2 ** attempt)  # 20 sec after first failed attempt, 40 after 2nd, 80 after 3rd, 160 after 4th
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"    ⚠️ Rate limited (attempt {attempt + 1}/{MAX_ATTEMPTS}). Waiting {wait_time}s...")
                else:
                    print(f"    ⚠️ API error (attempt {attempt + 1}/{MAX_ATTEMPTS}): {error_msg[:120]}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Dont't wait after last attempt failed
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"    ❌ Rate limited (final attempt {MAX_ATTEMPTS}/{MAX_ATTEMPTS}).")
                else:
                    print(f"    ❌ API error (final attempt {MAX_ATTEMPTS}/{MAX_ATTEMPTS}): {error_msg[:120]}.")

    print(f"    ❌ Batch {batch_num} failed after {MAX_ATTEMPTS} attempts")
    return [np.nan] * len(profiles)


# =========================
# Main
# =========================

def main():
    # --- 0. API Key Check ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set.")
        print("   Get a free API key at: https://aistudio.google.com/apikey")
        print("   Then set it in .env: GEMINI_API_KEY=your_gemini_api_key_here")
        sys.exit(1)

    print(f"LLM Benchmark (EV-04): {LLM_MODEL}")
    print(f"{'='*60}\n")

    # --- MLflow Setup ---
    print("Step 0: Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Points to running MLflow UI server
    mlflow.set_experiment("LLM Benchmarks")
    print(f"  Set up 'LLM Benchmarks' experiment in MLflow with URI '{mlflow.get_tracking_uri()}'\n")

    # --- 1. Data Preparation ---
    print("Step 1: Preparing human-readable validation data...")
    df_raw_val, y_val, w_val = prepare_human_readable_validation_data()

    # Align all arrays by common indices
    common_ids = df_raw_val.dropna(how="all").index.intersection(y_val.index)
    df_raw_val = df_raw_val.loc[common_ids]
    y_val = y_val.loc[common_ids]
    w_val = w_val.loc[common_ids]
    print(f"  Benchmarking on {len(common_ids):,} validation rows\n")

    # --- 2. Build Natural Language Profiles ---
    print("Step 2: Converting features to natural language profiles...")
    profiles = [row_to_profile(row) for _, row in df_raw_val.iterrows()]
    print(f"  Created {len(profiles):,} profiles\n")

    # --- 3. Query LLM in Batches ---
    print(f"Step 3: Querying {LLM_MODEL} ({len(profiles):,} profiles in batches of {BATCH_SIZE})...")
    client = genai.Client(api_key=api_key)

    all_predictions = []
    batches = [profiles[i : i + BATCH_SIZE] for i in range(0, len(profiles), BATCH_SIZE)]
    total_batches = len(batches)
    start_time = time.time()

    for i, batch in enumerate(batches):
        start_idx = i * BATCH_SIZE
        elapsed = time.time() - start_time
        
        # Calculate ETA: (time per batch so far) * (batches remaining)
        # We use max(i, 1) to avoid division by zero on the first batch
        eta = (elapsed / max(i, 1)) * (total_batches - i)
        
        print(
            f"  Batch {i + 1:>2}/{total_batches} | Elapsed: {elapsed:>4.0f}s | ETA: {eta:>4.0f}s"
        )

        predictions = query_llm_batch(client, batch, start_idx, i + 1)
        all_predictions.extend(predictions)

        # Rate-limiting delay (skip after last batch)
        if i < total_batches - 1:
            time.sleep(DELAY_SECONDS)

    total_time = time.time() - start_time
    client.close()  # Close the API client to release resources

    y_llm_pred = np.array(all_predictions, dtype=float)

    n_failed = np.isnan(y_llm_pred).sum()
    n_success = len(y_llm_pred) - n_failed
    print(f"\n  Completed in {total_time:.0f}s | Parsed: {n_success:,}/{len(y_llm_pred):,} | Failed: {n_failed:,}\n")

    # --- 4. Compute Weighted Metrics ---
    print("Step 4: Computing weighted evaluation metrics...")

    # Exclude rows where LLM failed to produce a prediction
    valid_mask = ~np.isnan(y_llm_pred)
    y_true = y_val.values[valid_mask]
    y_pred = y_llm_pred[valid_mask]
    weights = w_val.values[valid_mask]

    if n_success == 0:
        print("  ❌ No valid predictions. Cannot compute metrics.")
        sys.exit(1)

    llm_mdae = weighted_median_absolute_error(y_true, y_pred, sample_weight=weights)
    llm_mae = mean_absolute_error(y_true, y_pred, sample_weight=weights)
    llm_r2 = r2_score(y_true, y_pred, sample_weight=weights)

    print(f"  LLM Benchmark Results (n={n_success:,} rows)")
    print(f"    {LLM_MODEL:<25} → MdAE: {llm_mdae:8.2f} | MAE: {llm_mae:8.2f} | "
          f"R²: {llm_r2:.4f} | Inference Time: {total_time:.2f}s")

    # MLflow Logging
    print("  Logging results to MLflow...")
    with mlflow.start_run(run_name=LLM_MODEL):
        # Log Parameters
        mlflow.log_param("llm_model", LLM_MODEL)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("delay_seconds", DELAY_SECONDS)
        
        # Log Metrics
        mlflow.log_metric("val_mdae", float(llm_mdae))
        mlflow.log_metric("val_mae", float(llm_mae))
        mlflow.log_metric("val_r2", float(llm_r2))
        mlflow.log_metric("inference_time_s", total_time)
        mlflow.log_metric("n_success", int(n_success))
        mlflow.log_metric("n_failed", int(n_failed))

    # --- 5. Persistence ---
    print("Step 5: Saving results...")
    metrics_dict = {
        f"LLM ({LLM_MODEL})": {
            "val_mdae": float(llm_mdae),
            "val_mae": float(llm_mae),
            "val_r2": float(llm_r2),
            "n_predictions": int(n_success),
            "n_failed_predictions": int(n_failed),
            "inference_time_seconds": round(total_time, 2),
            "batch_size": BATCH_SIZE,
        }
    }
    save_metrics(metrics_dict, "models/llm_benchmark_metrics.json")
    print(f"  Saved evaluation metrics of LLM ({LLM_MODEL}) to 'models/llm_benchmark_metrics.json'")
    
    save_model(y_llm_pred, "models/llm_benchmark_predictions.joblib", verbose=False)
    print(f"  Saved LLM predictions to 'models/llm_benchmark_predictions.joblib'")

    print("\n✅ LLM benchmark complete.")


if __name__ == "__main__":
    main()
