"""Create the medical cost inflation factor artifact used by the deployed app.

Fetches the Bureau of Labour Statistics (BLS) CPI-U Medical Care series 
(CUUR0000SAM) and writes the ``app/data/medical_inflation.json`` artifact. 
The factor is the latest valid monthly index divided by the 2023 annual-average index.

Run this script before an app release, review and commit the resulting JSON, then
deploy that exact commit. It does not retrain or change the model, and the
deployed app must read the artifact rather than call the BLS API at prediction time.

Run:
    .venv-app/Scripts/python scripts/update_medical_inflation.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

APP_DATA_PATH = Path("app/data/medical_inflation.json")
BLS_API_URL_TEMPLATE = "https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
BASE_YEAR = 2023
SERIES_ID = "CUUR0000SAM"
SOURCE_NAME = "U.S. Bureau of Labor Statistics (BLS)"
SERIES_NAME = "Medical care in U.S. city average, all urban consumers, not seasonally adjusted"
MONTHLY_PERIODS = {f"M{month:02d}" for month in range(1, 13)}


def parse_args() -> argparse.Namespace:
    """Parse the output path and BLS API request timeout."""
    parser = argparse.ArgumentParser(
        description="Update the app medical-cost inflation artifact from BLS data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=APP_DATA_PATH,
        help=f"Artifact path (default: {APP_DATA_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="BLS request timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def fetch_bls_data(timeout: float) -> list[dict]:
    """Request data from the BLS API for the medical care series and 
    return the verified data.

    This function owns URL construction, timeout handling, network errors, and
    JSON loading. It delegates response-envelope validation to
    ``extract_bls_series_data`` so callers never receive an unverified BLS
    response.
    """
    if timeout <= 0:
        raise ValueError("Timeout must be greater than zero.")

    current_year = datetime.now(UTC).year
    url = (
        BLS_API_URL_TEMPLATE.format(series_id=SERIES_ID)
        + f"?startyear={BASE_YEAR}&endyear={current_year}"
    )
    try:
        with urlopen(url, timeout=timeout) as response:  # nosec B310: fixed HTTPS URL
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not retrieve BLS series {SERIES_ID}: {error}") from error

    return extract_bls_series_data(payload)


def extract_bls_series_data(payload: dict) -> list[dict]:
    """Validate the BLS response envelope and return the requested series data.

    The response must report success, contain exactly the expected CPI series,
    and expose that series' observations as a list. Anything else stops the
    release update rather than producing an artifact from the wrong data.
    """
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS series {SERIES_ID} request failed: {payload.get('message', [])}"
        )

    series = payload.get("Results", {}).get("series", [])
    if len(series) != 1 or series[0].get("seriesID") != SERIES_ID:
        raise RuntimeError(f"BLS response did not contain series {SERIES_ID}.")

    bls_data = series[0].get("data", [])
    if not isinstance(bls_data, list):
        raise RuntimeError(f"BLS series {SERIES_ID} contains invalid observation data.")
    return bls_data


def parse_monthly_bls_observations(bls_data: list[dict]) -> list[tuple[int, int, float]]:
    """Validate and parse usable monthly observations from extracted BLS data.

    Keep only M01-M12 observations with numeric positive index values and return
    them as ``(year, month, index)`` tuples. BLS may report unavailable values
    as ``"-"``; ignore those here. ``create_medical_inflation_artifact`` later
    requires all 12 base-year months before it can calculate a factor.
    """
    valid_monthly_observations = []
    observed_months = set()
    for observation in bls_data:
        period = observation.get("period")
        if period not in MONTHLY_PERIODS:
            continue
        value = observation.get("value")
        if value in (None, "-"):
            continue
        try:
            year = int(observation["year"])
            month = int(period[1:])
            index = float(value)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid BLS observation: {observation}") from error
        if not math.isfinite(index) or index <= 0:
            raise ValueError(f"Invalid BLS index value: {observation}")
        current_year_month = (year, month)
        if current_year_month in observed_months:
            raise ValueError(f"Duplicate BLS monthly observation: {observation}")
        observed_months.add(current_year_month)
        valid_monthly_observations.append((year, month, index))
    return valid_monthly_observations


def create_medical_inflation_artifact(bls_data: list[dict], generated_at: str) -> dict:
    """Apply the medical inflation business rule and return the artifact dict.

    The base index is the arithmetic mean of all 12 2023 monthly values. The
    target index is the latest valid published monthly value. The stored
    multiplier is ``target_index / base_index``. Missing any 2023 month is an
    error because it would silently change the baseline.
    """
    monthly_bls_data = parse_monthly_bls_observations(bls_data)
    base_months = {
        month: index
        for year, month, index in monthly_bls_data
        if year == BASE_YEAR
    }
    if set(base_months) != set(range(1, 13)):
        missing_months = sorted(set(range(1, 13)) - set(base_months))
        raise ValueError(f"Missing {BASE_YEAR} monthly BLS data: {missing_months}")

    base_index = sum(base_months.values()) / len(base_months)
    target_year, target_month, target_index = max(monthly_bls_data)
    factor = target_index / base_index

    return {
        "schema_version": 1,
        "source": SOURCE_NAME,
        "series_id": SERIES_ID,
        "series_name": SERIES_NAME,
        "base_period": f"{BASE_YEAR} annual average",
        "base_index": round(base_index, 3),
        "target_period": f"{target_year}-{target_month:02d}",
        "target_index": round(target_index, 3),
        "medical_cost_inflation_factor": round(factor, 6),
        "generated_at": generated_at,
    }


def write_json_atomically(path: Path, payload: dict) -> None:
    """Serialize the artifact dict to JSON without exposing a partial file.

    Write to a temporary file first, then rename it to the target path. This
    prevents other processes, such as the web app, from reading a half-written
    or empty artifact if the script is interrupted or a read happens mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    """Run the release update: fetch, validate, calculate, write, and report.

    This command intentionally overwrites the existing artifact. Git history
    preserves the factor used by each deployed release.
    """
    args = parse_args()
    bls_data = fetch_bls_data(args.timeout)
    artifact = create_medical_inflation_artifact(
        bls_data,
        generated_at=datetime.now(UTC).date().isoformat(),
    )
    write_json_atomically(args.output, artifact)

    print(f"Created '{args.output}'.")
    print(
        f"Medical Care CPI {artifact['base_period']}: {artifact['base_index']:.2f}\n"
        f"Medical Care CPI {artifact['target_period']}: {artifact['target_index']:.2f}\n"
        f"Medical Inflation Factor: {artifact['medical_cost_inflation_factor']:.3f}"
    )


if __name__ == "__main__":
    main()