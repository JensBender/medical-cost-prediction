"""Create the medical cost inflation factor artifact used by the deployed app.

Fetches the BLS CPI-U Medical Care series (CUUR0000SAM) and writes the 
``app/data/medical_inflation.json`` artifact. The factor is the latest valid 
monthly index divided by the 2023 annual-average index.

Run this script before an app release, review and commit the resulting JSON, then
deploy that exact commit. It does not retrain or change the model, and the
deployed app must read the artifact rather than call BLS at prediction time.

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
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
BASE_YEAR = 2023
SERIES_ID = "CUUR0000SAM"
SOURCE_NAME = "U.S. Bureau of Labor Statistics (BLS)"
SERIES_NAME = "Medical care in U.S. city average, all urban consumers, not seasonally adjusted"
MONTHLY_PERIODS = {f"M{month:02d}" for month in range(1, 13)}


def parse_args() -> argparse.Namespace:
    """Parse the release output path and BLS request timeout."""
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


def fetch_bls_observations(timeout: float) -> list[dict]:
    """Fetch raw 2023-to-current observations from the fixed BLS series.

    A failed request, failed BLS status, or unexpected series ID stops the
    update. The caller must never write an artifact from an incomplete or
    unverified BLS response.
    """
    if timeout <= 0:
        raise ValueError("Timeout must be greater than zero.")

    current_year = datetime.now(UTC).year
    url = (
        BLS_API_URL.format(series_id=SERIES_ID)
        + f"?startyear={BASE_YEAR}&endyear={current_year}"
    )
    try:
        with urlopen(url, timeout=timeout) as response:  # nosec B310: fixed HTTPS URL
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not retrieve BLS series {SERIES_ID}: {error}") from error

    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS series {SERIES_ID} request failed: {payload.get('message', [])}"
        )

    series = payload.get("Results", {}).get("series", [])
    if len(series) != 1 or series[0].get("seriesID") != SERIES_ID:
        raise RuntimeError(f"BLS response did not contain series {SERIES_ID}.")

    observations = series[0].get("data", [])
    if not isinstance(observations, list):
        raise RuntimeError(f"BLS series {SERIES_ID} contains invalid observation data.")
    return observations


def valid_monthly_observations(observations: list[dict]) -> list[tuple[int, int, float]]:
    """Return valid M01-M12 BLS values as ``(year, month, index)`` tuples.

    BLS may report an unavailable value as ``"-"``. Ignore unavailable
    months here; ``build_artifact`` separately requires all 12 base-year
    observations before it can calculate a factor.
    """
    valid_observations = []
    for observation in observations:
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
        valid_observations.append((year, month, index))
    return valid_observations


def build_artifact(observations: list[dict], generated_at: str) -> dict:
    """Build the auditable release artifact from validated BLS observations.

    The base index is the arithmetic mean of all 12 2023 monthly values.
    The target index is the latest valid published monthly value. The stored
    multiplier is ``target_index / base_index``. Missing any 2023 month is
    an error because it would silently change the baseline.
    """
    monthly_observations = valid_monthly_observations(observations)
    base_months = {
        month: index
        for year, month, index in monthly_observations
        if year == BASE_YEAR
    }
    if set(base_months) != set(range(1, 13)):
        missing_months = sorted(set(range(1, 13)) - set(base_months))
        raise ValueError(f"Missing {BASE_YEAR} monthly BLS observations: {missing_months}")

    base_index = sum(base_months.values()) / len(base_months)
    target_year, target_month, target_index = max(monthly_observations)
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
    """Atomically replace the release artifact with formatted JSON.

    The temporary file is written in the target directory and renamed only
    after serialization completes, so the app cannot observe a partial JSON
    file. Any temporary file is removed if writing or replacement fails.
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
    observations = fetch_bls_observations(args.timeout)
    artifact = build_artifact(
        observations,
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