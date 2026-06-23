"""Unit tests for the medical-cost inflation artifact updater.

Run from the project root:
    .venv-test/Scripts/python -m pytest tests/unit/test_update_medical_inflation.py
"""

import pytest

from scripts import update_medical_inflation as updater


def create_bls_data(base_values, target_year=2026, target_month=5, target_value=593.239):
    """Create Bureau of Labor Statistics (BLS)-style data with 2023 baseline and one target month."""
    bls_data = [
        {"year": "2023", "period": f"M{month:02d}", "value": str(value)}
        for month, value in enumerate(base_values, start=1)
    ]
    bls_data.append(
        {
            "year": str(target_year),
            "period": f"M{target_month:02d}",
            "value": str(target_value),
        }
    )
    return bls_data


def test_create_medical_inflation_artifact_uses_2023_average_and_latest_month():
    artifact = updater.create_medical_inflation_artifact(
        create_bls_data([500.0] * 12),
        generated_at="2026-06-23",
    )

    assert artifact["base_period"] == "2023 annual average"
    assert artifact["base_index"] == 500.0
    assert artifact["target_period"] == "2026-05"
    assert artifact["target_index"] == 593.239
    assert artifact["medical_cost_inflation_factor"] == 1.186478
    assert artifact["source"] == "U.S. Bureau of Labor Statistics (BLS)"
    assert artifact["series_id"] == "CUUR0000SAM"


def test_create_medical_inflation_artifact_requires_all_2023_months():
    bls_data = create_bls_data([500.0] * 11)

    with pytest.raises(ValueError, match="Missing 2023 monthly BLS data"):
        updater.create_medical_inflation_artifact(
            bls_data,
            generated_at="2026-06-23",
        )


def test_write_json_atomically_replaces_existing_artifact(tmp_path):
    output_path = tmp_path / "medical_inflation.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    payload = {"medical_cost_inflation_factor": 1.0804}

    updater.write_json_atomically(output_path, payload)

    assert output_path.read_text(encoding="utf-8") == (
        '{\n  "medical_cost_inflation_factor": 1.0804\n}\n'
    )
    assert not (tmp_path / ".medical_inflation.json.tmp").exists()


def test_create_medical_inflation_artifact_uses_latest_available_month():
    bls_data = create_bls_data([500.0] * 12)
    bls_data.append({"year": "2025", "period": "M10", "value": "-"})

    artifact = updater.create_medical_inflation_artifact(
        bls_data,
        generated_at="2026-06-23",
    )

    assert artifact["target_period"] == "2026-05"