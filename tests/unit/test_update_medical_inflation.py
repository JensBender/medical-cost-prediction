"""Unit tests for the medical-cost inflation artifact updater.

Run from the project root:
    .venv-test/Scripts/python -m pytest tests/unit/test_update_medical_inflation.py
"""

import pytest

from scripts import update_medical_inflation as updater


def create_bls_data(
    base_year_monthly_values,
    latest_year=2026,
    latest_month=5,
    latest_index=593.239,
):
    """Create BLS-style data with a 2023 baseline and one latest month."""
    bls_data = [
        {"year": "2023", "period": f"M{month:02d}", "value": str(value)}
        for month, value in enumerate(base_year_monthly_values, start=1)
    ]
    bls_data.append(
        {
            "year": str(latest_year),
            "period": f"M{latest_month:02d}",
            "value": str(latest_index),
        }
    )
    return bls_data


def test_create_medical_inflation_artifact_uses_2023_average_and_latest_month():
    inflation_artifact = updater.create_medical_inflation_artifact(
        create_bls_data([500.0] * 12),
        generated_at="2026-06-23",
    )

    assert inflation_artifact["base_period"] == "2023 annual average"
    assert inflation_artifact["base_index"] == 500.0
    assert inflation_artifact["target_period"] == "2026-05"
    assert inflation_artifact["target_index"] == 593.239
    assert inflation_artifact["medical_cost_inflation_factor"] == 1.186478


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
    artifact_payload = {"medical_cost_inflation_factor": 1.0804}

    updater.write_json_atomically(output_path, artifact_payload)

    assert output_path.read_text(encoding="utf-8") == (
        '{\n  "medical_cost_inflation_factor": 1.0804\n}\n'
    )
    assert not (tmp_path / ".medical_inflation.json.tmp").exists()


def test_create_medical_inflation_artifact_uses_latest_available_month():
    bls_data = create_bls_data([500.0] * 12)
    bls_data.append({"year": "2026", "period": "M06", "value": "-"})

    inflation_artifact = updater.create_medical_inflation_artifact(
        bls_data,
        generated_at="2026-06-23",
    )

    assert inflation_artifact["target_period"] == "2026-05"
