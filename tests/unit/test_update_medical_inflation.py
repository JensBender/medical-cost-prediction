"""Tests for the script to update the medical cost inflation factor."""

import pytest

from scripts import update_medical_inflation as updater


def make_observations(base_values, target_year=2026, target_month=5, target_value=593.239):
    """Build a minimal BLS-style observation list."""
    observations = [
        {"year": "2023", "period": f"M{month:02d}", "value": str(value)}
        for month, value in enumerate(base_values, start=1)
    ]
    observations.append(
        {
            "year": str(target_year),
            "period": f"M{target_month:02d}",
            "value": str(target_value),
        }
    )
    return observations


def test_build_artifact_calculates_auditable_factor():
    """The artifact stores the source values as well as the resulting multiplier."""
    artifact = updater.build_artifact(
        make_observations([500.0] * 12),
        generated_at="2026-06-23",
    )

    assert artifact["base_period"] == "2023 annual average"
    assert artifact["base_index"] == 500.0
    assert artifact["target_period"] == "2026-05"
    assert artifact["target_index"] == 593.239
    assert artifact["medical_cost_inflation_factor"] == 1.186478
    assert artifact["source"] == "U.S. Bureau of Labor Statistics (BLS)"
    assert artifact["series_id"] == "CUUR0000SAM"


def test_build_artifact_rejects_missing_base_month():
    """The annual 2023 baseline requires every monthly observation."""
    observations = make_observations([500.0] * 11)

    with pytest.raises(ValueError, match="Missing 2023 monthly BLS observations"):
        updater.build_artifact(observations, generated_at="2026-06-23")


def test_write_json_atomically_replaces_existing_artifact(tmp_path):
    """Writing a new artifact leaves valid JSON at the output path."""
    output_path = tmp_path / "medical_inflation.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    payload = {"medical_cost_inflation_factor": 1.0804}

    updater.write_json_atomically(output_path, payload)

    assert output_path.read_text(encoding="utf-8") == (
        '{\n  "medical_cost_inflation_factor": 1.0804\n}\n'
    )
    assert not (tmp_path / ".medical_inflation.json.tmp").exists()

def test_build_artifact_ignores_unavailable_nonbase_observation():
    """A later unavailable BLS month does not hide the latest valid observation."""
    observations = make_observations([500.0] * 12)
    observations.append({"year": "2025", "period": "M10", "value": "-"})

    artifact = updater.build_artifact(observations, generated_at="2026-06-23")

    assert artifact["target_period"] == "2026-05"
