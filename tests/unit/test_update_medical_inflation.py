"""Unit tests for the script to update the medical inflation artifact.

Run from the project root:
    .venv-test/Scripts/python -m pytest tests/unit/test_update_medical_inflation.py
"""

import json
from io import StringIO

import pytest

from scripts import update_medical_inflation as updater

pytestmark = pytest.mark.unit


class JsonResponse(StringIO):
    """Minimal context-managed response containing JSON data."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_bls_response(data):
    """Create a successful BLS API response for the fixed series."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": updater.SERIES_ID,
                    "data": data,
                }
            ]
        },
    }


def test_create_medical_inflation_artifact_uses_2023_average_and_latest_month():
    bls_data = [
        {"year": "2023", "period": "M01", "value": "500"},
        {"year": "2023", "period": "M02", "value": "501"},
        {"year": "2023", "period": "M03", "value": "502"},
        {"year": "2023", "period": "M04", "value": "503"},
        {"year": "2023", "period": "M05", "value": "504"},
        {"year": "2023", "period": "M06", "value": "505"},
        {"year": "2023", "period": "M07", "value": "506"},
        {"year": "2023", "period": "M08", "value": "507"},
        {"year": "2023", "period": "M09", "value": "508"},
        {"year": "2023", "period": "M10", "value": "509"},
        {"year": "2023", "period": "M11", "value": "510"},
        {"year": "2023", "period": "M12", "value": "511"},
        {"year": "2026", "period": "M05", "value": "593.239"},
    ]

    inflation_artifact = updater.create_medical_inflation_artifact(
        bls_data,
        generated_at="2026-06-23",
    )

    assert inflation_artifact["base_period"] == "2023 annual average"
    assert inflation_artifact["base_index"] == 505.5
    assert inflation_artifact["target_period"] == "2026-05"
    assert inflation_artifact["target_index"] == 593.239
    assert inflation_artifact["medical_cost_inflation_factor"] == 1.173569


def test_create_medical_inflation_artifact_requires_all_2023_months():
    bls_data = [
        {"year": "2023", "period": "M01", "value": "500.0"},
        {"year": "2023", "period": "M02", "value": "500.0"},
        {"year": "2023", "period": "M03", "value": "500.0"},
        {"year": "2023", "period": "M04", "value": "500.0"},
        {"year": "2023", "period": "M05", "value": "500.0"},
        {"year": "2023", "period": "M06", "value": "500.0"},
        {"year": "2023", "period": "M07", "value": "500.0"},
        {"year": "2023", "period": "M08", "value": "500.0"},
        {"year": "2023", "period": "M09", "value": "500.0"},
        {"year": "2023", "period": "M10", "value": "500.0"},
        {"year": "2023", "period": "M11", "value": "500.0"},
        {"year": "2026", "period": "M05", "value": "593.239"},
    ]

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


def test_write_json_atomically_removes_temporary_file_after_write_failure(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "nested" / "medical_inflation.json"

    def raise_write_error(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(updater.json, "dump", raise_write_error)

    with pytest.raises(OSError, match="disk full"):
        updater.write_json_atomically(output_path, {"schema_version": 1})

    assert not output_path.exists()
    assert not (output_path.parent / ".medical_inflation.json.tmp").exists()


def test_create_medical_inflation_artifact_uses_latest_available_month():
    bls_data = [
        {"year": "2023", "period": "M01", "value": "500.0"},
        {"year": "2023", "period": "M02", "value": "500.0"},
        {"year": "2023", "period": "M03", "value": "500.0"},
        {"year": "2023", "period": "M04", "value": "500.0"},
        {"year": "2023", "period": "M05", "value": "500.0"},
        {"year": "2023", "period": "M06", "value": "500.0"},
        {"year": "2023", "period": "M07", "value": "500.0"},
        {"year": "2023", "period": "M08", "value": "500.0"},
        {"year": "2023", "period": "M09", "value": "500.0"},
        {"year": "2023", "period": "M10", "value": "500.0"},
        {"year": "2023", "period": "M11", "value": "500.0"},
        {"year": "2023", "period": "M12", "value": "500.0"},
        {"year": "2026", "period": "M05", "value": "593.239"},
        {"year": "2026", "period": "M06", "value": "-"},
    ]

    inflation_artifact = updater.create_medical_inflation_artifact(
        bls_data,
        generated_at="2026-06-23",
    )

    assert inflation_artifact["target_period"] == "2026-05"


def test_create_medical_inflation_artifact_uses_latest_chronological_month():
    bls_data = [
        {"year": "2023", "period": "M01", "value": "500.0"},
        {"year": "2023", "period": "M02", "value": "500.0"},
        {"year": "2023", "period": "M03", "value": "500.0"},
        {"year": "2023", "period": "M04", "value": "500.0"},
        {"year": "2023", "period": "M05", "value": "500.0"},
        {"year": "2023", "period": "M06", "value": "500.0"},
        {"year": "2023", "period": "M07", "value": "500.0"},
        {"year": "2023", "period": "M08", "value": "500.0"},
        {"year": "2023", "period": "M09", "value": "500.0"},
        {"year": "2023", "period": "M10", "value": "500.0"},
        {"year": "2023", "period": "M11", "value": "500.0"},
        {"year": "2023", "period": "M12", "value": "500.0"},
        {"year": "2026", "period": "M05", "value": "593.239"},
        {"year": "2025", "period": "M12", "value": "800.0"},
    ]

    inflation_artifact = updater.create_medical_inflation_artifact(
        bls_data,
        generated_at="2026-06-23",
    )

    assert inflation_artifact["target_period"] == "2026-05"
    assert inflation_artifact["target_index"] == 593.239



def test_fetch_bls_data_returns_verified_series(monkeypatch):
    expected_data = [
        {"year": "2023", "period": "M01", "value": "500.0"},
        {"year": "2023", "period": "M02", "value": "500.0"},
        {"year": "2023", "period": "M03", "value": "500.0"},
        {"year": "2023", "period": "M04", "value": "500.0"},
        {"year": "2023", "period": "M05", "value": "500.0"},
        {"year": "2023", "period": "M06", "value": "500.0"},
        {"year": "2023", "period": "M07", "value": "500.0"},
        {"year": "2023", "period": "M08", "value": "500.0"},
        {"year": "2023", "period": "M09", "value": "500.0"},
        {"year": "2023", "period": "M10", "value": "500.0"},
        {"year": "2023", "period": "M11", "value": "500.0"},
        {"year": "2023", "period": "M12", "value": "500.0"},
        {"year": "2026", "period": "M05", "value": "593.239"},
    ]
    request = {}

    def fake_urlopen(url, timeout):
        request["url"] = url
        request["timeout"] = timeout
        return JsonResponse(json.dumps(create_bls_response(expected_data)))

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)

    assert updater.fetch_bls_data(timeout=12.5) == expected_data
    assert updater.SERIES_ID in request["url"]
    assert f"startyear={updater.BASE_YEAR}" in request["url"]
    assert request["timeout"] == 12.5


def test_fetch_bls_data_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="Timeout must be greater than zero"):
        updater.fetch_bls_data(timeout=0)


def test_fetch_bls_data_wraps_retrieval_errors(monkeypatch):
    def raise_network_error(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(updater, "urlopen", raise_network_error)

    with pytest.raises(RuntimeError, match="Could not retrieve BLS series"):
        updater.fetch_bls_data(timeout=30)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": "REQUEST_NOT_PROCESSED"}, "request failed"),
        (
            {
                "status": "REQUEST_SUCCEEDED",
                "Results": {"series": []},
            },
            "did not contain series",
        ),
        (
            {
                "status": "REQUEST_SUCCEEDED",
                "Results": {"seriesID": updater.SERIES_ID, "data": {}},
            },
            "did not contain series",
        ),
        (
            {
                "status": "REQUEST_SUCCEEDED",
                "Results": {"series": [{"seriesID": updater.SERIES_ID, "data": {}}]},
            },
            "invalid observation data",
        ),
    ],
)
def test_fetch_bls_data_rejects_invalid_response(monkeypatch, payload, message):
    monkeypatch.setattr(
        updater,
        "urlopen",
        lambda *args, **kwargs: JsonResponse(json.dumps(payload)),
    )

    with pytest.raises(RuntimeError, match=message):
        updater.fetch_bls_data(timeout=30)


def test_parse_valid_monthly_bls_data_ignores_non_monthly_and_unavailable_data():
    parsed_data = updater.parse_valid_monthly_bls_data(
        [
            {"year": "2026", "period": "M05", "value": "593.239"},
            {"year": "2026", "period": "M13", "value": "600"},
            {"year": "2026", "period": "M06", "value": "-"},
            {"year": "2026", "period": "M07", "value": None},
        ]
    )

    assert parsed_data == [(2026, 5, 593.239)]


@pytest.mark.parametrize(
    "observation",
    [
        {"period": "M01", "value": "500"},
        {"year": "2026", "period": "M01", "value": "not-a-number"},
        {"year": "2026", "period": "M01", "value": "0"},
        {"year": "2026", "period": "M01", "value": "nan"},
    ],
)
def test_parse_valid_monthly_bls_data_rejects_invalid_observations(observation):
    with pytest.raises(ValueError, match="Invalid BLS"):
        updater.parse_valid_monthly_bls_data([observation])
