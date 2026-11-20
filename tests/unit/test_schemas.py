from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def current_year(monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 1, 1)

    monkeypatch.setattr("app.schemas.date", FixedDate)


@pytest.fixture
def valid_prediction_input(current_year):
    return {
        "birth_year": 1984,
        "sex": "female",
        "insurance_status": "any_private",
        "family_size": 2,
        "education": "bachelors_degree",
    }


@pytest.mark.parametrize(
    "birth_year, expected_age",
    [
        pytest.param(1984, 42, id="typical_age"),
        pytest.param(2008, 18, id="minimum_age"),
        pytest.param(1940, 86, id="above_model_age_cap"),
        pytest.param(1901, 125, id="maximum_age"),
    ],
)
def test_prediction_request_derives_age_from_birth_year(
    valid_prediction_input, birth_year, expected_age,
):
    # Someone born in 2008 gets age 18, even if their birthday is after
    # January 1, 2026. We use age at the end of the year.
    prediction_input = valid_prediction_input.copy()
    prediction_input["birth_year"] = birth_year

    request = PredictionRequest.model_validate(prediction_input)

    assert request.birth_year == birth_year
    assert request.age == expected_age


@pytest.mark.parametrize(
    "birth_year, expected_error_type, expected_error_context",
    [
        pytest.param(2009, "less_than_equal", {"le": 2008}, id="below_minimum_age"),
        pytest.param(1900, "greater_than_equal", {"ge": 1901}, id="above_maximum_age"),
    ],
)
def test_prediction_request_raises_birth_year_errors_with_range_details(
    valid_prediction_input, birth_year, expected_error_type, expected_error_context,
):
    prediction_input = valid_prediction_input.copy()
    prediction_input["birth_year"] = birth_year

    with pytest.raises(ValidationError) as validation_error_info:
        PredictionRequest.model_validate(prediction_input)

    validation_errors = validation_error_info.value.errors()
    assert len(validation_errors) == 1
    birth_year_error = validation_errors[0]
    assert birth_year_error["loc"] == ("birth_year",)
    assert birth_year_error["type"] == expected_error_type
    assert birth_year_error["ctx"] == expected_error_context


@pytest.mark.parametrize("field", ["birth_year", "family_size"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(42.5, id="fractional_number"),
        pytest.param(42.0, id="integer_as_float"),
        pytest.param("42", id="numeric_string"),
        pytest.param(True, id="boolean"),
    ],
)
def test_prediction_request_requires_integer_types_for_integer_fields(
    valid_prediction_input, field, value,
):
    prediction_input = valid_prediction_input.copy()
    prediction_input[field] = value

    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest.model_validate(prediction_input)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field,)
    assert errors[0]["type"] == "int_type"
