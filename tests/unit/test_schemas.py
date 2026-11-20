import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def valid_prediction_input():
    return {
        "age": 42,
        "sex": "female",
        "insurance_status": "any_private",
        "family_size": 2,
        "education": "bachelors_degree",
    }


@pytest.mark.parametrize(
    "age",
    [
        pytest.param(18, id="minimum_age"),
        pytest.param(86, id="above_model_age_cap"),
        pytest.param(125, id="maximum_age"),
    ],
)
def test_prediction_request_accepts_ages_in_supported_range(valid_prediction_input, age):
    prediction_input = valid_prediction_input.copy()
    prediction_input["age"] = age

    request = PredictionRequest.model_validate(prediction_input)

    assert request.age == age


@pytest.mark.parametrize(
    "age, expected_error_type, expected_error_context",
    [
        pytest.param(17, "greater_than_equal", {"ge": 18}, id="below_minimum"),
        pytest.param(126, "less_than_equal", {"le": 125}, id="above_maximum"),
    ],
)
def test_prediction_request_rejects_ages_outside_supported_range(
    valid_prediction_input, age, expected_error_type, expected_error_context,
):
    prediction_input = valid_prediction_input.copy()
    prediction_input["age"] = age

    with pytest.raises(ValidationError) as validation_error_info:
        PredictionRequest.model_validate(prediction_input)

    validation_errors = validation_error_info.value.errors()
    assert len(validation_errors) == 1
    age_error = validation_errors[0]
    assert age_error["loc"] == ("age",)
    assert age_error["type"] == expected_error_type
    assert age_error["ctx"] == expected_error_context
