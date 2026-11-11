"""Request validation for the prediction app."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


HealthRating = Literal["excellent", "very_good", "good", "fair", "poor"]


class PredictionRequest(BaseModel):
    """Validate inputs for a single person's out-of-pockets cost prediction."""

    # strict=True rejects strings such as "42" or "false" for integer or boolean fields.
    # extra="forbid" raises an error for any unknown field, such as "diabtes".
    model_config = ConfigDict(strict=True, extra="forbid")

    age: int = Field(..., ge=18, le=85)
    sex: Literal["female", "male"]
    insurance_status: Literal["any_private", "public_only", "uninsured"]
    # TODO: Decide whether to allow family sizes above the training maximum.
    family_size: int = Field(..., ge=1, le=14)
    education: Literal[
        "no_degree",
        "ged",
        "high_school_diploma",
        "associate_degree",
        "bachelors_degree",
        "masters_degree",
        "doctorate",
        "other",
    ]

    region: Literal["northeast", "midwest", "south", "west"] | None = None
    physical_health: HealthRating | None = None
    mental_health: HealthRating | None = None
    income_bracket: Literal[
        "poor", "near_poor", "low_income", "middle_income", "high_income"
    ] | None = None
    marital_status: Literal[
        "married", "widowed", "divorced", "separated", "never_married"
    ] | None = None
    employment_status: Literal["employed", "not_employed"] | None = None
    recent_life_transition: bool | None = None
    usual_source_of_care: bool | None = None
    smoker: bool | None = None

    # TODO: Decide whether omitted condition and limitation fields mean No or unanswered.
    hypertension: bool | None = None
    high_cholesterol: bool | None = None
    diabetes: bool | None = None
    coronary_heart_disease: bool | None = None
    stroke: bool | None = None
    cancer: bool | None = None
    arthritis: bool | None = None
    asthma: bool | None = None

    receives_personal_care_help: bool | None = None
    receives_daily_tasks_help: bool | None = None
    walking_limitation: bool | None = None
    cognitive_limitation: bool | None = None
    joint_pain: bool | None = None
