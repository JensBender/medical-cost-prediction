"""Request validation for the prediction API."""

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
    insurance_status: Literal["any_private", "public_only", "uninsured"] = Field(
        description="Use any_private for private insurance, even alongside public coverage."
    )
    # TODO: Decide whether to allow family sizes above the training maximum.
    family_size: int = Field(
        ...,
        ge=1,
        le=14,
        description="Number of people in the family unit, including the person whose costs are being predicted.",
    )
    education: Literal[
        "no_degree",
        "ged",
        "high_school_diploma",
        "associate_degree",
        "bachelors_degree",
        "masters_degree",
        "doctorate",
        "other",
    ] = Field(
        description="Highest completed qualification. Doctorate includes PhD, MD, and JD."
    )

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
    smoker: bool | None = Field(
        default=None,
        description="Whether the person currently smokes cigarettes.",
    )

    # TODO: Decide whether omitted condition and limitation fields mean No or unanswered.
    hypertension: bool | None = Field(
        default=None,
        description="Ever diagnosed with high blood pressure by a health professional.",
    )
    high_cholesterol: bool | None = Field(
        default=None,
        description="Ever diagnosed with high cholesterol by a health professional.",
    )
    diabetes: bool | None = Field(
        default=None,
        description="Ever diagnosed with diabetes by a health professional.",
    )
    coronary_heart_disease: bool | None = Field(
        default=None,
        description="Ever diagnosed with coronary heart disease by a health professional.",
    )
    stroke: bool | None = Field(
        default=None,
        description="Ever diagnosed with a stroke by a health professional.",
    )
    cancer: bool | None = Field(
        default=None,
        description="Ever diagnosed with cancer by a health professional.",
    )
    arthritis: bool | None = Field(
        default=None,
        description="Ever diagnosed with arthritis by a health professional.",
    )
    asthma: bool | None = Field(
        default=None,
        description="Ever diagnosed with asthma by a health professional.",
    )

    receives_personal_care_help: bool | None = Field(
        default=None,
        description="Receives help or supervision with personal care, such as bathing or dressing.",
    )
    receives_daily_tasks_help: bool | None = Field(
        default=None,
        description="Receives help or supervision with daily tasks, such as managing medication, bills, or meals.",
    )
    walking_limitation: bool | None = Field(
        default=None,
        description="Difficulty with physical activities such as walking, climbing stairs, lifting, or bending.",
    )
    cognitive_limitation: bool | None = Field(
        default=None,
        description="Confusion, memory loss, problems making decisions, or a need for supervision for safety.",
    )
    joint_pain: bool | None = Field(
        default=None,
        description="Pain, aching, stiffness, or swelling in or around a joint during the past 12 months.",
    )
