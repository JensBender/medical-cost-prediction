"""Prediction input validation for the UI and API."""

from datetime import date
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticKnownError


HealthRating = Literal["excellent", "very_good", "good", "fair", "poor"]


class PredictionRequest(BaseModel):
    # strict=True rejects strings such as "42" or "false" for integer or boolean fields.
    # extra="forbid" raises an error for any unknown field, such as "diabtes".
    model_config = ConfigDict(strict=True, extra="forbid")

    birth_year: int = Field(
        ...,
        frozen=True,  # Keep the stored age consistent with its birth year.
        description="Birth year corresponding to an age of 18–125 at the end of the current year.",
    )
    _age: int = PrivateAttr()  # Excluded from input validation.
    sex: Literal["female", "male"]
    insurance_status: Literal["any_private", "public_only", "uninsured"] = Field(
        description="If the person has both private and public insurance, choose any_private."
    )
    # TODO: Decide whether to allow family sizes above the training maximum.
    family_size: int = Field(
        ...,
        ge=1,
        le=14,
        description="Number of people living together as a family, including the person.",
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
    # TODO: Finalize the UI income ranges and how API callers should provide
    # family income, including how to map their input to these categories.
    family_income: Literal[
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

    @field_validator("birth_year")
    @classmethod
    def validate_birth_year(cls, birth_year: int) -> int:
        current_year = date.today().year
        age = current_year - birth_year
        # Pydantic catches these known errors and includes them in ValidationError,
        # adding the birth_year field location and rejected input automatically.
        if age < 18:
            raise PydanticKnownError("less_than_equal", {"le": current_year - 18})
        if age > 125:
            raise PydanticKnownError("greater_than_equal", {"ge": current_year - 125})
        return birth_year

    @model_validator(mode="after")
    def store_age(self) -> Self:
        """Store the derived age (at end of current year) after all input fields pass validation."""
        self._age = date.today().year - self.birth_year
        return self

    @property
    def age(self) -> int:
        """Expose _age as read-only request.age without storing a second copy."""
        return self._age
