"""FIDUCIA — shared input schema.

Single source of truth for what the system collects. Both entry modes
(chat extraction and, later, the manual form) validate against these
exact models, so the two modes can never validate differently.

`FinancialProfile` is the complete, required record that scoring.py
consumes. `PartialProfile` carries the same fields and constraints but
everything optional — it validates the per-turn extractions coming out
of Ollama, where any subset of fields may arrive.

No demographic / protected-characteristic field exists in this schema.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EmploymentStatus(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    self_employed = "self_employed"
    unemployed = "unemployed"
    retired = "retired"
    student = "student"


class HousingStatus(str, Enum):
    own_outright = "own_outright"
    own_with_mortgage = "own_with_mortgage"
    renting = "renting"


class SavingsTrend(str, Enum):
    building_up = "building_up"
    stable = "stable"
    drawing_down = "drawing_down"


class IncomeVariability(str, Enum):
    fixed = "fixed"
    variable = "variable"


class FinancialProfile(BaseModel):
    """The complete validated record. scoring.py reads exactly these fields."""

    # --- core inputs from the original brief ---
    monthly_salary: float = Field(ge=0, le=1_000_000, description="Net monthly salary/income")
    current_savings: float = Field(ge=0, le=100_000_000, description="Total current savings")
    monthly_mortgage: float = Field(ge=0, le=1_000_000, description="Monthly mortgage payment (0 if none)")
    num_dependents: int = Field(ge=0, le=20, description="Number of financial dependents")
    employment_status: EmploymentStatus
    monthly_credit_card_spending: float = Field(ge=0, le=1_000_000)
    other_monthly_loan_repayments: float = Field(ge=0, le=1_000_000, description="Car loan, student loan, etc.")

    # --- expanded inputs ---
    dependents_ages: list[int] = Field(default_factory=list, description="Approximate ages of dependents")
    employment_sector: str = Field(min_length=1, max_length=100, description="e.g. healthcare, tech, retail")
    job_tenure_years: float = Field(ge=0, le=80, description="Years in current job/status")
    housing_status: HousingStatus
    savings_trend: SavingsTrend
    income_variability: IncomeVariability
    missed_payments_12m: int = Field(ge=0, le=100, description="Self-declared missed payments, last 12 months")
    credit_history_years: float = Field(ge=0, le=80, description="Self-declared years holding any credit account")
    credit_applications_6m: int = Field(ge=0, le=50, description="Self-declared credit applications, last 6 months")

    @field_validator("dependents_ages")
    @classmethod
    def ages_sane(cls, v: list[int]) -> list[int]:
        for age in v:
            if not 0 <= age <= 120:
                raise ValueError("dependent age out of range 0-120")
        return v


class PartialProfile(BaseModel):
    """Per-turn extraction: any subset of fields, same constraints as above."""

    monthly_salary: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    current_savings: Optional[float] = Field(default=None, ge=0, le=100_000_000)
    monthly_mortgage: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    num_dependents: Optional[int] = Field(default=None, ge=0, le=20)
    employment_status: Optional[EmploymentStatus] = None
    monthly_credit_card_spending: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    other_monthly_loan_repayments: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    dependents_ages: Optional[list[int]] = None
    employment_sector: Optional[str] = Field(default=None, min_length=1, max_length=100)
    job_tenure_years: Optional[float] = Field(default=None, ge=0, le=80)
    housing_status: Optional[HousingStatus] = None
    savings_trend: Optional[SavingsTrend] = None
    income_variability: Optional[IncomeVariability] = None
    missed_payments_12m: Optional[int] = Field(default=None, ge=0, le=100)
    credit_history_years: Optional[float] = Field(default=None, ge=0, le=80)
    credit_applications_6m: Optional[int] = Field(default=None, ge=0, le=50)

    @field_validator("dependents_ages")
    @classmethod
    def ages_sane(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            for age in v:
                if not 0 <= age <= 120:
                    raise ValueError("dependent age out of range 0-120")
        return v


# Field names in the order the conversation should roughly collect them.
ALL_FIELDS: list[str] = list(FinancialProfile.model_fields.keys())

# Human-readable labels for the front end and the report.
FIELD_LABELS: dict[str, str] = {
    "monthly_salary": "Monthly salary",
    "current_savings": "Current savings",
    "monthly_mortgage": "Monthly mortgage payment",
    "num_dependents": "Number of dependents",
    "dependents_ages": "Dependents' ages",
    "employment_status": "Employment status",
    "employment_sector": "Employment sector",
    "job_tenure_years": "Job tenure (years)",
    "monthly_credit_card_spending": "Monthly credit card spending",
    "other_monthly_loan_repayments": "Other monthly loan repayments",
    "housing_status": "Housing status",
    "savings_trend": "Savings trend",
    "income_variability": "Income variability",
    "missed_payments_12m": "Missed payments (last 12 months)",
    "credit_history_years": "Years of credit history",
    "credit_applications_6m": "Credit applications (last 6 months)",
}


def missing_fields(fields: dict) -> list[str]:
    """Which required fields are still unfilled.

    dependents_ages is only required when num_dependents > 0; when
    num_dependents == 0 it is auto-set to [] by the caller.
    """
    missing = []
    for name in ALL_FIELDS:
        if fields.get(name) is None:
            if name == "dependents_ages" and fields.get("num_dependents") == 0:
                continue
            missing.append(name)
    return missing
