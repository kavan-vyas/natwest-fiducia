"""FIDUCIA — deterministic scoring engine.

Pure functions only. No AI, no database access, no framework imports,
no demographic fields. This module is auditable in isolation: the score
is a function of exactly the fields listed in WEIGHTS' category scorers
below and nothing else.

Weights come from the Weighted Scoring Model derivation in
methodology.md (importance 1-5 x reliability multiplier, normalised to
100%). They must not be changed here without changing the methodology.

Every category sub-score is on a 0-100 scale where HIGHER = LOWER RISK.
The total is the weighted sum, mapped to a category:
    total >= 70  -> Low risk
    total >= 45  -> Medium risk
    otherwise    -> High risk

Note on employment_sector: it is collected for context and shown on the
report, but deliberately NOT scored — ranking sectors by "stability"
would be a subjective judgement with no defensible basis, exactly the
kind of opaque bias this project exists to avoid.
"""

# Category -> weight in percent. Sums to exactly 100.00.
WEIGHTS: dict[str, float] = {
    "affordability": 22.32,
    "employment_stability": 17.86,
    "payment_history": 13.39,
    "savings_buffer": 13.39,
    "credit_history_length": 8.04,
    "housing_status": 8.93,
    "debt_composition": 8.93,
    "dependents_burden": 4.46,
    "recent_credit_activity": 2.68,
}

CATEGORY_LABELS: dict[str, str] = {
    "affordability": "Affordability (DTI)",
    "employment_stability": "Employment Stability",
    "payment_history": "Self-Declared Payment History",
    "savings_buffer": "Savings Buffer & Trend",
    "credit_history_length": "Self-Declared Credit History Length",
    "housing_status": "Housing Status",
    "debt_composition": "Debt Composition (Credit Mix)",
    "dependents_burden": "Dependents Burden",
    "recent_credit_activity": "Self-Declared Recent Credit Activity",
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _linear(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation of x from [x0, x1] onto [y0, y1], clamped."""
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# ---------- category scorers ----------
# Each returns (sub_score 0-100, short human explanation).

def score_affordability(monthly_salary: float, monthly_mortgage: float,
                        monthly_credit_card_spending: float,
                        other_monthly_loan_repayments: float) -> tuple[float, str]:
    """Debt-to-income ratio. 0.36 is the conventional DTI ceiling in
    lending practice; below 0.15 is treated as fully comfortable."""
    debt = monthly_mortgage + monthly_credit_card_spending + other_monthly_loan_repayments
    if monthly_salary <= 0:
        return 0.0, "No monthly income declared; repayments cannot be serviced from income."
    dti = debt / monthly_salary
    if dti <= 0.15:
        score = 100.0
    elif dti <= 0.36:
        score = _linear(dti, 0.15, 0.36, 100.0, 60.0)
    elif dti <= 0.60:
        score = _linear(dti, 0.36, 0.60, 60.0, 20.0)
    else:
        score = _linear(dti, 0.60, 1.00, 20.0, 0.0)
    return round(score, 2), f"Monthly debt {debt:,.0f} against salary {monthly_salary:,.0f}: DTI = {dti:.0%}."


def score_employment_stability(employment_status: str, job_tenure_years: float,
                               income_variability: str) -> tuple[float, str]:
    """Blend: status 55%, tenure 30%, income variability 15%."""
    status_base = {
        "full_time": 90.0, "retired": 75.0, "self_employed": 65.0,
        "part_time": 60.0, "student": 45.0, "unemployed": 10.0,
    }[employment_status]
    tenure = _linear(job_tenure_years, 0.0, 8.0, 30.0, 100.0)
    variability = 100.0 if income_variability == "fixed" else 45.0
    score = 0.55 * status_base + 0.30 * tenure + 0.15 * variability
    return round(score, 2), (
        f"Status: {employment_status.replace('_', ' ')}, "
        f"{job_tenure_years:g} yr tenure, {income_variability} income."
    )


def score_payment_history(missed_payments_12m: int) -> tuple[float, str]:
    """Self-declared, so already discounted at the weight level (0.6
    reliability multiplier in the methodology)."""
    n = missed_payments_12m
    if n == 0:
        score = 100.0
    elif n == 1:
        score = 60.0
    elif n == 2:
        score = 35.0
    else:
        score = _clamp(20.0 - 10.0 * (n - 3))
    return score, f"{n} self-declared missed payment(s) in the last 12 months."


def score_savings_buffer(current_savings: float, monthly_salary: float,
                         monthly_mortgage: float, monthly_credit_card_spending: float,
                         other_monthly_loan_repayments: float,
                         savings_trend: str) -> tuple[float, str]:
    """Blend: months-of-cover 70%, trend 30%. Cover is savings divided by
    monthly income (or by outgoings when there is no income); 6+ months
    of cover scores full marks."""
    outgoings = monthly_mortgage + monthly_credit_card_spending + other_monthly_loan_repayments
    denominator = monthly_salary if monthly_salary > 0 else max(outgoings, 1.0)
    months = current_savings / max(denominator, 1.0)
    buffer_score = _linear(months, 0.0, 6.0, 0.0, 100.0)
    trend_score = {"building_up": 100.0, "stable": 70.0, "drawing_down": 25.0}[savings_trend]
    score = 0.70 * buffer_score + 0.30 * trend_score
    return round(score, 2), f"~{months:.1f} months of cover; savings {savings_trend.replace('_', ' ')}."


def score_credit_history_length(credit_history_years: float) -> tuple[float, str]:
    """Thin file scores low, 10+ years scores full."""
    score = _linear(credit_history_years, 0.0, 10.0, 20.0, 100.0)
    return round(score, 2), f"{credit_history_years:g} self-declared year(s) holding credit."


def score_housing_status(housing_status: str) -> tuple[float, str]:
    score = {"own_outright": 100.0, "own_with_mortgage": 75.0, "renting": 50.0}[housing_status]
    return score, f"Housing: {housing_status.replace('_', ' ')}."


def score_debt_composition(monthly_mortgage: float, monthly_credit_card_spending: float,
                           other_monthly_loan_repayments: float) -> tuple[float, str]:
    """A mix of instalment and revolving credit is healthiest; debt
    dominated by revolving (credit card) spend is the riskiest shape;
    no debt at all is a thin record, mildly discounted."""
    parts = {
        "mortgage": monthly_mortgage,
        "credit card": monthly_credit_card_spending,
        "other loans": other_monthly_loan_repayments,
    }
    total = sum(parts.values())
    active = sum(1 for v in parts.values() if v > 0)
    base = {0: 65.0, 1: 75.0, 2: 90.0, 3: 80.0}[active]
    note = f"{active} active debt type(s)."
    if total > 0:
        cc_share = monthly_credit_card_spending / total
        if cc_share > 0.85:
            base -= 30.0
            note += f" Credit card is {cc_share:.0%} of debt (revolving-dominated)."
        elif cc_share > 0.60:
            base -= 20.0
            note += f" Credit card is {cc_share:.0%} of debt (revolving-heavy)."
    return _clamp(base), note


def score_dependents_burden(num_dependents: int, dependents_ages: list[int]) -> tuple[float, str]:
    """More dependents = more fixed outgoings; minors add a small extra
    discount as the support obligation runs for years."""
    base_table = {0: 100.0, 1: 82.0, 2: 66.0, 3: 52.0, 4: 40.0}
    base = base_table.get(num_dependents, _clamp(40.0 - 10.0 * (num_dependents - 4), 10.0, 40.0))
    minors = sum(1 for a in dependents_ages if a < 18)
    score = _clamp(base - min(3.0 * minors, 12.0))
    return score, f"{num_dependents} dependent(s), {minors} under 18."


def score_recent_credit_activity(credit_applications_6m: int) -> tuple[float, str]:
    n = credit_applications_6m
    table = {0: 100.0, 1: 80.0, 2: 60.0, 3: 40.0}
    score = table.get(n, _clamp(25.0 - 10.0 * (n - 4)))
    return score, f"{n} self-declared credit application(s) in the last 6 months."


# ---------- top-level entry point ----------

def score_profile(p: dict) -> dict:
    """Take a validated profile (plain dict of numbers/strings/list) and
    return total score, risk category, and per-category breakdown.

    Deterministic: identical input dicts always produce identical output.
    """
    subs: dict[str, tuple[float, str]] = {
        "affordability": score_affordability(
            p["monthly_salary"], p["monthly_mortgage"],
            p["monthly_credit_card_spending"], p["other_monthly_loan_repayments"]),
        "employment_stability": score_employment_stability(
            p["employment_status"], p["job_tenure_years"], p["income_variability"]),
        "payment_history": score_payment_history(p["missed_payments_12m"]),
        "savings_buffer": score_savings_buffer(
            p["current_savings"], p["monthly_salary"], p["monthly_mortgage"],
            p["monthly_credit_card_spending"], p["other_monthly_loan_repayments"],
            p["savings_trend"]),
        "credit_history_length": score_credit_history_length(p["credit_history_years"]),
        "housing_status": score_housing_status(p["housing_status"]),
        "debt_composition": score_debt_composition(
            p["monthly_mortgage"], p["monthly_credit_card_spending"],
            p["other_monthly_loan_repayments"]),
        "dependents_burden": score_dependents_burden(
            p["num_dependents"], p["dependents_ages"]),
        "recent_credit_activity": score_recent_credit_activity(p["credit_applications_6m"]),
    }

    breakdown = []
    total = 0.0
    for key, weight in WEIGHTS.items():
        sub_score, note = subs[key]
        weighted = sub_score * weight / 100.0
        total += weighted
        breakdown.append({
            "key": key,
            "label": CATEGORY_LABELS[key],
            "weight": weight,
            "sub_score": round(sub_score, 2),
            "weighted_points": round(weighted, 2),
            "note": note,
        })

    total = round(total, 2)
    if total >= 70.0:
        category = "Low"
    elif total >= 45.0:
        category = "Medium"
    else:
        category = "High"

    return {"total_score": total, "category": category, "breakdown": breakdown}
