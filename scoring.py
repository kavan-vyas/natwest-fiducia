# deterministic scoring engine. pure functions, no ai, no db, no demographics.
# each category sub-score is 0-100 (higher = lower risk); weighted sum scaled
# to 0-1000 and mapped to five bands. employment_sector is collected but not
# scored on purpose (ranking sectors by "stability" would be biased).

# category -> weight %, sums to 100
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

# weighted sub-score sum (0-100) times this reaches the 0-1000 scale
SCALE: float = 10.0

# bands high -> low: (lower_bound, key, label, blurb). key is a css/js-safe slug.
RISK_BANDS: list[tuple[int, str, str, str]] = [
    (811, "excellent", "Excellent",
     "Lowest risk. Instant approvals, the highest credit limits, and the lowest interest rates."),
    (671, "very_good", "Very Good",
     "Highly reliable. You easily qualify for most standard loans, cards, and leases at competitive rates."),
    (531, "good", "Good",
     "Acceptable history. Approved for most standard credit, but at average interest rates rather than top discounts."),
    (439, "fair", "Fair",
     "Moderate risk. Approval is harder, requires closer scrutiny/paperwork, and comes with higher interest rates."),
    (0, "poor", "Poor",
     "High risk. Standard approvals are very difficult to get; you must actively rebuild your score to secure credit."),
]


def classify(total: float) -> dict:
    # first band whose lower bound the score meets
    for i, (low, key, label, blurb) in enumerate(RISK_BANDS):
        if total >= low:
            high = 1000 if i == 0 else RISK_BANDS[i - 1][0] - 1
            return {"category_key": key, "category": label, "category_blurb": blurb,
                    "band_low": low, "band_high": high}
    low, key, label, blurb = RISK_BANDS[-1]
    return {"category_key": key, "category": label, "category_blurb": blurb,
            "band_low": low, "band_high": RISK_BANDS[-2][0] - 1}

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
    # linear interp of x from [x0,x1] onto [y0,y1], clamped
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# category scorers: each returns (sub_score 0-100, short note)

def score_affordability(monthly_salary: float, monthly_mortgage: float,
                        monthly_credit_card_spending: float,
                        other_monthly_loan_repayments: float) -> tuple[float, str]:
    # debt-to-income. 0.36 conventional ceiling, below 0.15 fully comfortable
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
    # status 55%, tenure 30%, variability 15%
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
    # self-declared, already discounted at the weight level
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
    # months-of-cover 70%, trend 30%; 6+ months = full marks
    outgoings = monthly_mortgage + monthly_credit_card_spending + other_monthly_loan_repayments
    denominator = monthly_salary if monthly_salary > 0 else max(outgoings, 1.0)
    months = current_savings / max(denominator, 1.0)
    buffer_score = _linear(months, 0.0, 6.0, 0.0, 100.0)
    trend_score = {"building_up": 100.0, "stable": 70.0, "drawing_down": 25.0}[savings_trend]
    score = 0.70 * buffer_score + 0.30 * trend_score
    return round(score, 2), f"~{months:.1f} months of cover; savings {savings_trend.replace('_', ' ')}."


def score_credit_history_length(credit_history_years: float) -> tuple[float, str]:
    # thin file scores low, 10+ years scores full
    score = _linear(credit_history_years, 0.0, 10.0, 20.0, 100.0)
    return round(score, 2), f"{credit_history_years:g} self-declared year(s) holding credit."


def score_housing_status(housing_status: str) -> tuple[float, str]:
    score = {"own_outright": 100.0, "own_with_mortgage": 75.0, "renting": 50.0}[housing_status]
    return score, f"Housing: {housing_status.replace('_', ' ')}."


def score_debt_composition(monthly_mortgage: float, monthly_credit_card_spending: float,
                           other_monthly_loan_repayments: float) -> tuple[float, str]:
    # a mix is healthiest; revolving-dominated is riskiest; no debt is thin
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
    # more dependents = more fixed outgoings; minors add a small extra discount
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


# entry point: validated profile dict -> total, band, per-category breakdown
def score_profile(p: dict) -> dict:
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
        weighted = sub_score * weight / 100.0 * SCALE  # points on 0-1000 scale
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
    return {"total_score": total, "breakdown": breakdown, **classify(total)}
