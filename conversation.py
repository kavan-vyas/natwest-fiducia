"""FIDUCIA — conversational data-collection layer.

Two separate Ollama calls per turn, because extraction and conversation
have opposite needs:

1. EXTRACTION — sees only the last assistant question and the user's
   answer (no chat noise), temperature 0, output constrained to a JSON
   schema at the Ollama API level. Guaranteed parseable; validated by
   Pydantic (schema.PartialProfile) before a single value is accepted.
   Invalid values are dropped, which leaves the field missing, which
   makes the assistant re-ask — a re-prompt, not a guess.

2. REPLY — a plain free-text chat call that is told exactly what was
   just recorded and which 1-2 fields to ask for next. It never sees
   the scoring formula and has no influence on the score.
"""

import json

import httpx

from schema import ALL_FIELDS, FIELD_LABELS, PartialProfile, missing_fields

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
TIMEOUT = 300.0
HISTORY_WINDOW = 10  # recent messages passed to the reply call

GREETING = (
    "Hello! I'm the FIDUCIA assistant. I'll ask you a few questions about your "
    "finances to build your credit risk report. Everything stays on this machine. "
    "To start: what is your net monthly salary?"
)


def greeting_for(name: str) -> str:
    """Opening line for a fresh profile, addressed to a known user. Identity
    (name + email) is captured deterministically by the form beforehand, so
    the assistant never asks for it and jumps straight to finances."""
    first = (name or "").strip().split(" ")[0] or "there"
    return f"Thanks, {first}. To start: what's your net monthly salary?"


def resume_greeting_for(name: str) -> str:
    """Opening line when an existing user chooses to update their profile.
    Their previous answers are already loaded, so only changes need stating."""
    first = (name or "").strip().split(" ")[0] or "there"
    return (
        f"Welcome back, {first}. Just tell me what's changed and I'll rebuild your "
        "report — or say 'nothing's changed' to refresh it."
    )

# ---------- call 1: extraction ----------

# Every field is REQUIRED with a null union: the constrained decoder then
# walks all 16 keys explicitly, which (tested against qwen3:8b) is what
# makes multi-field extraction reliable. Optional-key schemas made the
# model close the object after one field; unconstrained "null everything
# unless stated" prompting made it hallucinate defaults.
_EXTRACT_PROPS: dict = {
    "monthly_salary": {"type": ["number", "null"]},
    "current_savings": {"type": ["number", "null"]},
    "monthly_mortgage": {"type": ["number", "null"]},
    "num_dependents": {"type": ["integer", "null"]},
    "dependents_ages": {"type": ["array", "null"], "items": {"type": "integer"}},
    "employment_status": {"type": ["string", "null"], "enum": [
        "full_time", "part_time", "self_employed", "unemployed", "retired", "student", None]},
    "employment_sector": {"type": ["string", "null"]},
    "job_tenure_years": {"type": ["number", "null"]},
    "monthly_credit_card_spending": {"type": ["number", "null"]},
    "other_monthly_loan_repayments": {"type": ["number", "null"]},
    "housing_status": {"type": ["string", "null"], "enum": [
        "own_outright", "own_with_mortgage", "renting", None]},
    "savings_trend": {"type": ["string", "null"], "enum": [
        "building_up", "stable", "drawing_down", None]},
    "income_variability": {"type": ["string", "null"], "enum": ["fixed", "variable", None]},
    "missed_payments_12m": {"type": ["integer", "null"]},
    "credit_history_years": {"type": ["number", "null"]},
    "credit_applications_6m": {"type": ["integer", "null"]},
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": _EXTRACT_PROPS,
    "required": list(_EXTRACT_PROPS.keys()),
}

EXTRACT_PROMPT = """You extract financial data from one user message into JSON. Every key is present in your output: set a VALUE only for fields the user explicitly stated in their reply, and null for every other field. Most fields will be null — that is normal and correct. Never guess, infer, or default a value the user did not state.

Normalisation:
- "three grand" -> 3000, "2.5k" -> 2500, "none"/"no"/"nothing" -> 0 for the amount the question was about.
- Amounts are monthly unless the user clearly says otherwise; convert yearly to monthly (divide by 12).
- savings_trend: growing/saving more -> building_up; shrinking/dipping into savings -> drawing_down; unchanged -> stable.
- income_variability: same amount every month -> fixed; changes month to month -> variable.
- dependents_ages: integers, one per dependent.
- If the message is ambiguous about a field, leave that field null.

Example:
Assistant asked: do you have a mortgage?
User replied: no mortgage, but I spend maybe 600 a month on my card. oh and I rent
Correct: monthly_mortgage=0, monthly_credit_card_spending=600, housing_status="renting", ALL other fields null.
"""


async def _extract(prev_question: str, user_message: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content":
                f"The assistant asked: {prev_question}\n\nThe user replied: {user_message}"},
        ],
        "stream": False,
        "think": False,
        "format": EXTRACT_SCHEMA,
        "options": {"temperature": 0.0},
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    try:
        return json.loads(data["message"]["content"]) or {}
    except (KeyError, json.JSONDecodeError):
        return {}


# ---------- call 2: reply ----------

REPLY_PROMPT = """You are the data-collection assistant for FIDUCIA, a local credit-risk prototype. Your ONLY function is gathering the user's financial details through friendly, plain, professional conversation. You have no other capabilities and no general knowledge.

Hard rules:
- You never calculate, estimate, hint at, or discuss the user's score or risk level — a separate deterministic system does that after all data is collected.
- STAY ON TASK. If the user asks about anything unrelated to this data collection (news, sport, trivia, coding, advice, opinions, anything), do NOT answer it — not even partially. Warmly say that's outside what you're here for (you only collect the financial details for their report), then immediately re-ask the pending question. Example: "That one's outside my lane, I'm afraid — I'm only here to collect the details for your credit risk report. So, what is your net monthly salary?"
- The two allowed exceptions: (a) the user asks why a field is needed — explain briefly (it feeds a transparent, affordability-style formula), then re-ask; (b) the user asks what FIDUCIA is or what happens to their data — answer briefly (local prototype, data stays on this machine, a fixed formula does the scoring), then re-ask.
- Never ask about age, gender, race, nationality, religion, or any personal characteristic. If offered, say it is not used and move on.
- BE BRIEF. One short line only: a two-word acknowledgement ("Thanks." / "Got it.") then the next question. No preamble, no restating what you recorded, no explaining what you are doing, no "your X is recorded as Y". Do not list. Just: acknowledge + ask.
- Ask ONLY for the fields named in your instruction block. Do not ask about anything else.
- FOLLOW-UP: if the instruction block says nothing new was recorded (the message was empty, vague, or off-topic), do NOT move on — re-ask the SAME pending field with a short example (e.g. "roughly how much a month — 800? 1200?")."""


async def _compose_reply(messages: list, user_message: str, accepted: dict,
                         next_targets: list[str], rejected: list[str]) -> str:
    if accepted:
        recorded = ", ".join(
            f"{FIELD_LABELS[k]} = {v}" for k, v in accepted.items())
    else:
        recorded = "nothing new"
    if next_targets:
        ask = "Now ask the user for: " + " and ".join(FIELD_LABELS[t] for t in next_targets) + "."
    else:
        ask = ("Every required field is now collected. Briefly thank the user and say "
               "their report is being prepared. Ask nothing further.")
    note = ""
    if rejected:
        note = ("\nThese values looked out of range and were NOT saved — ask the user "
                "to re-check them: " + ", ".join(rejected) + ".")

    instruction = (f"[INSTRUCTION — not from the user]\n"
                   f"Just recorded from the user's last message: {recorded}.\n{ask}{note}")

    payload = {
        "model": MODEL,
        "messages": (
            [{"role": "system", "content": REPLY_PROMPT}]
            + messages[-HISTORY_WINDOW:]
            + [{"role": "user", "content": user_message},
               {"role": "system", "content": instruction}]
        ),
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3},
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    reply = (data.get("message") or {}).get("content", "").strip()
    return reply or "Could you tell me a bit more?"


# ---------- per-turn orchestration ----------

def _validate(raw_updates: dict) -> tuple[dict, list[str]]:
    """Validate each extracted field independently through Pydantic, so one
    bad value doesn't discard the rest of the turn's extractions."""
    accepted: dict = {}
    rejected: list[str] = []
    for name, value in raw_updates.items():
        if value is None or name not in ALL_FIELDS:
            continue
        try:
            model = PartialProfile(**{name: value})
            validated = getattr(model, name)
            accepted[name] = getattr(validated, "value", validated)  # enum -> str
        except Exception:
            rejected.append(FIELD_LABELS.get(name, name))
    return accepted, rejected


async def run_turn(fields: dict, messages: list, user_message: str) -> dict:
    """One conversational turn. Returns:
    {reply, accepted: {field: value}, rejected: [label, ...]}
    Raises httpx errors upward for the route to turn into a 502.
    """
    prev_question = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
        GREETING,
    )

    raw_updates = await _extract(prev_question, user_message)
    accepted, rejected = _validate(raw_updates)

    # Work out what to ask next from the post-update state.
    tentative = dict(fields)
    tentative.update(accepted)
    if tentative.get("num_dependents") == 0:
        tentative["dependents_ages"] = []
    next_targets = missing_fields(tentative)[:2]

    reply = await _compose_reply(messages, user_message, accepted, next_targets, rejected)

    return {"reply": reply, "accepted": accepted, "rejected": rejected}
