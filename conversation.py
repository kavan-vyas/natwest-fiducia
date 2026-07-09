# conversational data-collection layer. two ollama calls per turn:
# 1. extraction - last question + user reply, temp 0, json-schema constrained,
#    pydantic-validated. bad values dropped -> field stays missing -> re-asked.
# 2. reply - free-text chat told what was recorded and what to ask next.
#    never sees the scoring formula.

import json
import re

import httpx

from schema import ALL_FIELDS, FIELD_LABELS, PartialProfile, missing_fields

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
TIMEOUT = 300.0
HISTORY_WINDOW = 10  # recent messages passed to the reply call
KEEP_ALIVE = "30m"   # keep model resident so turns stay warm


async def warmup() -> None:
    # preload the model on startup so the first turn isn't a cold load
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(OLLAMA_URL, json={
                "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "think": False, "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1},
            })
    except Exception:
        pass

GREETING = (
    "Hello! I'm the FIDUCIA assistant. I'll ask you a few questions about your "
    "finances to build your credit score report. Everything stays on this machine. "
    "To start: what is your net monthly salary?"
)


def greeting_for(name: str) -> str:
    # fresh profile; identity was captured by the form so jump straight to money
    first = (name or "").strip().split(" ")[0] or "there"
    return f"Thanks, {first}. To start: what's your net monthly salary?"


def resume_greeting_for(name: str) -> str:
    # returning user updating a profile; prior answers already loaded
    first = (name or "").strip().split(" ")[0] or "there"
    return (
        f"Welcome back, {first}. Just tell me what's changed and I'll rebuild your "
        "report — or say 'nothing's changed' to refresh it."
    )

# call 1: extraction

# every field required with a null union; the constrained decoder walks all 16
# keys, which is what makes multi-field extraction reliable on qwen3:8b
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
        "keep_alive": KEEP_ALIVE,
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


# call 2: reply

REPLY_PROMPT = """You are the data-collection assistant for FIDUCIA, a local credit score prototype. Your ONLY function is gathering the user's financial details through friendly, plain, professional conversation. You have no other capabilities and no general knowledge.

Hard rules:
- You never calculate, estimate, hint at, or discuss the user's score or risk level — a separate deterministic system does that after all data is collected.
- STAY ON TASK. If the user asks about anything unrelated to this data collection (news, sport, trivia, coding, advice, opinions, anything), do NOT answer it — not even partially. Warmly say that's outside what you're here for (you only collect the financial details for their report), then immediately re-ask the pending question. Example: "That one's outside my lane, I'm afraid — I'm only here to collect the details for your credit score report. So, what is your net monthly salary?"
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
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.3, "num_predict": 80},  # reply is one short line
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    reply = (data.get("message") or {}).get("content", "").strip()
    return reply or "Could you tell me a bit more?"


# grounding guard: the schema forces all 16 keys every turn, so on a form-shaped
# message the model can pattern-complete an unstated field with an in-range
# number that pydantic accepts and the profile silently completes. so trust a
# numeric value only when that number actually appears in the message; otherwise
# drop it (fabrication) and re-ask.

_NUMERIC_FIELDS = {
    "monthly_salary", "current_savings", "monthly_mortgage", "num_dependents",
    "job_tenure_years", "monthly_credit_card_spending",
    "other_monthly_loan_repayments", "missed_payments_12m",
    "credit_history_years", "credit_applications_6m",
}
# trailing \b so a bare "m" doesn't eat the next word ("34,000 Monthly" = 34000,
# not 34 million); "2.5k" still works since k->space/end is a boundary
_NUM_TOKEN_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k|grand|m|mn|million)?\b", re.I)
_MULT = {"k": 1_000, "grand": 1_000, "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000}
_ZERO_WORDS = ("none", "nothing", "n/a", "nil", "zero", "no ")


def _message_numbers(message: str) -> set[float]:
    # every number in the message, suffixes/separators resolved (5,400->5400, 2.5k->2500)
    vals: set[float] = set()
    for m in _NUM_TOKEN_RE.finditer(message):
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        suffix = (m.group(2) or "").lower()
        if suffix:
            n *= _MULT[suffix]
        vals.add(n)
    return vals


def _grounded(field: str, value, message: str) -> bool:
    # numeric fields must be backed by a number in the message; others pass
    if field not in _NUMERIC_FIELDS:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    low = message.lower()
    if v == 0:
        return "0" in low or any(w in low for w in _ZERO_WORDS)
    # direct match, or a yearly figure the extractor divided by 12
    return any(abs(v - c) < 0.01 or abs(v * 12 - c) < 0.01
               for c in _message_numbers(message))


# advice chat (post-report, streamed)

ADVICE_PROMPT = """You are the **Personal Financial Assistant** — a warm, sharp one-to-one guide who helps ONE person understand their finished credit score report and genuinely improve their whole financial life. You are shown their real report (scores + the actual figures they entered) in a [REPORT] block. Ground every answer in those real numbers and in what the person tells you about their situation.

How you work:
- READ THE PERSON. Pay attention to what they say about their job, income, goals, worries, and life stage. If they mention their profession (e.g. "I'm a nurse", "I run a small business", "I'm a contractor"), tailor your advice to how income, stability, and taxes typically work in that line of work. Ask a brief clarifying question when it would make your advice sharper.
- BE SPECIFIC AND PERSONAL. Give concrete, one-to-one guidance using their actual figures — not generic platitudes. Prioritise their lowest-scoring, highest-weight factors first, then broaden into holistic money guidance: budgeting, cutting the debt-to-income ratio, building an emergency buffer (aim ~3-6 months of expenses), managing revolving vs instalment debt, and sensible next steps for their goals.
- INVESTING GUIDANCE. You may give general investing education tailored to their situation — the usual sequence (clear high-interest debt → emergency fund → tax-advantaged/retirement accounts → diversified low-cost index funds), how risk relates to time horizon, why diversification matters, pound-cost averaging, and how their job's income stability should shape how much risk/liquidity they hold. Explain concepts; do not name specific stocks/tickers/funds to buy or promise returns.
- Explain the score bands and, in general terms, what kinds of behaviour move them.

Formatting — USE MARKDOWN so answers look clean and are easy to scan:
- Short **bold** for key terms and numbers, `-` bullet lists for steps, `###` mini-headings when an answer has parts, and short paragraphs. Keep it tight and readable — no walls of text, no giant tables.
- BE CONCISE: aim for roughly 90-160 words. Lead with the answer, give at most 3-5 crisp bullets, and stop. Do not pad or repeat. A short, sharp reply is better than a long one.

Guardrails — never break these:
- You NEVER recompute, re-estimate, predict, or promise a new credit score or approval odds. The score is fixed by a separate deterministic formula; you only explain and advise around it.
- Stay within personal finance (their report, money, budgeting, debt, saving, investing, and directly related life planning). If asked about something clearly unrelated (news, sport, coding, trivia, other people), warmly decline in one line and steer back to their finances.
- Never discuss or ask about age, gender, race, nationality, religion, or any protected characteristic; none of these affect the score, and you do not use them.
- Never invent figures the report does not contain; if something isn't in the report, say so and ask.
- This is educational guidance, not regulated financial, tax, or legal advice. When you give investing or tax pointers, add a brief reminder to confirm with a qualified professional before acting on big decisions."""


def _report_context(result: dict, profile: dict) -> str:
    # factual summary for the assistant: score, drivers (worst first), and the
    # person's own declared figures. no identity/demographic data.
    lines = [
        f"Total score: {round(result['total_score'])} / 1000 "
        f"({result['category']}, band {result['band_low']}-{result['band_high']}).",
        f"Band meaning: {result['category_blurb']}",
        "",
        "Factors (sub-score 0-100, higher = lower risk; weight = influence):",
    ]
    for row in sorted(result["breakdown"], key=lambda r: r["sub_score"]):
        lines.append(
            f"- {row['label']}: {round(row['sub_score'])}/100, weight {row['weight']:.1f}%. {row['note']}")

    facts = [(FIELD_LABELS.get(k, k), profile[k]) for k in ALL_FIELDS if k in profile]
    lines += ["", "Their declared details:"]
    lines += [f"- {label}: {value}" for label, value in facts]
    return "\n".join(lines)


def _advice_payload(result: dict, profile: dict, history: list, user_message: str) -> dict:
    # scoring facts injected fresh each call so the model always sees the real report
    report_block = f"[REPORT — the user's finished credit score report]\n{_report_context(result, profile)}"
    return {
        "model": MODEL,
        "messages": (
            [{"role": "system", "content": ADVICE_PROMPT},
             {"role": "system", "content": report_block}]
            + history[-HISTORY_WINDOW:]
            + [{"role": "user", "content": user_message}]
        ),
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.4, "num_predict": 420},
    }


async def advice_stream(result: dict, profile: dict, history: list, user_message: str):
    # yield reply pieces as produced so the ui renders live (8b model is slow to finish)
    payload = {**_advice_payload(result, profile, history, user_message), "stream": True}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece


# per-turn orchestration

def _validate(raw_updates: dict) -> tuple[dict, list[str]]:
    # validate each field independently so one bad value doesn't drop the rest
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
    # one turn -> {reply, accepted, rejected}; httpx errors bubble up as 502
    prev_question = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
        GREETING,
    )

    raw_updates = await _extract(prev_question, user_message)
    accepted, rejected = _validate(raw_updates)

    # drop numbers the user never typed (fabrication); silently, so it re-asks
    for name in list(accepted):
        if not _grounded(name, accepted[name], user_message):
            del accepted[name]

    # work out what to ask next from the post-update state
    tentative = dict(fields)
    tentative.update(accepted)
    if tentative.get("num_dependents") == 0:
        tentative["dependents_ages"] = []
    next_targets = missing_fields(tentative)[:2]

    reply = await _compose_reply(messages, user_message, accepted, next_targets, rejected)

    return {"reply": reply, "accepted": accepted, "rejected": rejected}
