"""FIDUCIA — FastAPI application.

Run:  uvicorn main:app --reload
Open: http://localhost:8000

Routes orchestrate; they contain no business logic. Conversation logic
lives in conversation.py, persistence in db.py, scoring in scoring.py,
and the shared input schema in schema.py.
"""

import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import conversation
import db
import scoring
from schema import ALL_FIELDS, FIELD_LABELS, FinancialProfile, missing_fields


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="FIDUCIA", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


def _session_status(fields: dict, completed: bool, session_id: str) -> dict:
    missing = missing_fields(fields)
    return {
        "fields": fields,
        "filled": [f for f in ALL_FIELDS if fields.get(f) is not None],
        "missing": missing,
        "labels": FIELD_LABELS,
        "complete": completed,
        "report_url": f"/report/{session_id}" if completed else None,
    }


# ---------- pages ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/report/{session_id}", response_class=HTMLResponse)
async def report(request: Request, session_id: str):
    record = db.load_structured(session_id)
    if record is None:
        raise HTTPException(404, "No completed profile for this session yet.")
    created_at = record.pop("created_at")
    result = scoring.score_profile(record)
    return templates.TemplateResponse(request, "report.html", {
        "session_id": session_id,
        "created_at": created_at,
        "profile": record,
        "labels": FIELD_LABELS,
        "result": result,
    })


# ---------- API ----------

@app.post("/api/session")
async def new_session():
    session_id = uuid.uuid4().hex[:12]
    fields = {name: None for name in ALL_FIELDS}
    messages = [{"role": "assistant", "content": conversation.GREETING}]
    db.create_session(session_id, fields, messages)
    db.log_message(session_id, "assistant", conversation.GREETING)
    return {"session_id": session_id, "reply": conversation.GREETING,
            **_session_status(fields, False, session_id)}


@app.get("/api/session/{session_id}")
async def resume_session(session_id: str):
    state = db.load_session(session_id)
    if state is None:
        raise HTTPException(404, "Unknown session.")
    return {"session_id": session_id, "messages": state["messages"],
            **_session_status(state["fields"], state["completed"], session_id)}


@app.post("/api/chat")
async def chat(body: ChatIn):
    state = db.load_session(body.session_id)
    if state is None:
        raise HTTPException(404, "Unknown session — start a new one.")
    if state["completed"]:
        return {"reply": "This session is complete — your report is ready.",
                **_session_status(state["fields"], True, body.session_id)}

    fields, messages = state["fields"], state["messages"]
    db.log_message(body.session_id, "user", body.message)

    try:
        turn = await conversation.run_turn(fields, messages, body.message)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama unavailable: {exc}") from exc

    fields.update(turn["accepted"])
    # dependents_ages is meaningless with zero dependents; auto-fill it
    if fields.get("num_dependents") == 0:
        fields["dependents_ages"] = []

    reply = turn["reply"]

    completed = False
    if not missing_fields(fields):
        # Full validation through the same schema the manual form will use.
        profile = FinancialProfile(**fields)
        record = profile.model_dump()
        for key in ("employment_status", "housing_status", "savings_trend", "income_variability"):
            record[key] = record[key].value  # enum -> plain string for storage
        db.insert_structured(body.session_id, record)
        completed = True
        reply += ("\n\nThat's everything I need. Your report is ready: "
                  f"open /report/{body.session_id}")

    messages.append({"role": "user", "content": body.message})
    messages.append({"role": "assistant", "content": reply})
    db.save_session(body.session_id, fields, messages, completed)
    db.log_message(body.session_id, "assistant", reply)

    return {"reply": reply, "accepted": turn["accepted"],
            **_session_status(fields, completed, body.session_id)}
