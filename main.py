# fiducia fastapi app. routes orchestrate, logic lives in the other modules.
# run: uvicorn main:app --reload

import csv
import io
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                              StreamingResponse)
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

import conversation
import db
import scoring
from schema import (ALL_FIELDS, FIELD_LABELS, GENDER_OPTIONS, FinancialProfile,
                    Identity, missing_fields)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    await conversation.warmup()  # warm the model so first turn isn't a cold load
    yield


app = FastAPI(title="FIDUCIA", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


class IdentifyIn(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=200)
    gender: str = Field(min_length=1, max_length=40)


class SessionIn(BaseModel):
    user_id: int
    mode: str = Field(default="new")  # new | continue


class AdviceMsg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class AdviceIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    history: list[AdviceMsg] = Field(default_factory=list, max_length=40)


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


# pages

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


def _report_payload(session_id: str) -> dict | None:
    # shared by html page, json api and csv export
    record = db.load_structured(session_id)
    if record is None:
        return None
    created_at = record.pop("created_at")
    result = scoring.score_profile(record)
    return {"session_id": session_id, "created_at": created_at,
            "profile": record, "labels": FIELD_LABELS, "result": result,
            "user": db.user_for_session(session_id)}


@app.get("/api/report/{session_id}")
async def report_json(session_id: str):
    payload = _report_payload(session_id)
    if payload is None:
        raise HTTPException(404, "No completed profile for this session yet.")
    return JSONResponse(payload)


# before the html route so ".csv" isn't swallowed by the {session_id} param
@app.get("/report/{session_id}.csv")
async def report_csv(session_id: str):
    payload = _report_payload(session_id)
    if payload is None:
        raise HTTPException(404, "No completed profile for this session yet.")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["FIDUCIA credit score report"])
    w.writerow(["Session", session_id])
    if payload["user"]:
        w.writerow(["Name", payload["user"]["full_name"]])
        w.writerow(["Email", payload["user"]["email"]])
    w.writerow(["Total score", payload["result"]["total_score"]])
    w.writerow(["Score band", payload["result"]["category"]])
    w.writerow([])
    w.writerow(["Category", "Weight %", "Sub-score", "Weighted points", "Basis"])
    for row in payload["result"]["breakdown"]:
        w.writerow([row["label"], row["weight"], row["sub_score"],
                    row["weighted_points"], row["note"]])
    w.writerow([])
    w.writerow(["Input", "Value"])
    for key, value in payload["profile"].items():
        w.writerow([FIELD_LABELS.get(key, key), value])
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="fiducia_{session_id}.csv"'},
    )


@app.get("/report/{session_id}", response_class=HTMLResponse)
async def report(request: Request, session_id: str):
    payload = _report_payload(session_id)
    if payload is None:
        raise HTTPException(404, "No completed profile for this session yet.")
    return templates.TemplateResponse(request, "report.html", {"request": request, **payload})


# api

@app.get("/api/meta")
async def meta():
    return {"gender_options": GENDER_OPTIONS}


@app.post("/api/identify")
async def identify(body: IdentifyIn):
    try:
        ident = Identity(full_name=body.full_name, email=body.email, gender=body.gender)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from exc
    existed = db.find_user(ident.full_name, ident.email) is not None
    user = db.get_or_create_user(ident.full_name, ident.email, ident.gender)
    has_profile = db.latest_profile_for_user(user["id"]) is not None
    return {"user_id": user["id"], "full_name": user["full_name"],
            "returning": existed, "has_profile": has_profile}


@app.post("/api/session")
async def new_session(body: SessionIn):
    user = db.get_user(body.user_id)
    if user is None:
        raise HTTPException(404, "Unknown user — identify first.")

    session_id = uuid.uuid4().hex[:12]
    prior = db.latest_profile_for_user(body.user_id) if body.mode == "continue" else None

    if prior:
        fields = {name: prior.get(name) for name in ALL_FIELDS}
        greeting = conversation.resume_greeting_for(user["full_name"])
    else:
        fields = {name: None for name in ALL_FIELDS}
        greeting = conversation.greeting_for(user["full_name"])

    messages = [{"role": "assistant", "content": greeting}]
    db.create_session(session_id, fields, messages, user_id=body.user_id)
    db.log_message(session_id, "assistant", greeting)
    return {"session_id": session_id, "reply": greeting,
            **_session_status(fields, False, session_id)}


@app.get("/api/session/{session_id}")
async def resume_session(session_id: str):
    state = db.load_session(session_id)
    if state is None:
        raise HTTPException(404, "Unknown session.")
    return {"session_id": session_id, "messages": state["messages"],
            **_session_status(state["fields"], state["completed"], session_id)}


@app.post("/api/advice")
async def advice(body: AdviceIn):
    # post-report assistant. needs a finished report. reply is streamed.
    payload = _report_payload(body.session_id)
    if payload is None:
        raise HTTPException(404, "No completed report for this session yet.")
    history = [m.model_dump() for m in body.history]

    async def gen():
        try:
            async for piece in conversation.advice_stream(
                    payload["result"], payload["profile"], history, body.message):
                yield piece
        except httpx.HTTPError as exc:
            yield f"\n\n_(The assistant is unavailable right now: {exc})_"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


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
    if fields.get("num_dependents") == 0:  # no dependents -> no ages needed
        fields["dependents_ages"] = []

    reply = turn["reply"]

    completed = False
    if not missing_fields(fields):
        # full validation through the same schema before storing
        profile = FinancialProfile(**fields)
        record = profile.model_dump()
        for key in ("employment_status", "housing_status", "savings_trend", "income_variability"):
            record[key] = record[key].value  # enum -> str
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
