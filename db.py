"""FIDUCIA — persistence layer.

All state lives in one permanent SQLite file on disk. Nothing is held in
in-process memory between requests, so conversations survive a server
restart and concurrent sessions never collide.

Three tables:
  sessions          — live conversation memory, keyed by session id
  conversation_log  — raw append-only record of every exchange (audit)
  structured_inputs — clean validated data; the ONLY table scoring reads
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "fiducia.db"

_STRUCTURED_COLUMNS = [
    "monthly_salary", "current_savings", "monthly_mortgage", "num_dependents",
    "dependents_ages", "employment_status", "employment_sector",
    "job_tenure_years", "monthly_credit_card_spending",
    "other_monthly_loan_repayments", "housing_status", "savings_trend",
    "income_variability", "missed_payments_12m", "credit_history_years",
    "credit_applications_6m",
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                fields TEXT NOT NULL,        -- JSON: field name -> value or null
                messages TEXT NOT NULL,      -- JSON: [{role, content}, ...]
                completed INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                role TEXT NOT NULL,          -- 'user' | 'assistant' | 'system'
                content TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS structured_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                monthly_salary REAL NOT NULL,
                current_savings REAL NOT NULL,
                monthly_mortgage REAL NOT NULL,
                num_dependents INTEGER NOT NULL,
                dependents_ages TEXT NOT NULL,      -- JSON list of ints
                employment_status TEXT NOT NULL,
                employment_sector TEXT NOT NULL,
                job_tenure_years REAL NOT NULL,
                monthly_credit_card_spending REAL NOT NULL,
                other_monthly_loan_repayments REAL NOT NULL,
                housing_status TEXT NOT NULL,
                savings_trend TEXT NOT NULL,
                income_variability TEXT NOT NULL,
                missed_payments_12m INTEGER NOT NULL,
                credit_history_years REAL NOT NULL,
                credit_applications_6m INTEGER NOT NULL
            )
        """)


# ---------- sessions (live persistent memory) ----------

def create_session(session_id: str, fields: dict, messages: list) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at, fields, messages, completed) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (session_id, now, now, json.dumps(fields), json.dumps(messages)),
        )


def load_session(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "fields": json.loads(row["fields"]),
        "messages": json.loads(row["messages"]),
        "completed": bool(row["completed"]),
    }


def save_session(session_id: str, fields: dict, messages: list, completed: bool) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET fields = ?, messages = ?, completed = ?, updated_at = ? "
            "WHERE session_id = ?",
            (json.dumps(fields), json.dumps(messages), int(completed), time.time(), session_id),
        )


# ---------- conversation_log (audit trail) ----------

def log_message(session_id: str, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversation_log (session_id, ts, role, content) VALUES (?, ?, ?, ?)",
            (session_id, time.time(), role, content),
        )


# ---------- structured_inputs (the only thing scoring reads) ----------

def insert_structured(session_id: str, profile: dict) -> None:
    """Write a completed, validated profile. profile values are plain
    Python types (enums already serialised to their string values)."""
    record = dict(profile)
    record["dependents_ages"] = json.dumps(record["dependents_ages"])
    cols = ", ".join(_STRUCTURED_COLUMNS)
    placeholders = ", ".join("?" for _ in _STRUCTURED_COLUMNS)
    with _connect() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO structured_inputs (session_id, created_at, {cols}) "
            f"VALUES (?, ?, {placeholders})",
            (session_id, time.time(), *[record[c] for c in _STRUCTURED_COLUMNS]),
        )


def load_structured(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM structured_inputs WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    record = {c: row[c] for c in _STRUCTURED_COLUMNS}
    record["dependents_ages"] = json.loads(record["dependents_ages"])
    record["created_at"] = row["created_at"]
    return record
