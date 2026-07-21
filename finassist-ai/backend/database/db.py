"""SQLite persistence layer for FinAssist AI.

Stores users (auth), chat history, document ingestion records, per-query
logs (for the admin dashboard), and evaluation results. SQLite is used
for simplicity/portability; the schema is small enough to swap for
Postgres later without touching calling code much.
"""

import hashlib
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

_DB_PATH = Path("./data/finassist.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    username TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    chunks_created INTEGER NOT NULL,
    uploaded_by TEXT,
    uploaded_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    username TEXT,
    question TEXT NOT NULL,
    route TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    confidence REAL NOT NULL,
    retrieval_score_avg REAL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieval_precision REAL,
    retrieval_recall REAL,
    latency_ms REAL,
    hallucination_rate REAL,
    context_relevance REAL,
    answer_relevance REAL,
    created_at REAL NOT NULL
);
"""


def init_db() -> None:
    """Create the SQLite database and tables if they do not already exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("Database initialized at %s", _DB_PATH)


@contextmanager
def _connect():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------- #
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def create_user(username: str, password: str) -> bool:
    """Create a new user. Returns False if the username already exists."""
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, time.time()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username: str, password: str) -> bool:
    """Check a username/password pair against stored credentials."""
    with _connect() as conn:
        row = conn.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return False
    return _hash_password(password, row["salt"]) == row["password_hash"]


def create_session(username: str) -> str:
    """Create a new auth session token for a user."""
    token = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, username, created_at) VALUES (?, ?, ?)",
            (token, username, time.time()),
        )
    return token


def get_username_for_token(token: str) -> str | None:
    """Resolve an auth token to a username, or None if invalid."""
    with _connect() as conn:
        row = conn.execute("SELECT username FROM sessions WHERE token = ?", (token,)).fetchone()
    return row["username"] if row else None


# --------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------- #
def log_chat_message(session_id: str, role: str, content: str, username: str | None = None) -> None:
    """Persist a single chat turn."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, username, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, username, role, content, time.time()),
        )


def get_chat_history(session_id: str) -> list[dict]:
    """Retrieve stored chat history for a session, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_history WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------- #
def log_document(document_id: str, filename: str, chunks_created: int, uploaded_by: str | None = None) -> None:
    """Record a successfully ingested document."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (document_id, filename, chunks_created, uploaded_by, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, filename, chunks_created, uploaded_by, time.time()),
        )


def count_documents() -> int:
    """Return the number of ingested documents."""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
    return row["c"]


# --------------------------------------------------------------------- #
# Query logging (drives the admin dashboard)
# --------------------------------------------------------------------- #
def log_query(
    session_id: str,
    question: str,
    route: str,
    latency_ms: float,
    confidence: float,
    retrieval_score_avg: float | None,
    username: str | None = None,
) -> None:
    """Record metadata about a chat query for analytics."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO queries (session_id, username, question, route, latency_ms, confidence, "
            "retrieval_score_avg, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, username, question, route, latency_ms, confidence, retrieval_score_avg, time.time()),
        )


def get_query_stats() -> dict:
    """Aggregate query statistics for the admin dashboard."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, AVG(latency_ms) AS avg_latency, "
            "AVG(confidence) AS avg_confidence, AVG(retrieval_score_avg) AS avg_retrieval_score "
            "FROM queries"
        ).fetchone()
        by_route = conn.execute("SELECT route, COUNT(*) AS c FROM queries GROUP BY route").fetchall()
        recent = conn.execute(
            "SELECT session_id, question, route, latency_ms, confidence, created_at "
            "FROM queries ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return {
        "total_queries": row["total"] or 0,
        "avg_latency_ms": row["avg_latency"] or 0.0,
        "avg_confidence": row["avg_confidence"] or 0.0,
        "avg_retrieval_score": row["avg_retrieval_score"] or 0.0,
        "by_route": {r["route"]: r["c"] for r in by_route},
        "recent": [dict(r) for r in recent],
    }


# --------------------------------------------------------------------- #
# Evaluations
# --------------------------------------------------------------------- #
def log_evaluation(
    question: str,
    answer: str,
    retrieval_precision: float | None,
    retrieval_recall: float | None,
    latency_ms: float,
    hallucination_rate: float | None,
    context_relevance: float | None,
    answer_relevance: float | None,
) -> None:
    """Persist a single evaluation run's metrics."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO evaluations (question, answer, retrieval_precision, retrieval_recall, latency_ms, "
            "hallucination_rate, context_relevance, answer_relevance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question,
                answer,
                retrieval_precision,
                retrieval_recall,
                latency_ms,
                hallucination_rate,
                context_relevance,
                answer_relevance,
                time.time(),
            ),
        )


def get_evaluation_stats() -> dict:
    """Aggregate evaluation metrics for the admin dashboard."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, AVG(retrieval_precision) AS avg_precision, "
            "AVG(retrieval_recall) AS avg_recall, AVG(latency_ms) AS avg_latency, "
            "AVG(hallucination_rate) AS avg_hallucination, AVG(context_relevance) AS avg_context_relevance, "
            "AVG(answer_relevance) AS avg_answer_relevance FROM evaluations"
        ).fetchone()
        recent = conn.execute(
            "SELECT question, retrieval_precision, retrieval_recall, hallucination_rate, "
            "context_relevance, answer_relevance, created_at FROM evaluations ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return {
        "total_evaluations": row["total"] or 0,
        "avg_precision": row["avg_precision"] or 0.0,
        "avg_recall": row["avg_recall"] or 0.0,
        "avg_latency_ms": row["avg_latency"] or 0.0,
        "avg_hallucination_rate": row["avg_hallucination"] or 0.0,
        "avg_context_relevance": row["avg_context_relevance"] or 0.0,
        "avg_answer_relevance": row["avg_answer_relevance"] or 0.0,
        "recent": [dict(r) for r in recent],
    }
