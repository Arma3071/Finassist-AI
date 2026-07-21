"""Unit tests for backend.database.db."""

from backend.database import db


def test_init_db_creates_tables():
    db.init_db()
    # Re-running init should be idempotent (CREATE TABLE IF NOT EXISTS).
    db.init_db()


def test_create_and_verify_user():
    db.init_db()
    assert db.create_user("alice", "hunter2pass") is True
    assert db.verify_user("alice", "hunter2pass") is True
    assert db.verify_user("alice", "wrongpass") is False


def test_duplicate_username_rejected():
    db.init_db()
    assert db.create_user("bob", "password123") is True
    assert db.create_user("bob", "password123") is False


def test_session_token_roundtrip():
    db.init_db()
    db.create_user("carol", "password123")
    token = db.create_session("carol")
    assert db.get_username_for_token(token) == "carol"
    assert db.get_username_for_token("nonexistent-token") is None


def test_chat_history_roundtrip():
    db.init_db()
    db.log_chat_message("session-1", "user", "Hello", username="dave")
    db.log_chat_message("session-1", "assistant", "Hi there!", username="dave")

    history = db.get_chat_history("session-1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_query_stats_aggregation():
    db.init_db()
    db.log_query("s1", "What is X?", "retrieval", 120.5, 0.8, 0.7, username="erin")
    db.log_query("s1", "Price of AAPL?", "tool", 80.0, 0.9, None, username="erin")

    stats = db.get_query_stats()
    assert stats["total_queries"] == 2
    assert stats["by_route"]["retrieval"] == 1
    assert stats["by_route"]["tool"] == 1


def test_evaluation_stats_aggregation():
    db.init_db()
    db.log_evaluation(
        question="Q1",
        answer="A1",
        retrieval_precision=0.8,
        retrieval_recall=0.6,
        latency_ms=200.0,
        hallucination_rate=0.1,
        context_relevance=0.75,
        answer_relevance=0.9,
    )
    stats = db.get_evaluation_stats()
    assert stats["total_evaluations"] == 1
    assert stats["avg_precision"] == 0.8
