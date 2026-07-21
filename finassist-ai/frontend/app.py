"""Streamlit frontend for FinAssist AI.

Provides: a simple login/register screen, a chat + document upload
interface, and an admin dashboard with usage and evaluation metrics.

Run with: streamlit run frontend/app.py
Expects the backend to be running at BACKEND_URL (default http://localhost:8000).
"""

import os
import uuid

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="FinAssist AI", page_icon="💹", layout="wide")

# --------------------------------------------------------------------- #
# Session state defaults
# --------------------------------------------------------------------- #
defaults = {
    "auth_token": None,
    "username": None,
    "session_id": str(uuid.uuid4()),
    "messages": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def auth_headers() -> dict:
    if st.session_state.auth_token:
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}


# --------------------------------------------------------------------- #
# Login / register screen
# --------------------------------------------------------------------- #
def login_screen() -> None:
    st.title("💹 FinAssist AI")
    st.caption("Sign in to start chatting with your financial research assistant.")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            resp = requests.post(
                f"{BACKEND_URL}/api/auth/login",
                json={"username": username, "password": password},
                timeout=15,
            )
            if resp.ok:
                data = resp.json()
                st.session_state.auth_token = data["token"]
                st.session_state.username = data["username"]
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            registered = st.form_submit_button("Register")
        if registered:
            resp = requests.post(
                f"{BACKEND_URL}/api/auth/register",
                json={"username": new_username, "password": new_password},
                timeout=15,
            )
            if resp.ok:
                st.success("Account created — you can log in now.")
            else:
                st.error(resp.json().get("detail", "Registration failed."))


# --------------------------------------------------------------------- #
# Main app (chat + admin) once authenticated
# --------------------------------------------------------------------- #
def chat_tab() -> None:
    with st.sidebar:
        st.subheader("📄 Documents")
        uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md"])
        if uploaded and st.button("Ingest document"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            with st.spinner("Ingesting..."):
                resp = requests.post(
                    f"{BACKEND_URL}/api/upload", files=files, headers=auth_headers(), timeout=120
                )
            if resp.ok:
                data = resp.json()
                st.success(f"Ingested {data['chunks_created']} chunks from {data['filename']}")
            else:
                st.error(f"Upload failed: {resp.text}")

        st.divider()
        if st.button("New session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['document_name']}** (score {s['score']:.2f})")
                        st.caption(s["text"][:300])
            if msg.get("tool_calls"):
                with st.expander(f"Tool calls ({len(msg['tool_calls'])})"):
                    for tc in msg["tool_calls"]:
                        st.json(tc)

    prompt = st.chat_input("Ask about a document, a stock, or financial data...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    headers=auth_headers(),
                    timeout=120,
                )
            if resp.ok:
                data = resp.json()
                st.markdown(data["answer"])
                confidence = data["confidence"]
                st.progress(confidence, text=f"Confidence: {confidence:.0%}")
                st.caption(f"Route: {data['route']} · {data['latency_ms']:.0f}ms")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "tool_calls": data.get("tool_calls", []),
                    }
                )
            else:
                st.error(f"Request failed: {resp.text}")


def admin_tab() -> None:
    st.subheader("📊 Admin Dashboard")

    try:
        metrics = requests.get(f"{BACKEND_URL}/api/metrics", headers=auth_headers(), timeout=15).json()
    except requests.RequestException:
        st.warning("Backend not reachable.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents uploaded", metrics["documents_uploaded"])
    col2.metric("Embedding chunks", metrics["embedding_count"])
    col3.metric("Total queries", metrics["total_queries"])
    col4.metric("Avg latency (ms)", f"{metrics['avg_latency_ms']:.0f}")

    col5, col6 = st.columns(2)
    col5.metric("Avg retrieval score", f"{metrics['avg_retrieval_score']:.2f}")
    col6.metric("Avg confidence", f"{metrics['avg_confidence']:.2f}")

    if metrics["queries_by_route"]:
        st.markdown("**Queries by route**")
        df_routes = pd.DataFrame(
            list(metrics["queries_by_route"].items()), columns=["route", "count"]
        ).set_index("route")
        st.bar_chart(df_routes)

    st.divider()
    st.markdown("### Evaluation results")
    ev = metrics["evaluation"]
    if ev["total_evaluations"] == 0:
        st.info("No evaluations run yet.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Avg precision", ev["avg_precision"])
        e2.metric("Avg recall", ev["avg_recall"])
        e3.metric("Avg hallucination rate", ev["avg_hallucination_rate"])
        e4, e5 = st.columns(2)
        e4.metric("Avg context relevance", ev["avg_context_relevance"])
        e5.metric("Avg answer relevance", ev["avg_answer_relevance"])

    if st.button("Run evaluation now"):
        with st.spinner("Running golden evaluation dataset..."):
            resp = requests.post(f"{BACKEND_URL}/api/evaluate", headers=auth_headers(), timeout=180)
        if resp.ok:
            data = resp.json()
            st.success(f"Evaluated {data['items_evaluated']} questions.")
            st.dataframe(pd.DataFrame(data["results"]))
            st.json(data["aggregate"])
        else:
            st.error(f"Evaluation failed: {resp.text}")


# --------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------- #
if not st.session_state.auth_token:
    login_screen()
else:
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Log out"):
            st.session_state.auth_token = None
            st.session_state.username = None
            st.rerun()

    tab_chat, tab_admin = st.tabs(["💬 Chat", "📊 Admin Dashboard"])
    with tab_chat:
        chat_tab()
    with tab_admin:
        admin_tab()
