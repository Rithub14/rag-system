import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/api/query"
INGEST_URL = f"{API_BASE_URL}/api/ingest/file"
SESSION_URL = f"{API_BASE_URL}/api/session"


def post_query(payload: Dict[str, Any]) -> Dict[str, Any]:
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            API_URL, json=payload, headers={"X-Session-Id": st.session_state.user_id}
        )
        resp.raise_for_status()
        return resp.json()

def ingest_pdf(user_id: str, file) -> Dict[str, Any]:
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            INGEST_URL, files=files, headers={"X-Session-Id": user_id}
        )
        resp.raise_for_status()
        return resp.json()

def ensure_user_id() -> str:
    params = st.query_params
    if "uid" in params and params["uid"]:
        return params["uid"]
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(SESSION_URL)
                resp.raise_for_status()
                user_id = resp.json()["user_id"]
                st.query_params["uid"] = user_id
                return user_id
        except Exception as exc:
            last_exc = exc
            time.sleep(1 + attempt)
    fallback = str(uuid.uuid4())
    st.warning(
        "Session service is slow/unavailable right now, so a temporary session "
        "ID was created. Please refresh later to restore normal limits."
    )
    st.query_params["uid"] = fallback
    return fallback

st.set_page_config(page_title="RAG App", layout="wide")
st.title("RAG App")

if "user_id" not in st.session_state:
    st.session_state.user_id = ensure_user_id()
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = []
if "chat_turns" not in st.session_state:
    st.session_state.chat_turns = []
if "chat_started_at" not in st.session_state:
    st.session_state.chat_started_at = datetime.now(timezone.utc)

if datetime.now(timezone.utc) - st.session_state.chat_started_at > timedelta(hours=1):
    st.session_state.chat_turns = []
    st.session_state.chat_started_at = datetime.now(timezone.utc)

with st.sidebar:
    st.header("Upload File and Set Parameters")
    st.caption(f"Session ID: {st.session_state.user_id}")
    uploaded = st.file_uploader(
        "Upload file", type=["pdf", "docx", "pptx", "xlsx"],
        help="Limit 10MB per file • PDF, DOCX, PPTX, XLSX"
    )
    if st.button("Ingest"):
        if not uploaded:
            st.error("Please select a PDF.")
        else:
            try:
                result = ingest_pdf(st.session_state.user_id, uploaded)
                st.session_state.doc_ids.append(result["doc_id"])
                st.success(f"Ingested {result['chunks']} chunks.")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    st.warning(
                        "Upload limit reached: this demo allows 1 PDF per hour per browser. "
                        "This helps keep costs low on the free tier. Please try again later."
                    )
                else:
                    st.error(f"Ingest failed: {exc}")
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

    st.header("Parameters")
    k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1)
    max_context_tokens = st.number_input(
        "Max context tokens", min_value=200, max_value=6000, value=1500, step=100
    )
    max_answer_tokens = st.number_input(
        "Max answer tokens", min_value=50, max_value=1000, value=300, step=50
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    rerank = st.checkbox("Rerank", value=True)
    enable_tools = st.checkbox("Enable tool router", value=True)
    enable_followups = st.checkbox("Generate follow-ups", value=True)
    enable_planning = st.checkbox("Enable planning (slower)", value=False)
    include_citations = st.checkbox("Include citations", value=True)
    show_raw = st.checkbox("Show raw results", value=False)

query = st.text_area("Query", placeholder="Enter text", height=120)

if st.button("Search"):
    payload = {
        "query": query,
        "k": int(k),
        "max_context_tokens": int(max_context_tokens),
        "max_answer_tokens": int(max_answer_tokens),
        "temperature": float(temperature),
        "rerank": bool(rerank),
        "enable_tools": bool(enable_tools),
        "enable_followups": bool(enable_followups),
        "enable_planning": bool(enable_planning),
        "include_citations": bool(include_citations),
    }
    try:
        data = post_query(payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            st.warning(
                "Query limit reached: this demo allows 10 questions per hour per browser "
                "to keep free-tier costs under control. Please try again later."
            )
        else:
            st.error(f"Request failed: {exc}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    else:
        st.subheader("Answer")
        st.write(data.get("answer", ""))
        st.session_state.chat_turns.append(
            {"query": query, "answer": data.get("answer", "")}
        )
        if len(st.session_state.chat_turns) > 10:
            st.session_state.chat_turns = st.session_state.chat_turns[-10:]

        if include_citations:
            st.subheader("Citations (used)")
            st.json(data.get("citations", {}).get("used", []))

        st.subheader("Context")
        st.code(data.get("context", ""), language="text")

        if data.get("tool_used") and data.get("tool_output"):
            st.subheader(f"Tool output: {data['tool_used']}")
            st.code(data.get("tool_output", ""), language="text")

        followups = data.get("follow_ups") or []
        if followups:
            st.subheader("Follow-up questions")
            for item in followups:
                st.markdown(f"- {item}")

        if include_citations:
            st.subheader("Related citations")
            st.json(data.get("citations", {}).get("related", []))

        if show_raw:
            st.subheader("Raw results")
            st.json(data.get("results", []))

        with st.expander("Full response JSON"):
            st.json(data)

st.subheader("Chat history")
if st.session_state.chat_turns:
    for turn in st.session_state.chat_turns:
        st.markdown(f"**You:** {turn['query']}")
        st.markdown(f"**Answer:** {turn['answer']}")
        st.divider()
else:
    st.caption("No messages yet.")
