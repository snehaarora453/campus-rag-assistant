"""
Streamlit UI for the Campus RAG Assistant.

Talks to the FastAPI backend over HTTP. Run the backend first:
    uvicorn main:app --reload --port 8000   (from backend/)
Then run this:
    streamlit run streamlit_app.py          (from frontend/)
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Campus RAG Assistant", page_icon="🎓", layout="wide")
st.title("🎓 Campus RAG Assistant")
st.caption("Ask questions about your notes — answers are grounded and cited.")

# --- Backend health check ---
with st.sidebar:
    st.header("Backend status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        if health.get("status") == "ok":
            st.success("Connected to backend")
            if not health.get("openai_key_configured"):
                st.error("OPENAI_API_KEY is not set on the backend (.env).")
        else:
            st.warning("Backend reachable but unhealthy.")
    except requests.exceptions.RequestException:
        st.error(
            f"Cannot reach backend at {BACKEND_URL}.\n\n"
            "Make sure it's running:\n`uvicorn main:app --reload --port 8000`"
        )
        st.stop()

    st.markdown("---")
    if st.button("⚠️ Reset entire knowledge base"):
        try:
            requests.post(f"{BACKEND_URL}/reset", timeout=10)
            st.success("Knowledge base cleared.")
        except requests.exceptions.RequestException as e:
            st.error(f"Reset failed: {e}")

# --- Upload & ingest ---
st.subheader("1. Upload your notes")
uploaded_files = st.file_uploader(
    "PDF, TXT, or Markdown files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Process documents", type="primary"):
    if not uploaded_files:
        st.error("Upload at least one file first.")
    else:
        progress = st.progress(0.0)
        errors = []
        for i, f in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Indexing {f.name}...")
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ingest",
                    files={"file": (f.name, f.getvalue())},
                    timeout=120,
                )
                if resp.status_code != 200:
                    errors.append(f"{f.name}: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.RequestException as e:
                errors.append(f"{f.name}: {e}")
        progress.progress(1.0, text="Done")
        if errors:
            for e in errors:
                st.error(e)
        st.success(f"Processed {len(uploaded_files) - len(errors)} of {len(uploaded_files)} file(s).")

# --- Show current knowledge base ---
try:
    sources_resp = requests.get(f"{BACKEND_URL}/sources", timeout=5).json()
    sources_list = sources_resp.get("sources", [])
except requests.exceptions.RequestException:
    sources_list = []

if sources_list:
    with st.expander(f"📚 Notes in knowledge base ({len(sources_list)})"):
        for s in sources_list:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{s['source']}** — {s['chunks']} chunks")
            if col2.button("Remove", key=f"remove_{s['source']}"):
                requests.delete(f"{BACKEND_URL}/sources/{s['source']}", timeout=10)
                st.rerun()

st.markdown("---")

# --- Chat ---
st.subheader("2. Ask a question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source']}**, chunk {s['chunk_index']}\n\n> {s['excerpt']}..."
                    )

question = st.chat_input("Ask something about your notes...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query", json={"question": question}, timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer, sources = data["answer"], data["sources"]
                else:
                    answer, sources = f"Error: {resp.json().get('detail', resp.text)}", []
            except requests.exceptions.RequestException as e:
                answer, sources = f"Request failed: {e}", []

        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for s in sources:
                    st.markdown(
                        f"**{s['source']}**, chunk {s['chunk_index']}\n\n> {s['excerpt']}..."
                    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
