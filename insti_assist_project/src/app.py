"""
app.py
Streamlit UI for IITB Insti-Assist (Hostel & Campus Life scope).

Run with:
    streamlit run src/app.py
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))
from rag import RAGPipeline

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🏠", layout="centered")

st.title("🏠 IITB Insti-Assist")
st.caption("Hostel & Campus Life Assistant — grounded answers from real IIT Bombay documents.")


@st.cache_resource(show_spinner="Loading models and index (first run only)...")
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except EnvironmentError as e:
    st.error(str(e))
    st.info("Create a `.env` file in the project root with: `GEMINI_API_KEY=your_key_here`")
    st.stop()

# --- Chat history (Bonus: multi-turn memory support) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources used"):
                seen = set()
                for s in msg["sources"]:
                    if s["source_file"] not in seen:
                        st.markdown(f"**{s['source_file']}** (similarity: {s['score']:.2f})")
                        st.caption(s["source_url"])
                        st.text(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))
                        seen.add(s["source_file"])

query = st.chat_input("Ask about hostels, mess, campus facilities...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking..."):
            result = pipeline.rag_answer(query)
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📄 Sources used"):
                seen = set()
                for s in result["sources"]:
                    if s["source_file"] not in seen:
                        st.markdown(f"**{s['source_file']}** (similarity: {s['score']:.2f})")
                        st.caption(s["source_url"])
                        st.text(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))
                        seen.add(s["source_file"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant answers questions about **IIT Bombay Hostel & Campus Life** "
        "using Retrieval-Augmented Generation (RAG) over real institute documents "
        "(hostel rules, mess rules, campus code of conduct, infrastructure info)."
    )
    st.markdown("It will say **\"I don't know\"** if the answer isn't in its knowledge base.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
