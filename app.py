"""Web interface for grounded-rag: upload PDFs, index them, chat with cited answers.

Save this file in the project root (next to the `src` folder), then run:
    streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from src.graph import build_graph
from src.ingest import ingest
from src.llm import get_vectorstore

st.set_page_config(page_title="Grounded RAG", page_icon="📚", layout="wide")
st.title("📚 Grounded RAG")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = set()


@st.cache_resource
def get_graph():
    return build_graph()


def human_page(doc) -> str:
    """PyPDFLoader numbers pages from 0, so add 1 to match the page you see in a PDF viewer."""
    page = doc.metadata.get("page", "?")
    return str(page + 1) if isinstance(page, int) else str(page)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Documents")
    files = st.file_uploader("Upload PDFs (with a real text layer)", type="pdf", accept_multiple_files=True)
    if st.button("Index uploaded PDFs", disabled=not files):
        for f in files:
            if f.name in st.session_state.indexed:
                st.info(f"{f.name}: already indexed in this session")
                continue
            path = UPLOAD_DIR / f.name
            path.write_bytes(f.getbuffer())
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    _, n_chunks = ingest(str(path))
                    st.session_state.indexed.add(f.name)
                    st.success(f"{f.name}: {n_chunks} chunks indexed")
                except Exception as e:  # show the real error instead of crashing
                    st.error(f"{f.name}: {e}")

    if st.session_state.indexed:
        st.caption("Indexed in this session: " + ", ".join(sorted(st.session_state.indexed)))

    st.header("2. Options")
    show_details = st.checkbox("Show critic details", value=False)
    if st.button("Clear index"):
        try:
            get_vectorstore().delete_collection()
            st.session_state.indexed.clear()
            st.success("Index cleared.")
        except Exception as e:
            st.error(str(e))

# ------------------------------------------------------------------- chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for line in m["sources"]:
                    st.markdown(line)

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (this can take a while on a computer without a GPU)"):
            try:
                state = get_graph().invoke({"question": question})
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        answer = state.get("answer", "")
        st.markdown(answer)

        sources = []
        for i, d in enumerate(state.get("documents", []), start=1):
            snippet = d.page_content[:300].replace("\n", " ")
            sources.append(f"**[{i}]** {d.metadata.get('source', '?')}, page {human_page(d)}  \n> {snippet}...")
        if sources:
            with st.expander("Sources"):
                for line in sources:
                    st.markdown(line)

        if show_details:
            with st.expander("Critic details"):
                st.write(f"Retrieval rounds: {state.get('attempts', 1)}")
                st.write(f"Grounded: {state.get('grounded')}")
                st.code(state.get("critique", "(none)"))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})