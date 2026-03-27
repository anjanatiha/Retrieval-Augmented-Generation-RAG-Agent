"""app.py — Streamlit UI entry point (under 50 lines).

This file only wires things together.
All rendering and event logic lives in ui/handlers.py.
All constants live in src/rag/config.py.
"""

import streamlit as st

from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from ui.handlers import (
    handle_file_upload, handle_url_ingestion, handle_user_input,
    render_chat_history, render_clear_button, render_header,
    render_mode_selector, render_sidebar,
)
from ui.session import get_active_bm25, init_session_state
from ui.theme import CSS


@st.cache_resource
def initialize():
    """Load documents and build the vector index once, then cache the result."""
    loader = DocumentLoader()
    loader.ensure_folders()
    chunks = loader.chunk_all_documents()
    store  = VectorStore()
    store.build_or_load(chunks)
    return loader, store


loader, store = initialize()
chunks        = store.chunks

st.set_page_config(page_title="Ask Your Documents", page_icon="🐱", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
init_session_state()
get_active_bm25(store.bm25_index)

_needs_rerun    = False
col_main, col_side = st.columns([3, 1])

with col_main:
    render_header()
    render_mode_selector()
    _needs_rerun |= handle_url_ingestion(loader, store)
    _needs_rerun |= handle_file_upload(loader, store)
    render_chat_history()
    render_clear_button(store)

# Refresh after ingestion — deferred so both columns finish rendering first
if _needs_rerun:
    st.rerun()

placeholder = "Ask a question..." if st.session_state.mode == 'chat' else "Give the agent a task..."
user_input  = st.chat_input(placeholder)
if user_input and user_input.strip():
    handle_user_input(user_input.strip(), store)
    st.rerun()

with col_side:
    render_sidebar(store, chunks)
