"""app.py — Streamlit entry point. Wires handlers, renderers, and session together."""

import streamlit as st

from src.ui.handlers import (
    initialize,
    handle_file_upload, handle_topic_search, handle_url_ingestion,
    handle_user_input, render_sidebar,
)
from src.ui.renderers import (
    render_chat_history, render_clear_button,
    render_footer, render_header, render_mode_selector,
)
from src.ui.session import get_active_bm25, init_session_state
from src.ui.theme import CSS


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
    _needs_rerun |= handle_topic_search(loader, store)
    _needs_rerun |= handle_url_ingestion(loader, store)
    _needs_rerun |= handle_file_upload(loader, store)
    render_chat_history()
    render_clear_button(store)
    render_footer()

if _needs_rerun:
    st.rerun()

placeholder = "Ask a question..." if st.session_state.mode == 'chat' else "Give the agent a task..."
user_input  = st.chat_input(placeholder)
if user_input and user_input.strip():
    handle_user_input(user_input.strip(), store)
    st.rerun()

with col_side:
    render_sidebar(store, chunks)
