"""ui/session.py — Streamlit session state helpers."""

import streamlit as st

__all__ = ['SESSION_DEFAULTS', 'init_session_state', 'get_active_bm25']

SESSION_DEFAULTS: dict = {
    'conv':        [],
    'display':     [],
    'total':       0,
    'last':        None,
    'mode':        'chat',
    'url_chunks':  [],
    'bm25_index':  None,
    'url_msg':     None,
    'file_msg':    None,
}


def init_session_state() -> None:
    """Set defaults for missing session state keys only."""
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_active_bm25(base_bm25):
    """Return session BM25 if updated after upload, else the base index."""
    return st.session_state.bm25_index if st.session_state.bm25_index is not None else base_bm25
