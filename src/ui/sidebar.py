"""ui/sidebar.py — Sidebar rendering functions for the Streamlit UI.

WHY THIS FILE EXISTS:
    handlers.py was approaching the 500-line limit. The sidebar rendering
    functions (pipeline info, session stats, document type counts) are a
    natural, self-contained unit that can live in their own module.

ADDING A NEW SIDEBAR PANEL:
    1. Write a new _render_* function here.
    2. Call it from render_sidebar() below.
    You never need to change handlers.py for sidebar-only changes.
"""

import streamlit as st

from src.rag.vector_store import VectorStore
from src.ui.theme import BADGE_CLASSES, CONFIDENCE_BADGE

__all__ = ['render_sidebar']


def render_sidebar(store: VectorStore, local_chunks: list) -> None:
    """Render the right-hand sidebar: pipeline info, session stats, chunk counts.

    Args:
        store:        VectorStore — used to generate source labels for chunks.
        local_chunks: All chunks loaded from the local ./docs folder.
    """
    st.markdown("### Pipeline")

    if st.session_state.last:
        pipeline_data = st.session_state.last['data']
        if st.session_state.last['type'] == 'chat':
            _render_pipeline_chat_info(pipeline_data, store)
        else:
            _render_pipeline_agent_info(pipeline_data)
        st.markdown("---")

    _render_session_stats(local_chunks)
    _render_document_type_counts(local_chunks)


def _render_pipeline_chat_info(data: dict, store: VectorStore) -> None:
    """Show query type badge, confidence badge, and pre/post rerank chunks.

    Args:
        data:  The result dict returned by store.run_pipeline().
        store: VectorStore — used to generate human-readable source labels.
    """
    query_type    = data['query_type']
    badge_class   = BADGE_CLASSES.get(query_type, 'b-gen')
    st.markdown(f'<span class="badge {badge_class}">{query_type}</span>', unsafe_allow_html=True)

    confidence_class = CONFIDENCE_BADGE[data['is_confident']]
    confidence_label = (
        f"conf:{data['best_score']:.2f}" if data['is_confident']
        else f"low:{data['best_score']:.2f}"
    )
    st.markdown(f'<span class="badge {confidence_class}">{confidence_label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Before rerank**")
    for chunk_entry, similarity_score in data['retrieved'][:4]:
        source_label = store._source_label(chunk_entry)
        st.markdown(
            f'<div class="chunk"><span class="cs">{similarity_score:.3f}</span> '
            f'<span class="src">[{chunk_entry["source"]} {source_label}]</span><br/>'
            f'{chunk_entry["text"][:55]}...</div>',
            unsafe_allow_html=True,
        )

    st.markdown("**After rerank**")
    for chunk_entry, similarity_score, rerank_score in data['reranked']:
        source_label = store._source_label(chunk_entry)
        st.markdown(
            f'<div class="chunk">'
            f'<span class="cs">sim:{similarity_score:.2f} re:{rerank_score:.0f}</span> '
            f'<span class="src">[{chunk_entry["source"]} {source_label}]</span><br/>'
            f'{chunk_entry["text"][:55]}...</div>',
            unsafe_allow_html=True,
        )


def _render_pipeline_agent_info(data: dict) -> None:
    """Show a compact list of tool calls the agent made.

    Args:
        data: The result dict returned by agent.run().
    """
    st.markdown("**Agent Steps**")
    for step in data['steps']:
        st.markdown(
            f'<div class="step">{step["step"]}. {step["tool"]}</div>',
            unsafe_allow_html=True,
        )


def _render_session_stats(local_chunks: list) -> None:
    """Show query count, memory turns, total chunks, URL chunks, and mode.

    Args:
        local_chunks: Chunks loaded from the local ./docs folder (not uploads).
    """
    url_chunk_count = len(st.session_state.url_chunks)
    total_chunks    = len(local_chunks) + url_chunk_count

    st.markdown("**Session**")
    stats = [
        ("Queries",    st.session_state.total),
        ("Memory",     f"{len(st.session_state.conv) // 2} turns"),
        ("Chunks",     total_chunks),
        ("URL chunks", url_chunk_count),
        ("Mode",       st.session_state.mode),
    ]
    for label, value in stats:
        st.markdown(
            f'<div class="stat">{label} <span class="sv">{value}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")


def _render_document_type_counts(local_chunks: list) -> None:
    """Show how many chunks belong to each document type (PDF, DOCX, etc.).

    Args:
        local_chunks: Chunks loaded from the local ./docs folder.
    """
    # Count how many chunks are from each document type
    type_counts: dict = {}
    for chunk in local_chunks:
        document_type = chunk.get('type', '?')
        type_counts[document_type] = type_counts.get(document_type, 0) + 1

    st.markdown("**Document Types**")
    for document_type, count in sorted(type_counts.items()):
        st.markdown(
            f'<div class="stat">{document_type.upper()} '
            f'<span class="sv">{count}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
