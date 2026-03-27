"""ui/handlers.py — All Streamlit rendering and event-handler functions.

WHY THIS FILE EXISTS:
    app.py must stay under 50 lines so it is easy to read at a glance.
    All the real UI work — showing chat messages, processing file uploads,
    handling URL fetches, rendering the sidebar — lives here instead.

HOW IT IS ORGANISED:
    Public functions (no leading underscore) are called directly from app.py.
    Private helpers (leading underscore) are used only inside this module.

ADDING A NEW PANEL OR FEATURE:
    1. Write a new function here with a clear docstring.
    2. Import it in app.py and call it in the right place.
    You never need to touch the existing functions to add something new.
"""

import logging
import os
import tempfile
from typing import Optional

import streamlit as st

from src.rag.agent import Agent
from src.rag.vector_store import VectorStore
from ui.theme import BADGE_CLASSES, CONFIDENCE_BADGE

# This module's logger — errors go to the logging system, not the terminal
logger = logging.getLogger(__name__)

__all__ = [
    'render_header',
    'render_mode_selector',
    'handle_url_ingestion',
    'handle_file_upload',
    'render_chat_history',
    'render_clear_button',
    'handle_user_input',
    'render_sidebar',
]


# ── Public rendering functions ─────────────────────────────────────────────────


def render_header() -> None:
    """Show the app title and tagline at the top of the main column."""
    st.markdown('<div class="rag-title">Ask Your Documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rag-sub">'
        'chunking · hybrid search · reranking · agent · '
        'PDF · TXT · DOCX · XLSX · PPTX · CSV · MD · HTML · URL'
        '</div>',
        unsafe_allow_html=True,
    )


def render_mode_selector() -> None:
    """Show the Chat / Agent radio and save the user's choice in session state.

    When Agent mode is selected, also shows a panel listing the available tools
    so the user knows what they can ask the agent to do.
    """
    chosen_mode = st.radio(
        "Mode:",
        ["Chat", "Agent"],
        horizontal=True,
        index=0 if st.session_state.mode == 'chat' else 1,
    )
    # Store as lowercase so the rest of the code can compare with 'chat' / 'agent'
    st.session_state.mode = chosen_mode.lower()

    if st.session_state.mode == 'agent':
        _render_agent_tools_panel()


def handle_url_ingestion(loader, store: VectorStore) -> bool:
    """Show the URL input form and process any URL the user submits.

    Fetches the URL, detects its file type, chunks the content, embeds the
    chunks, and adds them to the live knowledge base.

    Args:
        loader: DocumentLoader — handles fetching and chunking the URL.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if a URL was successfully submitted and the page should refresh.
        False if nothing happened (form not submitted or already shown result).
    """
    needs_rerun = False

    with st.expander("Add a URL to knowledge base", expanded=False):
        with st.form('url_form', clear_on_submit=True):
            url_input = st.text_input(
                "URL:",
                placeholder="https://example.com/page  or  https://example.com/file.pdf",
            )
            submitted = st.form_submit_button("Fetch & index →")

        if submitted and url_input.strip():
            _process_url(url_input.strip(), loader, store)
            needs_rerun = True

        # Show the result message from the most recent URL submission
        if st.session_state.url_msg:
            kind, message = st.session_state.url_msg
            st.success(message) if kind == 'ok' else st.error(message)

    return needs_rerun


def handle_file_upload(loader, store: VectorStore) -> bool:
    """Show the file upload panel and index any file the user uploads.

    Saves the uploaded file to a temporary path, dispatches it through the
    correct chunker, embeds the chunks, and adds them to the knowledge base.

    Args:
        loader: DocumentLoader — handles chunking the file by its type.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if a file was indexed and the page should refresh. False otherwise.
    """
    needs_rerun = False

    with st.expander("Upload a file to knowledge base", expanded=False):
        uploaded_file = st.file_uploader(
            "Supported: PDF, TXT, DOCX, XLSX, PPTX, CSV, MD, HTML",
            type=[
                "pdf", "txt", "docx", "doc", "xlsx", "xls",
                "pptx", "ppt", "csv", "md", "markdown", "html", "htm",
            ],
            key="file_uploader",
        )

        if uploaded_file and st.button("Index file →", key="file_index_btn"):
            _process_uploaded_file(uploaded_file, loader, store)
            needs_rerun = True

        # Show the result message from the most recent upload
        if st.session_state.get('file_msg'):
            kind, message = st.session_state.file_msg
            st.success(message) if kind == 'ok' else st.error(message)

    return needs_rerun


def render_chat_history() -> None:
    """Display all previous messages in the conversation as chat bubbles."""
    for message in st.session_state.display:
        avatar = _pick_avatar(message['role'])
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'], unsafe_allow_html=True)


def render_clear_button(store: VectorStore) -> None:
    """Show a Clear button below the chat when there are messages to clear.

    Clicking it wipes the on-screen conversation and the store's memory so
    the next question starts with a clean slate.

    Args:
        store: VectorStore whose conversation history should also be cleared.
    """
    if not st.session_state.display:
        return  # Nothing to clear yet — hide the button

    _, button_column = st.columns([6, 1])
    with button_column:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.conv    = []
            st.session_state.display = []
            st.session_state.last    = None
            st.session_state.total   = 0
            store.clear_conversation()
            st.rerun()


def handle_user_input(user_input: str, store: VectorStore) -> None:
    """Process a question or task from the chat input and store the result.

    Routes to agent mode or chat (pipeline) mode depending on session state.
    Shows a progress bar while running. Stores the result so app.py can call
    st.rerun() and display it.

    Args:
        user_input: The text the user typed into the chat box.
        store:      VectorStore used for retrieval and response generation.
    """
    # Clear any stale URL message so it doesn't reappear after a query
    st.session_state.url_msg = None
    st.session_state.display.append({'role': 'user', 'content': user_input})

    progress_slot = st.empty()

    if st.session_state.mode == 'agent':
        result = _run_agent(user_input, store, progress_slot)
        answer_html = _format_agent_steps_html(result['steps'])
        content = f"{answer_html}<br/><strong>Answer:</strong> {result['answer']}"
        st.session_state.display.append({'role': 'agent', 'content': content})
        st.session_state.last = {'type': 'agent', 'data': result}
    else:
        result = _run_pipeline(user_input, store, progress_slot)
        st.session_state.display.append({'role': 'assistant', 'content': result['response']})
        st.session_state.last = {'type': 'chat', 'data': result}

    st.session_state.total += 1


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


# ── Private helpers ────────────────────────────────────────────────────────────


def _render_agent_tools_panel() -> None:
    """Show the agent tools reference card when agent mode is active."""
    st.markdown(
        """
        <div style="background:#e8f4fd;border-left:3px solid #1565a0;
                    border-radius:6px;padding:10px 14px;margin:8px 0;font-size:0.82rem;">
        <b>🤖 Agent Tools Available</b><br><br>
        <b>🔍 rag_search</b> — search your documents<br>
        <i>e.g. "what skills does the resume mention?"</i><br><br>
        <b>🧮 calculator</b> — evaluate math expressions<br>
        <i>e.g. "what is 15% of 85000?"</i><br><br>
        <b>📝 summarise</b> — summarise any document or section<br>
        <i>e.g. "summarise the resume"</i><br><br>
        <b>💬 sentiment</b> — analyse tone &amp; sentiment of content<br>
        <i>e.g. "what is the sentiment of the resume?"</i><br><br>
        <b>✅ finish</b> — returns the final answer
        </div>
        """,
        unsafe_allow_html=True,
    )


def _process_url(url: str, loader, store: VectorStore) -> None:
    """Fetch a URL, chunk it, and add the chunks to the knowledge base.

    Updates st.session_state.url_msg with 'ok' or 'err' and a message string.

    Args:
        url:    The public URL to fetch.
        loader: DocumentLoader — does the fetching and chunking.
        store:  VectorStore   — stores the chunks and rebuilds BM25.
    """
    try:
        with st.spinner(f"Fetching {url}..."):
            new_chunks = loader.chunk_url(url)

        if new_chunks:
            with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                store.add_chunks(new_chunks, id_prefix='url')

            # BM25 must be rebuilt so the new chunks are included in keyword search
            st.session_state.url_chunks.extend(new_chunks)
            store.rebuild_bm25(store.chunks)
            st.session_state.bm25_index = store.bm25_index

            total_chunks = store.collection.count()
            st.session_state.url_msg = (
                'ok',
                f"Added {len(new_chunks)} chunks. Total in knowledge base: {total_chunks}",
            )
        else:
            st.session_state.url_msg = (
                'err',
                "Could not fetch or parse the URL. Check it is publicly accessible.",
            )

    except Exception as error:
        logger.error("URL ingestion failed for '%s': %s", url, error, exc_info=True)
        st.session_state.url_msg = ('err', f"Error fetching URL: {error}")


def _process_uploaded_file(uploaded_file, loader, store: VectorStore) -> None:
    """Write an uploaded file to a temp path, chunk it, and index the chunks.

    Updates st.session_state.file_msg with 'ok' or 'err' and a message string.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        loader:        DocumentLoader — dispatches to the right chunker.
        store:         VectorStore   — stores and indexes the chunks.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    document_type  = loader.ext_to_type.get(file_extension, 'txt')

    # Chunkers need a real file path, so write to a temporary file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            file_info = {
                'filepath':      tmp_path,
                'filename':      uploaded_file.name,
                'detected_type': document_type,
                'is_misplaced':  False,
            }
            new_chunks = loader._dispatch_chunker(file_info)

        if new_chunks:
            with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                store.add_chunks(new_chunks, id_prefix='file')

            # Rebuild BM25 so the uploaded file appears in keyword search
            st.session_state.url_chunks.extend(new_chunks)
            store.rebuild_bm25(store.chunks)
            st.session_state.bm25_index = store.bm25_index

            total_chunks = store.collection.count()
            st.session_state.file_msg = (
                'ok',
                f"Indexed '{uploaded_file.name}' — {len(new_chunks)} chunks added. "
                f"Total: {total_chunks}",
            )
        else:
            st.session_state.file_msg = (
                'err',
                f"Could not extract text from '{uploaded_file.name}'.",
            )

    except Exception as error:
        logger.error(
            "File indexing failed for '%s': %s",
            uploaded_file.name,
            error,
            exc_info=True,
        )
        st.session_state.file_msg = ('err', f"Error indexing '{uploaded_file.name}': {error}")

    finally:
        # Always delete the temp file — even if something went wrong above
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_error:
            logger.warning("Could not delete temp file '%s': %s", tmp_path, cleanup_error)


def _run_agent(user_input: str, store: VectorStore, progress_slot) -> dict:
    """Run the agent loop and return the result dict.

    Args:
        user_input:    The task the user typed.
        store:         VectorStore the agent uses for rag_search.
        progress_slot: A st.empty() slot for showing a progress bar.

    Returns:
        Agent result dict with keys: 'answer', 'steps'.
    """
    progress_bar = progress_slot.progress(0, text="Agent starting...")
    progress_bar.progress(30, text="Agent: searching knowledge base...")
    result = Agent(store).run(user_input, streamlit_mode=True)
    progress_bar.progress(100, text="Agent: done!")
    progress_slot.empty()
    return result


def _run_pipeline(user_input: str, store: VectorStore, progress_slot) -> dict:
    """Run the RAG pipeline and return the result dict.

    Args:
        user_input:    The question the user typed.
        store:         VectorStore that runs the full pipeline.
        progress_slot: A st.empty() slot for showing a progress bar.

    Returns:
        Pipeline result dict with keys: 'response', 'retrieved', 'reranked',
        'is_confident', 'best_score', 'query_type'.
    """
    progress_bar = progress_slot.progress(0, text="Classifying query...")
    progress_bar.progress(25, text="Retrieving documents...")
    progress_bar.progress(55, text="Reranking results...")
    progress_bar.progress(75, text="Generating answer...")
    result = store.run_pipeline(user_input, streamlit_mode=True)
    progress_bar.progress(100, text="Done!")
    progress_slot.empty()
    return result


def _format_agent_steps_html(steps: list) -> str:
    """Turn a list of agent step dicts into an HTML string for the chat bubble.

    Args:
        steps: List of dicts with keys: 'step', 'tool', 'arg', 'result'.

    Returns:
        HTML string with one <div class="step"> per step.
    """
    html_parts = []
    for step in steps:
        # Truncate long arguments so they don't overflow the chat bubble
        short_arg    = step["arg"][:50]    + "..." if len(step["arg"]) > 50    else step["arg"]
        short_result = step["result"][:80] + "..." if len(step["result"]) > 80 else step["result"]
        html_parts.append(
            f'<div class="step">'
            f'Step {step["step"]}: {step["tool"]}({short_arg}) → {short_result}'
            f'</div>'
        )
    return "".join(html_parts)


def _pick_avatar(role: str) -> str:
    """Return the emoji avatar for a given message role.

    Args:
        role: 'user', 'agent', or 'assistant'.

    Returns:
        An emoji string.
    """
    avatars = {'user': '🧑', 'agent': '🤖', 'assistant': '💬'}
    return avatars.get(role, '💬')


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
