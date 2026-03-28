"""ui/handlers.py — All Streamlit rendering and event-handler functions.

WHY THIS FILE EXISTS:
    app.py must stay under 50 lines so it is easy to read at a glance.
    All the real UI work — showing chat messages, processing file uploads,
    handling URL fetches — lives here instead.

HOW IT IS ORGANISED:
    Public functions (no leading underscore) are called directly from app.py.
    Private helpers (leading underscore) are used only inside this module.
    Sidebar rendering is in src/ui/sidebar.py (extracted to keep this file
    under the 500-line limit).

ADDING A NEW PANEL OR FEATURE:
    1. Write a new function here with a clear docstring.
    2. Import it in app.py and call it in the right place.
    You never need to touch the existing functions to add something new.
"""

import logging
import os
import tempfile

import streamlit as st

from src.rag.agent import Agent
from src.rag.config import URL_CRAWL_MAX_DEPTH, URL_CRAWL_MAX_PAGES
from src.rag.vector_store import VectorStore
from src.ui.ingestion import process_url, process_url_recursive
from src.ui.sidebar import render_sidebar

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

    The recursive crawl toggle is placed OUTSIDE the form so that checking
    it immediately shows or hides the depth, max-pages, and type controls
    without requiring a form submit first. The URL text box and the submit
    button stay inside the form so the URL is cleared after each submission.

    Args:
        loader: DocumentLoader — handles fetching and chunking the URL.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if a URL was successfully submitted and the page should refresh.
        False if nothing happened (form not submitted or already shown result).
    """
    needs_rerun = False

    with st.expander("🌐 Add a URL to knowledge base", expanded=False):

        # ── Toggle outside the form so it triggers an immediate rerun ──────
        use_recursive = st.checkbox(
            "🕷️ Recursive crawl — follow links and index linked pages",
            value=False,
            key='use_recursive',
        )

        # ── Crawl settings — visible only when recursive mode is on ────────
        crawl_depth     = URL_CRAWL_MAX_DEPTH
        crawl_max_pages = URL_CRAWL_MAX_PAGES
        crawl_topic     = ''
        allowed_types   = None

        if use_recursive:
            col_depth, col_pages = st.columns(2)
            with col_depth:
                crawl_depth = st.number_input(
                    "Depth (1 = direct links only)",
                    min_value=1, max_value=3, value=URL_CRAWL_MAX_DEPTH,
                    key='crawl_depth',
                )
            with col_pages:
                crawl_max_pages = st.number_input(
                    "Max pages",
                    min_value=1, max_value=50, value=URL_CRAWL_MAX_PAGES,
                    key='crawl_max_pages',
                )

            # Topic filter — only crawl pages whose URL path contains this word
            crawl_topic = st.text_input(
                "Topic filter (optional)",
                placeholder="e.g. python  or  machine-learning  or  api",
                help="Only crawl pages whose URL path contains this keyword. "
                     "Leave empty to crawl all pages on the domain.",
                key='crawl_topic',
            )

            # Document type filter — choose which types to index during the crawl
            st.caption("Index these document types:")
            type_cols   = st.columns(7)
            type_labels = ['HTML', 'PDF', 'DOCX', 'XLSX', 'CSV', 'PPTX', 'MD']
            type_keys   = ['html', 'pdf', 'docx', 'xlsx', 'csv', 'pptx', 'md']
            checked_types = []
            for col, label, key in zip(type_cols, type_labels, type_keys):
                with col:
                    if st.checkbox(label, value=True, key=f'crawl_type_{key}'):
                        checked_types.append(key)
            # None means "accept all types" — only filter if at least one is unchecked
            allowed_types = set(checked_types) if len(checked_types) < 7 else None

        # ── URL input + submit button inside the form (cleared on submit) ──
        with st.form('url_form', clear_on_submit=True):
            url_input = st.text_input(
                "URL:",
                placeholder="https://example.com/page  or  https://example.com/file.pdf",
            )
            submitted = st.form_submit_button("Fetch & index →")

        if submitted and url_input.strip():
            if use_recursive:
                process_url_recursive(
                    url_input.strip(), loader, store,
                    depth=int(crawl_depth),
                    max_pages=int(crawl_max_pages),
                    allowed_types=allowed_types,
                    topic_filter=crawl_topic,
                )
            else:
                process_url(url_input.strip(), loader, store)
            needs_rerun = True

        # Show the result message from the most recent URL submission
        if st.session_state.url_msg:
            kind, message = st.session_state.url_msg
            st.success(message) if kind == 'ok' else st.error(message)

    return needs_rerun


def handle_file_upload(loader, store: VectorStore) -> bool:
    """Show the file upload panel and index any files the user uploads.

    Accepts one or many files at once. Each file is saved to a temporary
    path, dispatched through the correct chunker, embedded, and added to
    the knowledge base. BM25 is rebuilt after every indexed file.

    Args:
        loader: DocumentLoader — handles chunking each file by its type.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if a file was indexed and the page should refresh. False otherwise.
    """
    needs_rerun = False

    with st.expander("📎 Upload files or a folder to knowledge base", expanded=False):
        st.caption("Select one file, multiple files, or all files from a folder at once.")
        uploaded_files = st.file_uploader(
            "Supported: PDF, TXT, DOCX, XLSX, PPTX, CSV, MD, HTML",
            type=[
                "pdf", "txt", "docx", "doc", "xlsx", "xls",
                "pptx", "ppt", "csv", "md", "markdown", "html", "htm",
            ],
            key="file_uploader",
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Index files →", key="file_index_btn"):
            for uploaded_file in uploaded_files:
                _process_uploaded_file(uploaded_file, loader, store)
            needs_rerun = True

        # Show the result message from the most recent upload batch
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


# ── Private helpers ────────────────────────────────────────────────────────────


def _render_agent_tools_panel() -> None:
    """Show the agent tools reference card when agent mode is active."""
    st.markdown(
        """
        <div class="tools-panel">
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


def _process_uploaded_file(uploaded_file, loader, store: VectorStore) -> None:
    """Write an uploaded file to a temp path, chunk it, and index the chunks.

    Called once per file when the user uploads one or many files at once.
    Updates st.session_state.file_msg after each file so the user can see
    rolling progress when a batch of files is being indexed.

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
            with st.spinner(f"Embedding {len(new_chunks)} chunks from {uploaded_file.name}..."):
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
