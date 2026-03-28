"""handlers.py — All Gradio event handlers and the UI builder for the HF Space.

WHY THIS FILE EXISTS:
    app.py must stay under 50 lines so it is easy to read at a glance.
    All the real work — initialising singletons, handling chat, file uploads,
    URL fetches, and building the Gradio UI — lives here instead.

HOW IT IS ORGANISED:
    _initialize()   — creates DocumentLoader and VectorStore once (lazy singleton)
    _chunk_count()  — quick helper to read the current chunk count
    Helpers         — _pipeline_summary(), _agent_steps_md() for formatting output
    Handlers        — chat(), upload_file(), fetch_url(), clear_chat()
    build_demo()    — assembles and returns the Gradio Blocks app

ADDING A NEW FEATURE:
    1. Write a new handler function here with a clear docstring.
    2. Add its inputs/outputs to build_demo().
    You never need to change app.py.
"""

import logging
import os
import re
import tempfile

import gradio as gr

from src.rag.agent import Agent
from src.rag.config import EXT_TO_TYPE, URL_CRAWL_MAX_DEPTH, URL_CRAWL_MAX_PAGES
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore

# Module-level logger — errors go to the logging system, not the terminal
logger = logging.getLogger(__name__)

__all__ = ['build_demo']

# ── Singletons ─────────────────────────────────────────────────────────────────
# These are created once on first use and reused for the lifetime of the Space.

_loader: DocumentLoader = None   # type: ignore[assignment]
_store:  VectorStore    = None   # type: ignore[assignment]


def _initialize():
    """Create the DocumentLoader and VectorStore on first call, then reuse them.

    Returns:
        Tuple of (DocumentLoader, VectorStore).
    """
    global _loader, _store
    if _loader is None:
        _loader = DocumentLoader()
    if _store is None:
        _store = VectorStore()
        _store.build_or_load([])   # start with an empty knowledge base
    return _loader, _store


def _chunk_count() -> int:
    """Return the number of chunks currently in the VectorStore collection."""
    return _store.collection.count() if _store and _store.collection else 0


# ── Formatting helpers ─────────────────────────────────────────────────────────


def _pipeline_summary(data: dict) -> str:
    """Format run_pipeline output as a markdown string for the pipeline info panel.

    Args:
        data: The dict returned by store.run_pipeline(streamlit_mode=True).

    Returns:
        A markdown string with query type, confidence score, and chunk previews.
    """
    if not data:
        return ""

    query_type  = data.get('query_type', '')
    best_score  = data.get('best_score', 0.0)
    is_confident = data.get('is_confident', False)

    lines = [
        f"**Query type:** `{query_type}`",
        f"**Confidence:** {'✅' if is_confident else '⚠️'} `{best_score:.3f}`",
        "",
        "**Before rerank (top 4)**",
    ]

    for chunk_entry, similarity_score in data.get('retrieved', [])[:4]:
        source_label = _store._source_label(chunk_entry)
        lines.append(
            f"- `{similarity_score:.3f}` "
            f"[{chunk_entry['source']} {source_label}] "
            f"{chunk_entry['text'][:60]}..."
        )

    lines += ["", "**After rerank**"]
    for chunk_entry, similarity_score, rerank_score in data.get('reranked', []):
        source_label = _store._source_label(chunk_entry)
        lines.append(
            f"- sim:`{similarity_score:.2f}` re:`{rerank_score:.0f}` "
            f"[{chunk_entry['source']} {source_label}] "
            f"{chunk_entry['text'][:60]}..."
        )

    return '\n'.join(lines)


def _agent_steps_md(steps: list) -> str:
    """Format agent steps as a markdown string for display in the chat.

    Args:
        steps: List of step dicts with keys: 'step', 'tool', 'arg', 'result'.

    Returns:
        Markdown string with one bold heading and one blockquote per step.
    """
    lines = []
    for step in steps:
        short_arg    = step['arg'][:60]    + '...' if len(step['arg']) > 60    else step['arg']
        short_result = step['result'][:80] + '...' if len(step['result']) > 80 else step['result']
        lines.append(f"**Step {step['step']}** `{step['tool']}({short_arg})`")
        lines.append(f"> {short_result}")
        lines.append("")
    return '\n'.join(lines)


# ── Core event handlers ────────────────────────────────────────────────────────


def chat(message: str, history: list, mode: str):
    """Handle one chat turn — route to Agent or pipeline and return the result.

    Args:
        message: The question or task the user typed.
        history: Gradio message history (list of role/content dicts).
        mode:    'Chat' or 'Agent'.

    Returns:
        Tuple of (updated_history, pipeline_info_markdown).
    """
    loader, store = _initialize()

    if not message or not message.strip():
        return history, ""

    # Allow pure-math questions even when the knowledge base is empty
    is_math_question = bool(re.search(r'[\d].*[\+\-\*\/\%]|[\+\-\*\/\%].*[\d]', message))
    knowledge_base_empty = (store.collection is None or store.collection.count() == 0)

    if not is_math_question and knowledge_base_empty:
        empty_notice = "⚠️ No documents in the knowledge base yet. Please upload a file or add a URL first."
        updated_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": empty_notice},
        ]
        return updated_history, ""

    if mode == "Agent":
        response, pipeline_info = _run_agent_mode(message, store)
    else:
        response, pipeline_info = _run_chat_mode(message, store)

    updated_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response},
    ]
    return updated_history, pipeline_info


def upload_file(file_objs, progress=None):
    """Chunk and index one or more uploaded files into the live knowledge base.

    Accepts a single file or a list of files so users can select multiple files
    (or all files from a folder) in one pick.

    Args:
        file_objs: A single Gradio filepath string / file-like object, or a list of them.
        progress:  Optional gr.Progress() for showing a progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if file_objs is None:
        return "No file selected.", f"Chunks in knowledge base: {_chunk_count()}"

    # Normalise to a list so single-file and multi-file uploads use the same code path
    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    total_chunks  = 0
    indexed_names = []
    errors        = []

    for i, file_obj in enumerate(file_objs):
        # Gradio 5 passes a plain filepath string; older versions pass a file-like object
        filepath  = file_obj if isinstance(file_obj, str) else file_obj.name
        filename  = os.path.basename(filepath)
        extension = os.path.splitext(filename)[1].lower()
        doc_type  = loader.ext_to_type.get(extension, 'txt')

        step = (i + 1) / len(file_objs)
        progress(step * 0.5, desc=f"Reading {filename}...")

        try:
            file_info = {
                'filepath':      filepath,
                'filename':      filename,
                'detected_type': doc_type,
                'is_misplaced':  False,
            }
            new_chunks = loader._dispatch_chunker(file_info)

            if new_chunks:
                progress(step * 0.5 + 0.4, desc=f"Embedding {len(new_chunks)} chunks from {filename}...")
                store.add_chunks(new_chunks, id_prefix='file')
                total_chunks += len(new_chunks)
                indexed_names.append(filename)
            else:
                errors.append(f"⚠️ No text extracted from **{filename}**.")

        except Exception as error:
            logger.error("File upload failed for '%s': %s", filename, error, exc_info=True)
            errors.append(f"❌ Error indexing **{filename}**: {error}")

    if total_chunks > 0:
        # Rebuild BM25 once after all files are indexed (more efficient than per-file)
        progress(0.95, desc="Rebuilding search index...")
        store.rebuild_bm25(store.chunks)
        progress(1.0, desc="Done")

    # Build a readable status message summarising the whole batch
    parts = []
    if indexed_names:
        names_str = ", ".join(f"**{n}**" for n in indexed_names)
        parts.append(f"✅ Indexed {names_str} — {total_chunks} chunks added.")
    parts.extend(errors)

    status = "\n".join(parts) if parts else "No files processed."
    return status, f"Chunks in knowledge base: **{_chunk_count()}**"


def fetch_url(url: str, progress=None):
    """Fetch a public URL, chunk its content, and add it to the knowledge base.

    Args:
        url:      The public HTTP/HTTPS URL to fetch.
        progress: Optional gr.Progress() for showing a progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if not url or not url.strip():
        return "No URL provided.", f"Chunks in knowledge base: {_chunk_count()}"

    clean_url = url.strip()
    try:
        progress(0.2, desc="Fetching URL...")
        new_chunks = loader.chunk_url(clean_url)

        if new_chunks:
            progress(0.5, desc=f"Embedding {len(new_chunks)} chunks on CPU — please wait...")
            store.add_chunks(new_chunks, id_prefix='url')
            progress(0.9, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (
                f"✅ Indexed **{clean_url[:60]}** — {len(new_chunks)} chunks added.",
                f"Chunks in knowledge base: **{_chunk_count()}**",
            )
        else:
            return (
                "⚠️ No content extracted from URL.",
                f"Chunks in knowledge base: {_chunk_count()}",
            )

    except Exception as error:
        logger.error("URL fetch failed for '%s': %s", clean_url, error, exc_info=True)
        return (
            f"❌ Error fetching URL: {error}",
            f"Chunks in knowledge base: {_chunk_count()}",
        )


def fetch_url_recursive(url: str, depth: int, max_pages: int,
                        topic_filter: str,
                        use_html: bool, use_pdf: bool, use_docx: bool,
                        use_xlsx: bool, use_csv: bool, use_pptx: bool,
                        use_md: bool, progress=None):
    """Crawl a seed URL recursively and index all discovered pages.

    Args:
        url:          The seed URL to crawl from.
        depth:        How many link-levels deep to follow (1–3).
        max_pages:    Maximum total pages to fetch (1–50).
        topic_filter: Optional keyword the URL path must contain to be crawled.
                      Empty string means no filter — crawl everything on the domain.
        use_html, use_pdf, use_docx, use_xlsx, use_csv, use_pptx, use_md:
                      Which document types to include in the crawl.
        progress:     Optional gr.Progress() for a progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if not url or not url.strip():
        return "No URL provided.", f"Chunks in knowledge base: {_chunk_count()}"

    # Build the set of allowed document types from the checkboxes
    type_map    = {'html': use_html, 'pdf': use_pdf, 'docx': use_docx,
                   'xlsx': use_xlsx, 'csv': use_csv, 'pptx': use_pptx, 'md': use_md}
    allowed     = {t for t, checked in type_map.items() if checked}
    allowed_set = allowed if allowed else None   # None means "all types"

    clean_url   = url.strip()
    pages_done  = [0]   # mutable list so the callback can update it

    def _progress_callback(page_url: str, dtype: str, chunk_count: int) -> None:
        """Update the Gradio progress bar after each page is crawled."""
        pages_done[0] += 1
        # Progress is approximate — we don't know the total in advance
        fraction = min(0.9, pages_done[0] / max(max_pages, 1))
        short    = page_url[:60] + '...' if len(page_url) > 60 else page_url
        progress(fraction, desc=f"[{dtype.upper()}] {short}")

    try:
        progress(0.05, desc="Starting crawl...")
        new_chunks = loader.chunk_url_recursive(
            clean_url,
            depth=int(depth),
            max_pages=int(max_pages),
            allowed_types=allowed_set,
            topic_filter=topic_filter,
            progress_callback=_progress_callback,
        )

        if new_chunks:
            progress(0.92, desc=f"Embedding {len(new_chunks)} chunks on CPU — please wait...")
            store.add_chunks(new_chunks, id_prefix='url')
            progress(0.97, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (
                f"✅ Crawled {pages_done[0]} pages — {len(new_chunks)} chunks added.",
                f"Chunks in knowledge base: **{_chunk_count()}**",
            )
        else:
            return (
                "⚠️ No content extracted from crawl.",
                f"Chunks in knowledge base: {_chunk_count()}",
            )

    except Exception as error:
        logger.error("Recursive crawl failed for '%s': %s", clean_url, error, exc_info=True)
        return (
            f"❌ Crawl error: {error}",
            f"Chunks in knowledge base: {_chunk_count()}",
        )


def clear_chat():
    """Reset the chat history and the store's conversation memory.

    Returns:
        Tuple of (empty_history, empty_pipeline_info).
    """
    _store.clear_conversation()
    return [], ""


# ── Private pipeline runners ───────────────────────────────────────────────────


def _run_agent_mode(message: str, store: VectorStore):
    """Run the agent loop and return (response_markdown, pipeline_info).

    Args:
        message: The user's task or question.
        store:   VectorStore the agent uses for rag_search.

    Returns:
        Tuple of (formatted_response_string, pipeline_info_string).
    """
    agent        = Agent(store)
    result       = agent.run(message, streamlit_mode=True)
    steps_md     = _agent_steps_md(result['steps'])
    response     = f"{steps_md}\n**Answer:** {result['answer']}"
    pipeline_info = f"**Agent mode** — {len(result['steps'])} steps"
    return response, pipeline_info


def _run_chat_mode(message: str, store: VectorStore):
    """Run the RAG pipeline and return (response_text, pipeline_info_markdown).

    Args:
        message: The user's question.
        store:   VectorStore that runs the full retrieval + generation pipeline.

    Returns:
        Tuple of (response_text, pipeline_info_markdown).
    """
    result        = store.run_pipeline(message, streamlit_mode=True)
    pipeline_info = _pipeline_summary(result)
    return result['response'], pipeline_info


# ── Gradio UI builder ──────────────────────────────────────────────────────────

# IBM Plex Mono stylesheet — polished to match Streamlit version
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ── Base ────────────────────────────────────────────────────────────────── */
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    background: #f8fbfd;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.rag-title {
    font-family: 'IBM Plex Mono', monospace;
    color: #1565a0;
    font-size: 1.8rem;
    font-weight: 600;
    padding-left: .6rem;
    border-left: 4px solid #1565a0;
    margin-bottom: .25rem;
}
.rag-sub {
    color: #7aafc8;
    font-size: 0.75rem;
    padding-left: .6rem;
    margin-bottom: 1rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Chatbot ─────────────────────────────────────────────────────────────── */
.chatbot .message {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    line-height: 1.6;
    border-radius: 8px !important;
}
.chatbot .user { background: #eaf4fd !important; border: 1px solid #b3d4e8 !important; }
.chatbot .bot  { background: #ffffff !important; border: 1px solid #daeaf4 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
button.primary {
    background: #1565a0 !important;
    color: #ffffff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    transition: background .15s ease !important;
}
button.primary:hover { background: #0d47a1 !important; }
button.secondary {
    background: #ffffff !important;
    color: #1565a0 !important;
    border: 1px solid #1565a0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
button.secondary:hover { background: #eaf4fd !important; }

/* ── Accordions / panels ─────────────────────────────────────────────────── */
.accordion {
    border: 1px solid #b3d4e8 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    margin-bottom: .5rem;
}

/* ── Text inputs ─────────────────────────────────────────────────────────── */
input[type="text"], textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    border: 1px solid #b3d4e8 !important;
    border-radius: 6px !important;
    background: #ffffff !important;
    color: #0d2b45 !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #1565a0 !important;
    box-shadow: 0 0 0 2px rgba(21,101,160,.2) !important;
    outline: none !important;
}

/* ── Hide Gradio footer ──────────────────────────────────────────────────── */
footer { display: none !important; }
"""


def build_demo():
    """Assemble and return the complete Gradio Blocks app.

    All UI components, layout, and event wiring are defined here.
    app.py calls this once and launches the returned demo object.

    Returns:
        A configured gradio.Blocks instance ready to launch.
    """
    with gr.Blocks(css=_CSS, title="RAG Agent — Ask Your Documents") as demo:

        gr.HTML("""
        <div class="rag-title">Ask Your Documents</div>
        <div class="rag-sub">
            chunking · hybrid search · reranking · agent ·
            PDF · TXT · DOCX · XLSX · PPTX · CSV · MD · HTML · URL
        </div>
        """)

        with gr.Row():

            # ── Left column: chat + ingestion ──────────────────────────────
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    show_copy_button=True,
                    type='messages',
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="",
                        show_label=False,
                        scale=5,
                    )
                    mode_radio = gr.Radio(
                        choices=["Chat", "Agent"],
                        value="Chat",
                        label="Mode",
                        scale=1,
                    )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=4)
                    clear_btn  = gr.Button("🗑 Clear", scale=1)

                gr.Markdown("---")

                with gr.Accordion("📎 Upload files or a folder", open=False):
                    gr.Markdown("Select one file, multiple files, or all files from a folder.")
                    file_upload = gr.File(
                        label="Supported: PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, MD, HTML",
                        file_types=[
                            ".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls",
                            ".pptx", ".ppt", ".csv", ".md", ".markdown", ".html", ".htm",
                        ],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("Index files →", variant="secondary")
                    upload_msg = gr.Markdown("")

                with gr.Accordion("🌐 Add a URL", open=False):
                    url_input = gr.Textbox(
                        placeholder="https://example.com/page  or  https://example.com/file.pdf",
                        label="Public URL",
                    )
                    url_btn = gr.Button("Fetch & index →", variant="secondary")
                    url_msg = gr.Markdown("")

                with gr.Accordion("🕷️ Recursive web crawl", open=False):
                    gr.Markdown(
                        "Follow links on the page and index all discovered pages. "
                        "⚠️ Depth 1, max 10 pages on HF free CPU to avoid timeout."
                    )
                    crawl_url_input = gr.Textbox(
                        placeholder="https://example.com  (seed URL to start crawling from)",
                        label="Seed URL",
                    )
                    with gr.Row():
                        crawl_depth = gr.Slider(
                            minimum=1, maximum=3, value=URL_CRAWL_MAX_DEPTH, step=1,
                            label="Depth",
                        )
                        crawl_max_pages = gr.Slider(
                            minimum=1, maximum=50, value=URL_CRAWL_MAX_PAGES, step=1,
                            label="Max pages",
                        )
                    crawl_topic_input = gr.Textbox(
                        placeholder="e.g. python  or  machine-learning  or  api",
                        label="Topic filter (optional) — only crawl pages whose URL contains this word",
                    )
                    gr.Markdown("**Index these types:**")
                    with gr.Row():
                        crawl_html  = gr.Checkbox(value=True,  label="HTML")
                        crawl_pdf   = gr.Checkbox(value=True,  label="PDF")
                        crawl_docx  = gr.Checkbox(value=True,  label="DOCX")
                        crawl_xlsx  = gr.Checkbox(value=True,  label="XLSX")
                        crawl_csv   = gr.Checkbox(value=True,  label="CSV")
                        crawl_pptx  = gr.Checkbox(value=True,  label="PPTX")
                        crawl_md    = gr.Checkbox(value=True,  label="MD")
                    crawl_btn = gr.Button("Start crawl →", variant="secondary")
                    crawl_msg = gr.Markdown("")

            # ── Right column: pipeline info ────────────────────────────────
            with gr.Column(scale=1):

                startup_status = gr.Markdown(value="⏳ Initializing...", label="")
                chunk_counter  = gr.Markdown(value="Chunks in knowledge base: **0**", label="")

                gr.Markdown("### Agent Tools")
                gr.Markdown("""
**🔍 rag_search** — search your documents
*e.g. "what skills does the resume mention?"*

**🧮 calculator** — evaluate math expressions
*e.g. "what is 15% of 85000?"*

**📝 summarise** — summarise a document section
*e.g. "summarise the resume"*

**💬 sentiment** — analyse tone & sentiment
*e.g. "what is the sentiment of the resume?"*

**✅ finish** — return the final answer
""")

                gr.Markdown("---")
                gr.Markdown("### Pipeline")
                pipeline_box = gr.Markdown(
                    value="*Pipeline info will appear here after your first query.*",
                    label="",
                )

        # ── Event wiring ──────────────────────────────────────────────────

        def _submit(message, history, mode):
            """Handle both the Send button and pressing Enter in the text box."""
            new_history, info = chat(message, history, mode)
            return new_history, info, ""   # the "" clears the message box

        submit_btn.click(
            fn=_submit,
            inputs=[msg_box, chatbot, mode_radio],
            outputs=[chatbot, pipeline_box, msg_box],
        )
        msg_box.submit(
            fn=_submit,
            inputs=[msg_box, chatbot, mode_radio],
            outputs=[chatbot, pipeline_box, msg_box],
        )
        clear_btn.click(fn=clear_chat, outputs=[chatbot, pipeline_box])

        def _upload(file_objs):
            """Thin wrapper to map upload_file outputs to Gradio components."""
            status_message, counter_text = upload_file(file_objs)
            return status_message, counter_text

        upload_btn.click(
            fn=_upload,
            inputs=[file_upload],
            outputs=[upload_msg, chunk_counter],
        )

        def _fetch(url):
            """Wrapper that also clears the URL input box after fetching."""
            status_message, counter_text = fetch_url(url)
            return status_message, counter_text, ""  # "" clears the URL box

        url_btn.click(
            fn=_fetch,
            inputs=[url_input],
            outputs=[url_msg, chunk_counter, url_input],
        )

        def _crawl(url, depth, max_pages, topic,
                   use_html, use_pdf, use_docx, use_xlsx, use_csv, use_pptx, use_md):
            """Wrapper that maps Gradio inputs to fetch_url_recursive and clears the URL box."""
            status, counter = fetch_url_recursive(
                url, depth, max_pages, topic,
                use_html, use_pdf, use_docx, use_xlsx, use_csv, use_pptx, use_md,
            )
            return status, counter, ""   # "" clears the crawl URL input

        crawl_btn.click(
            fn=_crawl,
            inputs=[
                crawl_url_input, crawl_depth, crawl_max_pages, crawl_topic_input,
                crawl_html, crawl_pdf, crawl_docx,
                crawl_xlsx, crawl_csv, crawl_pptx, crawl_md,
            ],
            outputs=[crawl_msg, chunk_counter, crawl_url_input],
        )

        def _on_load(progress=None):
            """Eagerly initialise both singletons on page load so the first query is fast."""
            if progress is None:
                progress = gr.Progress()
            progress(0.1, desc="Starting up...")
            global _loader, _store
            if _loader is None:
                progress(0.3, desc="Loading document processor...")
                _loader = DocumentLoader()
            if _store is None:
                progress(0.5, desc="Loading embedding model (first run may take 1-2 min)...")
                _store = VectorStore()
                progress(0.8, desc="Initializing vector store...")
                _store.build_or_load([])
            progress(1.0, desc="Ready!")
            return "✅ Ready", f"Chunks in knowledge base: **{_chunk_count()}**"

        demo.load(fn=_on_load, outputs=[startup_status, chunk_counter])

    return demo
