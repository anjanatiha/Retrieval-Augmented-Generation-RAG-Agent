"""app.py — Gradio UI for HF Space. RAG Agent."""

import os
import re
import tempfile
import gradio as gr

from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.rag.agent import Agent
from src.rag.config import EXT_TO_TYPE

# ── Initialization ────────────────────────────────────────────────────────────

_loader: DocumentLoader = None
_store:  VectorStore    = None


def _initialize():
    global _loader, _store
    if _loader is None:
        _loader = DocumentLoader()
    if _store is None:
        _store = VectorStore()
        _store.build_or_load([])   # start with empty knowledge base
    return _loader, _store


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_count():
    return _store.collection.count() if _store and _store.collection else 0


def _pipeline_summary(data):
    """Format pipeline info from run_pipeline result for the info panel."""
    if not data:
        return ""
    lines = []
    qt = data.get('query_type', '')
    bs = data.get('best_score', 0.0)
    ok = data.get('is_confident', False)
    lines.append(f"**Query type:** `{qt}`")
    lines.append(f"**Confidence:** {'✅' if ok else '⚠️'} `{bs:.3f}`")
    lines.append("")
    lines.append("**Before rerank (top 4)**")
    for e, s in data.get('retrieved', [])[:4]:
        label = _store._source_label(e)
        lines.append(f"- `{s:.3f}` [{e['source']} {label}] {e['text'][:60]}...")
    lines.append("")
    lines.append("**After rerank**")
    for e, sim, rs in data.get('reranked', []):
        label = _store._source_label(e)
        lines.append(f"- sim:`{sim:.2f}` re:`{rs:.0f}` [{e['source']} {label}] {e['text'][:60]}...")
    return '\n'.join(lines)


def _agent_steps_md(steps):
    """Format agent steps as markdown."""
    lines = []
    for s in steps:
        arg_short = s['arg'][:60] + '...' if len(s['arg']) > 60 else s['arg']
        res_short = s['result'][:80] + '...' if len(s['result']) > 80 else s['result']
        lines.append(f"**Step {s['step']}** `{s['tool']}({arg_short})`")
        lines.append(f"> {res_short}")
        lines.append("")
    return '\n'.join(lines)


# ── Core handlers ─────────────────────────────────────────────────────────────

def chat(message, history, mode):
    """Main chat handler. Returns updated history + pipeline info."""
    loader, store = _initialize()

    if not message or not message.strip():
        return history, ""

    _is_math = bool(re.search(r'[\d].*[\+\-\*\/\%]|[\+\-\*\/\%].*[\d]', message))
    if not _is_math and (store.collection is None or store.collection.count() == 0):
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ No documents in the knowledge base yet. Please upload a file or add a URL first."}]
        return history, ""

    pipeline_info = ""

    if mode == "Agent":
        agent = Agent(store)
        res   = agent.run(message, streamlit_mode=True)
        steps_md = _agent_steps_md(res['steps'])
        answer   = res['answer']
        response = f"{steps_md}\n**Answer:** {answer}"
        pipeline_info = f"**Agent mode** — {len(res['steps'])} steps"
    else:
        res = store.run_pipeline(message, streamlit_mode=True)
        response = res['response']
        pipeline_info = _pipeline_summary(res)

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
    return history, pipeline_info


def upload_file(file_obj, progress=gr.Progress()):
    """Index an uploaded file into the knowledge base."""
    loader, store = _initialize()
    if file_obj is None:
        return "No file selected.", f"Chunks in knowledge base: {_chunk_count()}"

    # Gradio 5: file_obj is a filepath string
    filepath = file_obj if isinstance(file_obj, str) else file_obj.name
    filename = os.path.basename(filepath)
    ext      = os.path.splitext(filename)[1].lower()
    dtype    = loader.ext_to_type.get(ext, 'txt')

    try:
        progress(0.2, desc="Reading file...")
        file_info = {
            'filepath':      filepath,
            'filename':      filename,
            'detected_type': dtype,
            'is_misplaced':  False,
        }
        new_chunks = loader._dispatch_chunker(file_info)
        if new_chunks:
            progress(0.5, desc=f"Embedding {len(new_chunks)} chunks on CPU — please wait...")
            store.add_chunks(new_chunks, id_prefix='file')
            progress(0.9, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (f"✅ Indexed **{filename}** — {len(new_chunks)} chunks added.",
                    f"Chunks in knowledge base: **{_chunk_count()}**")
        else:
            return f"⚠️ No text extracted from **{filename}**.", f"Chunks in knowledge base: {_chunk_count()}"
    except Exception as e:
        return f"❌ Error indexing **{filename}**: {e}", f"Chunks in knowledge base: {_chunk_count()}"


def fetch_url(url, progress=gr.Progress()):
    """Fetch and index a URL into the knowledge base."""
    loader, store = _initialize()
    if not url or not url.strip():
        return "No URL provided.", f"Chunks in knowledge base: {_chunk_count()}"
    try:
        progress(0.2, desc="Fetching URL...")
        new_chunks = loader.chunk_url(url.strip())
        if new_chunks:
            progress(0.5, desc=f"Embedding {len(new_chunks)} chunks on CPU — please wait...")
            store.add_chunks(new_chunks, id_prefix='url')
            progress(0.9, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (f"✅ Indexed **{url.strip()[:60]}** — {len(new_chunks)} chunks added.",
                    f"Chunks in knowledge base: **{_chunk_count()}**")
        else:
            return f"⚠️ No content extracted from URL.", f"Chunks in knowledge base: {_chunk_count()}"
    except Exception as e:
        return f"❌ Error fetching URL: {e}", f"Chunks in knowledge base: {_chunk_count()}"


def clear_chat():
    """Reset chat history and conversation memory."""
    _store.clear_conversation()
    return [], ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; max-width: 1200px; margin: 0 auto; }
.title { font-family: 'IBM Plex Mono', monospace; color: #1565a0; font-size: 1.8rem; font-weight: 600; }
.subtitle { color: #7aafc8; font-size: 0.8rem; margin-bottom: 1rem; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="RAG Agent — Ask Your Documents") as demo:

    gr.HTML("""
    <div class="title">Ask Your Documents</div>
    <div class="subtitle">
        chunking · hybrid search · reranking · agent · PDF · TXT · DOCX · XLSX · PPTX · CSV · MD · HTML · URL
    </div>
    """)

    with gr.Row():
        # ── Left column: chat + ingestion ──────────────────────────────────
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

            with gr.Accordion("📎 Upload a document", open=False):
                file_upload = gr.File(
                    label="Supported: PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, MD, HTML",
                    file_types=[".pdf",".txt",".docx",".doc",".xlsx",".xls",
                                ".pptx",".ppt",".csv",".md",".markdown",".html",".htm"],
                )
                upload_btn  = gr.Button("Index file →", variant="secondary")
                upload_msg  = gr.Markdown("")

            with gr.Accordion("🌐 Add a URL", open=False):
                url_input = gr.Textbox(
                    placeholder="https://example.com/page  or  https://example.com/file.pdf",
                    label="Public URL",
                )
                url_btn  = gr.Button("Fetch & index →", variant="secondary")
                url_msg  = gr.Markdown("")

        # ── Right column: pipeline info ────────────────────────────────────
        with gr.Column(scale=1):

            chunk_counter = gr.Markdown(
                value="Chunks in knowledge base: **0**",
                label="",
            )

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

    # ── Event wiring ──────────────────────────────────────────────────────────

    def _submit(message, history, mode):
        new_history, info = chat(message, history, mode)
        return new_history, info, ""   # clear the message box

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
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, pipeline_box],
    )

    def _upload(f):
        msg, counter = upload_file(f)
        return msg, counter

    upload_btn.click(
        fn=_upload,
        inputs=[file_upload],
        outputs=[upload_msg, chunk_counter],
    )

    def _fetch(url):
        msg, counter = fetch_url(url)
        return msg, counter, ""

    url_btn.click(
        fn=_fetch,
        inputs=[url_input],
        outputs=[url_msg, chunk_counter, url_input],
    )

    def _on_load():
        _initialize()
        return f"Chunks in knowledge base: **{_chunk_count()}**"

    demo.load(fn=_on_load, outputs=[chunk_counter])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
