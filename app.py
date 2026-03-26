"""app.py — Streamlit UI entry point."""

import os
import tempfile
import streamlit as st

from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.rag.agent import Agent
from src.rag.config import TOP_RERANK
from ui.theme import CSS, BADGE_CLASSES, CONFIDENCE_BADGE
from ui.session import init_session_state, get_active_bm25


@st.cache_resource
def initialize():
    loader = DocumentLoader()
    loader.ensure_folders()
    chunks = loader.chunk_all_documents()
    store  = VectorStore()
    store.build_or_load(chunks)
    return loader, store


loader, store = initialize()
chunks = store.chunks

st.set_page_config(page_title="Ask Your Documents", page_icon="🐱", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
init_session_state()

active_bm25  = get_active_bm25(store.bm25_index)
_needs_rerun = False

col_main, col_side = st.columns([3, 1])

with col_main:
    st.markdown('<div class="rag-title">Ask Your Documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rag-sub">chunking · hybrid search · reranking · agent · '
        'PDF · TXT · DOCX · XLSX · PPTX · CSV · MD · HTML · URL</div>',
        unsafe_allow_html=True
    )

    mode = st.radio("Mode:", ["Chat", "Agent"], horizontal=True,
                    index=0 if st.session_state.mode == 'chat' else 1)
    st.session_state.mode = mode.lower()

    if st.session_state.mode == 'agent':
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
            unsafe_allow_html=True
        )

    # ── URL ingestion ──
    with st.expander("Add a URL to knowledge base", expanded=False):
        with st.form('url_form', clear_on_submit=True):
            url_input  = st.text_input("URL:", placeholder="https://example.com/page or https://example.com/file.pdf")
            url_submit = st.form_submit_button("Fetch & index →")
        if url_submit and url_input.strip():
            try:
                with st.spinner(f"Fetching {url_input.strip()}..."):
                    new_chunks = loader.chunk_url(url_input.strip())
                if new_chunks:
                    with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                        store.add_chunks(new_chunks, id_prefix='url')
                    st.session_state.url_chunks.extend(new_chunks)
                    store.rebuild_bm25(store.chunks)
                    st.session_state.bm25_index = store.bm25_index
                    active_bm25 = store.bm25_index
                    st.session_state.url_msg = ('ok', f"Added {len(new_chunks)} chunks. Total in knowledge base: {store.collection.count()}")
                else:
                    st.session_state.url_msg = ('err', "Could not fetch or parse the URL. Check it's publicly accessible.")
            except Exception as e:
                st.session_state.url_msg = ('err', f"Error fetching URL: {e}")
            _needs_rerun = True

        if st.session_state.url_msg:
            kind, msg = st.session_state.url_msg
            if kind == 'ok':
                st.success(msg)
            else:
                st.error(msg)

    # ── File upload ingestion ──
    with st.expander("Upload a file to knowledge base", expanded=False):
        uploaded = st.file_uploader(
            "Supported: PDF, TXT, DOCX, XLSX, PPTX, CSV, MD, HTML",
            type=["pdf","txt","docx","doc","xlsx","xls","pptx","ppt","csv","md","markdown","html","htm"],
            key="file_uploader"
        )
        if uploaded:
            if st.button("Index file →", key="file_index_btn"):
                ext   = os.path.splitext(uploaded.name)[1].lower()
                dtype = loader.ext_to_type.get(ext, 'txt')
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    with st.spinner(f"Processing {uploaded.name}..."):
                        file_info = {
                            'filepath':      tmp_path,
                            'filename':      uploaded.name,
                            'detected_type': dtype,
                            'is_misplaced':  False,
                        }
                        new_chunks = loader._dispatch_chunker(file_info)
                    if new_chunks:
                        with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                            store.add_chunks(new_chunks, id_prefix='file')
                        st.session_state.url_chunks.extend(new_chunks)
                        store.rebuild_bm25(store.chunks)
                        st.session_state.bm25_index = store.bm25_index
                        active_bm25 = store.bm25_index
                        st.session_state.file_msg = ('ok', f"Indexed '{uploaded.name}' — {len(new_chunks)} chunks added. Total: {store.collection.count()}")
                    else:
                        st.session_state.file_msg = ('err', f"Could not extract text from '{uploaded.name}'.")
                except Exception as e:
                    st.session_state.file_msg = ('err', f"Error indexing '{uploaded.name}': {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                _needs_rerun = True
        if st.session_state.get('file_msg'):
            kind, msg = st.session_state.file_msg
            if kind == 'ok':
                st.success(msg)
            else:
                st.error(msg)

    # ── Chat message history ──
    for msg in st.session_state.display:
        avatar = "🧑" if msg['role'] == 'user' else ("🤖" if msg['role'] == 'agent' else "💬")
        with st.chat_message(msg['role'], avatar=avatar):
            st.markdown(msg['content'], unsafe_allow_html=True)

    # ── Clear button below chat ──
    if st.session_state.display:
        _, btn_col = st.columns([6, 1])
        with btn_col:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.conv    = []
                st.session_state.display = []
                st.session_state.last    = None
                st.session_state.total   = 0
                store.clear_conversation()
                st.rerun()

# Deferred rerun — after all columns finish rendering
if _needs_rerun:
    st.rerun()

# ── Chat input at page level ──
placeholder = "Ask a question..." if st.session_state.mode == 'chat' else "Give the agent a task..."
user_input  = st.chat_input(placeholder)
_progress_slot = st.empty()

if user_input and user_input.strip():
    st.session_state.url_msg = None
    st.session_state.display.append({'role': 'user', 'content': user_input})

    if st.session_state.mode == 'agent':
        bar = _progress_slot.progress(0, text="Agent starting...")
        bar.progress(30, text="Agent: searching knowledge base...")
        agent = Agent(store)
        res   = agent.run(user_input, streamlit_mode=True)
        bar.progress(100, text="Agent: done!")
        _progress_slot.empty()
        steps_html = ''.join(
            f'<div class="step">Step {s["step"]}: {s["tool"]}({s["arg"][:50]}...) → {s["result"][:80]}...</div>'
            if len(s["arg"]) > 50 else
            f'<div class="step">Step {s["step"]}: {s["tool"]}({s["arg"]}) → {s["result"][:80]}</div>'
            for s in res['steps']
        )
        content = f"{steps_html}<br/><strong>Answer:</strong> {res['answer']}"
        st.session_state.display.append({'role': 'agent', 'content': content})
        st.session_state.last = {'type': 'agent', 'data': res}

    else:
        bar = _progress_slot.progress(0, text="Classifying query...")
        bar.progress(25, text="Retrieving documents...")
        bar.progress(55, text="Reranking results...")
        bar.progress(75, text="Generating answer...")
        res = store.run_pipeline(user_input, streamlit_mode=True)
        bar.progress(100, text="Done!")
        _progress_slot.empty()
        st.session_state.display.append({'role': 'assistant', 'content': res['response']})
        st.session_state.last = {'type': 'chat', 'data': res}

    st.session_state.total += 1
    st.rerun()

with col_side:
    st.markdown("### Pipeline")
    if st.session_state.last:
        d = st.session_state.last['data']
        if st.session_state.last['type'] == 'chat':
            qt        = d['query_type']
            badge_cls = BADGE_CLASSES.get(qt, 'b-gen')
            st.markdown(f'<span class="badge {badge_cls}">{qt}</span>', unsafe_allow_html=True)
            cc = CONFIDENCE_BADGE[d['is_confident']]
            cl = f"conf:{d['best_score']:.2f}" if d['is_confident'] else f"low:{d['best_score']:.2f}"
            st.markdown(f'<span class="badge {cc}">{cl}</span>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Before rerank**")
            for e, s in d['retrieved'][:4]:
                label = store._source_label(e)
                st.markdown(
                    f'<div class="chunk"><span class="cs">{s:.3f}</span> '
                    f'<span class="src">[{e["source"]} {label}]</span><br/>'
                    f'{e["text"][:55]}...</div>',
                    unsafe_allow_html=True
                )
            st.markdown("**After rerank**")
            for e, sim, rs in d['reranked']:
                label = store._source_label(e)
                st.markdown(
                    f'<div class="chunk"><span class="cs">sim:{sim:.2f} re:{rs:.0f}</span> '
                    f'<span class="src">[{e["source"]} {label}]</span><br/>'
                    f'{e["text"][:55]}...</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("**Agent Steps**")
            for s in d['steps']:
                st.markdown(f'<div class="step">{s["step"]}. {s["tool"]}</div>', unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("**Session**")
    st.markdown(f'<div class="stat">Queries <span class="sv">{st.session_state.total}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">Memory <span class="sv">{len(st.session_state.conv)//2} turns</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">Chunks <span class="sv">{len(chunks) + len(st.session_state.url_chunks)}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">URL chunks <span class="sv">{len(st.session_state.url_chunks)}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">Mode <span class="sv">{st.session_state.mode}</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    type_counts = {}
    for c in chunks:
        t = c.get('type', '?')
        type_counts[t] = type_counts.get(t, 0) + 1
    st.markdown("**Document Types**")
    for t, cnt in sorted(type_counts.items()):
        st.markdown(f'<div class="stat">{t.upper()} <span class="sv">{cnt}</span></div>', unsafe_allow_html=True)
    st.markdown("---")
