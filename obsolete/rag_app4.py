import ollama
import json
import os
import sys
import time
from datetime import datetime
from rank_bm25 import BM25Okapi

# ============================================================
# CONFIGURATION
# ============================================================
EMBEDDING_MODEL      = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL       = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
DATA_FOLDER          = '.'
CACHE_FILE           = 'embeddings_cache.json'
LOG_FILE             = 'rag_logs.json'
SIMILARITY_THRESHOLD = 0.4
TOP_RETRIEVE         = 10
TOP_RERANK           = 3

# ============================================================
# 1. LOAD DOCUMENTS (multiple .txt files)
# ============================================================
def load_documents(folder_path):
    dataset = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in '{folder_path}'")
        sys.exit(1)
    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                dataset.append({'text': line, 'source': filename, 'line': i + 1})
        print(f"Loaded '{filename}' — {len(lines)} lines")
    print(f"Total chunks: {len(dataset)}\n")
    return dataset

# ============================================================
# 2. EMBEDDINGS WITH CACHE
# ============================================================
def load_or_compute_embeddings(dataset):
    if os.path.exists(CACHE_FILE):
        print("Loading embeddings from cache...")
        with open(CACHE_FILE, 'r') as f:
            cached = json.load(f)
        if len(cached) == len(dataset):
            print(f"Cache loaded ({len(cached)} embeddings)\n")
            return [(dataset[i], emb) for i, emb in enumerate(cached)]
        print("Cache mismatch — recomputing...")
    print("Computing embeddings...")
    embeddings = []
    raw_embeddings = []
    for i, entry in enumerate(dataset):
        emb = ollama.embed(model=EMBEDDING_MODEL, input=entry['text'])['embeddings'][0]
        embeddings.append((entry, emb))
        raw_embeddings.append(emb)
        print(f"  Embedded {i+1}/{len(dataset)}", end='\r')
    with open(CACHE_FILE, 'w') as f:
        json.dump(raw_embeddings, f)
    print(f"\nEmbeddings cached\n")
    return embeddings

# ============================================================
# 3. COSINE SIMILARITY
# ============================================================
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x ** 2 for x in a) ** 0.5
    nb  = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

# ============================================================
# 4. HYBRID SEARCH (BM25 + Dense)
# ============================================================
def build_bm25_index(dataset):
    return BM25Okapi([e['text'].lower().split() for e in dataset])

def hybrid_retrieve(query, vector_db, bm25_index, top_n=TOP_RETRIEVE, alpha=0.5):
    query_emb    = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    dense_scores = [cosine_similarity(query_emb, emb) for _, emb in vector_db]
    bm25_scores  = bm25_index.get_scores(query.lower().split()).tolist()
    max_dense    = max(dense_scores) or 1
    max_bm25     = max(bm25_scores)  or 1
    combined = []
    for i, (entry, _) in enumerate(vector_db):
        score = (alpha * dense_scores[i] / max_dense +
                 (1 - alpha) * bm25_scores[i] / max_bm25)
        combined.append((entry, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_n]

# ============================================================
# 5. LLM RERANKER (no extra library needed)
# ============================================================
def rerank(query, chunks, top_n=TOP_RERANK):
    scored = []
    for entry, similarity in chunks:
        resp = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content':
                f"Rate relevance 0-10.\nQuery: {query}\nText: {entry['text']}\nReply with only a number."}]
        )
        try:
            score = float(resp['message']['content'].strip().split()[0])
        except Exception:
            score = 0.0
        scored.append((entry, similarity, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]

# ============================================================
# 6. QUERY CLASSIFICATION
# ============================================================
def classify_query(query):
    resp = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content':
            f"Classify into one word: factual, comparison, or general.\nQuery: {query}\nReply with only the word."}]
    )
    c = resp['message']['content'].strip().lower()
    if 'factual'    in c: return 'factual'
    if 'comparison' in c: return 'comparison'
    return 'general'

def smart_top_n(qtype):
    return {'factual': 5, 'comparison': 15, 'general': 10}.get(qtype, TOP_RETRIEVE)

# ============================================================
# 7. CONFIDENCE CHECK
# ============================================================
def check_confidence(chunks):
    if not chunks: return False, 0.0
    best = chunks[0][1]
    return best >= SIMILARITY_THRESHOLD, best

# ============================================================
# 8. LOGGING
# ============================================================
def log_interaction(query, qtype, chunks_used, sim_scores, response):
    entry = {
        'timestamp':       datetime.now().isoformat(),
        'query':           query,
        'query_type':      qtype,
        'chunks_used':     chunks_used,
        'top_similarity':  round(sim_scores[0], 4) if sim_scores else 0,
        'avg_similarity':  round(sum(sim_scores)/len(sim_scores), 4) if sim_scores else 0,
        'response_length': len(response),
    }
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try: logs = json.load(f)
            except Exception: logs = []
    logs.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

# ============================================================
# 9. STREAM WITH TYPING INDICATOR
# ============================================================
def stream_response(stream):
    print("\nChatbot: ", end='', flush=True)
    for _ in range(3):
        print('.', end='', flush=True)
        time.sleep(0.3)
    print('\r' + ' ' * 30 + '\r', end='', flush=True)
    print("Chatbot: ", end='', flush=True)
    full = ''
    for chunk in stream:
        c = chunk['message']['content']
        sys.stdout.write(c)
        sys.stdout.flush()
        full += c
    print()
    return full

# ============================================================
# CORE PIPELINE (shared by terminal + streamlit)
# ============================================================
def run_pipeline(query, vector_db, bm25_index, conversation_history, streamlit_mode=False):
    qtype = classify_query(query)
    top_n = smart_top_n(qtype)
    retrieved = hybrid_retrieve(query, vector_db, bm25_index, top_n=top_n)
    is_confident, best_score = check_confidence(retrieved)
    reranked = rerank(query, retrieved, top_n=TOP_RERANK)

    context_lines = [
        f" - [{e['source']}, line {e['line']}] {e['text']}"
        for e, _, _ in reranked
    ]
    context = '\n'.join(context_lines)

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following context to answer the question.\n"
        "Do not make up new information.\n"
        "Cite the sources at the end of your answer.\n\n"
        f"{context}"
    )

    conversation_history.append({'role': 'user', 'content': query})

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            *conversation_history,
        ],
        stream=True,
    )

    if streamlit_mode:
        full_response = ''.join(chunk['message']['content'] for chunk in stream)
    else:
        full_response = stream_response(stream)

    conversation_history.append({'role': 'assistant', 'content': full_response})

    sim_scores = [s for _, s, _ in reranked]
    log_interaction(query, qtype, len(reranked), sim_scores, full_response)

    return {
        'response':     full_response,
        'query_type':   qtype,
        'is_confident': is_confident,
        'best_score':   best_score,
        'retrieved':    retrieved,
        'reranked':     reranked,
    }

# ============================================================
# TERMINAL CHATBOT (loop)
# ============================================================
def run_terminal(vector_db, bm25_index):
    conversation_history = []
    print("=" * 60)
    print("        RAG Chatbot - Enhanced Pipeline")
    print("=" * 60)
    print("Type your question. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break

        result = run_pipeline(query, vector_db, bm25_index, conversation_history)

        print(f"\n[Query type: {result['query_type']}]")

        if not result['is_confident']:
            print(f"[Warning] Low confidence ({result['best_score']:.2f}) — answer may be unreliable")

        print(f"\nBefore reranking (top {len(result['retrieved'])}):")
        for entry, score in result['retrieved']:
            print(f"  (score: {score:.3f}) [{entry['source']} L{entry['line']}] {entry['text'][:70]}...")

        print(f"\nAfter reranking (top {TOP_RERANK}):")
        for entry, sim, rscore in result['reranked']:
            print(f"  (sim: {sim:.3f} | rerank: {rscore:.1f}) [{entry['source']} L{entry['line']}] {entry['text'][:70]}...")

        print(f"\n[Logged to '{LOG_FILE}']")
        print("-" * 60)

# ============================================================
# STREAMLIT UI
# ============================================================
def run_streamlit(vector_db, bm25_index):
    import streamlit as st

    st.set_page_config(page_title="RAG Chatbot", page_icon="🐱", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0d0d0d; color: #e8e8e8; }
    .stApp { background-color: #0d0d0d; }
    .rag-title { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #f0c040; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    .rag-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #555; margin-bottom: 2rem; }
    .msg-user { background: #1a1a1a; border-left: 3px solid #f0c040; padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }
    .msg-bot  { background: #141414; border-left: 3px solid #3a9ad9; padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; line-height: 1.6; }
    .msg-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #555; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .chunk-card { background: #111; border: 1px solid #222; border-radius: 6px; padding: 0.6rem 0.8rem; margin: 0.3rem 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #aaa; }
    .chunk-score { color: #f0c040; font-weight: 600; }
    .chunk-source { color: #3a9ad9; }
    .badge { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.3rem; }
    .badge-factual    { background: #1a3a1a; color: #4caf50; }
    .badge-comparison { background: #1a2a3a; color: #3a9ad9; }
    .badge-general    { background: #2a1a2a; color: #ce93d8; }
    .badge-warning    { background: #3a2a00; color: #f0c040; }
    .badge-ok         { background: #1a3a1a; color: #4caf50; }
    .stat-row { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #555; padding: 0.3rem 0; border-bottom: 1px solid #1a1a1a; display: flex; justify-content: space-between; }
    .stat-val { color: #f0c040; }
    .stTextInput > div > div > input { background: #1a1a1a !important; border: 1px solid #333 !important; color: #e8e8e8 !important; font-family: 'IBM Plex Mono', monospace !important; border-radius: 6px !important; }
    .stButton > button { background: #f0c040 !important; color: #0d0d0d !important; font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important; border: none !important; border-radius: 6px !important; }
    [data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 1px solid #1a1a1a; }
    hr { border-color: #1a1a1a !important; }
    </style>
    """, unsafe_allow_html=True)

    # Session state init
    for key, val in [('conv_history', []), ('chat_display', []),
                     ('total_queries', 0), ('last_result', None)]:
        if key not in st.session_state:
            st.session_state[key] = val

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown('<div class="rag-title">// RAG Chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="rag-subtitle">hybrid search · reranking · source citation · memory</div>', unsafe_allow_html=True)

        # Chat history display
        for msg in st.session_state.chat_display:
            role_class = "msg-user" if msg['role'] == 'user' else "msg-bot"
            label      = "you"       if msg['role'] == 'user' else "assistant"
            st.markdown(f"""
            <div class="{role_class}">
                <div class="msg-label">{label}</div>
                {msg['content']}
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Input form
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Ask a question:", placeholder="e.g. Do cats sleep a lot?", label_visibility='collapsed')
            submitted  = st.form_submit_button("Send →")

        if submitted and user_input.strip():
            with st.spinner("Thinking..."):
                result = run_pipeline(
                    user_input, vector_db, bm25_index,
                    st.session_state.conv_history,
                    streamlit_mode=True
                )
            st.session_state.chat_display.append({'role': 'user',      'content': user_input})
            st.session_state.chat_display.append({'role': 'assistant', 'content': result['response']})
            st.session_state.total_queries += 1
            st.session_state.last_result = result
            st.rerun()

    with col_side:
        st.markdown("### Pipeline Info")

        if st.session_state.last_result:
            r = st.session_state.last_result
            qtype = r['query_type']
            st.markdown(f'<span class="badge badge-{qtype}">{qtype}</span>', unsafe_allow_html=True)
            conf_cls   = "badge-ok" if r['is_confident'] else "badge-warning"
            conf_label = f"conf: {r['best_score']:.2f}" if r['is_confident'] else f"low: {r['best_score']:.2f}"
            st.markdown(f'<span class="badge {conf_cls}">{conf_label}</span>', unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("**Before reranking**")
            for entry, score in r['retrieved'][:5]:
                st.markdown(f"""<div class="chunk-card">
                    <span class="chunk-score">{score:.3f}</span>
                    <span class="chunk-source"> [{entry['source']} L{entry['line']}]</span><br/>
                    {entry['text'][:60]}...
                </div>""", unsafe_allow_html=True)

            st.markdown("**After reranking**")
            for entry, sim, rscore in r['reranked']:
                st.markdown(f"""<div class="chunk-card">
                    <span class="chunk-score">sim:{sim:.2f} re:{rscore:.0f}</span>
                    <span class="chunk-source"> [{entry['source']} L{entry['line']}]</span><br/>
                    {entry['text'][:60]}...
                </div>""", unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("**Session**")
        st.markdown(f'<div class="stat-row">Queries <span class="stat-val">{st.session_state.total_queries}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-row">Memory <span class="stat-val">{len(st.session_state.conv_history)//2} turns</span></div>', unsafe_allow_html=True)
        st.markdown("---")

        if st.button("Clear Chat"):
            st.session_state.conv_history  = []
            st.session_state.chat_display  = []
            st.session_state.last_result   = None
            st.session_state.total_queries = 0
            st.rerun()

# ============================================================
# BENCHMARKING
# ============================================================

BENCHMARK_FILE = 'benchmark_results.json'

# Default test questions for cat-facts.txt
# Edit these to match your dataset — add expected keywords per answer
DEFAULT_TEST_CASES = [
    {
        'question': 'How long do cats sleep each day?',
        'expected_keywords': ['sleep', 'hours', '12', '16'],
    },
    {
        'question': 'Do cats have good night vision?',
        'expected_keywords': ['night', 'vision', 'dark', 'see'],
    },
    {
        'question': 'How many toes does a cat have?',
        'expected_keywords': ['toes', 'paws', '18'],
    },
    {
        'question': 'What do cats use their whiskers for?',
        'expected_keywords': ['whiskers', 'balance', 'navigate', 'sense'],
    },
    {
        'question': 'Can cats taste sweet things?',
        'expected_keywords': ['sweet', 'taste', 'cannot', 'receptors'],
    },
]

def score_faithfulness(response, context_chunks):
    """
    LLM-based faithfulness: checks if response is grounded in retrieved chunks.
    Returns score 0.0 - 1.0
    """
    context = '\n'.join([e['text'] for e, _, _ in context_chunks])
    resp = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content':
            f"Given this context:\n{context}\n\n"
            f"And this answer:\n{response}\n\n"
            f"Rate from 0 to 10 how faithful the answer is to the context only. "
            f"10 means every claim in the answer is supported by context. "
            f"Reply with only a number."}]
    )
    try:
        return min(float(resp['message']['content'].strip().split()[0]), 10) / 10
    except Exception:
        return 0.0

def score_relevancy(question, response):
    """
    LLM-based answer relevancy: checks if response actually addresses the question.
    Returns score 0.0 - 1.0
    """
    resp = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content':
            f"Question: {question}\nAnswer: {response}\n\n"
            f"Rate from 0 to 10 how well the answer addresses the question. "
            f"Reply with only a number."}]
    )
    try:
        return min(float(resp['message']['content'].strip().split()[0]), 10) / 10
    except Exception:
        return 0.0

def score_keyword_recall(response, expected_keywords):
    """
    Simple keyword recall: what fraction of expected keywords appear in response.
    Returns score 0.0 - 1.0
    """
    if not expected_keywords:
        return 1.0
    response_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return hits / len(expected_keywords)

def score_context_relevance(question, retrieved_chunks):
    """
    Checks if retrieved chunks are relevant to the question.
    Returns score 0.0 - 1.0 based on average top similarity scores.
    """
    if not retrieved_chunks:
        return 0.0
    scores = [score for _, score in retrieved_chunks[:TOP_RERANK]]
    return sum(scores) / len(scores)

def run_benchmark(vector_db, bm25_index, test_cases=None):
    """
    Runs full benchmark across all test cases.
    Saves results to benchmark_results.json and prints a summary table.
    """
    if test_cases is None:
        test_cases = DEFAULT_TEST_CASES

    print("\n" + "=" * 70)
    print("  BENCHMARKING RAG PIPELINE")
    print("=" * 70)
    print(f"  Running {len(test_cases)} test cases...\n")

    results = []

    for i, tc in enumerate(test_cases):
        question = tc['question']
        keywords = tc.get('expected_keywords', [])

        print(f"[{i+1}/{len(test_cases)}] {question}")

        # Run pipeline (no conversation history needed for benchmarking)
        history   = []
        retrieved = hybrid_retrieve(question, vector_db, bm25_index, top_n=TOP_RETRIEVE)
        reranked  = rerank(question, retrieved, top_n=TOP_RERANK)

        context = '\n'.join([f" - {e['text']}" for e, _, _ in reranked])
        instruction_prompt = (
            "You are a helpful chatbot.\n"
            "Use only the following context to answer the question.\n"
            "Do not make up new information.\n\n"
            f"{context}"
        )
        history.append({'role': 'user', 'content': question})
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'system', 'content': instruction_prompt}, *history],
            stream=True,
        )
        response = ''.join(chunk['message']['content'] for chunk in stream)

        # Score all metrics
        faithfulness      = score_faithfulness(response, reranked)
        relevancy         = score_relevancy(question, response)
        keyword_recall    = score_keyword_recall(response, keywords)
        context_relevance = score_context_relevance(question, retrieved)
        overall           = (faithfulness + relevancy + keyword_recall + context_relevance) / 4

        result = {
            'question':          question,
            'response':          response,
            'faithfulness':      round(faithfulness,      3),
            'answer_relevancy':  round(relevancy,         3),
            'keyword_recall':    round(keyword_recall,    3),
            'context_relevance': round(context_relevance, 3),
            'overall':           round(overall,           3),
            'top_chunk':         retrieved[0][0]['text'][:80] if retrieved else '',
        }
        results.append(result)

        print(f"  faithfulness={faithfulness:.2f}  relevancy={relevancy:.2f}  "
              f"keyword_recall={keyword_recall:.2f}  ctx_rel={context_relevance:.2f}  "
              f"overall={overall:.2f}\n")

    # Summary
    avg_faith   = sum(r['faithfulness']      for r in results) / len(results)
    avg_rel     = sum(r['answer_relevancy']  for r in results) / len(results)
    avg_kw      = sum(r['keyword_recall']    for r in results) / len(results)
    avg_ctx     = sum(r['context_relevance'] for r in results) / len(results)
    avg_overall = sum(r['overall']           for r in results) / len(results)

    print("=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Score':>8}  {'Bar'}")
    print(f"  {'-'*25} {'-'*8}  {'-'*20}")

    def bar(score):
        filled = int(score * 20)
        return '[' + '█' * filled + '░' * (20 - filled) + ']'

    print(f"  {'Faithfulness':<25} {avg_faith:>8.3f}  {bar(avg_faith)}")
    print(f"  {'Answer Relevancy':<25} {avg_rel:>8.3f}  {bar(avg_rel)}")
    print(f"  {'Keyword Recall':<25} {avg_kw:>8.3f}  {bar(avg_kw)}")
    print(f"  {'Context Relevance':<25} {avg_ctx:>8.3f}  {bar(avg_ctx)}")
    print(f"  {'─'*25} {'─'*8}")
    print(f"  {'OVERALL':<25} {avg_overall:>8.3f}  {bar(avg_overall)}")
    print("=" * 70)

    # Save results
    output = {
        'timestamp':  datetime.now().isoformat(),
        'num_cases':  len(results),
        'summary': {
            'faithfulness':      round(avg_faith,   3),
            'answer_relevancy':  round(avg_rel,     3),
            'keyword_recall':    round(avg_kw,      3),
            'context_relevance': round(avg_ctx,     3),
            'overall':           round(avg_overall, 3),
        },
        'results': results,
    }

    # Load previous runs for comparison
    history_runs = []
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, 'r') as f:
            try:
                existing = json.load(f)
                # Support both single run and list of runs
                if isinstance(existing, list):
                    history_runs = existing
                else:
                    history_runs = [existing]
            except Exception:
                history_runs = []

    # Show comparison if previous run exists
    if history_runs:
        prev = history_runs[-1]['summary']
        print("\n  COMPARISON vs PREVIOUS RUN")
        print(f"  {'Metric':<25} {'Previous':>10} {'Current':>10} {'Change':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for metric in ['faithfulness', 'answer_relevancy', 'keyword_recall', 'context_relevance', 'overall']:
            prev_val = prev.get(metric, 0)
            curr_val = output['summary'][metric]
            delta    = curr_val - prev_val
            arrow    = '▲' if delta > 0 else ('▼' if delta < 0 else '─')
            print(f"  {metric:<25} {prev_val:>10.3f} {curr_val:>10.3f} {arrow}{abs(delta):>9.3f}")
        print("=" * 70)

    history_runs.append(output)
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(history_runs, f, indent=2)

    print(f"\n  Results saved to '{BENCHMARK_FILE}'")
    print("=" * 70 + "\n")
    return output

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RAG Chatbot')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark evaluation')
    parser.add_argument('--streamlit', action='store_true', help='Launch Streamlit UI')
    args = parser.parse_args()

    print("Initializing RAG pipeline...")
    dataset   = load_documents(DATA_FOLDER)
    vector_db = load_or_compute_embeddings(dataset)
    bm25      = build_bm25_index(dataset)

    if args.benchmark:
        run_benchmark(vector_db, bm25)
    elif args.streamlit:
        run_streamlit(vector_db, bm25)
    else:
        run_terminal(vector_db, bm25)
