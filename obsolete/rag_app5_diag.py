"""
RAG Chatbot — Full Enhanced Pipeline
=====================================
Features:
  1.  Sliding window chunking with overlap
  2.  Persistent vector DB (ChromaDB)
  3.  Hybrid search (BM25 + dense vector)
  4.  Query expansion
  5.  LLM reranker
  6.  Query classification
  7.  Confidence / hallucination filter
  8.  Source citation
  9.  Conversation memory
  10. Logging & analytics
  11. Streaming with typing indicator
  12. Benchmarking with before/after comparison
  13. Agent with tool calling (RAG, calculator, summarise)
  14. Streamlit UI

Run modes:
  python3 rag_app.py                  # terminal chatbot
  python3 rag_app.py --agent          # agent mode (terminal)
  python3 rag_app.py --benchmark      # benchmark evaluation
  streamlit run rag_app.py            # Streamlit UI
"""

import ollama, json, os, sys, re, time, argparse
from datetime import datetime
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

# ============================================================
# CONFIGURATION
# ============================================================
EMBEDDING_MODEL      = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL       = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
DATA_FOLDER          = '.'
CHROMA_DIR           = './chroma_db'
CHROMA_COLLECTION    = 'rag_docs'
LOG_FILE             = 'rag_logs.json'
BENCHMARK_FILE       = 'benchmark_results.json'
SIMILARITY_THRESHOLD = 0.4
TOP_RETRIEVE         = 10
TOP_RERANK           = 3
CHUNK_SIZE           = 3       # lines per chunk
CHUNK_OVERLAP        = 1       # overlapping lines between chunks

# ============================================================
# 1. CHUNKING — Sliding Window with Overlap
# ============================================================
def chunk_documents(folder_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Loads all .txt files and splits into overlapping sliding window chunks.
    Example with chunk_size=3, overlap=1:
      lines [1,2,3] → chunk 1
      lines [3,4,5] → chunk 2  (line 3 repeated = overlap)
      lines [5,6,7] → chunk 3
    """
    all_chunks = []
    txt_files  = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in '{folder_path}'")
        sys.exit(1)

    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        step = max(1, chunk_size - overlap)
        for i in range(0, len(lines), step):
            window = lines[i : i + chunk_size]
            if not window:
                continue
            chunk_text = ' '.join(window)
            all_chunks.append({
                'text':       chunk_text,
                'source':     filename,
                'start_line': i + 1,
                'end_line':   i + len(window),
            })

        print(f"  '{filename}': {len(lines)} lines → {len(all_chunks)} chunks so far")

    print(f"Total chunks after sliding window: {len(all_chunks)}\n")
    return all_chunks

# ============================================================
# 2. PERSISTENT VECTOR DB — ChromaDB
# ============================================================
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

def build_or_load_chroma(chunks):
    """
    Adds chunks to ChromaDB if not already stored.
    ChromaDB persists to disk so embeddings survive restarts.
    """
    collection = get_chroma_collection()
    existing   = collection.count()

    if existing == len(chunks):
        print(f"ChromaDB loaded — {existing} chunks already stored.\n")
        return collection

    if existing > 0:
        print(f"ChromaDB has {existing} chunks but dataset has {len(chunks)} — rebuilding...")
        collection.delete(ids=collection.get()['ids'])

    print(f"Embedding {len(chunks)} chunks into ChromaDB...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch  = chunks[i : i + batch_size]
        ids    = [f"chunk_{i+j}" for j in range(len(batch))]
        texts  = [c['text'] for c in batch]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line']} for c in batch]
        embeds = [ollama.embed(model=EMBEDDING_MODEL, input=t)['embeddings'][0]
                  for t in texts]
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
        print(f"  Stored {min(i+batch_size, len(chunks))}/{len(chunks)}", end='\r')

    print(f"\nChromaDB ready — {collection.count()} chunks stored.\n")
    return collection

# ============================================================
# 3. COSINE SIMILARITY (for in-memory BM25 fusion)
# ============================================================
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x**2 for x in a)**0.5
    nb  = sum(x**2 for x in b)**0.5
    return dot / (na * nb) if na and nb else 0.0

# ============================================================
# 4. QUERY EXPANSION
# ============================================================
def expand_query(query):
    """
    Asks LLM to rewrite the query in 2 alternative ways.
    All 3 versions are searched and results merged for better recall.
    """
    resp = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content':
            f"Rewrite this question in 2 different ways to improve search recall.\n"
            f"Return only the 2 questions, one per line, no numbering:\n{query}"}]
    )
    lines    = [l.strip() for l in resp['message']['content'].strip().split('\n') if l.strip()]
    expanded = [query] + lines[:2]
    return expanded

# ============================================================
# 5. HYBRID SEARCH (BM25 + ChromaDB Dense)
# ============================================================
def build_bm25_index(chunks):
    return BM25Okapi([c['text'].lower().split() for c in chunks])

def hybrid_retrieve(queries, collection, chunks, bm25_index, top_n=TOP_RETRIEVE, alpha=0.5):
    """
    Runs hybrid BM25 + dense retrieval for all expanded queries.
    Merges results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}   # chunk_id → accumulated RRF score

    for query in queries:
        # --- Dense retrieval via ChromaDB ---
        q_emb    = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        results  = collection.query(query_embeddings=[q_emb], n_results=min(top_n, collection.count()))
        dense_ids   = results['ids'][0]
        dense_docs  = results['documents'][0]
        dense_metas = results['metadatas'][0]
        dense_dists = results['distances'][0]  # cosine distance (lower = better)

        # DIAGNOSTIC — show raw distances and similarities
        print(f"\n[DIAG] Query: {query[:70]}")
        for doc, dist in zip(dense_docs, dense_dists):
            print(f"  dist={dist:.4f}  sim={1-dist:.4f}  | {doc[:70]}")

        # Build lookup: id → (text, meta, similarity)
        dense_map = {}
        for cid, doc, meta, dist in zip(dense_ids, dense_docs, dense_metas, dense_dists):
            dense_map[cid] = {'text': doc, 'meta': meta, 'similarity': 1 - dist}

        # --- Sparse retrieval via BM25 ---
        bm25_scores  = bm25_index.get_scores(query.lower().split()).tolist()
        bm25_ranked  = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_n]

        # --- RRF fusion ---
        K = 60  # RRF constant
        for rank, (cid) in enumerate(dense_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (K + rank + 1)

        for rank, (idx, _) in enumerate(bm25_ranked):
            cid = f"chunk_{idx}"
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (K + rank + 1)

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]

    # Resolve chunk objects
    retrieved = []
    for cid in sorted_ids:
        res = collection.get(ids=[cid])
        if res['ids']:
            entry = {
                'text':       res['documents'][0],
                'source':     res['metadatas'][0].get('source', '?'),
                'start_line': res['metadatas'][0].get('start_line', 0),
                'end_line':   res['metadatas'][0].get('end_line', 0),
            }
            retrieved.append((entry, rrf_scores[cid]))

    return retrieved[:top_n]

# ============================================================
# 6. LLM RERANKER
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
# 7. QUERY CLASSIFICATION
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
# 8. CONFIDENCE CHECK
# ============================================================
def check_confidence(chunks):
    if not chunks: return False, 0.0
    best = chunks[0][1]
    return best >= SIMILARITY_THRESHOLD, best

# ============================================================
# 9. LOGGING
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
# 10. STREAMING WITH TYPING INDICATOR
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
# 11. CORE PIPELINE
# ============================================================
def run_pipeline(query, collection, chunks, bm25_index, conversation_history, streamlit_mode=False):
    qtype    = classify_query(query)
    top_n    = smart_top_n(qtype)
    queries  = expand_query(query)           # query expansion
    retrieved = hybrid_retrieve(queries, collection, chunks, bm25_index, top_n=top_n)
    is_confident, best_score = check_confidence(retrieved)
    reranked = rerank(query, retrieved, top_n=TOP_RERANK)

    context_lines = [
        f" - [{e['source']} L{e['start_line']}-{e['end_line']}] {e['text']}"
        for e, _, _ in reranked
    ]
    context = '\n'.join(context_lines)

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following context to answer the question.\n"
        "Do not make up new information.\n"
        "Cite sources at the end of your answer.\n\n"
        f"{context}"
    )

    conversation_history.append({'role': 'user', 'content': query})
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'system', 'content': instruction_prompt}, *conversation_history],
        stream=True,
    )

    full_response = (''.join(c['message']['content'] for c in stream)
                     if streamlit_mode else stream_response(stream))

    conversation_history.append({'role': 'assistant', 'content': full_response})

    sim_scores = [s for _, s, _ in reranked]
    log_interaction(query, qtype, len(reranked), sim_scores, full_response)

    return {
        'response':     full_response,
        'query_type':   qtype,
        'queries':      queries,
        'is_confident': is_confident,
        'best_score':   best_score,
        'retrieved':    retrieved,
        'reranked':     reranked,
    }

# ============================================================
# 12. AGENT — Tool Calling
# ============================================================
AGENT_SYSTEM_PROMPT = """You are a helpful agent with access to the following tools:

1. rag_search(query) — search the knowledge base for information
2. calculator(expression) — evaluate a math expression e.g. calculator(12 * 7)
3. summarise(text) — summarise a long piece of text
4. finish(answer) — return final answer to the user

To use a tool, respond EXACTLY in this format (nothing else):
TOOL: tool_name(argument)

When you have enough information to answer, use:
TOOL: finish(your final answer here)

Think step by step. Use tools as many times as needed before finishing.
"""

def tool_rag_search(query, collection, chunks, bm25_index):
    queries   = expand_query(query)
    retrieved = hybrid_retrieve(queries, collection, chunks, bm25_index, top_n=5)
    reranked  = rerank(query, retrieved, top_n=3)
    return '\n'.join([f"- {e['text']}" for e, _, _ in reranked])

def tool_calculator(expression):
    try:
        # Safe eval — only allow numbers and basic operators
        allowed = set('0123456789+-*/(). ')
        if not all(c in allowed for c in expression):
            return "Error: unsafe expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def tool_summarise(text):
    resp = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content': f"Summarise this in 2-3 sentences:\n{text}"}]
    )
    return resp['message']['content'].strip()

def parse_tool_call(response_text):
    """Parse TOOL: tool_name(argument) from LLM response."""
    match = re.search(r'TOOL:\s*(\w+)\((.+)\)', response_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def run_agent(user_query, collection, chunks, bm25_index, max_steps=6, streamlit_mode=False):
    """
    ReAct-style agent loop:
    1. LLM decides which tool to call
    2. Tool runs and returns result
    3. Result fed back to LLM
    4. Repeat until LLM calls finish()
    """
    messages = [
        {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
        {'role': 'user',   'content': user_query},
    ]

    steps   = []
    answer  = None

    for step in range(max_steps):
        resp     = ollama.chat(model=LANGUAGE_MODEL, messages=messages)
        raw_text = resp['message']['content'].strip()

        tool_name, tool_arg = parse_tool_call(raw_text)

        if not tool_name:
            # LLM didn't follow format — treat as final answer
            answer = raw_text
            steps.append({'step': step+1, 'tool': 'none', 'arg': '', 'result': raw_text})
            break

        if tool_name == 'finish':
            answer = tool_arg
            steps.append({'step': step+1, 'tool': 'finish', 'arg': tool_arg, 'result': tool_arg})
            break

        # Execute tool
        if tool_name == 'rag_search':
            result = tool_rag_search(tool_arg, collection, chunks, bm25_index)
        elif tool_name == 'calculator':
            result = tool_calculator(tool_arg)
        elif tool_name == 'summarise':
            result = tool_summarise(tool_arg)
        else:
            result = f"Unknown tool: {tool_name}"

        steps.append({'step': step+1, 'tool': tool_name, 'arg': tool_arg, 'result': result})

        if not streamlit_mode:
            print(f"\n  [Agent Step {step+1}] {tool_name}({tool_arg[:60]}...)"
                  if len(tool_arg) > 60 else f"\n  [Agent Step {step+1}] {tool_name}({tool_arg})")
            print(f"  → {result[:120]}..." if len(result) > 120 else f"  → {result}")

        # Feed result back
        messages.append({'role': 'assistant', 'content': raw_text})
        messages.append({'role': 'user',      'content': f"Tool result: {result}\nContinue."})

    if answer is None:
        answer = "Agent reached max steps without a final answer."

    return {'answer': answer, 'steps': steps}

# ============================================================
# 13. BENCHMARKING
# ============================================================
BENCHMARK_FILE   = 'benchmark_results.json'
DEFAULT_TEST_CASES = [
    {'question': 'How long do cats sleep each day?',         'expected_keywords': ['sleep', 'hours']},
    {'question': 'Do cats have good night vision?',          'expected_keywords': ['night', 'vision', 'dark']},
    {'question': 'How many toes does a cat have?',           'expected_keywords': ['toes', '18']},
    {'question': 'What do cats use their whiskers for?',     'expected_keywords': ['whiskers', 'navigate']},
    {'question': 'Can cats taste sweet things?',             'expected_keywords': ['sweet', 'taste']},
]

def score_faithfulness(response, reranked):
    context = '\n'.join(e['text'] for e, _, _ in reranked)
    resp = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content':
        f"Context:\n{context}\nAnswer:\n{response}\n"
        f"Rate 0-10 how faithful the answer is to the context. Reply with only a number."}])
    try:    return min(float(resp['message']['content'].strip().split()[0]), 10) / 10
    except: return 0.0

def score_relevancy(question, response):
    resp = ollama.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content':
        f"Question: {question}\nAnswer: {response}\n"
        f"Rate 0-10 how well the answer addresses the question. Reply with only a number."}])
    try:    return min(float(resp['message']['content'].strip().split()[0]), 10) / 10
    except: return 0.0

def score_keyword_recall(response, keywords):
    if not keywords: return 1.0
    rl = response.lower()
    return sum(1 for kw in keywords if kw.lower() in rl) / len(keywords)

def score_context_relevance(retrieved):
    if not retrieved: return 0.0
    scores = [s for _, s in retrieved[:TOP_RERANK]]
    return sum(scores) / len(scores)

def run_benchmark(collection, chunks, bm25_index, test_cases=None):
    test_cases = test_cases or DEFAULT_TEST_CASES
    print("\n" + "="*70)
    print("  BENCHMARKING RAG PIPELINE")
    print("="*70)
    results = []

    for i, tc in enumerate(test_cases):
        q, kw = tc['question'], tc.get('expected_keywords', [])
        print(f"\n[{i+1}/{len(test_cases)}] {q}")

        queries   = expand_query(q)
        retrieved = hybrid_retrieve(queries, collection, chunks, bm25_index, top_n=TOP_RETRIEVE)
        reranked  = rerank(q, retrieved, top_n=TOP_RERANK)

        context = '\n'.join(f" - {e['text']}" for e, _, _ in reranked)
        stream  = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content':
                    f"Answer using only this context:\n{context}"},
                {'role': 'user', 'content': q},
            ], stream=True)
        response = ''.join(c['message']['content'] for c in stream)

        faith  = score_faithfulness(response, reranked)
        rel    = score_relevancy(q, response)
        kwr    = score_keyword_recall(response, kw)
        ctx    = score_context_relevance(retrieved)
        ovr    = (faith + rel + kwr + ctx) / 4

        results.append({'question': q, 'faithfulness': round(faith,3),
                        'answer_relevancy': round(rel,3), 'keyword_recall': round(kwr,3),
                        'context_relevance': round(ctx,3), 'overall': round(ovr,3)})

        print(f"  faith={faith:.2f} rel={rel:.2f} kw={kwr:.2f} ctx={ctx:.2f} overall={ovr:.2f}")

    def avg(k): return sum(r[k] for r in results) / len(results)
    summary = {k: round(avg(k), 3) for k in
               ['faithfulness','answer_relevancy','keyword_recall','context_relevance','overall']}

    def bar(s): return '[' + '█'*int(s*20) + '░'*(20-int(s*20)) + ']'

    print("\n" + "="*70 + "\n  SUMMARY\n" + "="*70)
    for k, v in summary.items():
        print(f"  {k:<25} {v:>6.3f}  {bar(v)}")

    # Compare to previous run
    runs = []
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE) as f:
            try: runs = json.load(f)
            except: runs = []
    if runs:
        prev = runs[-1]['summary']
        print("\n  vs PREVIOUS RUN")
        for k in summary:
            d = summary[k] - prev.get(k, 0)
            print(f"  {k:<25} {prev.get(k,0):>6.3f} → {summary[k]:>6.3f}  "
                  f"{'▲' if d>0 else '▼' if d<0 else '─'}{abs(d):.3f}")

    runs.append({'timestamp': datetime.now().isoformat(), 'summary': summary, 'results': results})
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(runs, f, indent=2)
    print(f"\n  Saved to '{BENCHMARK_FILE}'\n" + "="*70)
    return summary

# ============================================================
# 14. TERMINAL CHATBOT (loop)
# ============================================================
def run_terminal(collection, chunks, bm25_index):
    conv = []
    print("="*60 + "\n  RAG Chatbot — Full Pipeline\n" + "="*60)
    print("Commands: 'exit' quit | 'agent: <q>' use agent mode\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break

        if not query: continue
        if query.lower() in ['exit','quit','bye']:
            print("Goodbye!"); break

        if query.lower().startswith('agent:'):
            q = query[6:].strip()
            print("\n[Agent mode]")
            result = run_agent(q, collection, chunks, bm25_index)
            print(f"\nAgent answer: {result['answer']}")
        else:
            result = run_pipeline(query, collection, chunks, bm25_index, conv)
            if not result['is_confident']:
                print(f"[Warning] Low confidence ({result['best_score']:.2f})")
            print(f"\n[type:{result['query_type']} | expanded:{len(result['queries'])} queries]")
            print(f"Before rerank: {len(result['retrieved'])} chunks | After: {TOP_RERANK} chunks")
        print("-"*60)

# ============================================================
# 15. STREAMLIT UI
# ============================================================
def run_streamlit(collection, chunks, bm25_index):
    import streamlit as st

    st.set_page_config(page_title="RAG Chatbot", page_icon="🐱", layout="wide")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#0d0d0d;color:#e8e8e8}
    .stApp{background:#0d0d0d}
    .rag-title{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;color:#f0c040;letter-spacing:-.02em}
    .rag-sub{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#444;margin-bottom:1.5rem}
    .msg-user{background:#1a1a1a;border-left:3px solid #f0c040;padding:.8rem 1rem;margin:.4rem 0;border-radius:0 8px 8px 0}
    .msg-bot{background:#141414;border-left:3px solid #3a9ad9;padding:.8rem 1rem;margin:.4rem 0;border-radius:0 8px 8px 0;line-height:1.6}
    .msg-agent{background:#0f1a0f;border-left:3px solid #4caf50;padding:.8rem 1rem;margin:.4rem 0;border-radius:0 8px 8px 0}
    .msg-label{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#444;margin-bottom:.2rem;text-transform:uppercase;letter-spacing:.1em}
    .chunk{background:#111;border:1px solid #1e1e1e;border-radius:6px;padding:.5rem .7rem;margin:.25rem 0;font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:#888}
    .cs{color:#f0c040;font-weight:600}.src{color:#3a9ad9}
    .step{background:#0a1a0a;border:1px solid #1a2a1a;border-radius:6px;padding:.5rem .7rem;margin:.2rem 0;font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:#4caf50}
    .badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.65rem;padding:.15rem .4rem;border-radius:3px;margin:.1rem}
    .b-fact{background:#1a3a1a;color:#4caf50}.b-comp{background:#1a2a3a;color:#3a9ad9}.b-gen{background:#2a1a2a;color:#ce93d8}
    .b-ok{background:#1a3a1a;color:#4caf50}.b-warn{background:#3a2a00;color:#f0c040}
    .stat{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#444;padding:.25rem 0;border-bottom:1px solid #151515;display:flex;justify-content:space-between}
    .sv{color:#f0c040}
    .stTextInput>div>div>input{background:#1a1a1a!important;border:1px solid #2a2a2a!important;color:#e8e8e8!important;font-family:'IBM Plex Mono',monospace!important;border-radius:6px!important}
    .stButton>button{background:#f0c040!important;color:#0d0d0d!important;font-family:'IBM Plex Mono',monospace!important;font-weight:600!important;border:none!important;border-radius:6px!important}
    [data-testid="stSidebar"]{background:#0a0a0a!important;border-right:1px solid #151515}
    hr{border-color:#1a1a1a!important}
    </style>
    """, unsafe_allow_html=True)

    for k,v in [('conv',[]),('display',[]),('total',0),('last',None),('mode','chat')]:
        if k not in st.session_state: st.session_state[k] = v

    col_main, col_side = st.columns([3,1])

    with col_main:
        st.markdown('<div class="rag-title">// RAG Chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="rag-sub">chunking · hybrid search · query expansion · reranking · agent</div>', unsafe_allow_html=True)

        mode = st.radio("Mode:", ["Chat", "Agent"], horizontal=True,
                        index=0 if st.session_state.mode=='chat' else 1)
        st.session_state.mode = mode.lower()
        st.markdown("---")

        for msg in st.session_state.display:
            css = {'user':'msg-user','assistant':'msg-bot','agent':'msg-agent'}.get(msg['role'],'msg-bot')
            lbl = msg['role']
            st.markdown(f'<div class="{css}"><div class="msg-label">{lbl}</div>{msg["content"]}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        with st.form('chat', clear_on_submit=True):
            placeholder = "Ask a question..." if st.session_state.mode == 'chat' else "Give the agent a task..."
            user_input = st.text_input("Input:", placeholder=placeholder, label_visibility='collapsed')
            submitted  = st.form_submit_button("Send →")

        if submitted and user_input.strip():
            st.session_state.display.append({'role':'user','content': user_input})
            if st.session_state.mode == 'agent':
                with st.spinner("Agent thinking..."):
                    res = run_agent(user_input, collection, chunks, bm25_index, streamlit_mode=True)
                steps_html = ''.join(
                    f'<div class="step">Step {s["step"]}: {s["tool"]}({s["arg"][:50]}...) → {s["result"][:80]}...</div>'
                    if len(s["arg"])>50 else
                    f'<div class="step">Step {s["step"]}: {s["tool"]}({s["arg"]}) → {s["result"][:80]}</div>'
                    for s in res['steps']
                )
                content = f"{steps_html}<br/><strong>Answer:</strong> {res['answer']}"
                st.session_state.display.append({'role':'agent','content': content})
                st.session_state.last = {'type':'agent','data': res}
            else:
                with st.spinner("Thinking..."):
                    res = run_pipeline(user_input, collection, chunks, bm25_index,
                                       st.session_state.conv, streamlit_mode=True)
                st.session_state.display.append({'role':'assistant','content': res['response']})
                st.session_state.last = {'type':'chat','data': res}
            st.session_state.total += 1
            st.rerun()

    with col_side:
        st.markdown("### Pipeline")
        if st.session_state.last:
            d = st.session_state.last['data']
            if st.session_state.last['type'] == 'chat':
                qt = d['query_type']
                badge_cls = {'factual':'b-fact','comparison':'b-comp','general':'b-gen'}.get(qt,'b-gen')
                st.markdown(f'<span class="badge {badge_cls}">{qt}</span>', unsafe_allow_html=True)
                cc = 'b-ok' if d['is_confident'] else 'b-warn'
                cl = f"conf:{d['best_score']:.2f}" if d['is_confident'] else f"low:{d['best_score']:.2f}"
                st.markdown(f'<span class="badge {cc}">{cl}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="badge" style="background:#1a1a2a;color:#888">queries:{len(d["queries"])}</span>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**Before rerank**")
                for e,s in d['retrieved'][:4]:
                    st.markdown(f'<div class="chunk"><span class="cs">{s:.3f}</span> <span class="src">[{e["source"]} L{e["start_line"]}]</span><br/>{e["text"][:55]}...</div>', unsafe_allow_html=True)
                st.markdown("**After rerank**")
                for e,sim,rs in d['reranked']:
                    st.markdown(f'<div class="chunk"><span class="cs">sim:{sim:.2f} re:{rs:.0f}</span> <span class="src">[{e["source"]}]</span><br/>{e["text"][:55]}...</div>', unsafe_allow_html=True)
            else:
                st.markdown("**Agent Steps**")
                for s in d['steps']:
                    st.markdown(f'<div class="step">{s["step"]}. {s["tool"]}</div>', unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("**Session**")
        st.markdown(f'<div class="stat">Queries <span class="sv">{st.session_state.total}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat">Memory <span class="sv">{len(st.session_state.conv)//2} turns</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat">Chunks <span class="sv">{len(chunks)}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat">Mode <span class="sv">{st.session_state.mode}</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.conv=[]; st.session_state.display=[]
            st.session_state.last=None; st.session_state.total=0
            st.rerun()

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG Chatbot')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--agent',     action='store_true', help='Agent mode in terminal')
    args = parser.parse_args()

    print("="*60)
    print("  Initializing RAG Pipeline")
    print("="*60)
    print(f"  Chunk size:    {CHUNK_SIZE} lines, overlap {CHUNK_OVERLAP}")
    print(f"  Vector DB:     ChromaDB (persistent @ {CHROMA_DIR})")
    print(f"  Retrieval:     Hybrid BM25 + Dense + RRF fusion")
    print(f"  Query expand:  {True}")
    print(f"  Reranker:      LLM-based")
    print("="*60 + "\n")

    chunks     = chunk_documents(DATA_FOLDER)
    collection = build_or_load_chroma(chunks)
    bm25       = build_bm25_index(chunks)

    if args.benchmark:
        run_benchmark(collection, chunks, bm25)
    elif args.agent:
        print("Agent mode — type your task:\n")
        while True:
            try: q = input("Task: ").strip()
            except (EOFError, KeyboardInterrupt): print("\nGoodbye!"); break
            if not q: continue
            if q.lower() in ['exit','quit']: print("Goodbye!"); break
            res = run_agent(q, collection, chunks, bm25)
            print(f"\nFinal answer: {res['answer']}\n" + "-"*60)
    else:
        run_terminal(collection, chunks, bm25)
