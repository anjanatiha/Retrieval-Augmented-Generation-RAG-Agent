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
EMBEDDING_MODEL  = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL   = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
DATA_FOLDER      = '.'          # folder with .txt files (change if needed)
CACHE_FILE       = 'embeddings_cache.json'
LOG_FILE         = 'rag_logs.json'
SIMILARITY_THRESHOLD = 0.4      # below this = low confidence warning
TOP_RETRIEVE     = 10           # how many chunks to retrieve before reranking
TOP_RERANK       = 3            # how many chunks to keep after reranking

# ============================================================
# 1. LOAD DOCUMENTS (multiple .txt files)
# ============================================================
def load_documents(folder_path):
    dataset = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"⚠️  No .txt files found in '{folder_path}'")
        sys.exit(1)

    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    dataset.append({
                        'text': line,
                        'source': filename,
                        'line': i + 1
                    })
        print(f"📄 Loaded '{filename}' — {i+1} lines")

    print(f"✅ Total chunks loaded: {len(dataset)}\n")
    return dataset


# ============================================================
# 2. EMBEDDINGS WITH CACHE
# ============================================================
def load_or_compute_embeddings(dataset):
    # Try loading from cache
    if os.path.exists(CACHE_FILE):
        print("⚡ Loading embeddings from cache...")
        with open(CACHE_FILE, 'r') as f:
            cached = json.load(f)
        # Validate cache matches current dataset
        if len(cached) == len(dataset):
            print(f"✅ Cache loaded ({len(cached)} embeddings)\n")
            return [(dataset[i], emb) for i, emb in enumerate(cached)]
        else:
            print("⚠️  Cache mismatch — recomputing embeddings...")

    # Compute fresh embeddings
    print("🔄 Computing embeddings (this may take a moment)...")
    embeddings = []
    raw_embeddings = []
    for i, entry in enumerate(dataset):
        emb = ollama.embed(model=EMBEDDING_MODEL, input=entry['text'])['embeddings'][0]
        embeddings.append((entry, emb))
        raw_embeddings.append(emb)
        print(f"  Embedded {i+1}/{len(dataset)}", end='\r')

    # Save cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(raw_embeddings, f)
    print(f"\n✅ Embeddings computed and cached\n")
    return embeddings


# ============================================================
# 3. COSINE SIMILARITY
# ============================================================
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ============================================================
# 4. HYBRID SEARCH (BM25 + Dense Vector)
# ============================================================
def build_bm25_index(dataset):
    tokenized = [entry['text'].lower().split() for entry in dataset]
    return BM25Okapi(tokenized)

def hybrid_retrieve(query, vector_db, bm25_index, dataset, top_n=TOP_RETRIEVE, alpha=0.5):
    # --- Dense retrieval ---
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    dense_scores = [cosine_similarity(query_embedding, emb) for _, emb in vector_db]

    # --- Sparse retrieval (BM25) ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query).tolist()

    # --- Normalize scores ---
    max_dense = max(dense_scores) or 1
    max_bm25  = max(bm25_scores)  or 1

    # --- Combine ---
    combined = []
    for i, (entry, _) in enumerate(vector_db):
        score = (alpha * dense_scores[i] / max_dense +
                 (1 - alpha) * bm25_scores[i] / max_bm25)
        combined.append((entry, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_n]


# ============================================================
# 5. SIMPLE RERANKER (LLM-based, no extra library)
# ============================================================
def rerank(query, chunks, top_n=TOP_RERANK):
    scored = []
    for entry, similarity in chunks:
        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{
                'role': 'user',
                'content': (
                    f"Rate how relevant this text is to the query on a scale of 0 to 10.\n"
                    f"Query: {query}\n"
                    f"Text: {entry['text']}\n"
                    f"Reply with only a single number between 0 and 10."
                )
            }]
        )
        try:
            score = float(response['message']['content'].strip().split()[0])
        except Exception:
            score = 0.0
        scored.append((entry, similarity, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


# ============================================================
# 6. QUERY CLASSIFICATION
# ============================================================
def classify_query(query):
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{
            'role': 'user',
            'content': (
                f"Classify this query into exactly one word:\n"
                f"- factual\n- comparison\n- general\n\n"
                f"Query: {query}\n"
                f"Reply with only the single category word."
            )
        }]
    )
    category = response['message']['content'].strip().lower()
    if 'factual' in category:
        return 'factual'
    elif 'comparison' in category:
        return 'comparison'
    else:
        return 'general'

def smart_top_n(query_type):
    return {'factual': 5, 'comparison': 15, 'general': 10}.get(query_type, TOP_RETRIEVE)


# ============================================================
# 7. CONFIDENCE CHECK
# ============================================================
def check_confidence(chunks):
    if not chunks:
        return False, 0.0
    best_similarity = chunks[0][1]
    return best_similarity >= SIMILARITY_THRESHOLD, best_similarity


# ============================================================
# 8. LOGGING
# ============================================================
def log_interaction(query, query_type, chunks_used, similarity_scores, response):
    entry = {
        'timestamp':      datetime.now().isoformat(),
        'query':          query,
        'query_type':     query_type,
        'chunks_used':    chunks_used,
        'top_similarity': round(similarity_scores[0], 4) if similarity_scores else 0,
        'avg_similarity': round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0,
        'response_length': len(response),
    }
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


# ============================================================
# 9. STREAMING WITH TYPING INDICATOR
# ============================================================
def stream_response(stream):
    # Typing indicator
    print("\nChatbot: ", end='', flush=True)
    for _ in range(3):
        print('.', end='', flush=True)
        time.sleep(0.3)
    print('\r' + ' ' * 30 + '\r', end='', flush=True)
    print("Chatbot: ", end='', flush=True)

    full_response = ''
    for chunk in stream:
        content = chunk['message']['content']
        sys.stdout.write(content)
        sys.stdout.flush()
        full_response += content
    print()
    return full_response


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("        🐱 RAG Chatbot — Enhanced Pipeline")
    print("=" * 60 + "\n")

    # --- Load & embed ---
    dataset    = load_documents(DATA_FOLDER)
    vector_db  = load_or_compute_embeddings(dataset)
    bm25_index = build_bm25_index(dataset)

    # --- Conversation memory ---
    conversation_history = []

    print("💬 Type your question (or 'exit' to quit)\n")

    # --- Single question (no loop) ---
    input_query = input("You: ").strip()

    if not input_query or input_query.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        return

    # Step 1 — Classify query
    print("\n🔍 Classifying query...")
    query_type = classify_query(input_query)
    top_n      = smart_top_n(query_type)
    print(f"   Query type: {query_type} → retrieving top {top_n} chunks")

    # Step 2 — Hybrid retrieval
    print("\n📚 Retrieving relevant chunks (hybrid BM25 + vector)...")
    retrieved = hybrid_retrieve(input_query, vector_db, bm25_index, dataset, top_n=top_n)

    print(f"\nBefore reranking (top {len(retrieved)}):")
    for entry, score in retrieved:
        print(f"  (score: {score:.3f}) [{entry['source']} L{entry['line']}] {entry['text'][:80]}...")

    # Step 3 — Confidence check
    is_confident, best_score = check_confidence(retrieved)
    if not is_confident:
        print(f"\n⚠️  Low confidence ({best_score:.2f}) — answer may be unreliable")

    # Step 4 — Rerank
    print(f"\n🔄 Reranking to top {TOP_RERANK}...")
    reranked = rerank(input_query, retrieved, top_n=TOP_RERANK)

    print(f"\nAfter reranking (top {TOP_RERANK}):")
    for entry, similarity, rerank_score in reranked:
        print(f"  (similarity: {similarity:.3f} | rerank: {rerank_score:.1f}) "
              f"[{entry['source']} L{entry['line']}] {entry['text'][:80]}...")

    # Step 5 — Build context with source citation
    context_lines = []
    for entry, similarity, rerank_score in reranked:
        context_lines.append(
            f" - [{entry['source']}, line {entry['line']}] {entry['text']}"
        )
    context = '\n'.join(context_lines)

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question.\n"
        "Do not make up any new information.\n"
        "At the end of your answer, cite the sources you used.\n\n"
        f"{context}"
    )

    # Step 6 — Add to conversation memory
    conversation_history.append({'role': 'user', 'content': input_query})

    # Step 7 — Stream response
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            *conversation_history,
        ],
        stream=True,
    )

    full_response = stream_response(stream)

    # Step 8 — Save to conversation memory
    conversation_history.append({'role': 'assistant', 'content': full_response})

    # Step 9 — Log interaction
    similarity_scores = [sim for _, sim, _ in reranked]
    log_interaction(input_query, query_type, len(reranked), similarity_scores, full_response)
    print(f"\n📝 Interaction logged to '{LOG_FILE}'")
    print("=" * 60)


if __name__ == '__main__':
    main()
