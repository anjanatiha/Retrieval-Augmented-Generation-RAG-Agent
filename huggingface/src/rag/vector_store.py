"""vector_store.py — VectorStore class. HF Space version.

Replaces Ollama with:
  - sentence-transformers (BAAI/bge-base-en-v1.5) for embeddings
  - HF Inference Providers router (OpenAI-compatible) for LLM calls
  - chromadb.EphemeralClient for in-memory vector store
"""

import os
import re
import chromadb
from rank_bm25 import BM25Okapi

from src.rag.config import (
    EMBEDDING_MODEL, LANGUAGE_MODEL, LANGUAGE_MODEL_FALLBACKS,
    CHROMA_COLLECTION,
    SIMILARITY_THRESHOLD, TOP_RETRIEVE, TOP_RERANK,
)

__all__ = ['VectorStore']


def _load_st_model():
    """Load sentence-transformers model once at module level."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device='cpu')


# Module-level singletons — loaded once when the module is first imported
_ST_MODEL = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = _load_st_model()
    return _ST_MODEL


def _llm_call(prompt, max_tokens=512, temperature=0.1):
    """Call HF Inference API. Tries chat completions first, then text-generation fallback."""
    import requests
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("[WARNING] HF_TOKEN not set.")
        return "[LLM error: HF_TOKEN not set]"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    last_error = ""

    for model in LANGUAGE_MODEL_FALLBACKS:
        # ── Try 1: router chat completions (OpenAI-compatible) ──────────────
        chat_url = "https://router.huggingface.co/v1/chat/completions"
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.1),
            "stream": False,
        }
        try:
            resp = requests.post(chat_url, headers=headers, json=chat_payload, timeout=60)
            if resp.ok:
                result = resp.json()["choices"][0]["message"]["content"].strip()
                if result:
                    print(f"[LLM] Used model (chat): {model}")
                    return result
            error_body = resp.text[:300]
            print(f"[LLM] {model} chat HTTP {resp.status_code}: {error_body}")
            # If not a chat model, try text-generation format on the same model
            if "not a chat model" not in error_body:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"[LLM] {model} chat failed: {last_error}")
            continue

        # ── Try 2: text-generation endpoint (for non-chat models) ───────────
        gen_url = f"https://api-inference.huggingface.co/models/{model}"
        gen_payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.1),
                "return_full_text": False,
            },
        }
        try:
            resp = requests.post(gen_url, headers=headers, json=gen_payload, timeout=60)
            if resp.ok:
                data = resp.json()
                if isinstance(data, list) and data:
                    result = data[0].get("generated_text", "").strip()
                    if result:
                        print(f"[LLM] Used model (text-gen): {model}")
                        return result
            print(f"[LLM] {model} text-gen HTTP {resp.status_code}: {resp.text[:200]}")
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"[LLM] {model} text-gen failed: {last_error}")

    return f"[LLM error: all models failed. Last: {last_error}]"


class VectorStore:
    """Owns ChromaDB, BM25, hybrid search, reranking, query pipeline,
    response generation, and conversation history."""

    def __init__(self):
        self.collection          = None
        self.chunks              = []
        self.bm25_index          = None
        self.conversation_history = []

    # ── Public ──────────────────────────────────────────────────────────────

    def build_or_load(self, chunks):
        """Initialize an in-memory ChromaDB collection and embed any provided chunks."""
        client     = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        if chunks:
            print(f"Embedding {len(chunks)} chunks...")
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch  = chunks[i: i + batch_size]
                ids    = [f"chunk_{i+j}" for j in range(len(batch))]
                texts  = [c['text'] for c in batch]
                metas  = [{'source': c['source'], 'start_line': c['start_line'],
                           'end_line': c['end_line'], 'type': c.get('type', 'txt')}
                          for c in batch]
                embeds = [self._embed(self._truncate_for_embedding(t)) for t in texts]
                collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
            print(f"Ready — {collection.count()} chunks stored.\n")

        self.collection = collection
        self.chunks     = list(chunks)
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in chunks]) if chunks else None

    def add_chunks(self, chunks, id_prefix):
        """Add new chunks (e.g. from URL / file upload) to the live collection."""
        if not chunks:
            return
        offset = self.collection.count()
        ids    = [f"{id_prefix}_{offset+i}" for i in range(len(chunks))]
        texts  = [c['text'] for c in chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c.get('type', 'txt')}
                  for c in chunks]
        embeds = [self._embed(self._truncate_for_embedding(t)) for t in texts]
        self.collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
        self.chunks = self.chunks + list(chunks)

    def rebuild_bm25(self, all_chunks):
        """Rebuild the BM25 index after adding new chunks."""
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in all_chunks]) if all_chunks else None

    def run_pipeline(self, query, streamlit_mode=False):
        """Full RAG pipeline: classify → expand → retrieve → rerank → generate."""
        qtype      = self._classify_query(query)
        top_n      = self._smart_top_n(qtype)
        top_rerank = 10 if qtype == 'summarise' else TOP_RERANK
        queries    = self._expand_query(query)
        retrieved  = self._hybrid_retrieve(queries, top_n=top_n)
        is_confident, best_score = self._check_confidence(retrieved)
        reranked   = self._rerank(query, retrieved, top_n=top_rerank)

        context_lines = []
        for e, _, _ in reranked:
            label = self._source_label(e)
            context_lines.append(f" - [{e['source']} {label}] {e['text']}")
        context = '\n'.join(context_lines)

        if not is_confident:
            full_response = (
                "I could not find relevant information in the provided documents to answer this question. "
                "Please upload a document or add a URL that contains the relevant information."
            )
            self.conversation_history.append({'role': 'user', 'content': query})
            self.conversation_history.append({'role': 'assistant', 'content': full_response})
            return {
                'response':     full_response,
                'query_type':   qtype,
                'queries':      queries,
                'is_confident': False,
                'best_score':   best_score,
                'retrieved':    retrieved,
                'reranked':     reranked,
            }

        instruction_prompt = self._build_instruction_prompt(context)
        self.conversation_history.append({'role': 'user', 'content': query})

        messages = [{'role': 'system', 'content': instruction_prompt},
                    *self.conversation_history]
        full_response = self._llm_chat(messages, temperature=0.1)
        full_response = self._filter_hallucination(full_response)

        self.conversation_history.append({'role': 'assistant', 'content': full_response})

        return {
            'response':     full_response,
            'query_type':   qtype,
            'queries':      queries,
            'is_confident': is_confident,
            'best_score':   best_score,
            'retrieved':    retrieved,
            'reranked':     reranked,
        }

    def clear_conversation(self):
        self.conversation_history = []

    # ── Private — LLM ────────────────────────────────────────────────────────

    def _llm_chat(self, messages, temperature=0.0, max_tokens=512):
        """Single point of contact for all LLM calls via HF Inference API."""
        try:
            prompt = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in messages
            ) + "\nAssistant:"
            content = _llm_call(prompt, max_tokens=max_tokens, temperature=temperature)
            if not content:
                print("[WARNING] LLM returned empty response.")
                return "[LLM error: empty response from model]"
            return content.strip()
        except Exception as e:
            print(f"[ERROR] LLM call failed: {type(e).__name__}: {e}")
            return f"[LLM error: {type(e).__name__}: {e}]"

    # ── Private — vector/search ──────────────────────────────────────────────

    def _embed(self, text):
        """Embed text using sentence-transformers (runs locally)."""
        model = _get_st_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def _truncate_for_embedding(self, text, max_words=200, max_chars=1200):
        """Truncate to stay within bge-base-en 512 token limit."""
        words     = text.split()
        truncated = ' '.join(words[:max_words]) if len(words) > max_words else text
        return truncated[:max_chars] if len(truncated) > max_chars else truncated

    def _cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x**2 for x in a)**0.5
        nb  = sum(x**2 for x in b)**0.5
        return dot / (na * nb) if na and nb else 0.0

    def _hybrid_retrieve(self, queries, top_n, alpha=0.5):
        """True hybrid search: fuses BM25 (lexical) + ChromaDB (dense) scores."""
        if self.collection is None or self.collection.count() == 0:
            return []

        fused = {}

        for query in queries:
            q_emb   = self._embed(query)
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_n * 2, self.collection.count())
            )

            dense_map = {}
            for doc, meta, dist in zip(results['documents'][0],
                                        results['metadatas'][0],
                                        results['distances'][0]):
                entry = {
                    'text':       doc,
                    'source':     meta.get('source', '?'),
                    'start_line': meta.get('start_line', 0),
                    'end_line':   meta.get('end_line', 0),
                    'type':       meta.get('type', 'txt'),
                }
                dense_map[doc] = (entry, 1 - dist)

            if self.bm25_index is not None:
                tokenized       = query.lower().split()
                bm25_scores_raw = self.bm25_index.get_scores(tokenized)
                bm25_max        = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
                bm25_norm       = [s / bm25_max for s in bm25_scores_raw]
            else:
                bm25_norm = [0.0] * len(self.chunks)

            for doc, (entry, dense_score) in dense_map.items():
                bm25_score = 0.0
                for idx, c in enumerate(self.chunks):
                    if c['text'] == doc:
                        bm25_score = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
                        break
                score = alpha * dense_score + (1 - alpha) * bm25_score
                if doc not in fused or score > fused[doc][1]:
                    fused[doc] = (entry, score)

        return sorted(fused.values(), key=lambda x: x[1], reverse=True)[:top_n]

    def _rerank(self, query, candidates, top_n):
        # LLM reranking disabled on HF free CPU — too slow (N × API call = timeout).
        # Use hybrid similarity score directly instead.
        scored = [(entry, sim, sim) for entry, sim in candidates]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    def _rerank_prompt(self, query, entry):
        """Returns a reranking prompt tailored to the document type."""
        text     = entry['text']
        doc_type = entry.get('type', 'txt')

        if doc_type in ('xlsx', 'csv'):
            return (
                f"A user is searching for: {query}\n"
                f"Does this spreadsheet row contain relevant information to answer the query?\n"
                f"Row data: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )
        elif doc_type == 'pptx':
            return (
                f"A user is searching for: {query}\n"
                f"Does this presentation slide contain relevant information to answer the query?\n"
                f"Slide text: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )
        elif doc_type == 'pdf':
            return (
                f"A user is searching for: {query}\n"
                f"Does this PDF page extract contain relevant information to answer the query?\n"
                f"Page text: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )
        elif doc_type == 'docx':
            return (
                f"A user is searching for: {query}\n"
                f"Does this document paragraph contain relevant information to answer the query?\n"
                f"Paragraph: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )
        elif doc_type == 'html':
            return (
                f"A user is searching for: {query}\n"
                f"Does this webpage content contain relevant information to answer the query?\n"
                f"Content: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )
        elif doc_type == 'md':
            return (
                f"A user is searching for: {query}\n"
                f"Does this markdown document section contain relevant information to answer the query?\n"
                f"Section: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )
        else:
            return (
                f"On a scale of 1-10, how relevant is the following text to the query?\n"
                f"Query: {query}\nText: {text}\n"
                f"Reply with a single integer from 1 to 10 and nothing else."
            )

    # ── Private — query ──────────────────────────────────────────────────────

    def _classify_query(self, query):
        """Classifies query as summarise / comparison / factual / general."""
        q = query.lower()
        summarise_signals  = ['summarise', 'summarize', 'summary', 'overview',
                              'tell me about', 'what is in', 'describe', 'explain',
                              'give me a summary', 'resume']
        comparison_signals = ['compare', 'difference', 'vs', 'versus', 'better', 'worse',
                              'pros and cons', 'which is', 'how does', 'contrast']
        factual_signals    = ['what is', 'what are', 'who is', 'who are', 'when did',
                              'where is', 'how many', 'how much', 'does', 'did', 'has',
                              'have', 'list', 'name', 'define', 'tell me']
        if any(s in q for s in summarise_signals):
            return 'summarise'
        if any(s in q for s in comparison_signals):
            return 'comparison'
        if any(s in q for s in factual_signals):
            return 'factual'
        return 'general'

    def _expand_query(self, query):
        """Query expansion disabled on HF free CPU — saves 1 LLM call per query."""
        return [query]

    def _check_confidence(self, results):
        if not results:
            return False, 0.0
        best = results[0][1]
        return best >= SIMILARITY_THRESHOLD, best

    def _smart_top_n(self, query_type):
        return {'factual': 5, 'comparison': 15, 'general': 10,
                'summarise': TOP_RETRIEVE}.get(query_type, TOP_RETRIEVE)

    # ── Private — response ───────────────────────────────────────────────────

    def _build_instruction_prompt(self, context):
        return (
            "You are a document question-answering assistant.\n"
            "Answer the question using ONLY the context passages provided below.\n"
            "STRICT RULES:\n"
            "- Do NOT use your training data or general knowledge under any circumstances.\n"
            "- If the context does not contain the answer, say exactly: "
            "'The provided documents do not contain information about this topic.'\n"
            "- Do NOT speculate, infer, or elaborate beyond what the context states.\n"
            "- At the end of your answer, cite ONLY the bracketed source labels from the context "
            "(e.g. [filename.pdf p3] or [example.com/page s12]). "
            "Do NOT copy any bibliographic references, footnotes, or citations that appear "
            "inside the text.\n\n"
            f"CONTEXT:\n{context}"
        )

    def _source_label(self, entry):
        """Returns a consistent location label for any doc type."""
        t = entry.get('type', 'txt')
        if t == 'pdf':
            return f"p{entry['start_line']}"
        elif t in ('xlsx', 'csv'):
            return f"row{entry['start_line']}"
        elif t == 'pptx':
            return f"slide{entry['start_line']}"
        elif t == 'html':
            return f"s{entry['start_line']}"
        else:
            return f"L{entry['start_line']}-{entry['end_line']}"

    def _synthesize(self, question, context):
        """Takes raw retrieved context and asks LLM to produce a clean direct answer."""
        prompt = (
            "You are a helpful assistant. Answer the question below using ONLY the "
            "provided context. Be concise and direct. Do not repeat the context — "
            "just answer the question. Cite the source filename at the end.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        try:
            return self._llm_chat([{'role': 'user', 'content': prompt}], temperature=0)
        except Exception:
            return context

    def _filter_hallucination(self, response):
        """Truncate at hallucination pivot if model admitted no-info then hallucinated."""
        _no_info_phrases = [
            "there is no information",
            "i couldn't find",
            "i could not find",
            "the provided context does not",
            "the provided documents do not",
            "no information in the provided",
            "not mentioned in the",
            "not found in the",
        ]
        _hallucination_pivots = [
            "however,", "but i can", "but,", "that said,",
            "nevertheless,", "i can tell you", "i can provide",
        ]
        lower_resp = response.lower()
        if any(p in lower_resp for p in _no_info_phrases):
            for pivot in _hallucination_pivots:
                idx = lower_resp.find(pivot)
                if idx != -1:
                    return (
                        response[:idx].strip() + "\n\n"
                        "I can only answer based on the uploaded documents. "
                        "Please add a relevant document or URL to get an answer."
                    )
        return response
