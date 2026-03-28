"""vector_store.py — VectorStore class.

Owns ChromaDB, BM25, hybrid search, reranking, query pipeline,
response generation, and conversation history.
"""

import re
import sys
import time
import ollama
import chromadb
from rank_bm25 import BM25Okapi

from src.rag.config import (
    EMBEDDING_MODEL, LANGUAGE_MODEL,
    CHROMA_DIR, CHROMA_COLLECTION,
    SIMILARITY_THRESHOLD, TOP_RETRIEVE, TOP_RERANK,
)
from src.rag import logger
from src.rag.reranker import rerank_prompt


class VectorStore:
    """Owns all retrieval, search, and response generation.

    State:
        collection:           ChromaDB collection (persistent on disk)
        chunks:               list of all indexed chunk dicts (local + runtime-added)
        bm25_index:           BM25Okapi keyword index — rebuilt after every add_chunks call
        conversation_history: list of {'role', 'content'} dicts for multi-turn context

    Public API:
        build_or_load(chunks)               — embed and persist, or load from disk
        add_chunks(chunks, id_prefix)       — add URL/file-upload chunks at runtime
        rebuild_bm25(all_chunks)            — rebuild keyword index after runtime additions
        run_pipeline(query, streamlit_mode) — full chat pipeline: expand → retrieve → rerank → respond
        stream_response(stream)             — print tokens to terminal with typing indicator
        clear_conversation()                — reset multi-turn history
    """

    def __init__(self):
        """Initialise all state to empty; call build_or_load() before querying."""
        self.collection          = None
        self.chunks              = []
        self._local_chunks       = []   # snapshot of local-doc chunks set at build time
        self.bm25_index          = None
        self.conversation_history = []

    # ── Public ──────────────────────────────────────────────────────────────

    def build_or_load(self, chunks: list) -> None:
        """Load existing ChromaDB collection or embed all chunks and persist to disk.

        Args:
            chunks: list of chunk dicts from DocumentLoader.chunk_all_documents().
        """
        client     = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        existing = collection.count()

        if existing >= len(chunks):
            print(f"ChromaDB loaded — {existing} chunks already stored.\n")
        else:
            if existing > 0:
                print(f"ChromaDB has {existing} chunks but dataset has {len(chunks)} — rebuilding...")
                collection.delete(ids=collection.get()['ids'])

            print(f"Embedding {len(chunks)} chunks into ChromaDB...")
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
                print(f"  Stored {min(i+batch_size, len(chunks))}/{len(chunks)}", end='\r')
            print(f"\nChromaDB ready — {collection.count()} chunks stored.\n")

        self.collection    = collection
        self.chunks        = chunks
        self._local_chunks = list(chunks)   # frozen snapshot — used by clear_added_chunks()
        self.bm25_index    = BM25Okapi([c['text'].lower().split() for c in chunks])

    def add_chunks(self, chunks: list, id_prefix: str) -> None:
        """Add new chunks (e.g. from URL ingestion) to the live collection."""
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
        self.chunks = self.chunks + chunks

    def rebuild_bm25(self, all_chunks: list) -> None:
        """Rebuild the BM25 keyword index from scratch after chunks are added at runtime.

        Args:
            all_chunks: combined list of base chunks + newly added chunks.
        """
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in all_chunks])

    def run_pipeline(self, query: str, streamlit_mode: bool = False) -> dict:
        """Run the full RAG pipeline for a user query.

        Args:
            query:          The user's question.
            streamlit_mode: If True, collect tokens into a string and return a result dict.
                            If False, stream tokens to the terminal.

        Returns:
            dict with keys: response, query_type, queries, is_confident,
            best_score, retrieved, reranked.
        """
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
            logger.log_interaction(query, qtype, 0, [], full_response)
            return {
                'response':     full_response,
                'query_type':   qtype,
                'queries':      queries,
                'is_confident': False,
                'best_score':   best_score,
                'retrieved':    retrieved,
                'reranked':     reranked,
            }

        instruction_prompt = self._build_instruction_prompt(context, qtype)

        self.conversation_history.append({'role': 'user', 'content': query})
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'system', 'content': instruction_prompt},
                      *self.conversation_history],
            stream=True,
        )

        full_response = (''.join(c['message']['content'] for c in stream)
                         if streamlit_mode else self.stream_response(stream))

        full_response = self._filter_hallucination(full_response)

        self.conversation_history.append({'role': 'assistant', 'content': full_response})

        sim_scores = [s for _, s, _ in reranked]
        logger.log_interaction(query, qtype, len(reranked), sim_scores, full_response)

        return {
            'response':     full_response,
            'query_type':   qtype,
            'queries':      queries,
            'is_confident': is_confident,
            'best_score':   best_score,
            'retrieved':    retrieved,
            'reranked':     reranked,
        }

    def stream_response(self, stream) -> str:
        """Print a streaming Ollama response to the terminal token by token.

        Args:
            stream: the iterable returned by ollama.chat(..., stream=True).

        Returns:
            The full response string.
        """
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

    def clear_added_chunks(self) -> int:
        """Remove all URL and file-upload chunks added at runtime.

        Deletes every chunk whose ChromaDB ID starts with 'url_' or 'file_'
        (the prefixes used by add_chunks()). Local document chunks — loaded
        from ./docs/ at startup — are kept. BM25 is rebuilt from the remaining
        local chunks.

        Returns:
            Number of chunks removed.
        """
        # Get all IDs currently in the collection
        all_ids     = self.collection.get()['ids']
        runtime_ids = [id_ for id_ in all_ids
                       if id_.startswith('url_') or id_.startswith('file_')]

        # Delete runtime chunks from ChromaDB
        if runtime_ids:
            self.collection.delete(ids=runtime_ids)

        # Reset the in-memory chunk list to the local-only snapshot
        self.chunks = list(self._local_chunks)
        self.rebuild_bm25(self.chunks)

        return len(runtime_ids)

    def clear_conversation(self) -> None:
        """Reset multi-turn conversation history so the next query starts fresh."""
        self.conversation_history = []

    # ── Private — vector/search ──────────────────────────────────────────────

    def _embed(self, text: str) -> list:
        # ['embeddings'][0] — ollama.embed returns a list of embedding vectors; we always send one text.
        return ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]

    def _truncate_for_embedding(self, text: str, max_words: int = 200, max_chars: int = 1200) -> str:
        """Truncate to stay within bge-base-en 512 token limit.
        Caps on both word count and character count — whichever is shorter.
        """
        words     = text.split()
        truncated = ' '.join(words[:max_words]) if len(words) > max_words else text
        return truncated[:max_chars] if len(truncated) > max_chars else truncated

    def _cosine_similarity(self, a: list, b: list) -> float:
        # Manual dot product — avoids a numpy import for this single operation.
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x**2 for x in a)**0.5
        nb  = sum(x**2 for x in b)**0.5
        return dot / (na * nb) if na and nb else 0.0

    def _hybrid_retrieve(self, queries: list, top_n: int, alpha: float = 0.5) -> list:
        """True hybrid search: fuses BM25 (lexical) + ChromaDB (dense) scores."""
        fused = {}  # doc_text → (entry, best_score)

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

            tokenized       = query.lower().split()
            bm25_scores_raw = self.bm25_index.get_scores(tokenized)
            bm25_max        = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
            bm25_norm       = [s / bm25_max for s in bm25_scores_raw]

            for doc, (entry, dense_score) in dense_map.items():
                bm25_score = 0.0
                for idx, c in enumerate(self.chunks):
                    if c['text'] == doc:
                        bm25_score = bm25_norm[idx]
                        break
                # alpha=0.5 gives equal weight to semantic (dense) and exact-match (BM25) signals.
                # Keeping the best score per doc across all expanded queries avoids double-counting.
                score = alpha * dense_score + (1 - alpha) * bm25_score
                if doc not in fused or score > fused[doc][1]:
                    fused[doc] = (entry, score)

        return sorted(fused.values(), key=lambda x: x[1], reverse=True)[:top_n]

    def _rerank(self, query: str, candidates: list, top_n: int) -> list:
        # LLM gives a 1–10 relevance score; divide by 10 to normalise to [0, 1].
        # Falls back to the hybrid similarity score if the LLM call fails.
        scored = []
        for entry, sim in candidates:
            prompt = rerank_prompt(query, entry)
            try:
                resp = ollama.chat(
                    model=LANGUAGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0}
                )
                raw       = resp['message']['content'].strip()
                m         = re.search(r'\d+', raw)
                llm_score = float(m.group()) / 10.0 if m else sim
            except Exception:
                llm_score = sim
            scored.append((entry, sim, llm_score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    # ── Private — query ──────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
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

    def _expand_query(self, query: str) -> list:
        """Generates 2 alternative phrasings of the query using the LLM."""
        prompt = (
            "Rewrite the following search query in 2 different ways to improve document retrieval. "
            "Use synonyms, acronyms, and related terms. Output ONLY the 2 rewrites, one per line, "
            "no numbering, no explanation.\n\n"
            f"Query: {query}"
        )
        try:
            resp = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.3}
            )
            lines = [l.strip() for l in resp['message']['content'].strip().splitlines()
                     if l.strip()]
            expansions = lines[:2]
            return [query] + expansions
        except Exception:
            return [query]

    def _check_confidence(self, results: list) -> tuple:
        # Skip LLM call entirely when no chunk clears the threshold — prevents hallucination on empty context.
        if not results:
            return False, 0.0
        best = results[0][1]
        return best >= SIMILARITY_THRESHOLD, best

    def _smart_top_n(self, query_type: str) -> int:
        # Budget scales with query complexity — factual queries have a clear correct answer so 5 chunks is enough;
        # comparison needs both sides of the argument (15); summarise needs the full document scope (TOP_RETRIEVE).
        return {'factual': 5, 'comparison': 15, 'general': 10,
                'summarise': TOP_RETRIEVE}.get(query_type, TOP_RETRIEVE)

    # ── Private — response ───────────────────────────────────────────────────

    def _build_instruction_prompt(self, context: str, query_type: str = 'factual') -> str:
        # Length hints keep answers proportional to query complexity — factual stays short,
        # summarise gets room to cover all key points.
        _length_hints = {
            'factual':    '2-3 sentences',
            'general':    '3-4 sentences',
            'comparison': '4-6 sentences covering each item being compared',
            'summarise':  '6-8 sentences covering all key points',
        }
        length_hint = _length_hints.get(query_type, '2-3 sentences')
        return (
            "You are a document question-answering assistant.\n"
            "Answer the question using ONLY the context passages provided below.\n"
            "STRICT RULES:\n"
            "- Do NOT use your training data or general knowledge under any circumstances.\n"
            "- If the context does not contain the answer, say exactly: "
            "'The provided documents do not contain information about this topic.'\n"
            "- Do NOT speculate, infer, or elaborate beyond what the context states.\n"
            f"- Answer in {length_hint}.\n"
            "- At the end of your answer, cite ONLY the bracketed source labels from the context "
            "(e.g. [filename.pdf p3] or [example.com/page s12]). "
            "Do NOT copy any bibliographic references, footnotes, or citations that appear "
            "inside the text.\n\n"
            f"CONTEXT:\n{context}"
        )

    def _source_label(self, entry: dict) -> str:
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

    def _synthesize(self, question: str, context: str) -> str:
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
            resp = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0}
            )
            return resp['message']['content'].strip()
        except Exception:
            return context

    def _filter_hallucination(self, response: str) -> str:
        """Truncate at hallucination pivot if model admitted no-info then hallucinated.

        Pattern: model says "there is no information..." then continues with "however, I can..."
        We keep the no-info admission and drop everything from the pivot word onward,
        replacing it with a standard redirect message. This prevents the model from
        answering from training data after correctly admitting the context is missing.
        """
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
