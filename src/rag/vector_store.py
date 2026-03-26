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


class VectorStore:
    def __init__(self):
        self.collection          = None
        self.chunks              = []
        self.bm25_index          = None
        self.conversation_history = []

    # ── Public ──────────────────────────────────────────────────────────────

    def build_or_load(self, chunks):
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

        self.collection = collection
        self.chunks     = chunks
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in chunks])

    def add_chunks(self, chunks, id_prefix):
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

    def rebuild_bm25(self, all_chunks):
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in all_chunks])

    def run_pipeline(self, query, streamlit_mode=False):
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

        instruction_prompt = self._build_instruction_prompt(context)

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

    def stream_response(self, stream):
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

    def clear_conversation(self):
        self.conversation_history = []

    # ── Private — vector/search ──────────────────────────────────────────────

    def _embed(self, text):
        return ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]

    def _truncate_for_embedding(self, text, max_words=200, max_chars=1200):
        """Truncate to stay within bge-base-en 512 token limit.
        Caps on both word count and character count — whichever is shorter.
        """
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
                score = alpha * dense_score + (1 - alpha) * bm25_score
                if doc not in fused or score > fused[doc][1]:
                    fused[doc] = (entry, score)

        return sorted(fused.values(), key=lambda x: x[1], reverse=True)[:top_n]

    def _rerank(self, query, candidates, top_n):
        scored = []
        for entry, sim in candidates:
            prompt = self._rerank_prompt(query, entry)
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
            # txt and any other type — generic prompt
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
            resp = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0}
            )
            return resp['message']['content'].strip()
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
