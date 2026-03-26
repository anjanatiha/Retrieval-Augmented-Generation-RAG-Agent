# DESIGN.md — Architectural Decisions

## Why 4 Classes Instead of More Granular Decomposition

The system uses exactly **4 classes** and **4 modules**. This is a deliberate choice, not a limitation.

**Classes own state. Modules own constants and stateless functions.**

The 4 classes map directly to the 4 units of meaningful state in the system:

| Class | State it owns |
|-------|--------------|
| `DocumentLoader` | File system paths, extension map, chunk size config |
| `VectorStore` | ChromaDB collection, BM25 index, chunk list, conversation history |
| `Agent` | ReAct message history, collected context, tool dispatch |
| `Benchmarker` | VectorStore reference, results file path |

A finer decomposition (e.g. a `QueryExpander` class, a `Reranker` class, a `HybridSearcher` class) would split logic that shares the same state — the vector collection and BM25 index — across multiple objects that would all need to reference each other. This creates indirection and circular dependency risk without adding clarity.

---

## What Each Class Owns and Why

### `DocumentLoader`
Owns all ingestion logic: reading files from disk, parsing 9 formats, chunking, misplaced file detection, and URL fetching. These belong together because they all transform raw input into the same output structure (a list of chunk dicts). There is no meaningful reason to separate PDF parsing from DOCX parsing — they are just different implementations of the same operation.

### `VectorStore`
Owns ChromaDB, BM25, hybrid retrieval, reranking, query expansion, query classification, response generation, hallucination filtering, and conversation history. These all operate on the same two data structures — the vector collection and the chunk list. Splitting them would require passing the collection and chunks to every caller, which is worse than grouping them under one owner.

### `Agent`
Owns the ReAct loop and all 5 tools as private methods. The tools (calculator, summarise, sentiment, rag_search) have no independent state — they are pure operations that happen to live inside the agent's context. Making each tool a class would add 5 empty `__init__` methods and 5 objects that exist only to call one function.

### `Benchmarker`
Owns all evaluation logic: scoring, run comparison, result persistence. Takes a `VectorStore` as a dependency so it can run the same pipeline the user runs in production.

---

## Why Tools Are Private Methods on Agent, Not Separate Classes

The 5 agent tools have no state of their own. `_tool_calculator` is `eval()` with a safety check. `_tool_summarise` is one `ollama.chat` call. `_tool_sentiment` is one `ollama.chat` call with an optional prior RAG search. None of them maintain anything between calls.

Making them classes would require:
- An empty `__init__` or a store reference injected into each
- An `execute()` or `run()` method containing one function call
- An import chain that risks circular dependencies

Three similar lines of code is better than a premature abstraction. The private method prefix (`_`) makes the boundary clear: these are implementation details of the agent, not public API.

---

## Tradeoffs

### ChromaDB vs Pinecone / Weaviate
ChromaDB runs entirely locally with no API key, no network call, and no cost. For a fully local system this is the right choice. The tradeoff is that ChromaDB does not scale horizontally — at production scale with millions of chunks, a managed vector store (Pinecone, Weaviate, Qdrant) would be preferable.

### Local LLM (Ollama) vs API (OpenAI / Anthropic)
Local inference via Ollama means zero data leaves the machine. The tradeoff is speed and quality — `Llama-3.2-3B-Instruct` is much smaller than GPT-4 or Claude. For a demo or private document system this is acceptable. For production with latency SLAs, an API call to a larger model would produce better answers faster.

### BM25 + Dense Hybrid vs Pure Dense
Pure dense retrieval misses exact keyword matches — searching for "NLP" may not retrieve a chunk that says "natural language processing" at the same score as a semantically similar but lexically different chunk. BM25 is excellent at exact matches. The hybrid (alpha=0.5) gets the best of both. The tradeoff is that BM25 must be rebuilt in-memory when new chunks are added, which is O(n) in the number of chunks.

### Sentence-level Chunking vs Fixed-size Chunking
Sentence windows (PDF: 5 sentences, HTML: 5 sentences) produce chunks with complete thoughts rather than arbitrary byte cuts. This improves embedding quality because the model sees coherent context. The tradeoff is variable chunk size, which makes batch embedding slightly less predictable.

---

## What the Type-Aware Reranker Solves

A generic reranker prompt scores all chunks the same way. This works poorly for structured data:

- A spreadsheet row like `Name=Alice | Role=Engineer | Salary=90000` looks like a key-value dump to a generic prompt, which scores it low because it is not fluent prose.
- A slide title and three bullet points looks incomplete without its presentation context.
- A CSV row without column headers is meaningless unless framed correctly.

The type-aware reranker uses 7 different prompts — one per document type — that frame the content appropriately. The xlsx/csv prompt says "does this **spreadsheet row** contain relevant information?" The pptx prompt says "does this **presentation slide** contain relevant information?" This framing dramatically improves reranking accuracy on structured data.

---

## Benchmark Metrics

| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Word overlap between response and retrieved context — measures whether the model stayed grounded or hallucinated |
| `answer_relevancy` | F1 overlap between question keywords and response keywords — measures whether the answer addressed the question |
| `keyword_recall` | Fraction of expected keywords found in the response — measures factual completeness |
| `context_relevance` | Mean similarity score of the top-reranked chunks — measures whether the retrieval pipeline found relevant content |
| `overall` | Mean of the four above — single headline metric for pipeline health |

A low `faithfulness` with a high `context_relevance` indicates the model is hallucinating despite having good context. A low `context_relevance` with low `faithfulness` indicates the retrieval is broken upstream.

---

## How to Scale to Production

1. **Async embedding**: Replace sequential `ollama.embed` calls with a batched async queue. At 50 chunks/batch this is the main bottleneck at index time.

2. **Managed vector store**: Replace `chromadb.PersistentClient` with Pinecone or Weaviate for horizontal scaling and filtering. The `VectorStore` interface isolates this behind `build_or_load` and `add_chunks` — swapping the backend requires changing two methods.

3. **Streaming responses**: The pipeline already supports `stream=True`. In the Streamlit UI, pipe the stream token-by-token into `st.write_stream` for live token output.

4. **Caching embeddings**: Cache `_embed` results by text hash to avoid re-embedding duplicate content across sessions.

5. **Larger language model**: Swap `LANGUAGE_MODEL` in `config.py` to a larger Ollama model or an API model. The model name is the only change required.

6. **Distributed BM25**: At millions of documents, replace `BM25Okapi` with Elasticsearch or OpenSearch for distributed keyword search. The `_hybrid_retrieve` method isolates the BM25 call — swap the implementation behind the same interface.

7. **Authentication and multi-tenancy**: Add a user ID to ChromaDB metadata filters so each user only retrieves from their own documents.

---

## What Would Be Done Differently with More Time

- **Streaming in the Streamlit chat**: Currently the full response is collected before display. Token-level streaming would make the UI feel much faster.
- **Persistent conversation history**: Conversation history lives in memory and resets on restart. A database-backed store (SQLite or Redis) would enable cross-session memory.
- **Evaluation dataset**: The 5-question cat facts benchmark is a smoke test. A real evaluation dataset with 50-100 domain-specific questions and human-verified ground truth would give meaningful signal on retrieval quality.
- **Document update detection**: Currently, if a document changes on disk, the system does not detect it. A file hash comparison at startup would trigger targeted re-embedding of changed files without a full rebuild.
- **Reranker fine-tuning**: The LLM reranker uses a zero-shot prompt. Fine-tuning a small cross-encoder (e.g. `ms-marco-MiniLM`) on domain-specific relevance pairs would give faster and more accurate reranking than a generative model.
