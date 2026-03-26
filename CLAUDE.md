# CLAUDE.md — RAG Agent

This file tells Claude Code how to work with this codebase.
Read it fully before making any changes.

---

## Project Overview

RAG Agent is a fully local, production-grade Retrieval-Augmented Generation system.
It ingests documents in 8 formats, retrieves using hybrid BM25 + dense vector search,
reranks with a type-aware LLM reranker, and answers via LLaMA 3.2 — all on-device via Ollama.

**Entry points:**
- `python main.py` — terminal chatbot
- `python main.py --agent` — agent mode (ReAct loop)
- `python main.py --benchmark` — benchmark evaluation
- `streamlit run app.py` — Streamlit web UI

---

## Target Project Structure

```
rag/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py            ← all constants and settings
│       ├── document_loader.py   ← DocumentLoader class
│       ├── retriever.py         ← Retriever class
│       ├── agent.py             ← Agent class
│       └── benchmarker.py       ← Benchmarker class
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_retriever.py
│   ├── test_agent.py
│   └── test_benchmarker.py
├── app.py                       ← Streamlit UI (thin wrapper only)
├── main.py                      ← CLI entry point (thin wrapper only)
├── requirements.txt
├── pyproject.toml
├── CLAUDE.md                    ← this file
└── README.md
```

---

## Architecture

### Class Responsibilities

#### `config.py`
- All hardcoded constants: model names, paths, thresholds, chunk sizes
- No logic — pure configuration only
- Everything imported from here, never hardcoded elsewhere

#### `DocumentLoader`
Owns everything related to getting text into the system.
- `ensure_folders()` — create doc subfolders
- `scan_all_files()` — walk all subfolders, detect type by extension, flag misplaced files
- `chunk_all_documents()` — dispatch to correct chunker per type
- Per-type chunkers: `_chunk_txt`, `_chunk_pdf`, `_chunk_docx`, `_chunk_xlsx`, `_chunk_pptx`, `_chunk_csv`, `_chunk_md`, `_chunk_html`
- `chunk_url(url)` — fetch and chunk a remote URL
- `truncate_chunk(text)` — enforce 300-word limit before embedding

#### `Retriever`
Owns all retrieval, indexing, and ranking logic.
- `build_or_load_chroma(chunks)` — persistent ChromaDB setup
- `build_bm25_index(chunks)` — in-memory BM25 index
- `embed(text)` — call Ollama embedding model
- `hybrid_retrieve(queries, collection, chunks, bm25, top_n)` — fuse BM25 + vector results
- `rerank(query, candidates, top_n)` — LLM reranker with type-aware prompts
- `expand_query(query)` — LLM query expansion
- `classify_query(query)` — factual / comparison / general
- `check_confidence(results)` — similarity threshold filter
- `smart_top_n(query_type)` — dynamic retrieval depth

#### `Agent`
Owns the agentic ReAct loop and tool execution.
- `run_agent(task, collection, chunks, bm25)` — main ReAct loop
- `_parse_tool_call(text)` — extract tool name + argument from LLM output
- `_tool_rag_search(query, collection, chunks, bm25)` — calls Retriever
- `_tool_calculator(expr)` — safe arithmetic eval
- `_tool_summarise(passage)` — LLM summarisation pass
- `_build_system_prompt(tools)` — construct agent system prompt

#### `Benchmarker`
Owns evaluation logic.
- `run_benchmark(collection, chunks, bm25)` — run full eval suite
- `score_faithfulness(response, context)` — grounding check
- `score_answer_relevancy(query, response)` — on-topic check
- `score_keyword_recall(response, expected_keywords)` — keyword coverage
- `score_context_relevance(query, retrieved)` — retrieval quality
- `save_results(results)` — write to benchmark_results.json
- `compare_runs(current, previous)` — before/after delta

---

## Refactoring Rules

### Non-negotiable
- **Change structure, not behavior** — the app must work identically after every step
- **One class at a time** — extract, verify, commit, then move to the next
- **Run the app after every class extraction** before proceeding
- **Never break the Streamlit entry point** — `streamlit run app.py` must always work
- **No circular imports** — config → loader → retriever → agent → benchmarker (one direction only)

### Code style
- Full docstrings on every class and public method (Google style)
- Type hints on all method signatures
- All constants in `config.py` — no magic numbers or strings inline
- Private methods prefixed with `_`
- `__all__` defined in each module

### Testing rules
- Use `pytest` with `unittest.mock`
- Mock `ollama.chat`, `ollama.embeddings`, and `chromadb.Client` — never call real models in tests
- Mock file I/O for document loader tests
- Each test file maps 1:1 to a source file
- Tests must be runnable with `pytest` from project root with no extra setup

---

## Step-by-Step Refactor Order

Work through these in order. Do NOT skip ahead.

```
Step 0: Create folder structure + pyproject.toml + empty __init__.py files
        → verify: project imports correctly

Step 1: Extract config.py
        → move all constants from rag_app.py top section
        → verify: nothing breaks (just constants moved)
        → commit: "refactor: extract config.py"

Step 2: Extract DocumentLoader
        → all chunkers + scan_all_files + ensure_folders + chunk_url
        → update rag_app.py to import and use DocumentLoader
        → verify: streamlit run app.py loads documents correctly
        → commit: "refactor: extract DocumentLoader class"

Step 3: Extract Retriever
        → build_or_load_chroma, build_bm25_index, embed, hybrid_retrieve,
          rerank, expand_query, classify_query, check_confidence, smart_top_n
        → verify: queries return correct results
        → commit: "refactor: extract Retriever class"

Step 4: Extract Agent
        → run_agent, all tool methods, prompt builder
        → verify: python main.py --agent works end-to-end
        → commit: "refactor: extract Agent class"

Step 5: Extract Benchmarker
        → all scoring functions, run_benchmark, save/compare
        → verify: python main.py --benchmark completes
        → commit: "refactor: extract Benchmarker class"

Step 6: Slim down app.py and main.py
        → both files should be <50 lines — import classes, wire together, run
        → verify: all three entry points work (terminal, agent, streamlit)
        → commit: "refactor: slim entry points"

Step 7: Write unit tests
        → one test file per class, mock all external calls
        → verify: pytest passes with no real model calls
        → commit: "test: add unit tests for all classes"

Step 8: Final check
        → run full app end-to-end
        → run pytest
        → verify README matches new structure
        → commit: "refactor: complete class-based restructure"
```

---

## Key Implementation Details

### Streamlit cache
The `@st.cache_resource` on `_initialize()` is critical.
Do NOT refactor in a way that causes re-initialization on every Streamlit rerun.
The cached object must hold `(collection, chunks, bm25)` across reruns.

### Chunk truncation
All chunks must be truncated to 300 words before embedding.
This is a hard requirement — the BGE model has a 512-token context limit.
The truncation must happen in `DocumentLoader.truncate_chunk()` and be called
from every chunker before the chunk is returned.

### Misplaced file detection
`scan_all_files()` detects files dropped into the wrong subfolder by checking
the file extension against the folder it was found in.
This logic must be preserved exactly — do not simplify it.

### BM25 index
The BM25 index is built in-memory from chunks and must be rebuilt whenever
new chunks are added (URL ingestion, file upload).
`Retriever` must expose a `rebuild_bm25(chunks)` method for this.

### Type-aware reranker
The reranker uses a different prompt depending on the document type of each chunk.
The prompt selection logic is in `rerank()` and must be preserved exactly.
See the `_rerank_prompt_for_type()` helper in the original code.

### URL ingestion
`DocumentLoader.chunk_url(url)` detects the content type by URL extension first,
then falls back to the HTTP `Content-Type` header.
Both detection paths must be preserved.

---

## Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/rag

# Terminal chatbot
python main.py

# Agent mode
python main.py --agent

# Benchmark
python main.py --benchmark

# Streamlit UI
streamlit run app.py
```

---

## What NOT to Do

- Do not change any retrieval logic, prompt text, or scoring formulas
- Do not rename public-facing functions that are called from Streamlit session state
- Do not remove the misplaced file detection
- Do not remove the chunk truncation step
- Do not call real Ollama models in tests
- Do not put business logic in `app.py` or `main.py`
- Do not merge multiple class extractions into a single commit
