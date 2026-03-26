# Architect Mode Prompt — RAG Agent Refactor

Paste everything below the line into Claude Code after running `/architect`

---

I have a working 1600-line Python RAG application in `rag_app.py`.
I want to refactor it into a clean, production-standard Python package
with proper class encapsulation, separated files, and full unit tests.

Read `CLAUDE.md` first — it contains the full architecture plan,
class responsibilities, refactor rules, and step-by-step order.
Follow it exactly.

## Project structure to create

```
rag/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py
│       ├── document_loader.py
│       ├── retriever.py
│       ├── agent.py
│       └── benchmarker.py
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_retriever.py
│   ├── test_agent.py
│   └── test_benchmarker.py
├── app.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Classes to extract from rag_app.py

### 1. `config.py`
Move all top-level constants:
- `EMBEDDING_MODEL`, `LANGUAGE_MODEL`
- `DOCS_ROOT`, `DOC_FOLDERS`, `EXT_TO_TYPE`
- `CHROMA_DIR`, `CHROMA_COLLECTION`, `LOG_FILE`, `BENCHMARK_FILE`
- `SIMILARITY_THRESHOLD`, `TOP_RETRIEVE`, `TOP_RERANK`
- All chunk size constants (`TXT_CHUNK_SIZE`, `PDF_CHUNK_SENTENCES`, etc.)

### 2. `DocumentLoader` class
Extract these functions as methods:
- `ensure_folders()` → `self.ensure_folders()`
- `scan_all_files()` → `self.scan_all_files()`
- `chunk_all_documents()` → `self.chunk_all_documents()`
- `_chunk_txt()`, `_chunk_pdf()`, `_chunk_docx()`, `_chunk_xlsx()`
- `_chunk_pptx()`, `_chunk_csv()`, `_chunk_md()`, `_chunk_html()`
- `chunk_url(url)` → `self.chunk_url(url)`
- Truncation logic → `self.truncate_chunk(text)`

### 3. `Retriever` class
Extract these functions as methods:
- `build_or_load_chroma(chunks)` → `self.build_or_load_chroma(chunks)`
- `build_bm25_index(chunks)` → `self.build_bm25_index(chunks)`
- `embed(text)` → `self.embed(text)`
- `hybrid_retrieve(queries, collection, chunks, bm25, top_n)` → `self.hybrid_retrieve(...)`
- `rerank(query, candidates, top_n)` → `self.rerank(...)`
- `expand_query(query)` → `self.expand_query(query)`
- `classify_query(query)` → `self.classify_query(query)`
- `check_confidence(results)` → `self.check_confidence(results)`
- `smart_top_n(query_type)` → `self.smart_top_n(query_type)`
- `log_interaction(...)` → `self.log_interaction(...)`

### 4. `Agent` class
Extract these functions as methods:
- `run_agent(task, collection, chunks, bm25)` → `self.run(task)`
- `_parse_tool_call(text)` → `self._parse_tool_call(text)`
- All `_tool_*` methods
- `_build_system_prompt()` → `self._build_system_prompt()`
Constructor takes `retriever: Retriever` as dependency.

### 5. `Benchmarker` class
Extract these functions as methods:
- `run_benchmark(collection, chunks, bm25)` → `self.run()`
- All `score_*` functions → `self.score_*()`
- `_load_previous_results()`, `_save_results()`
Constructor takes `retriever: Retriever` as dependency.

## Entry points after refactor

### `main.py` (~40 lines max)
```python
from src.rag.config import Config
from src.rag.document_loader import DocumentLoader
from src.rag.retriever import Retriever
from src.rag.agent import Agent
from src.rag.benchmarker import Benchmarker
import argparse

def main():
    loader = DocumentLoader()
    loader.ensure_folders()
    chunks = loader.chunk_all_documents()

    retriever = Retriever()
    collection = retriever.build_or_load_chroma(chunks)
    bm25 = retriever.build_bm25_index(chunks)

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--agent', action='store_true')
    args = parser.parse_args()

    if args.benchmark:
        Benchmarker(retriever, collection, chunks, bm25).run()
    elif args.agent:
        agent = Agent(retriever, collection, chunks, bm25)
        # run terminal agent loop
    else:
        # run terminal chat loop

if __name__ == '__main__':
    main()
```

### `app.py` (~50 lines max)
- Import classes, wire with `@st.cache_resource`
- All Streamlit UI logic stays here (run_streamlit function)
- No business logic — only UI rendering and calling class methods

## Unit tests to write

### `tests/test_document_loader.py`
```python
# Mock file system and test:
- test_chunk_txt_basic()
- test_chunk_txt_empty_file()
- test_chunk_pdf_returns_chunks()
- test_chunk_xlsx_key_value_format()
- test_scan_detects_misplaced_file()
- test_truncate_chunk_enforces_300_words()
- test_chunk_url_html_content()
- test_chunk_url_unknown_type_raises()
```

### `tests/test_retriever.py`
```python
# Mock ollama.embeddings and chromadb:
- test_embed_returns_vector()
- test_classify_query_factual()
- test_classify_query_comparison()
- test_expand_query_returns_list()
- test_check_confidence_above_threshold()
- test_check_confidence_below_threshold()
- test_hybrid_retrieve_merges_results()
- test_rerank_returns_top_n()
- test_smart_top_n_factual()
```

### `tests/test_agent.py`
```python
# Mock Retriever and ollama.chat:
- test_parse_tool_call_rag_search()
- test_parse_tool_call_calculator()
- test_parse_tool_call_finish()
- test_tool_calculator_basic_arithmetic()
- test_tool_calculator_rejects_unsafe_input()
- test_run_agent_calls_finish()
- test_run_agent_step_limit()
- test_build_system_prompt_contains_tools()
```

### `tests/test_benchmarker.py`
```python
# Mock Retriever and ollama.chat:
- test_score_faithfulness_high()
- test_score_faithfulness_low()
- test_score_keyword_recall_full_match()
- test_score_keyword_recall_partial()
- test_score_answer_relevancy()
- test_save_and_load_results()
- test_compare_runs_shows_delta()
```

## pyproject.toml to create

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "rag-agent"
version = "1.0.0"
description = "A fully local, production-grade RAG system"
requires-python = ">=3.11"
dependencies = [
    "ollama",
    "rank-bm25",
    "chromadb",
    "streamlit",
    "requests",
    "pymupdf",
    "python-docx",
    "openpyxl",
    "xlrd",
    "python-pptx",
    "beautifulsoup4",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "unittest-mock"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.setuptools.packages.find]
where = ["src"]
```

## Absolute rules — do not violate

1. Change structure, not behavior — every step must leave the app working
2. One class per commit — do not combine extractions
3. Run `streamlit run app.py` after steps 2, 3, 6 to confirm UI works
4. Run `python main.py` after every step
5. Run `pytest` after step 7 — all tests must pass before final commit
6. Never call real Ollama models in tests — always mock
7. Keep the `@st.cache_resource` pattern on initialization in `app.py`
8. Keep chunk truncation (300 words) in every chunker path
9. Keep misplaced file detection logic intact
10. Keep type-aware reranker prompts exactly as written

## Start here

Begin with Step 0: create the folder structure, empty `__init__.py` files,
and `pyproject.toml`. Then read `CLAUDE.md` and confirm you understand
the full plan before touching any logic in `rag_app.py`.
