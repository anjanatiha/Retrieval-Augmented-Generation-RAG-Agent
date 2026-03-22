# Retrieval-Augmented Generation (RAG) Agent

A production-grade, fully local RAG chatbot and agent built incrementally with an advanced retrieval pipeline. Supports **PDF and TXT documents**, including structured/tabular formats like resumes, with hybrid search, LLM reranking, agentic tool calling, benchmarking, and a Streamlit UI — all running **100% on-device** with no API keys required.

---

## Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Sliding window chunking** | Configurable chunk size + overlap for TXT; sentence-based chunking for PDF pages |
| 2 | **PDF support** | Extracts and chunks text page-by-page from `.pdf` files via PyMuPDF |
| 3 | **Structured document retrieval** | Accurately retrieves from tabular/structured documents like resumes and tables — a known hard problem for basic RAG systems |
| 4 | **Persistent vector DB** | ChromaDB with cosine similarity; embeddings survive restarts — no re-embedding on reload |
| 5 | **Hybrid search** | BM25 (lexical) + dense vector (semantic) retrieval combined for higher recall |
| 6 | **Query expansion** | Multi-query support to improve retrieval coverage |
| 7 | **LLM reranker** | Secondary LLM pass scores and reranks retrieved chunks by relevance (1–10 scale) |
| 8 | **Query classification** | Auto-classifies queries as factual / comparison / general and adjusts retrieval depth |
| 9 | **Confidence / hallucination filter** | Similarity threshold check flags low-confidence answers before they reach the user |
| 10 | **Source citation** | Every answer cites its source document and line/page number |
| 11 | **Conversation memory** | Full multi-turn memory across the session |
| 12 | **Logging & analytics** | Every query logged to `rag_logs.json` with similarity scores, query type, and response length |
| 13 | **Streaming with typing indicator** | Token-level streaming with animated typing indicator in terminal |
| 14 | **Benchmarking** | Automated eval suite with faithfulness, answer relevancy, keyword recall, and context relevance scores — with before/after run comparison |
| 15 | **Agent with tool calling** | Agentic mode with `rag_search`, `calculator`, `summarise`, and `finish` tools; robust tool-call parsing and auto-finish logic |
| 16 | **Streamlit UI** | Dark-themed web UI with chat + agent mode toggle, live pipeline sidebar (pre/post rerank chunks, confidence badges, session stats) |

---

## Structured Document & PDF Retrieval

One of the most technically challenging aspects of this system is **accurate retrieval from structured and tabular documents** — a problem where standard RAG pipelines typically fail.

Most basic RAG systems flatten all content into plain text, destroying the relational structure of tables, resumes, and multi-column layouts in the process. This system addresses that through:

- **Page-level isolation** — PDF text is extracted page by page, preserving document structure
- **Sentence-aware chunking** — Pages are split into sentence-based windows rather than fixed character counts, keeping semantic units intact
- **Source + page metadata** — Every chunk stores its source filename and page number, enabling precise citation (e.g. `[resume.pdf p2]`)
- **LLM reranking** — A secondary LLM pass re-scores retrieved chunks by relevance to the query, correcting for embedding-level misses that commonly occur in structured content

This makes the system capable of accurately answering queries like *"What is the candidate's GPA?"* or *"Where did they work in 2021?"* from a resume PDF — queries that would produce incorrect or hallucinated answers in a naive RAG setup.

---

## Architecture

```
Documents (PDF / TXT)
        │
        ▼
  Chunking Layer
  ├── TXT: sliding window (line-based)
  └── PDF: page extraction → sentence-based chunks (PyMuPDF)
        │
        ▼
  ChromaDB (persistent vector store)
  + BM25 index (in-memory lexical index)
        │
        ▼
  Query Pipeline
  ├── Query classification (factual / comparison / general)
  ├── Query expansion (multi-query)
  ├── Hybrid retrieval (BM25 + dense vector, top-N)
  ├── Confidence filter (similarity threshold)
  └── LLM reranker (top-K)
        │
        ▼
  Response Generation
  ├── Context injection with source citations
  ├── Conversation memory (multi-turn)
  ├── Streaming output
  └── Logging
```

---

## Models Used

| Role | Model |
|------|-------|
| Embeddings | `bge-base-en-v1.5` (via Ollama) |
| Language / Reranker | `Llama-3.2-3B-Instruct` (via Ollama) |

All models run **locally via Ollama** — no internet connection or API key needed after setup.

---

## Folder Structure

```
project/
├── docs/               ← drop .txt files here (auto-created)
├── pdfs/               ← drop .pdf files here (auto-created)
├── chroma_db/          ← persistent vector store (auto-created)
├── rag_app11.py        ← main application
├── rag_logs.json       ← interaction logs (auto-generated)
└── benchmark_results.json  ← benchmark history (auto-generated)
```

---

## Installation

### Step 1 — Install Python 3.11

```bash
brew install python@3.11
python3.11 --version
```

### Step 2 — Create Virtual Environment

```bash
cd ~/Desktop/rag
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install ollama rank_bm25 streamlit chromadb pymupdf
```

### Step 4 — Pull Models via Ollama

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
```

---

## Usage

Drop your `.txt` files into `./docs/` and/or `.pdf` files into `./pdfs/`, then choose a run mode:

```bash
# Standard chatbot (terminal)
python3 rag_app11.py

# Agent mode — uses tool calling (terminal)
python3 rag_app11.py --agent

# Benchmark evaluation
python3 rag_app11.py --benchmark

# Streamlit web UI (chat + agent toggle)
streamlit run rag_app11.py
```

---

## Agent Mode

In agent mode the system acts as an autonomous agent with access to tools:

| Tool | Description |
|------|-------------|
| `rag_search` | Searches the knowledge base using the full retrieval pipeline |
| `calculator` | Evaluates safe arithmetic expressions |
| `summarise` | Summarises a passage using the LLM |
| `finish` | Returns the final answer to the user |

The agent runs a ReAct-style loop — calling tools, observing results, and deciding next steps — up to a configurable step limit. Includes robust format recovery and auto-finish logic to handle small model limitations.

---

## Benchmarking

The built-in benchmark evaluates retrieval and generation quality across four metrics:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | How much of the response is grounded in retrieved context |
| **Answer Relevancy** | How well the response addresses the question |
| **Keyword Recall** | Whether expected answer keywords are present |
| **Context Relevance** | Average similarity of retrieved chunks to the query |

Results are saved to `benchmark_results.json` with run-over-run comparison so you can track improvements as the pipeline evolves.

---

## Streamlit UI

The web UI features:
- Dark terminal-aesthetic theme
- Chat and Agent mode toggle
- Live pipeline sidebar showing pre/post rerank chunks with similarity scores
- Confidence and query-type badges
- Session stats (query count, conversation turns, chunk count)
- Clear chat button

---

## Built With

`Python` · `Ollama` · `ChromaDB` · `rank_bm25` · `PyMuPDF` · `Streamlit` · `LLaMA 3.2` · `BGE Embeddings`

---

## Based On

Started from: https://huggingface.co/blog/ngxson/make-your-own-rag  
Significantly enhanced with: hybrid search, LLM reranking, PDF/structured document support, agent tool calling, benchmarking, persistent vector DB, conversation memory, and Streamlit UI.
