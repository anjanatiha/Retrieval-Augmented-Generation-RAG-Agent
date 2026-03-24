# Retrieval-Augmented Generation (RAG) Agent

A production-grade, fully local RAG chatbot and agent built incrementally with an advanced retrieval pipeline. Supports **PDF, TXT, DOCX, XLSX, PPTX, CSV, Markdown, and HTML documents**, including structured/tabular formats like resumes, with hybrid search, LLM reranking, agentic tool calling, benchmarking, and a Streamlit UI — all running **100% on-device** with no API keys required.

**Live demo:** [ragdoll on Hugging Face Spaces](https://huggingface.co/spaces/anjanatiha2024/ragdoll) — upload any document and try it in your browser, no setup needed.

![Hugging Face Demo](assets/huggingface_ragdoll.png)

---

## Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Sliding window chunking** | Configurable chunk size + overlap for TXT/MD; sentence-based chunking for PDF/HTML pages; paragraph-based for DOCX; row-based for XLSX/CSV; slide-based for PPTX |
| 2 | **Multi-format document support** | PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, Markdown, HTML — each with a dedicated chunker and type-aware metadata |
| 3 | **Structured document retrieval** | Accurately retrieves from tabular/structured documents like resumes, spreadsheets, and tables — a known hard problem for basic RAG systems |
| 4 | **Smart misplaced file detection** | Files dropped into the wrong subfolder are auto-detected by extension, processed correctly, and flagged with a `[MISPLACED]` notice |
| 5 | **Chunk truncation** | Chunks automatically truncated to 300 words before embedding to stay within the `bge-base-en-v1.5` 512 token context limit |
| 6 | **Persistent vector DB** | ChromaDB with cosine similarity; embeddings survive restarts — no re-embedding on reload |
| 7 | **Hybrid search** | BM25 (lexical) + dense vector (semantic) retrieval combined for higher recall |
| 8 | **Query expansion** | LLM-generated query rewrites using synonyms and acronyms to improve retrieval coverage |
| 9 | **Document type-aware LLM reranker** | Secondary LLM pass scores retrieved chunks using prompts tailored per document type — spreadsheet rows, slides, PDF pages, DOCX paragraphs, webpages and Markdown sections each get type-specific relevance framing for more accurate scoring |
| 10 | **Query classification** | Auto-classifies queries as factual / comparison / general and adjusts retrieval depth |
| 11 | **Confidence / hallucination filter** | Similarity threshold check flags low-confidence answers before they reach the user |
| 12 | **Source citation** | Every answer cites its source document with type-aware location labels (line, page, row, slide) |
| 13 | **Conversation memory** | Full multi-turn memory across the session |
| 14 | **Logging & analytics** | Every query logged to `rag_logs.json` with similarity scores, query type, and response length |
| 15 | **Streaming with typing indicator** | Token-level streaming with animated typing indicator in terminal |
| 16 | **Benchmarking** | Automated eval suite with faithfulness, answer relevancy, keyword recall, and context relevance scores — with before/after run comparison |
| 17 | **Agent with tool calling** | Agentic mode with `rag_search`, `calculator`, `summarise`, and `finish` tools; robust tool-call parsing and auto-finish logic |
| 18 | **Streamlit UI** | Ocean Blue web UI with native chat bubbles, agent mode toggle, URL ingestion panel, file upload panel, live pipeline sidebar (pre/post rerank chunks, confidence badges, document type breakdown, session stats) |
| 19 | **URL ingestion** | Paste any public URL — webpage, PDF, DOCX, XLSX, CSV, PPTX — and it is fetched, auto-detected by type, chunked through the correct chunker, and added to the index alongside local files |
| 20 | **File upload ingestion** | Upload any supported file directly through the UI — chunked, embedded, and added to the live knowledge base without restarting |
| 21 | **Step-by-step progress bar** | Real-time progress bar showing each retrieval stage: classify → retrieve → rerank → generate |
| 22 | **Native chat interface** | Messages rendered with `st.chat_message` bubbles and persistent `st.chat_input` at the bottom; clear button appears below conversation |

---

## Structured Document & Multi-Format Retrieval

One of the most technically challenging aspects of this system is **accurate retrieval from structured and tabular documents** — a problem where standard RAG pipelines typically fail.

Most basic RAG systems flatten all content into plain text, destroying the relational structure of tables, resumes, spreadsheets, and multi-column layouts in the process. This system addresses that through:

- **Page-level isolation** — PDF and HTML text is extracted page/section by section, preserving document structure
- **Sentence-aware chunking** — Pages are split into sentence-based windows rather than fixed character counts, keeping semantic units intact
- **Row-level chunking for spreadsheets** — Each XLSX/CSV row becomes a `key=value` pair chunk, preserving column context across retrieval
- **Slide-level chunking for presentations** — Each PPTX slide's text shapes are extracted and chunked together
- **Markdown stripping** — MD files are cleaned of syntax markers before chunking for cleaner embeddings
- **Chunk truncation** — All chunks are truncated to 300 words before embedding, preventing `input length exceeds context length` errors on dense documents
- **Source + location metadata** — Every chunk stores its source filename and a type-aware location label (e.g. `[resume.pdf p2]`, `[data.xlsx row14]`, `[deck.pptx slide3]`)
- **Document type-aware reranking** — the reranker uses a different prompt per document type. Spreadsheet rows (`key=value` pairs), slide bullets, PDF page extracts, and webpage text each get framing suited to their structure — a generic prompt underscores structured data because it reads as data rather than natural language

This makes the system capable of accurately answering queries like *"What is the candidate's GPA?"* or *"What was Q3 revenue in the spreadsheet?"* — queries that would produce incorrect or hallucinated answers in a naive RAG setup.

---

## Architecture

```
Documents (PDF / TXT / DOCX / XLSX / PPTX / CSV / MD / HTML)
+ URLs  (any public webpage or document link)
        │
        ▼
  Smart File Scanner + URL Fetcher
  ├── Scans all subfolders under ./docs/
  ├── Detects real file type by extension (not by folder)
  ├── Flags misplaced files with [MISPLACED] notice; processes anyway
  └── URL fetcher detects type by extension or Content-Type header
        │
        ▼
  Chunking Layer (type-aware dispatch)
  ├── TXT / MD:  sliding window (line-based; MD syntax stripped)
  ├── PDF:       page extraction → sentence-based chunks (PyMuPDF)
  ├── DOCX:      paragraph-based chunks (python-docx)
  ├── XLSX/XLS:  row → key=value pair chunks (openpyxl / xlrd)
  ├── CSV:       row → key=value pair chunks (stdlib csv)
  ├── PPTX:      slide text shape extraction (python-pptx)
  └── HTML/URL:  tag-stripped → sentence-based chunks (BeautifulSoup)
        │
        ▼
  Truncation (300 words max per chunk → within 512 token limit)
        │
        ▼
  ChromaDB (persistent vector store)
  + BM25 index (in-memory lexical index)
        │
        ▼
  Query Pipeline
  ├── Query classification (factual / comparison / general)
  ├── Query expansion (LLM-generated rewrites)
  ├── Hybrid retrieval (BM25 + dense vector, top-N)
  ├── Confidence filter (similarity threshold)
  └── Document type-aware LLM reranker (top-K)
        │
        ▼
  Response Generation
  ├── Context injection with type-aware source citations
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
├── docs/                   ← root documents folder (auto-created, git-ignored)
│   ├── pdfs/               ← drop .pdf files here
│   ├── txts/               ← drop .txt files here
│   ├── docx/               ← drop .docx / .doc files here
│   ├── xlsx/               ← drop .xlsx / .xls files here
│   ├── pptx/               ← drop .pptx / .ppt files here
│   ├── csv/                ← drop .csv files here
│   ├── md/                 ← drop .md / .markdown files here
│   └── html/               ← drop .html / .htm files here
├── chroma_db/              ← persistent vector store (auto-created, git-ignored)
├── rag_app.py              ← main application
├── requirements.txt        ← Python dependencies
├── .gitignore              ← excludes env, chroma_db, and docs from git
├── rag_logs.json           ← interaction logs (auto-generated)
└── benchmark_results.json  ← benchmark history (auto-generated)
```

> **Tip:** All subfolders are created automatically on first run. You can drop a file into any subfolder — the smart file scanner will detect the correct type by extension and process it accordingly, printing a `[MISPLACED]` notice if the folder doesn't match.

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
pip install -r requirements.txt
```

### Step 4 — Pull Models via Ollama

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
```

### Step 5 — Start Ollama

Make sure Ollama is running before starting the app:

```bash
ollama serve
```

> **Note:** The embedding model (`bge-base-en-v1.5`) has a 512 token context limit. Chunks are automatically truncated to 300 words before embedding. If you still encounter a context length error, delete `./chroma_db/` and rerun.

---

## Usage

Drop your documents into the appropriate subfolder under `./docs/`, then choose a run mode:

```bash
# Standard chatbot (terminal)
python3 rag_app.py

# Agent mode — uses tool calling (terminal)
python3 rag_app.py --agent

# Benchmark evaluation
python3 rag_app.py --benchmark

# Streamlit web UI (chat + agent toggle)
streamlit run rag_app.py
```

---

## Supported File Types

### Local files — drop into `./docs/` subfolders

| Extension(s) | Type | Chunking Strategy | Library |
|---|---|---|---|
| `.pdf` | PDF | Sentence-based per page | `pymupdf` |
| `.txt` | Plain text | Sliding window (line-based) | stdlib |
| `.docx`, `.doc` | Word document | Paragraph groups | `python-docx` |
| `.xlsx`, `.xls` | Spreadsheet | Row → key=value pairs | `openpyxl`, `xlrd` |
| `.csv` | CSV | Row → key=value pairs | stdlib |
| `.pptx`, `.ppt` | Presentation | Text shapes per slide | `python-pptx` |
| `.md`, `.markdown` | Markdown | Line-based (syntax stripped) | stdlib |
| `.html`, `.htm` | HTML | Sentence-based (tags stripped) | `beautifulsoup4` |

### Remote URLs — paste in the Streamlit UI

Any public URL is also supported. Type is detected by file extension in the URL first, then Content-Type response header:

| URL type | Example | Handled by |
|---|---|---|
| Webpage | `https://example.com/about` | BeautifulSoup HTML chunker |
| Remote PDF | `https://example.com/report.pdf` | PyMuPDF chunker |
| Remote DOCX | `https://example.com/resume.docx` | python-docx chunker |
| Remote XLSX | `https://example.com/data.xlsx` | openpyxl chunker |
| Remote CSV | `https://example.com/data.csv` | csv chunker |
| Remote PPTX | `https://example.com/deck.pptx` | python-pptx chunker |

Source label uses the URL hostname + path (e.g. `[careers.company.com/job L1]`).

**Example:** load a resume PDF locally + paste a job description URL → ask *"Does the candidate meet the requirements for this role?"*

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

**Chat view** — native chat bubbles, file upload, URL ingestion, and clear button:

![Streamlit Chat](assets/streamlit_rag_before.png)

**Pipeline panel** — post-query sidebar showing reranked chunks, confidence scores, and session stats:

![Streamlit Pipeline](assets/streamlit_rag_after.png)

The web UI features:
- Ocean Blue theme — white background with deep navy and light blue accents
- Native `st.chat_message` bubbles with user / assistant / agent avatars
- Persistent `st.chat_input` bar always visible at the bottom of the page
- Chat and Agent mode toggle
- **URL ingestion panel** — paste any public URL to fetch and index it alongside local files
- **File upload panel** — upload PDF, TXT, DOCX, XLSX, PPTX, CSV, MD, or HTML directly through the UI
- **Step-by-step progress bar** — shows classify → retrieve → rerank → generate stages in real time
- **🗑 Clear button** — appears below conversation to wipe chat history and memory
- Live pipeline sidebar showing pre/post rerank chunks with similarity scores
- Confidence and query-type badges
- Document type breakdown (chunk counts per type: PDF, DOCX, XLSX, etc.)
- Session stats (query count, conversation turns, total chunk count, URL chunk count)

---

## Built With

`Python` · `Ollama` · `ChromaDB` · `rank_bm25` · `PyMuPDF` · `python-docx` · `openpyxl` · `python-pptx` · `BeautifulSoup4` · `requests` · `Streamlit` · `LLaMA 3.2` · `BGE Embeddings`

---

## Based On

Started from: https://huggingface.co/blog/ngxson/make-your-own-rag  
Significantly enhanced with: hybrid search, document type-aware LLM reranking, multi-format document support (PDF, DOCX, XLSX, PPTX, CSV, MD, HTML), URL ingestion, smart misplaced file detection, agent tool calling, benchmarking, persistent vector DB, conversation memory, and Streamlit UI.

---

## Related

- **ragdoll** — refactored version of this codebase using `DocumentLoader`, `Retriever`, and `Agent` classes
- **HF Spaces demo** — Gradio-based version running on Hugging Face with file upload UI (no Ollama required)
