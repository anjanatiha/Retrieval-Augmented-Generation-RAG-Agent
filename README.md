# RAG Agent — Retrieval-Augmented Generation System

A production-grade, fully local RAG chatbot and autonomous agent. Upload your documents, ask questions, and get grounded answers with source citations — all running **100% on your machine** with no API keys or cloud services required.

**[Try the live demo on Hugging Face →](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent)** — no installation needed.

![Hugging Face Demo](assets/huggingface_ragdoll.png)

---

## What it does

- **Chat with your documents** — ask questions about any PDF, Word doc, spreadsheet, presentation, CSV, Markdown, or HTML file
- **Works with structured data** — accurately retrieves from resumes, spreadsheets, and tables (where most RAG systems fail)
- **Agent mode** — autonomous ReAct agent with 5 tools: search, calculator, summarise, sentiment analysis, and finish
- **Multiple input methods** — drop files into a folder, upload via UI, or paste any public URL
- **Fully local** — LLaMA 3.2 and BGE embeddings run via Ollama — nothing leaves your machine

---

## Quick Start

> Already have Python 3.11 and Ollama installed? Run these 4 commands:

```bash
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent
pip install -r requirements.txt
streamlit run app.py
```

Don't have Ollama yet? Follow the [full installation guide](#installation) below.

---

## Try Without Installing

The fastest way to try the system is the **[Hugging Face Space](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent)**:

1. Open the link
2. Upload any supported file (PDF, DOCX, XLSX, PPTX, CSV, TXT, MD, HTML)
3. Ask a question

No Python, no Ollama, no setup required. Runs in your browser.

---

## How to Use

### Step 1 — Add your documents

Drop files into `./docs/` subfolders, or drop an entire folder (any depth, mixed file types):

```
docs/
  pdfs/      ← .pdf files
  txts/      ← .txt files
  docx/      ← .docx / .doc files
  xlsx/      ← .xlsx / .xls files
  pptx/      ← .pptx files
  csv/       ← .csv files
  md/        ← .md / .markdown files
  html/      ← .html files
```

> **Tip:** You can drop a folder with mixed file types anywhere under `./docs/` — the scanner walks recursively at any depth and detects every file by extension automatically.

### Step 2 — Choose a mode

**Web UI (recommended)**
```bash
streamlit run app.py
```

**Terminal chatbot**
```bash
python3 main.py          # chat mode
python3 main.py --agent  # agent mode with tool calling
```

**Benchmark evaluation**
```bash
python3 main.py --benchmark
```

### Step 3 — Ask questions

Example queries that work well:
- *"What is the candidate's most recent job title?"* — on a resume PDF
- *"What was the revenue in Q3?"* — on a spreadsheet
- *"Summarise the main points of this document"* — any format
- *"What is 15% of the salary mentioned in the resume?"* — agent mode with calculator
- *"What is the sentiment of the cover letter?"* — agent mode with sentiment tool

---

## Supported File Types

| Format | Extensions | Chunking strategy |
|--------|-----------|------------------|
| PDF | `.pdf` | Sentence-based per page (PyMuPDF) |
| Word | `.docx`, `.doc` | Paragraph groups + table rows |
| Spreadsheet | `.xlsx`, `.xls` | Row → key=value pairs |
| Presentation | `.pptx`, `.ppt` | Text shapes per slide |
| CSV | `.csv` | Row → key=value pairs |
| Plain text | `.txt` | Sliding window (line-based) |
| Markdown | `.md`, `.markdown` | Line-based, syntax stripped |
| HTML | `.html`, `.htm` | Sentence-based, tags stripped |

**Remote URLs** are also supported — paste any public URL in the UI and it is fetched, type-detected, and indexed automatically:

| URL type | Example |
|----------|---------|
| Webpage | `https://example.com/about` |
| Remote PDF | `https://example.com/report.pdf` |
| Remote DOCX | `https://example.com/resume.docx` |
| Remote XLSX | `https://example.com/data.xlsx` |
| Remote CSV | `https://example.com/data.csv` |
| Remote PPTX | `https://example.com/deck.pptx` |

---

## Agent Mode

In agent mode the system runs a ReAct loop — reasoning about which tool to use, calling it, observing the result, and deciding what to do next.

| Tool | What it does |
|------|-------------|
| `rag_search` | Searches your documents using the full retrieval pipeline |
| `calculator` | Evaluates safe arithmetic expressions |
| `summarise` | Summarises a passage with adaptive length |
| `sentiment` | Returns Sentiment, Tone, Key phrases, and Explanation |
| `finish` | Returns the final answer |

**Example agent queries:**
- *"Summarise the resume"*
- *"What is the sentiment of the introduction?"*
- *"What is 20% of the salary mentioned in the document?"*

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | Other versions not tested |
| Ollama | Latest | For running models locally |
| RAM | 8 GB+ | 16 GB recommended for smooth inference |

---

### Step 1 — Install Python 3.11

**macOS**
```bash
brew install python@3.11
python3.11 --version
```

**Windows**
Download Python 3.11 from [python.org](https://www.python.org/downloads/). Check **"Add Python to PATH"** during installation.

---

### Step 2 — Install Ollama

**macOS**
```bash
brew install ollama
```

**Windows**
Download and run the installer from [ollama.com/download](https://ollama.com/download).

---

### Step 3 — Clone and set up a virtual environment

**macOS**
```bash
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate
```

**Windows**
```cmd
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent
python -m venv rag_env_311
rag_env_311\Scripts\activate
```

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Pull the models

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
```

---

### Step 6 — Start Ollama

**macOS**
```bash
ollama serve
```

**Windows** — Ollama starts automatically. If not running, launch it from the Start menu or run `ollama serve`.

---

## Troubleshooting

**`context length` error on startup**
```bash
rm -rf ./chroma_db/
python3 main.py   # rebuilds from scratch
```

**`ModuleNotFoundError`**
```bash
pip install -r requirements.txt   # make sure your venv is activated
```

**Ollama not responding**
```bash
ollama serve   # start Ollama in a separate terminal
```

**No chunks found**
- Check that your files are under `./docs/` (any subfolder)
- Check the file extension is in the supported list above
- Unsupported extensions are silently skipped

**Model is slow**
- 8 GB RAM minimum; 16 GB recommended
- Close other applications to free memory
- The HF Space uses a cloud GPU — try it if local inference is too slow

---

## Architecture

The codebase uses **4 classes** and **4 modules**. Classes own state; modules own stateless functions. See [DESIGN.md](DESIGN.md) for the full rationale.

| Component | Responsibility |
|-----------|---------------|
| `DocumentLoader` | File scanning, URL fetching, chunker dispatch |
| `chunkers` module | 9 stateless format-specific chunker functions |
| `VectorStore` | ChromaDB, BM25, hybrid retrieval, reranking, response generation |
| `Agent` | ReAct loop and all 5 tools |
| `Benchmarker` | 4-metric evaluation and run comparison |

**Pipeline flow:**
```
Documents / URLs
      ↓
DocumentLoader  →  scan recursively, detect type, dispatch to chunker
      ↓
Truncation      →  300 words OR 1200 chars (whichever shorter)
      ↓
VectorStore     →  ChromaDB (dense) + BM25 (lexical) index
      ↓
Query pipeline  →  classify → expand → hybrid retrieve → confidence check → rerank → synthesize
      ↓
Response with source citations and conversation memory
```

**Models:**

| Role | Model |
|------|-------|
| Embeddings | `bge-base-en-v1.5` via Ollama |
| Language / Reranker | `Llama-3.2-3B-Instruct` via Ollama |

---

## Contributing

Contributions are welcome. Here is how to get started.

### Set up the development environment

```bash
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate     # Windows: rag_env_311\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"             # installs pytest, pytest-cov, pytest-mock
```

### Run the tests

```bash
pytest                        # all 566 local tests
pytest --cov=src              # with coverage report
pytest tests/test_agent.py    # one specific file
cd huggingface && pytest      # 262 HF Space tests
```

All tests must pass before submitting a pull request.

### Code structure

```
src/rag/
  config.py           ← all constants — edit here to change models, thresholds, chunk sizes
  chunkers.py         ← add a new file format here (one function per format)
  document_loader.py  ← ingestion orchestration and URL fetching
  vector_store.py     ← retrieval pipeline and response generation
  agent.py            ← ReAct loop and tools
  benchmarker.py      ← evaluation metrics
src/ui/
  handlers.py         ← Streamlit event handlers
  theme.py            ← CSS and style constants
  session.py          ← session state helpers
src/cli/
  runner.py           ← terminal chat, agent, and benchmark entry points
```

### How to add a new file format

1. Add the file extension to `EXT_TO_TYPE` in `src/rag/config.py`
2. Add a folder entry to `DOC_FOLDERS` in `src/rag/config.py`
3. Write a `chunk_yourformat(filepath, filename)` function in `src/rag/chunkers.py`
4. Add the routing case to `_dispatch_chunker()` in `src/rag/document_loader.py`
5. Write tests in `tests/test_document_loader.py` (unit) and `tests/test_integration_loader.py` (integration)

### How to add a new agent tool

1. Add a `_tool_yourname(self, arg)` private method to `Agent` in `src/rag/agent.py`
2. Add the tool name to `AGENT_SYSTEM_PROMPT` in the same file
3. Add the routing case to `_dispatch_tool()` in `src/rag/agent.py`
4. Write tests in `tests/test_agent.py`

### Pull request guidelines

- One focused change per PR — don't mix features and refactors
- All tests must pass: `pytest` green before opening a PR
- Follow the existing code style: plain English names, docstrings on every public method, type hints on all signatures
- No new packages in `requirements.txt` unless genuinely necessary — add dev-only packages to `pyproject.toml`
- See [DESIGN.md](DESIGN.md) before making architectural changes

---

## Benchmarking

```bash
python3 main.py --benchmark
```

Results are saved to `benchmark_results.json` with run-over-run comparison.

**Current scores (cat-facts.txt test set):**

| Metric | Score |
|--------|-------|
| Faithfulness | 0.798 |
| Answer Relevancy | 0.369 |
| Keyword Recall | 1.000 |
| Context Relevance | 0.719 |
| **Overall** | **0.721** |

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | How grounded the response is in retrieved context |
| **Answer Relevancy** | How well the response addresses the question |
| **Keyword Recall** | Whether expected keywords appear in the response |
| **Context Relevance** | Average similarity of retrieved chunks to the query |

---

## Streamlit UI

**Chat view:**

![Streamlit Chat](assets/streamlit_rag_before.png)

**Pipeline panel** — post-query sidebar with retrieved chunks, confidence scores, and session stats:

![Streamlit Pipeline](assets/streamlit_rag_after.png)

UI features:
- Chat and Agent mode toggle
- URL ingestion panel — paste any public URL to index it
- File upload panel — upload one or multiple files at once (Ctrl+A to select a whole folder)
- Step-by-step progress bar — classify → retrieve → rerank → generate
- Live pipeline sidebar with pre/post rerank chunks and similarity scores
- Confidence and query-type badges
- Document type breakdown and session stats
- Clear button to reset conversation

---

## Built With

`Python 3.11` · `Ollama` · `ChromaDB` · `rank-bm25` · `PyMuPDF` · `python-docx` · `openpyxl` · `xlrd` · `python-pptx` · `BeautifulSoup4` · `lxml` · `requests` · `Streamlit` · `LLaMA 3.2` · `BGE Embeddings`

---

## Related

- **[Live demo on Hugging Face](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent)** — try it in your browser, no setup needed
- **[DESIGN.md](DESIGN.md)** — architectural decisions, class ownership, tradeoffs, and production scaling path
- **[Based on](https://huggingface.co/blog/ngxson/make-your-own-rag)** — significantly extended with hybrid search, type-aware reranking, 9 format support, agent mode, benchmarking, persistent vector DB, and Streamlit UI
