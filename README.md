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

## Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **9 document formats** | PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, Markdown, HTML — each with a dedicated format-specific chunker |
| 2 | **Recursive folder scan** | Drop a folder anywhere under `./docs/` at any depth — every file is detected by extension and indexed automatically |
| 3 | **Smart misplaced file detection** | Files not in their canonical subfolder are still processed and flagged with a `[MISPLACED]` notice |
| 4 | **Structured document retrieval** | Row-level XLSX/CSV chunking and type-aware reranking make queries over spreadsheets and resumes accurate |
| 5 | **Hybrid search** | BM25 (lexical) + dense vector (semantic) retrieval fused for higher recall than either alone |
| 6 | **Query expansion** | LLM generates 2 rewrites of the query + the original = 3 queries run in parallel for better coverage |
| 7 | **Query classification** | Queries auto-classified as summarise / comparison / factual / general — retrieval depth adjusts per type |
| 8 | **Type-aware LLM reranker** | 7 different reranker prompts — one per document type — for more accurate relevance scoring on structured data |
| 9 | **Confidence / hallucination filter** | Similarity threshold check and pivot-phrase filter prevent low-confidence and hallucinated answers |
| 10 | **Source citations** | Every answer cites the source file with a type-aware location label (page, row, slide, line) |
| 11 | **Persistent vector DB** | ChromaDB stores embeddings on disk — no re-embedding on restart |
| 12 | **Conversation memory** | Full multi-turn memory across the session |
| 13 | **URL ingestion** | Paste any public URL — webpage, PDF, DOCX, XLSX, CSV, PPTX — auto-detected and indexed |
| 14 | **Multi-file upload** | Upload one file or many at once; select all files from a folder with Ctrl+A |
| 15 | **Agent with tool calling** | ReAct loop with `rag_search`, `calculator`, `summarise`, `sentiment`, and `finish` tools |
| 16 | **Benchmarking** | 4-metric automated evaluation suite with run-over-run comparison |
| 17 | **Logging & analytics** | Every query logged to `rag_logs.json` with similarity scores, query type, and response length |
| 18 | **Streamlit UI** | Ocean Blue web UI with chat bubbles, agent mode toggle, live pipeline sidebar, confidence badges |
| 19 | **HF Space deployment** | Same system deployed on Hugging Face using InferenceClient — no Ollama required |
| 20 | **Progress bar** | Real-time classify → retrieve → rerank → generate progress bar in the UI |

---

## How RAG Works

RAG stands for **Retrieval-Augmented Generation**. It solves a fundamental problem with large language models: they hallucinate answers when they don't know something, because they can only draw on what they learned during training.

RAG fixes this by giving the model a **reference library** — your documents — and forcing it to look things up before answering:

```
Without RAG:  Question → LLM → Answer (may be hallucinated)
With RAG:     Question → Search documents → Find relevant passages
                      → Feed passages to LLM → Grounded answer with citations
```

**The three steps:**

1. **Index** — Your documents are split into chunks, converted to numbers (embeddings), and stored in a database. This happens once at startup.

2. **Retrieve** — When you ask a question, the system finds the most relevant chunks from your documents using both keyword search (BM25) and semantic search (vector similarity).

3. **Generate** — The retrieved chunks are given to the LLM as context. The LLM reads them and writes an answer — grounded in your documents, not in its training data.

**Why this system goes further than basic RAG:**

Most RAG systems stop at step 2 — they retrieve and generate. This system adds:
- **Query expansion** — searches with 3 versions of your question for better recall
- **Query classification** — adjusts how many chunks to retrieve based on query type
- **Type-aware reranking** — a second LLM pass re-scores chunks with prompts tailored per document type
- **Confidence check** — skips the LLM entirely if no relevant chunks are found
- **Hallucination filter** — catches and truncates responses where the model starts fabricating

---

## How It Works — Algorithms

This section explains every algorithm in the retrieval pipeline, from chunking to the final answer.

### 1. Chunking

Before any search can happen, documents are split into chunks — small pieces of text that can be embedded and retrieved individually.

Each format uses a strategy suited to its structure:

| Format | Strategy | Why |
|--------|----------|-----|
| TXT / MD | 1 line per chunk | Each line is typically one fact or sentence |
| PDF | 5-sentence sliding window per page | Preserves sentence context; page boundaries prevent cross-page noise |
| DOCX | Groups of 3 paragraphs + table rows | Keeps related paragraphs together; table rows extracted as `key=value` pairs |
| XLSX / XLS | 1 row per chunk as `col=value \| col=value` | Preserves column context for structured queries |
| CSV | 1 row per chunk as `col=value \| col=value` | Same as XLSX — column headers give each value meaning |
| PPTX | 1 slide per chunk | Each slide is a self-contained idea |
| HTML | 5-sentence sliding window | Tags stripped; boilerplate filtered; sentence windows preserve flow |

All chunks are **truncated to 300 words OR 1200 characters** (whichever is shorter) before embedding — this prevents the BGE model's 512-token context limit from being exceeded on dense text.

---

### 2. Embedding

Each chunk is converted into a **768-dimensional vector** using `bge-base-en-v1.5`, a BERT-based bi-encoder trained on retrieval tasks. Similar text produces vectors that are close together in vector space.

```
"Cats sleep 16 hours a day" → [0.12, -0.34, 0.89, ..., 0.05]  # 768 numbers
```

Vectors are stored in **ChromaDB** with cosine similarity as the distance metric. Embeddings are computed in batches of 50 and persist on disk — no re-embedding on restart unless the document set changes.

---

### 3. Query Expansion

Instead of running one query, the system runs **3 queries**:

```
Original:  "How many hours do cats sleep?"
Rewrite 1: "What is the daily sleep duration of cats?"
Rewrite 2: "Cat sleeping habits hours per day"
```

The LLM generates 2 rewrites using synonyms and alternative phrasings. All 3 queries are run through the full retrieval pipeline and results are merged — the best score for each chunk across all 3 queries is kept.

**Why:** A single query can miss relevant chunks that use different vocabulary. Query expansion increases recall without sacrificing precision.

---

### 4. Query Classification

Before retrieval, the query is classified into one of 4 types:

| Type | Example | Effect |
|------|---------|--------|
| `summarise` | "Summarise the resume" | Retrieves top 20 chunks; agent uses fast path with 4 targeted searches |
| `comparison` | "Compare Python vs JavaScript" | Retrieves top 15 chunks for breadth |
| `factual` | "What is the candidate's GPA?" | Retrieves top 5 chunks for precision |
| `general` | "Tell me about machine learning" | Retrieves top 10 chunks |

Classification uses keyword matching in a strict priority order: summarise is checked first, then comparison, then factual, then general.

---

### 5. Hybrid Search

Two retrieval methods run in parallel and their scores are fused:

**Dense retrieval (semantic):**
- Query is embedded into a 768-dim vector
- ChromaDB finds the nearest chunks by cosine similarity
- Score = `1 - cosine_distance` (higher = more similar)

**BM25 retrieval (lexical):**
- BM25Okapi scores chunks by term frequency weighted by inverse document frequency
- Exact keyword matches score high even if semantically distant
- Scores are normalised to [0, 1] by dividing by the max BM25 score

**Fusion:**
```
final_score = 0.5 × dense_score + 0.5 × bm25_score
```

The alpha=0.5 weighting gives equal weight to semantic and lexical signals. The best score per chunk across all 3 expanded queries is kept, then the top 20 chunks are passed to the reranker.

**Why hybrid:** Dense search finds semantically similar text even with different vocabulary. BM25 finds exact keyword matches. Together they achieve higher recall than either alone — especially important for structured documents like spreadsheets where exact column names matter.

---

### 6. Type-Aware LLM Reranking

The top 20 retrieved chunks are reranked by the LLM using a **relevance scoring prompt**. The LLM reads each chunk and the query and returns a score from 1–10.

The key insight: a generic reranker prompt underscores structured data. A spreadsheet row like:

```
Name=Alice | Role=Engineer | Salary=90000
```

reads as a data dump — a generic prompt rates it low because it is not fluent prose. Instead, the system uses **7 different prompts**, one per document type:

| Document type | Prompt frames the chunk as... |
|--------------|------------------------------|
| PDF | "...this passage from a PDF document..." |
| DOCX | "...this paragraph from a Word document..." |
| XLSX / CSV | "...this spreadsheet row containing structured data..." |
| PPTX | "...this presentation slide..." |
| HTML | "...this section from a webpage..." |
| TXT | "...this text passage..." |
| MD | "...this section from a Markdown document..." |

The top 5 chunks after reranking go to the LLM for answer synthesis.

---

### 7. Confidence Check

Before calling the LLM, the system checks whether the best retrieved chunk is above the similarity threshold:

```
if best_score >= 0.40:
    is_confident = True  → proceed to synthesis
else:
    is_confident = False → return "I don't have enough information..."
```

**Why:** Without this check, the LLM would receive irrelevant chunks and hallucinate an answer. The threshold skips the LLM entirely for low-confidence queries.

---

### 8. Hallucination Filter

Even when the LLM is called, its response is checked for hallucination pivot phrases — patterns where the model starts with "I don't have information" but then continues with hallucinated content:

```
"There is no information in the documents... however, I can tell you that..."
                                             ↑
                               Pivot phrase detected → truncate here
```

The filter maintains two phrase lists: **no-info phrases** (signals the model has no context) and **pivot phrases** (signals it is about to hallucinate). If a no-info phrase is followed by a pivot phrase, the response is truncated at the pivot.

---

### 9. ReAct Agent Loop

The agent follows the **ReAct** (Reason + Act) pattern:

```
1. THINK  — the LLM decides what to do next
2. ACT    — it calls a tool: TOOL: rag_search(query)
3. OBSERVE — the tool result is added to the context
4. Repeat until TOOL: finish(answer)
```

Tool calls are parsed using two regex patterns:
```
Pattern 1 (with parentheses):   TOOL: tool_name(argument)
Pattern 2 (without parentheses): TOOL: tool_name argument
```

If the LLM produces a malformed response (neither pattern matches), the system retries up to 2 times with a correction prompt before falling back to the raw text as the answer.

**Fast paths** bypass the loop for common queries:
- **Summarise** — detected by keyword; runs 4 targeted searches (`work experience`, `education`, `skills projects`, `summary contact`) and synthesises directly
- **Sentiment** — detected by keyword; retrieves relevant chunks, strips metadata labels, then analyses sentiment

---

### 10. URL Type Detection — 4-Priority Pipeline

When a URL is fetched, the document type is determined in strict priority order:

```
1. Content-Type header    → "application/pdf" → pdf
2. File extension in URL  → "/report.pdf"     → pdf
3. PDF magic bytes        → content[:4] == b'%PDF' → pdf
4. Default               → html
```

This handles edge cases like servers that return `application/octet-stream` for all binary files regardless of actual format.

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

## Testing

The project has **566 local tests** and **262 HF Space tests** — 828 total. Every part of the system is covered.

### Run the tests

```bash
# All local tests
pytest

# With coverage report
pytest --cov=src

# One specific file
pytest tests/test_agent.py

# HF Space tests
cd huggingface && pytest
```

### Test categories

| Type | What it checks | Example |
|------|---------------|---------|
| **Unit** | One method at a time, all dependencies mocked | `test_expand_query_returns_3` |
| **Integration** | Two or more real components, no mocks for file libraries | `test_load_pdf`, `test_url_html_webpage` |
| **Contract** | Output shape — correct keys, types, structure | `test_chunk_has_5_keys` |
| **Regression** | Exact prompt text and phrase lists locked down | `test_hallucination_phrases_unchanged` |
| **Boundary** | Empty files, single-item inputs, at-limit sizes | `test_xlsx_header_only_returns_empty` |
| **Negative** | Wrong or missing input handled gracefully | `test_http_404_returns_empty_list` |
| **Combination** | Parametrized matrix — all modes × all doc types × all URL types | `test_text_doc_type_pipeline[pdf]` |

### Test files

**Local (`tests/`)** — 566 tests across 23 files:

| File | Covers |
|------|--------|
| `test_document_loader.py` | Chunkers, scan, misplaced detection (unit) |
| `test_vector_store.py` | BM25, dense retrieval, build logic (unit) |
| `test_vector_store_pipeline.py` | Full pipeline, rerank, classify (unit) |
| `test_agent.py` | ReAct loop, tools, fast paths (unit) |
| `test_benchmarker.py` | Scoring metrics (unit) |
| `test_integration_loader.py` | All 9 formats + URL ingestion (integration) |
| `test_integration_pipeline.py` | Full RAG pipeline end-to-end (integration) |
| `test_combinations.py` | Chat/Agent × all 8 doc types (parametrized) |
| `test_combinations_url.py` | URL × all content types (parametrized) |
| `test_contracts.py` | Output shape contracts |
| `test_regression.py` | Prompts and phrase lists locked down |
| `test_boundary_negative.py` | Empty files, wrong input |

**HF Space (`huggingface/tests/`)** — 262 tests across 13 files covering the same categories adapted for the Hugging Face deployment (InferenceClient instead of Ollama, EphemeralClient instead of persistent ChromaDB).

### Mock strategy

```python
# Always mock — these make network or model calls:
ollama.embed    → {'embeddings': [[0.1, 0.2, ...]]}
ollama.chat     → {'message': {'content': 'mock response'}}
requests.get    → Mock with .content, .headers, .encoding

# Never mock — use real libraries on real temp files:
fitz (PyMuPDF), python-docx, openpyxl, python-pptx, beautifulsoup4, BM25Okapi
```

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
