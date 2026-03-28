# RAG Agent — Retrieval-Augmented Generation System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Release](https://img.shields.io/badge/release-v1.2.1-2ea44f)](https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent/releases)
[![Tests](https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent/actions/workflows/test.yml/badge.svg)](https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent/actions/workflows/test.yml)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FF6B35)](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20LLaMA%203.2-black?logo=ollama)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Runs Locally](https://img.shields.io/badge/runs-100%25%20locally-success)](README.md)

Upload your documents. Ask questions in plain English. Get accurate answers with source citations — all running on your machine, no cloud, no API keys, no data leaving your device.

Supports PDF, Word, Excel, PowerPoint, CSV, Markdown, HTML, and plain text. Works with resumes, reports, spreadsheets, codebases, and any mixed folder of documents.

---

## Get Started

| | Option | Time |
|-|--------|------|
| **[► Try the live demo](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent)** | Open in browser, upload a file, ask a question — nothing to install | 30 seconds |
| **[► Install locally](#installation)** | Run on your own machine, keep documents private | ~10 minutes |

![Hugging Face Demo](assets/huggingface_ragdoll.png)

---

## What it does

- **Chat with your documents** — ask questions about PDFs, Word docs, spreadsheets, presentations, CSV, Markdown, or HTML files
- **Works with structured data** — accurately retrieves from resumes, spreadsheets, and tables (where most RAG systems fail)
- **Agent mode** — autonomous ReAct agent with 6 tools: search, calculator, summarise, sentiment, translate, and finish
- **Multiple input methods** — drop files into a folder, upload via UI, paste any public URL, or search a topic
- **Topic search** — search DuckDuckGo for a topic, crawl the top results, and index them automatically (no API key)
- **Recursive URL crawling** — follow links up to depth 3, filtered to same domain only, with optional keyword filter
- **Fully local** — LLaMA 3.2 and BGE embeddings run via Ollama — nothing leaves your machine

---

## Technical Highlights

Built from scratch as a production-grade NLP system — not a tutorial or notebook. Fully structured, tested, and deployed.

| Area | What was built |
|------|---------------|
| **NLP & Information Retrieval** | Hybrid BM25 + dense vector search, query expansion, query classification, type-aware LLM reranking, hallucination filtering |
| **LLM Application Engineering** | RAG pipeline design, ReAct agent loop with tool calling, prompt engineering across 7 document-type-specific reranker prompts |
| **Software Architecture** | 4-class design with strict separation of concerns, stateless module functions vs stateful class methods, 500-line file cap |
| **Testing** | 848 tests across 31+ files — unit, functional, integration, contract, regression, boundary, negative, parametrized combination, UI (AppTest + mocked st) |
| **Deployment** | Local Ollama + Hugging Face Space using InferenceClient, persistent ChromaDB vector store, CI/CD pipeline |
| **Data Engineering** | 9 format-specific chunkers split across two modules — text formats (txt, md, csv, html) in `chunkers.py`, binary formats (pdf, docx, xlsx, xls, pptx) in `binary_chunkers.py` |

**Key design decisions:**
- **Hybrid search over pure dense retrieval** — BM25 + dense fusion achieves higher recall than either alone, especially for structured documents
- **Type-aware reranking** — 7 different LLM reranker prompts (one per document type) ensure spreadsheet rows are evaluated as structured data, not penalised for not being prose
- **4-class architecture** — all state-carrying logic lives in exactly 4 classes; stateless operations live in modules
- **Confidence gate before every LLM call** — skips the LLM entirely when no relevant content exists rather than hallucinating

> For full algorithm explanations — hybrid search math, reranking, hallucination filter, ReAct loop, URL type detection — see [docs_technical/ARCHITECTURE.md](docs_technical/ARCHITECTURE.md).

---

## Benchmark Results

Measured against 15 questions across 4 domains (cat facts, Python language, team members CSV, machine learning).

| Metric | Score | Kind |
|--------|-------|------|
| Faithfulness | **0.802** | LLM-as-judge |
| Answer Relevancy | **0.828** | LLM-as-judge |
| Ground Truth Match | **0.640** | F1 word overlap |
| Keyword Recall | **0.934** | Fraction of expected keywords found |
| Context Relevance | **0.700** | Mean cosine similarity of retrieved chunks |
| Precision@5 | **0.720** | Relevant chunks in top 5 |
| MRR | **0.900** | Rank of first relevant chunk |
| **Overall** | **0.789** | Mean across all 7 metrics |

**Agent tool benchmark:** calculator 5/5 · sentiment 4/4 · summarise 3/3

> For full benchmark methodology, metric formulas, and how to add your own test cases, see [docs_technical/BENCHMARK.md](docs_technical/BENCHMARK.md).

---

## Deployments

| | URL | Stack |
|-|-----|-------|
| Hugging Face Space | [anjanatiha2024/Rag-Agent](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent) | Gradio + InferenceClient |
| Local web UI | `streamlit run app.py` | Streamlit + Ollama |
| Local terminal | `python main.py` | argparse + Ollama |

---

## Quick Start

Already have Python 3.11 and Ollama? Four commands:

```bash
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## How to Use

### Step 1 — Add your documents

Drop files into `./docs/` subfolders:

```
docs/
  pdfs/   txts/   docx/   xlsx/   pptx/   csv/   md/   html/
```

Or drop a whole mixed-type folder anywhere under `./docs/` — the scanner walks recursively at any depth.

### Step 2 — Choose a mode

```bash
streamlit run app.py          # web UI (recommended)
python3 main.py               # terminal chat
python3 main.py --agent       # agent mode with tool calling
python3 main.py --benchmark   # run the evaluation suite
```

### Step 3 — Ask questions

```
"What is the candidate's most recent job title?"       ← on a resume PDF
"What was the revenue in Q3?"                          ← on a spreadsheet
"Summarise the main points of this document"           ← any format
"What is 15% of the salary mentioned in the resume?"   ← agent + calculator
"What is the sentiment of the cover letter?"           ← agent + sentiment
```

---

## Supported File Types

| Format | Extensions | Chunking strategy |
|--------|-----------|------------------|
| PDF | `.pdf` | Sentence windows per page (PyMuPDF) |
| Word | `.docx`, `.doc` | Paragraph groups + table rows |
| Spreadsheet | `.xlsx`, `.xls` | Row → key=value pairs |
| Presentation | `.pptx` | Text shapes per slide |
| CSV | `.csv` | Row → key=value pairs |
| Plain text | `.txt` | Line-based |
| Markdown | `.md`, `.markdown` | Line-based, syntax stripped |
| HTML | `.html`, `.htm` | Sentence windows, tags stripped |

**Remote URLs** — paste any public URL in the UI: webpages, PDFs, DOCX, XLSX, CSV, PPTX are all fetched, type-detected, and indexed automatically.

**Topic search** — enter a search query; the top DuckDuckGo results are crawled and indexed in one click (no API key, no rate limits).

**Recursive crawl** — follow links from a seed URL up to depth 3, limited to the same domain, with an optional keyword filter to stay on topic.

---

## Agent Mode

The ReAct agent reasons about which tool to use, calls it, observes the result, and repeats.

| Tool | What it does |
|------|-------------|
| `rag_search` | Searches documents using the full retrieval pipeline |
| `calculator` | Evaluates safe arithmetic expressions |
| `summarise` | Summarises a passage with adaptive length |
| `sentiment` | Returns Sentiment, Tone, Key phrases, and Explanation |
| `translate` | Translates to any target language; short queries search the knowledge base first |
| `finish` | Returns the final answer |

---

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11 |
| Ollama | Latest |
| RAM | 8 GB+ (16 GB recommended) |

### Steps

```bash
# 1. Install Ollama — https://ollama.com/download

# 2. Pull the models
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF

# 3. Clone and set up
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate        # Windows: rag_env_311\Scripts\activate
pip install -r requirements.txt

# 4. Start Ollama (separate terminal)
ollama serve

# 5. Run
streamlit run app.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `context length` error | `rm -rf ./chroma_db/` then restart — rebuilds from scratch |
| `ModuleNotFoundError` | `pip install -r requirements.txt` (check venv is activated) |
| Ollama not responding | Run `ollama serve` in a separate terminal |
| No chunks found | Check files are under `./docs/` with a supported extension |
| Model is slow | Close other apps; or use the [HF Space](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent) (cloud GPU) |

---

## Architecture

4 classes and 4 modules. Classes own state; modules own stateless functions.

| Component | Responsibility |
|-----------|---------------|
| `DocumentLoader` | File scanning, URL fetching, chunker dispatch |
| `VectorStore` | ChromaDB, BM25, hybrid retrieval, reranking, response generation |
| `Agent` | ReAct loop and all 6 tools |
| `Benchmarker` | 7-metric evaluation, CSV export, run comparison |
| `chunkers` module | Stateless text-based chunker functions (txt, md, csv, html) |
| `binary_chunkers` module | Stateless binary format chunker functions (pdf, docx, xlsx, xls, pptx) |
| `url_crawl` module | Stateless URL crawl and DuckDuckGo search functions |
| `url_utils` module | URL type detection, source name building, link extraction, topic filtering |
| `metrics` module | 7 stateless scoring functions |
| `benchmark_report` module | Stateless terminal report formatting |
| `tool_benchmarks` module | Calculator / sentiment / summarise benchmark suite |

**Pipeline flow:**
```
Documents / URLs → DocumentLoader → VectorStore index (ChromaDB + BM25)
Query → classify → expand → hybrid retrieve → confidence check → rerank → synthesize → answer
```

> For architectural decisions, class ownership rationale, and the production scaling path, see [DESIGN.md](DESIGN.md).

---

## Folder Structure

```
├── app.py                    ← Streamlit UI entry point (<50 lines)
├── main.py                   ← CLI entry point (<50 lines)
├── src/rag/                  ← Core RAG system
│   ├── config.py             ← All constants
│   ├── logger.py             ← Interaction logging
│   ├── chunkers.py           ← Text-based chunker functions (txt, md, csv, html)
│   ├── binary_chunkers.py    ← Binary format chunker functions (pdf, docx, xlsx, xls, pptx)
│   ├── url_crawl.py          ← URL crawl and DuckDuckGo search functions
│   ├── url_utils.py          ← URL type detection, link extraction, topic filtering
│   ├── document_loader.py    ← DocumentLoader class
│   ├── vector_store.py       ← VectorStore class
│   ├── agent.py              ← Agent class
│   ├── benchmarker.py        ← Benchmarker class
│   ├── benchmark_report.py   ← Terminal report formatting
│   ├── metrics.py            ← 7 scoring functions
│   └── tool_benchmarks.py    ← Tool benchmark suite
├── src/ui/                   ← Streamlit UI modules
│   ├── handlers.py           ← Event handlers (url, file, topic, user input)
│   ├── renderers.py          ← Pure Streamlit render functions
│   ├── sidebar.py            ← Sidebar rendering
│   ├── session.py            ← Session state helpers
│   └── theme.py              ← CSS and style constants
├── src/cli/                  ← Terminal interface
├── tests/                    ← 848 local tests (31+ files)
├── huggingface/              ← HF Space deployment (385 tests)
├── benchmark_docs/           ← Sample files for self-contained benchmarking
│   ├── python-language.txt
│   ├── team-members.csv
│   └── machine-learning.md
├── docs_technical/           ← Deep-dive documentation
│   ├── ARCHITECTURE.md       ← Full pipeline algorithm explanations
│   ├── BENCHMARK.md          ← Benchmark methodology and metric formulas
│   └── CONTRIBUTING.md       ← Dev setup, code standards, PR guidelines
├── DESIGN.md                 ← Architectural decisions and tradeoffs
└── docs/                     ← Your documents go here (git-ignored)
```

---

## Testing

**848 local tests · 385 HF Space tests · 1233 total**

```bash
pytest                          # all local tests
pytest --cov=src                # with coverage report
cd huggingface && pytest        # HF Space tests
```

Test types: unit · functional · integration · contract · regression · boundary · negative · parametrized combination · UI (AppTest + mocked Streamlit).

> For the full test file breakdown, mock strategy, and category descriptions, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Benchmarking

```bash
python3 main.py --benchmark
```

Runs two phases automatically:
1. **RAG pipeline** — 15 questions across 4 domains, 7 metrics, LLM-as-judge + numeric scoring
2. **Agent tools** — 12 deterministic tests (calculator, sentiment, summarise)

Results saved to `benchmark_results.json`, `benchmark_results.csv`, and `tool_benchmark_results.json`.

> For full methodology, metric formulas, sample output, and how to add test cases, see [docs_technical/BENCHMARK.md](docs_technical/BENCHMARK.md).

---

## Streamlit UI

**Chat view:**

![Streamlit Chat](assets/streamlit_rag_before.png)

**Pipeline panel — post-query sidebar:**

![Streamlit Pipeline](assets/streamlit_rag_after.png)

Features: chat and agent mode toggle · URL ingestion · recursive URL crawl · topic search with live crawl log · file upload · step-by-step progress bar · pre/post rerank chunks · confidence and query-type badges · session stats · clear button.

---

## Built With

`Python 3.11` · `Ollama` · `ChromaDB` · `rank-bm25` · `PyMuPDF` · `python-docx` · `openpyxl` · `xlrd` · `python-pptx` · `BeautifulSoup4` · `lxml` · `requests` · `Streamlit` · `LLaMA 3.2` · `BGE Embeddings`

**Development:** Built with the assistance of [Claude](https://claude.ai) (Anthropic) — used for code generation, architecture review, test writing, and documentation. All design decisions and direction were set by the author.

---

## Related

- [Live demo on Hugging Face](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent) — try it in your browser
- [DESIGN.md](DESIGN.md) — architectural decisions, class ownership, tradeoffs, production scaling
- [docs_technical/ARCHITECTURE.md](docs_technical/ARCHITECTURE.md) — full pipeline and algorithm explanations
- [docs_technical/BENCHMARK.md](docs_technical/BENCHMARK.md) — benchmark methodology and metric formulas
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup and contribution guide
- [Based on](https://huggingface.co/blog/ngxson/make-your-own-rag) — significantly extended with hybrid search, type-aware reranking, 9 format support, agent mode, benchmarking, and Streamlit UI
