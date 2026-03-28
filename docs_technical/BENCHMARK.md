# BENCHMARK.md — Benchmark Methodology and Metric Reference

This document covers the full benchmarking system — why the metrics were chosen, how each one is calculated, how to interpret results, and how to add your own test cases.

**Quick navigation:**
- [Overview](#overview)
- [Why 7 metrics](#why-7-metrics)
- [The three quality dimensions](#the-three-quality-dimensions)
- [How each metric is calculated](#how-each-metric-is-calculated)
- [How to interpret the summary statistics](#how-to-interpret-the-summary-statistics)
- [How to interpret drops in the run comparison](#how-to-interpret-drops-in-the-run-comparison)
- [Agent tool benchmark](#agent-tool-benchmark)
- [Sample terminal output](#sample-terminal-output)
- [How to add your own test cases](#how-to-add-your-own-test-cases)
- [Test domains and sample files](#test-domains-and-sample-files)

---

## Overview

The benchmark suite has two phases, both run with one command:

```bash
python3 main.py --benchmark
```

**Phase 1 — RAG pipeline quality:**
- 15 questions across 4 domains (cat facts, Python language, team members CSV, machine learning)
- Sample files are committed to `benchmark_docs/` — no extra setup needed
- 7 metrics per question: 2 LLM-as-judge + 5 numeric
- Prints per-question table, summary statistics, by-query-type breakdown, and run-over-run delta
- Results saved to `benchmark_results.json` (full history) and `benchmark_results.csv` (latest run)

**Phase 2 — Agent tool correctness:**
- 12 deterministic tests: 5 calculator, 4 sentiment, 3 summarise
- Results saved to `tool_benchmark_results.json`

Every run is compared against the previous one with delta indicators (▲ improved / ▼ regressed / ─ unchanged).

---

## Why 7 Metrics

A RAG system can fail in three independent ways. Using only one or two metrics hides which part is broken.

**Example of why this matters:**

A high faithfulness score with a low MRR means the LLM is faithfully quoting whatever it retrieved — but it retrieved the wrong chunks. You cannot see this failure with faithfulness alone. You need both faithfulness (generation quality) and MRR (retrieval ranking quality) to pinpoint it.

---

## The Three Quality Dimensions

| Dimension | Metrics | What a low score here means |
|-----------|---------|----------------------------|
| **Retrieval quality** | Context Relevance, Precision@5, MRR | The system is fetching the wrong chunks — the answer exists in the documents but was not retrieved |
| **Generation quality** | Faithfulness, Answer Relevancy | The retrieved context was good but the LLM ignored it, hallucinated, or went off-topic |
| **Factual accuracy** | Ground Truth Match, Keyword Recall | The answer is roughly correct but missing specific facts, numbers, or expected phrasing |

**Diagnostic patterns:**

| Pattern | What it usually means |
|---------|----------------------|
| Low faithfulness + high context relevance | Model is hallucinating despite good chunks. Check the system prompt and SIMILARITY_THRESHOLD. |
| Low context relevance + low precision@5 | Retrieval is broken upstream. Check document indexing and embedding quality. |
| Low MRR + adequate context relevance | Relevant chunks exist but are ranking low. Reranking quality may have degraded. |
| Low ground truth match + high faithfulness | Answer is correct but phrased differently from the expected answer. May not be a real problem. |

---

## How Each Metric is Calculated

### Faithfulness (LLM-as-judge)

Asks the language model to grade whether every claim in the response is grounded in the retrieved context:

```
LLM rates 1–5 → normalised to 0.0–1.0

1 = answer invents facts not in the context (hallucination)
5 = every claim comes directly from the retrieved chunks
```

More reliable than word-overlap alone because it catches faithful paraphrases — which word-overlap would penalise as hallucination (*"Felines rest 12–16 hours daily"* faithfully captures *"cats sleep 16 hours"* but shares few words).

---

### Answer Relevancy (LLM-as-judge)

Asks the language model to grade whether the answer directly addresses the question:

```
LLM rates 1–5 → normalised to 0.0–1.0

1 = answer ignores the question entirely
5 = answer completely and directly addresses the question
```

---

### Ground Truth Match

F1 word overlap between the response and the known correct answer:

```
precision = words in common / words in response
recall    = words in common / words in ground truth
F1        = 2 × precision × recall / (precision + recall)
```

Requires a `ground_truth` field in each test case. Measures lexical similarity to the expected answer — not semantic similarity. A fluent paraphrase scores low here even if correct.

Common stopwords are excluded from the comparison (a, the, is, are, was, etc.).

---

### Keyword Recall

Fraction of expected answer keywords found in the response:

```
keyword_recall = keywords found in response / total expected keywords
```

Each test case supplies an `expected_keywords` list. For *"How many hours do cats sleep?"* the list is `['sleep', '16']`. If both appear in the response, recall = 1.0.

---

### Context Relevance

Mean cosine similarity of the top reranked chunks to the query:

```
context_relevance = mean(cosine_similarity(chunk_embedding, query_embedding))
                    over the top TOP_RERANK chunks
```

Measures retrieval quality independently of the LLM. A low score here means the problem is upstream in retrieval, not in generation.

---

### Precision@5

Fraction of the top-5 retrieved chunks that contain at least one expected keyword:

```
precision@5 = relevant chunks in top 5 / 5
```

A chunk is "relevant" if it contains at least one expected keyword from the test case.

**Why both Precision@5 and MRR:**

- **Precision@5:** *"Of the 5 chunks I retrieved, how many were useful?"* — measures coverage
- **MRR:** *"How far down did I have to look for the first useful chunk?"* — measures ranking quality

A system can have high Precision@5 but low MRR if it retrieves many relevant chunks but buries them behind irrelevant ones at the top. The reranker is specifically designed to fix this — a reranking improvement shows up as an MRR increase before Precision@5.

---

### MRR (Mean Reciprocal Rank)

How high up the list the first relevant chunk appeared:

```
MRR = 1 / rank_of_first_relevant_chunk

MRR = 1.00 → the very first chunk was relevant
MRR = 0.50 → the second chunk was the first relevant one
MRR = 0.33 → the third chunk was the first relevant one
MRR = 0.00 → no relevant chunk found
```

---

### Overall

Mean of all 7 scored metrics (latency is excluded):

```
overall = mean(faithfulness, answer_relevancy, ground_truth_match,
               keyword_recall, context_relevance, precision_at_5, mrr)
```

---

## How to Interpret the Summary Statistics

| Statistic | What it means |
|-----------|--------------|
| **Mean** | Average across all test questions — the headline score |
| **Std** | Standard deviation — how consistent the score is. High Std = pipeline performs very differently on different questions. Low Std with high Mean is ideal. |
| **Min / Max** | Worst and best individual question scores. A large gap between Min and Max confirms the inconsistency the Std hints at. |
| **latency_ms** | End-to-end wall-clock time per question, from query expansion to last generated token. Does not include LLM-as-judge scoring (evaluation overhead, not pipeline latency). |

---

## How to Interpret Drops in the Run Comparison

| Metric drops | What it usually means |
|---|---|
| **faithfulness_llm** | LLM generating content not grounded in documents. Try raising SIMILARITY_THRESHOLD (0.40 → 0.50) |
| **answer_relevancy_llm** | LLM is digressing. Often caused by ambiguous test questions or a recently changed system prompt |
| **ground_truth_match** | Response wording drifted from expected. The answer may still be correct but phrased differently |
| **keyword_recall** | Model is omitting expected facts. Check whether the keywords are present in the indexed documents |
| **context_relevance** | Retrieval is degraded. New documents may have diluted the BM25 index or the embedding model is struggling with a new content type |
| **precision_at_5** | Retrieved chunks no longer contain the answer. Check whether the right document is indexed |
| **mrr** | The first relevant chunk is ranking lower. Reranking quality may have dropped — check the reranker model |

---

## Agent Tool Benchmark

**Why separate from the RAG pipeline benchmark:**

The RAG pipeline benchmark tests retrieval and generation quality — it uses the LLM as both the system under test and the judge. The tool benchmark tests deterministic correctness of the agent tools — no LLM-as-judge, direct pass/fail checks.

**The 12 tests:**

| Tool | # Tests | What passes |
|------|---------|-------------|
| **calculator** | 5 | Arithmetic results match exactly (with tolerance for floats); unsafe characters (letters) are rejected with an error message |
| **sentiment** | 4 | Output contains all 4 required fields (`Sentiment:`, `Tone:`, `Key phrases:`, `Explanation:`) and a valid label (Positive/Negative/Neutral/Mixed) |
| **summarise** | 3 | Output is non-empty and contains at least one key term from the input |

**Calculator tests are fully deterministic** — no LLM call, direct eval. Sentiment and summarise tests call the language model once per test.

**Calculator allowed characters:** `0123456789+-*/(). ` — note that `**` (power) passes (two asterisks are both in the allowed set) but `sqrt(4)` fails (letters `s`, `q`, `r`, `t` are not in the allowed set).

---

## Sample Terminal Output

```
════════════════════════════════════════════════════════════════════════
  RAG PIPELINE BENCHMARK  ·  2026-03-27 14:32:01  ·  15 questions
════════════════════════════════════════════════════════════════════════

  [1/15] How many hours do cats sleep per day?
         faith=0.75  relev=0.88  gt=0.68  kw=1.00  ctx=0.71  p@5=0.80  mrr=1.00  1243ms
  ...

════════════════════════════════════════════════════════════════════════
  SUMMARY
════════════════════════════════════════════════════════════════════════

  Metric                    Mean    Std    Min    Max  Bar
  ──────────────────────────────────────────────────────────────────
  faithfulness (LLM)       0.802  0.068  0.750  0.880  [████████████████░░░░]
  answer_relevancy (LLM)   0.828  0.068  0.750  0.880  [████████████████░░░░]
  ground_truth_match       0.640  0.063  0.550  0.720  [████████████░░░░░░░░]
  keyword_recall           0.934  0.149  0.670  1.000  [██████████████████░░]
  context_relevance        0.700  0.015  0.680  0.720  [██████████████░░░░░░]
  precision_at_5           0.720  0.110  0.600  0.800  [██████████████░░░░░░]
  mrr                      0.900  0.224  0.500  1.000  [██████████████████░░]

  latency_ms               1155     62   1089   1243  ms
  ──────────────────────────────────────────────────────────────────
  overall                  0.789  0.039  0.747  0.841  [███████████████░░░░░]

════════════════════════════════════════════════════════════════════════
  vs PREVIOUS RUN
════════════════════════════════════════════════════════════════════════
  faithfulness_llm          0.741 → 0.802  ▲0.061
  answer_relevancy_llm      0.781 → 0.828  ▲0.047
  overall                   0.743 → 0.789  ▲0.046

════════════════════════════════════════════════════════════════════════
  AGENT TOOL BENCHMARK
════════════════════════════════════════════════════════════════════════
  #    Tool           Status  Input                                     Note
  ────────────────────────────────────────────────────────────────────────
  1    calculator     PASS    '6 * 7'                                   6 * 7 = 42
  2    calculator     PASS    '(100 + 50) / 3'                          ≈ 50.0
  3    calculator     PASS    'sqrt(4)'                                 letter chars rejected
  ...
  12   summarise      PASS    'The sky is blue. The sun is yellow.'     non-empty summary

────────────────────────────────────────────────────────────────────────
  Total: 12/12 passed  (100%)
    calculator      5/5
    sentiment       4/4
    summarise       3/3

  Saved  → tool_benchmark_results.json
════════════════════════════════════════════════════════════════════════
```

---

## How to Add Your Own Test Cases

```python
from src.rag.benchmarker import Benchmarker
from src.rag.vector_store import VectorStore

store = VectorStore()
bench = Benchmarker(store)

my_test_cases = [
    {
        'question':          "What is the candidate's most recent job title?",
        'ground_truth':      'The candidate is a Senior Machine Learning Engineer.',
        'expected_keywords': ['engineer', 'machine learning', 'senior'],
        'query_type':        'factual',
    },
    {
        'question':          'What year did the company reach $1M revenue?',
        'ground_truth':      'The company reached $1 million in revenue in 2022.',
        'expected_keywords': ['2022', 'million', 'revenue'],
        'query_type':        'factual',
    },
]

bench.run(test_cases=my_test_cases)
```

Or add them permanently to `DEFAULT_TEST_CASES` in `src/rag/benchmarker.py`.

**Tips for writing good test cases:**
- `ground_truth` — write the ideal one-sentence answer. Used by ground_truth_match and LLM-as-judge.
- `expected_keywords` — include the exact words or numbers you expect (2–4 is ideal). Too many keywords penalise natural phrasing.
- `query_type` — label as `'factual'`, `'comparison'`, or `'summarise'` to see per-type breakdowns.
- Mix question types to get a balanced overall score that reflects real usage.

---

## Test Domains and Sample Files

The 15 default test questions are spread across 4 domains. Sample files are committed to `benchmark_docs/` — no extra setup needed.

| Domain | File | Format | Questions |
|--------|------|--------|-----------|
| Cat facts | (indexed separately in `docs/`) | TXT | 5 — basic factual questions |
| Python language | `benchmark_docs/python-language.txt` | TXT | 4 — Python history and features |
| Team members | `benchmark_docs/team-members.csv` | CSV | 3 — structured data retrieval |
| Machine learning | `benchmark_docs/machine-learning.md` | MD | 3 — ML concept questions |

**Why these 4 domains:**

- **Cat facts** — simple factual retrieval baseline (backward compatible with all previous runs)
- **Python language** — tests TXT format retrieval with factual questions about a well-known domain
- **Team members CSV** — tests structured data retrieval where most RAG systems fail (spreadsheet rows as key=value pairs)
- **Machine learning MD** — tests Markdown format retrieval with conceptual questions

**Evaluating the benchmark with a real domain:**

The infrastructure is already in place — `ground_truth`, `query_type`, and `chunk_directory()`. Only the domain-specific documents and questions need to be added. A production evaluation dataset with 50–100 domain-specific questions, human-verified ground truth answers, and a mix of factual/comparison/summarise query types would give much stronger signal.
