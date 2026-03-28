"""tool_benchmarks.py — Stateless functions for benchmarking all 5 agent tools.

WHY THIS FILE EXISTS:
    The Benchmarker class tests the RAG pipeline (retrieval + generation quality).
    Agent tools have different evaluation criteria:
        - calculator: exact numeric correctness
        - sentiment:  format compliance + valid label
        - summarise:  keyword coverage in output

    These concerns belong in a module (no class state needed) so they do not
    push benchmarker.py over the 500-line file-size limit.

USAGE:
    from src.rag.tool_benchmarks import run_tool_benchmarks
    run_tool_benchmarks(store)   # prints results to the terminal and saves to JSON

OUTPUT FILES:
    tool_benchmark_results.json  — full per-test results; all runs are appended
                                   (same format as benchmark_results.json)

OUTPUT COLUMNS:
    Tool         — which tool was tested
    Input        — the input given to the tool
    Pass/Fail    — whether the result met the success criterion
    Note         — what a passing result should look like
"""

import json
import os
from datetime import datetime
from typing import List

from src.rag.agent import Agent
from src.rag.config import TOOL_BENCHMARK_FILE
from src.rag.vector_store import VectorStore

__all__ = ['TOOL_TEST_CASES', 'run_tool_benchmarks']

# Width of separator lines used in the terminal report
_LINE_WIDTH = 72


# ── Private check helpers ──────────────────────────────────────────────────────
# Defined before TOOL_TEST_CASES so they can be referenced directly in the list.

def _check_sentiment_format(result: str) -> bool:
    """Return True if the sentiment output contains all 4 required fields.

    The Agent's _tool_sentiment() must return a result with exactly these labels:
        Sentiment:   (Positive / Negative / Neutral / Mixed)
        Tone:        (one short phrase)
        Key phrases: (2-4 phrases)
        Explanation: (1-2 sentences)

    We check for the field labels (not the values) so the test works regardless
    of what the LLM says about a particular piece of text.
    """
    required_fields = ['Sentiment:', 'Tone:', 'Key phrases:', 'Explanation:']
    return all(field in result for field in required_fields)


def _check_valid_sentiment_label(result: str) -> bool:
    """Return True if the result contains at least one valid sentiment category label."""
    valid_labels = ['Positive', 'Negative', 'Neutral', 'Mixed']
    return any(label in result for label in valid_labels)


def _check_calculator_approx(expected: float, tolerance: float = 0.01):
    """Return a check function that verifies numeric output is within tolerance of expected.

    Args:
        expected:  The correct numeric answer.
        tolerance: How close the actual output must be to the expected value.

    Returns:
        A callable(result_str) → bool.
    """
    def check(result: str) -> bool:
        try:
            return abs(float(result.strip()) - expected) < tolerance
        except (ValueError, TypeError):
            return False
    return check


# ── Test case definitions ──────────────────────────────────────────────────────
# Each entry describes one test for one tool.
# Required keys:
#   tool      — which Agent tool to invoke ('calculator', 'sentiment', 'summarise')
#   input     — argument string passed to the tool
#   check     — callable(result_str) → bool; True means the test passed
#   note      — short human-readable description of the success criterion

TOOL_TEST_CASES: List[dict] = [

    # ── Calculator: deterministic arithmetic ────────────────────────────────
    # Results must match exactly because eval() is deterministic.
    {
        'tool':  'calculator',
        'input': '6 * 7',
        'check': lambda r: r.strip() == '42',
        'note':  '6 * 7 = 42',
    },
    {
        'tool':  'calculator',
        'input': '(100 + 50) / 3',
        'check': _check_calculator_approx(50.0),
        'note':  '(100 + 50) / 3 ≈ 50.0',
    },
    {
        'tool':  'calculator',
        'input': 'sqrt(4)',
        # Letters like 's','q','r','t' are not in the allowed character set
        # so the calculator must reject this and return an error message.
        'check': lambda r: 'error' in r.lower() or 'unsafe' in r.lower(),
        'note':  "letter chars not allowed — must return an error message",
    },
    {
        'tool':  'calculator',
        'input': '365 * 24',
        'check': lambda r: r.strip() == '8760',
        'note':  '365 * 24 = 8760',
    },
    {
        'tool':  'calculator',
        'input': '15% of 85000',
        # % of is normalised to (15/100*85000) = 12750.0 before eval
        'check': _check_calculator_approx(12750.0, tolerance=0.5),
        'note':  '15% of 85000 = 12750.0',
    },

    # ── Sentiment: output must have all 4 required fields and a valid label ──
    {
        'tool':  'sentiment',
        'input': 'I absolutely love this product. It works perfectly every time.',
        'check': _check_sentiment_format,
        'note':  'must contain Sentiment, Tone, Key phrases, Explanation fields',
    },
    {
        'tool':  'sentiment',
        'input': 'This is a terrible experience. Nothing works and support is useless.',
        'check': _check_sentiment_format,
        'note':  'must contain Sentiment, Tone, Key phrases, Explanation fields',
    },
    {
        'tool':  'sentiment',
        'input': 'Water boils at 100 degrees Celsius at sea level.',
        'check': _check_sentiment_format,
        'note':  'must contain Sentiment, Tone, Key phrases, Explanation fields',
    },
    {
        'tool':  'sentiment',
        'input': 'I absolutely love this product. It works perfectly every time.',
        'check': _check_valid_sentiment_label,
        'note':  'must contain a valid label (Positive/Negative/Neutral/Mixed)',
    },

    # ── Summarise: output must contain key terms from the input ─────────────
    {
        'tool':  'summarise',
        'input': (
            'Python was created by Guido van Rossum. '
            'It was first released in 1991. '
            'Python emphasizes readability and uses significant indentation. '
            'It supports object-oriented, procedural, and functional programming.'
        ),
        'check': lambda r: 'python' in r.lower() or 'guido' in r.lower(),
        'note':  'summary must mention Python or Guido',
    },
    {
        'tool':  'summarise',
        'input': (
            'Machine learning is a subset of artificial intelligence. '
            'It enables computers to learn from data without being explicitly programmed. '
            'Common algorithms include decision trees, neural networks, and support vector machines. '
            'Supervised learning uses labelled data. Unsupervised learning finds hidden patterns.'
        ),
        'check': lambda r: 'learn' in r.lower() or 'machine' in r.lower(),
        'note':  'summary must mention machine learning or learning',
    },
    {
        'tool':  'summarise',
        # Very short input — must still produce a non-empty response
        'input': 'The sky is blue. The sun is yellow.',
        'check': lambda r: len(r.strip()) > 0,
        'note':  'must return a non-empty summary for short input',
    },
]


# ── Public ────────────────────────────────────────────────────────────────────

def run_tool_benchmarks(store: VectorStore) -> dict:
    """Run all agent tool test cases and print a scored report to the terminal.

    Creates a fresh Agent for each test so tool state does not carry over
    between tests. Calculator tests do not call the LLM. Sentiment and
    summarise tests each call the language model once.

    Args:
        store: An already-initialised VectorStore. Needed by the sentiment
               tool (it calls _expand_query and _hybrid_retrieve internally).

    Returns:
        Dict with keys: total (int), passed (int), failed (int),
        pass_rate (float 0–1), results (list of per-test dicts).
    """
    print('\n' + '═' * _LINE_WIDTH)
    print('  AGENT TOOL BENCHMARK')
    print('═' * _LINE_WIDTH)
    print(f"  {'#':<4} {'Tool':<14} {'Status':<7} {'Input (truncated)':<40} Note")
    print('  ' + '─' * 68)

    results = []

    for index, tc in enumerate(TOOL_TEST_CASES, start=1):
        tool_name  = tc['tool']
        tool_input = tc['input']
        check_fn   = tc['check']
        note       = tc['note']

        # A fresh Agent per test ensures each test starts with clean state
        agent = Agent(store)

        try:
            result_text = _invoke_tool(agent, tool_name, tool_input)
            passed      = bool(check_fn(result_text))
            error_note  = ''
        except Exception as exc:
            result_text = ''
            passed      = False
            error_note  = f"Exception: {exc}"

        status = 'PASS' if passed else 'FAIL'
        # Truncate long inputs so the table stays within a readable width
        display_input = tool_input[:37] + '..' if len(tool_input) > 39 else tool_input
        display_note  = error_note if error_note else note

        print(
            f"  {index:<4} {tool_name:<14} {status:<7} "
            f"{display_input!r:<40}  {display_note}"
        )

        results.append({
            'tool':   tool_name,
            'input':  tool_input,
            'result': result_text,
            'passed': passed,
            'note':   display_note,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total     = len(results)
    n_passed  = sum(1 for r in results if r['passed'])
    n_failed  = total - n_passed
    pass_rate = n_passed / total if total > 0 else 0.0

    print('\n' + '─' * _LINE_WIDTH)
    print(f"  Total: {n_passed}/{total} passed  ({pass_rate:.0%})")

    # Per-tool breakdown so we can see which tool is weakest
    tool_names_seen = sorted({r['tool'] for r in results})
    for name in tool_names_seen:
        tool_results = [r for r in results if r['tool'] == name]
        tool_passed  = sum(1 for r in tool_results if r['passed'])
        print(f"    {name:<14} {tool_passed}/{len(tool_results)}")

    summary = {
        'total':     total,
        'passed':    n_passed,
        'failed':    n_failed,
        'pass_rate': round(pass_rate, 3),
        'results':   results,
    }

    # Append this run to the JSON history file so we can track trends over time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _save_tool_results(summary, timestamp)
    print(f"\n  Saved  → {TOOL_BENCHMARK_FILE}")
    print('═' * _LINE_WIDTH + '\n')

    return summary


# ── Private ───────────────────────────────────────────────────────────────────

def _save_tool_results(summary: dict, timestamp: str) -> None:
    """Append one benchmark run to the JSON history file.

    Each run is stored as a dict with 'timestamp', 'pass_rate', and 'results'.
    Older runs are never deleted so we can compare across time.
    If the file does not exist or is corrupt, starts a fresh list.

    Args:
        summary:   The return value of run_tool_benchmarks (total/passed/results etc.)
        timestamp: Human-readable timestamp string (e.g. '2024-01-15 14:30:00').
    """
    # Read existing history — start fresh if file is missing or unreadable
    existing_runs: list = []
    if os.path.isfile(TOOL_BENCHMARK_FILE):
        try:
            with open(TOOL_BENCHMARK_FILE, 'r', encoding='utf-8') as file_handle:
                existing_runs = json.load(file_handle)
        except (json.JSONDecodeError, OSError):
            existing_runs = []

    # Append the current run to the history list
    run_entry = {
        'timestamp': timestamp,
        'pass_rate': summary['pass_rate'],
        'passed':    summary['passed'],
        'total':     summary['total'],
        'results':   summary['results'],
    }
    existing_runs.append(run_entry)

    # Write the updated list back to disk
    with open(TOOL_BENCHMARK_FILE, 'w', encoding='utf-8') as file_handle:
        json.dump(existing_runs, file_handle, indent=2)


def _invoke_tool(agent: Agent, tool_name: str, tool_input: str) -> str:
    """Call one Agent private tool method directly and return its string output.

    Why call private methods here?  These benchmarks test individual tools in
    isolation — not the full ReAct loop. Calling the loop would add LLM round-
    trips to decide which tool to use, making the benchmark slow and non-
    deterministic. The private-method call is deliberate and analogous to how
    tests call private methods to verify internal behaviour.

    Args:
        agent:      A freshly created Agent instance.
        tool_name:  One of 'calculator', 'sentiment', 'summarise'.
        tool_input: The argument string to pass to the tool.

    Returns:
        The string output of the tool.

    Raises:
        ValueError: if tool_name is not one of the three supported direct tools.
    """
    if tool_name == 'calculator':
        return agent._tool_calculator(tool_input)
    if tool_name == 'sentiment':
        return agent._tool_sentiment(tool_input)
    if tool_name == 'summarise':
        return agent._tool_summarise(tool_input)
    raise ValueError(
        f"_invoke_tool: unsupported tool '{tool_name}'. "
        "Supported direct-call tools: 'calculator', 'sentiment', 'summarise'."
    )
