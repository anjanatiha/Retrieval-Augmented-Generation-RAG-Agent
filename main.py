"""main.py — Command-line entry point (under 50 lines).

This file only parses arguments and calls the right runner function.
All actual logic lives in src/cli/runner.py.

Usage:
    python main.py               # interactive chat mode
    python main.py --agent       # interactive agent mode (tool calling)
    python main.py --benchmark   # run automated benchmark evaluation
"""

import argparse

from src.cli.runner import initialize, run_agent, run_benchmark, run_chat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG Agent — Ask your documents')
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run automated benchmark evaluation (faithfulness, relevancy, recall)',
    )
    parser.add_argument(
        '--agent',
        action='store_true',
        help='Start in agent mode (rag_search, calculator, summarise, sentiment tools)',
    )
    args = parser.parse_args()

    loader, store = initialize()

    if args.benchmark:
        run_benchmark(loader, store)
    elif args.agent:
        run_agent(store)
    else:
        run_chat(store)
