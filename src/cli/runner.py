"""cli/runner.py — Logic for each command-line mode.

WHY THIS FILE EXISTS:
    main.py must stay under 50 lines so it is easy to read at a glance.
    The actual work — initialising the pipeline, running the chat loop,
    running the agent loop, running benchmarks — all lives here.

ADDING A NEW MODE:
    1. Write a new run_*() function in this file.
    2. Add a new --flag to main.py's argument parser.
    3. Call your new function from main.py.
    You never need to change the existing run_* functions.
"""

import logging
from typing import Tuple

from src.rag.agent import Agent
from src.rag.benchmarker import Benchmarker
from src.rag.config import CHROMA_DIR, DOC_FOLDERS, DOCS_ROOT, TOP_RERANK
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore

# Module-level logger — replaces bare print() for non-user-facing messages
logger = logging.getLogger(__name__)

__all__ = ['initialize', 'run_benchmark', 'run_agent', 'run_chat']


def initialize() -> Tuple[DocumentLoader, VectorStore]:
    """Load all documents, build the vector index, and return the two main objects.

    Prints a startup summary to the terminal so the user knows what is loading.

    Returns:
        A tuple of (DocumentLoader, VectorStore) ready to handle queries.
    """
    print("=" * 60)
    print("  Initializing RAG Pipeline")
    print("=" * 60)
    print(f"  Docs root:  {DOCS_ROOT}/")
    for document_type, folder_path in DOC_FOLDERS.items():
        print(f"    {document_type.upper():<8} → {folder_path}/")
    print(f"  Vector DB:  ChromaDB (persistent @ {CHROMA_DIR})")
    print(f"  Reranker:   LLM-based (document type-aware)")
    print(f"  Misplaced file detection: ENABLED")
    print("=" * 60 + "\n")

    loader = DocumentLoader()
    loader.ensure_folders()
    all_chunks = loader.chunk_all_documents()

    store = VectorStore()
    store.build_or_load(all_chunks)
    return loader, store


def run_benchmark(store: VectorStore) -> None:
    """Run the automated benchmark and print scores to the terminal.

    Uses 5 cat-facts questions by default. Results are saved to
    benchmark_results.json and compared against the previous run.

    Args:
        store: An already-initialised VectorStore with documents loaded.
    """
    Benchmarker(store).run()


def run_agent(store: VectorStore) -> None:
    """Start an interactive agent loop in the terminal.

    The agent can use rag_search, calculator, summarise, sentiment, and finish.
    Type 'exit' or press Ctrl+C to quit.

    Args:
        store: An already-initialised VectorStore with documents loaded.
    """
    agent = Agent(store)
    print("Agent mode — type your task (or 'exit' to quit):\n")

    while True:
        try:
            user_task = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_task:
            continue  # User pressed Enter with nothing typed — ask again

        if user_task.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        result = agent.run(user_task)
        print(f"\nFinal answer: {result['answer']}\n" + "-" * 60)


def run_chat(store: VectorStore) -> None:
    """Start an interactive chat loop in the terminal.

    Uses the full RAG pipeline: classify → expand → retrieve → rerank → generate.
    Prefix a message with 'agent:' to run one question through the agent instead.
    Type 'exit', 'quit', or 'bye' to quit.

    Args:
        store: An already-initialised VectorStore with documents loaded.
    """
    print("=" * 60 + "\n  RAG Chatbot — Full Pipeline\n" + "=" * 60)
    print("Commands: 'exit' to quit | 'agent: <question>' for one agent turn\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_query:
            continue  # Empty line — ask again

        if user_query.lower() in ('exit', 'quit', 'bye'):
            print("Goodbye!")
            break

        # Allow a one-off agent query without switching modes permanently
        if user_query.lower().startswith('agent:'):
            agent_query = user_query[6:].strip()
            print("\n[Agent mode]")
            agent_result = Agent(store).run(agent_query)
            print(f"\nAgent answer: {agent_result['answer']}")
        else:
            pipeline_result = store.run_pipeline(user_query)

            # Warn the user if the answer might be unreliable
            if not pipeline_result['is_confident']:
                print(f"[Warning] Low confidence ({pipeline_result['best_score']:.2f})")

            print(
                f"\n[type:{pipeline_result['query_type']} | "
                f"expanded:{len(pipeline_result['queries'])} queries]"
            )
            print(
                f"Before rerank: {len(pipeline_result['retrieved'])} chunks | "
                f"After: {TOP_RERANK} chunks"
            )

        print("-" * 60)
