"""main.py — CLI thin wrapper for the RAG pipeline."""

import argparse
import sys

from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.rag.agent import Agent
from src.rag.benchmarker import Benchmarker
from src.rag.config import DOCS_ROOT, DOC_FOLDERS, CHROMA_DIR, TOP_RERANK


def initialize():
    """Load documents, build vector index, return (loader, store)."""
    print("=" * 60)
    print("  Initializing RAG Pipeline")
    print("=" * 60)
    print(f"  Docs root:  {DOCS_ROOT}/")
    for t, folder in DOC_FOLDERS.items():
        print(f"    {t.upper():<8} → {folder}/")
    print(f"  Vector DB:  ChromaDB (persistent @ {CHROMA_DIR})")
    print(f"  Reranker:   LLM-based (document type-aware)")
    print(f"  Smart mis-placed file detection: ENABLED")
    print("=" * 60 + "\n")

    loader = DocumentLoader()
    loader.ensure_folders()
    chunks = loader.chunk_all_documents()

    store = VectorStore()
    store.build_or_load(chunks)
    return loader, store


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG Chatbot')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--agent',     action='store_true', help='Agent mode in terminal')
    args = parser.parse_args()

    loader, store = initialize()

    if args.benchmark:
        Benchmarker(store).run()

    elif args.agent:
        agent = Agent(store)
        print("Agent mode — type your task:\n")
        while True:
            try:
                q = input("Task: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not q:
                continue
            if q.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            res = agent.run(q)
            print(f"\nFinal answer: {res['answer']}\n" + "-" * 60)

    else:
        conv = store.conversation_history
        print("=" * 60 + "\n  RAG Chatbot — Full Pipeline\n" + "=" * 60)
        print("Commands: 'exit' quit | 'agent: <q>' use agent mode\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            if query.lower().startswith('agent:'):
                q = query[6:].strip()
                print("\n[Agent mode]")
                res = Agent(store).run(q)
                print(f"\nAgent answer: {res['answer']}")
            else:
                result = store.run_pipeline(query)
                if not result['is_confident']:
                    print(f"[Warning] Low confidence ({result['best_score']:.2f})")
                print(f"\n[type:{result['query_type']} | expanded:{len(result['queries'])} queries]")
                print(f"Before rerank: {len(result['retrieved'])} chunks | After: {TOP_RERANK} chunks")
            print("-" * 60)
