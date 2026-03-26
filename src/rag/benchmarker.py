"""benchmarker.py — Benchmarker class.

Owns all evaluation, scoring, and results management.
"""

import json
import os
import ollama
from datetime import datetime

from src.rag.config import LANGUAGE_MODEL, BENCHMARK_FILE, TOP_RERANK, TOP_RETRIEVE
from src.rag.vector_store import VectorStore


class Benchmarker:
    DEFAULT_TEST_CASES = [
        {'question': 'How many hours do cats sleep per day?',     'expected_keywords': ['sleep', '16']},
        {'question': 'Can cats see in dim light?',                'expected_keywords': ['dim', 'light', 'see']},
        {'question': 'How many toes do cats have on front paws?', 'expected_keywords': ['five', 'toes', 'front']},
        {'question': 'How many whiskers does a cat have?',        'expected_keywords': ['whiskers', '12']},
        {'question': 'Can cats taste sweet food?',                'expected_keywords': ['sweet', 'taste']},
    ]

    def __init__(self, store: VectorStore):
        self.store        = store
        self.results_file = BENCHMARK_FILE

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, test_cases=None):
        test_cases = test_cases or self.DEFAULT_TEST_CASES
        print("\n" + "="*70)
        print("  BENCHMARKING RAG PIPELINE")
        print("="*70)
        results = []

        for i, tc in enumerate(test_cases):
            q, kw = tc['question'], tc.get('expected_keywords', [])
            print(f"\n[{i+1}/{len(test_cases)}] {q}")

            queries   = self.store._expand_query(q)
            retrieved = self.store._hybrid_retrieve(queries, top_n=TOP_RETRIEVE)
            reranked  = self.store._rerank(q, retrieved, top_n=TOP_RERANK)

            context = '\n'.join(f" - {e['text']}" for e, _, _ in reranked)
            stream  = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content':
                        f"You are a factual assistant. Answer in 1-2 sentences "
                        f"using ONLY the facts below.\n\nFacts:\n{context}"},
                    {'role': 'user', 'content': q},
                ], stream=True)
            response = ''.join(c['message']['content'] for c in stream)

            faith = self._score_faithfulness(response, reranked)
            rel   = self._score_relevancy(q, response)
            kwr   = self._score_keyword_recall(response, kw)
            ctx   = self._score_context_relevance(reranked)
            ovr   = (faith + rel + kwr + ctx) / 4

            results.append({'question': q, 'faithfulness': round(faith, 3),
                            'answer_relevancy': round(rel, 3),
                            'keyword_recall': round(kwr, 3),
                            'context_relevance': round(ctx, 3),
                            'overall': round(ovr, 3)})
            print(f"  faith={faith:.2f} rel={rel:.2f} kw={kwr:.2f} ctx={ctx:.2f} overall={ovr:.2f}")

        def avg(k): return sum(r[k] for r in results) / len(results)
        summary = {k: round(avg(k), 3) for k in
                   ['faithfulness', 'answer_relevancy', 'keyword_recall',
                    'context_relevance', 'overall']}
        def bar(s): return '[' + '█'*int(s*20) + '░'*(20-int(s*20)) + ']'

        print("\n" + "="*70 + "\n  SUMMARY\n" + "="*70)
        for k, v in summary.items():
            print(f"  {k:<25} {v:>6.3f}  {bar(v)}")

        runs = self._read_results()
        if runs:
            prev = runs[-1]['summary']
            print(self._compare_runs(summary, prev))

        runs.append({'timestamp': datetime.now().isoformat(),
                     'summary': summary, 'results': results})
        self._save_results(runs)
        print(f"\n  Saved to '{self.results_file}'\n" + "="*70)
        return summary

    # ── Private ──────────────────────────────────────────────────────────────

    def _score_faithfulness(self, response, reranked):
        context = ' '.join(e['text'] for e, _, _ in reranked)
        stopwords = {'a','an','the','is','are','was','were','do','does','it','its',
                     'to','of','in','for','and','or','not','with','on','at','by',
                     'this','that','be','as','i','you','we','they','but','so','if'}
        context_words  = set(w for w in context.lower().split()  if w not in stopwords)
        response_words = set(w for w in response.lower().split() if w not in stopwords)
        if not response_words:
            return 0.0
        return min(len(response_words & context_words) / max(len(response_words), 1), 1.0)

    def _score_relevancy(self, question, response):
        stopwords = {'a','an','the','is','are','was','were','do','does','did','have',
                     'has','can','what','how','why','when','where','who','to','of','in',
                     'it','its','for','and','or','not','with','on','at','by','from'}
        q_words = set(question.lower().split()) - stopwords
        r_words = set(response.lower().split()) - stopwords
        if not q_words:
            return 0.0
        precision = len(q_words & r_words) / max(len(r_words), 1)
        recall    = len(q_words & r_words) / max(len(q_words), 1)
        if precision + recall == 0:
            return 0.0
        return min(2 * precision * recall / (precision + recall), 1.0)

    def _score_keyword_recall(self, response, keywords):
        if not keywords:
            return 1.0
        rl = response.lower()
        return sum(1 for kw in keywords if kw.lower() in rl) / len(keywords)

    def _score_context_relevance(self, reranked):
        if not reranked:
            return 0.0
        scores = [sim for _, sim, _ in reranked[:TOP_RERANK]]
        return sum(scores) / len(scores)

    def _save_results(self, results):
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _compare_runs(self, current, previous):
        lines = ["\n  vs PREVIOUS RUN"]
        for k in current:
            d = current[k] - previous.get(k, 0)
            lines.append(
                f"  {k:<25} {previous.get(k,0):>6.3f} → {current[k]:>6.3f}  "
                f"{'▲' if d>0 else '▼' if d<0 else '─'}{abs(d):.3f}"
            )
        return '\n'.join(lines)

    def _read_results(self):
        if not os.path.exists(self.results_file):
            return []
        with open(self.results_file) as f:
            try:
                return json.load(f)
            except Exception:
                return []
