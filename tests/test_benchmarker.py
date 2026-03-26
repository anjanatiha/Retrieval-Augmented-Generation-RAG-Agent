"""Unit tests for Benchmarker."""

import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    from src.rag.vector_store import VectorStore
    store = MagicMock(spec=VectorStore)
    entry = {'text': 'Cats sleep 16 hours a day.', 'source': 'cats.txt',
             'start_line': 1, 'end_line': 1, 'type': 'txt'}
    store._expand_query.return_value = ['cats sleep']
    store._hybrid_retrieve.return_value = [(entry, 0.9)]
    store._rerank.return_value = [(entry, 0.9, 0.9)]
    return store


@pytest.fixture
def benchmarker(mock_store):
    from src.rag.benchmarker import Benchmarker
    return Benchmarker(mock_store)


# ---------------------------------------------------------------------------
# DEFAULT_TEST_CASES
# ---------------------------------------------------------------------------

class TestDefaultTestCases:
    def test_has_5_cases(self):
        from src.rag.benchmarker import Benchmarker
        assert len(Benchmarker.DEFAULT_TEST_CASES) == 5

    def test_all_have_question(self):
        from src.rag.benchmarker import Benchmarker
        for tc in Benchmarker.DEFAULT_TEST_CASES:
            assert 'question' in tc

    def test_all_have_expected_keywords(self):
        from src.rag.benchmarker import Benchmarker
        for tc in Benchmarker.DEFAULT_TEST_CASES:
            assert 'expected_keywords' in tc

    def test_first_question_about_cats_sleep(self):
        from src.rag.benchmarker import Benchmarker
        assert 'sleep' in Benchmarker.DEFAULT_TEST_CASES[0]['question'].lower()

    def test_first_keywords_contain_16(self):
        from src.rag.benchmarker import Benchmarker
        assert '16' in Benchmarker.DEFAULT_TEST_CASES[0]['expected_keywords']


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_has_store(self, benchmarker, mock_store):
        assert benchmarker.store is mock_store

    def test_has_results_file(self, benchmarker):
        from src.rag.config import BENCHMARK_FILE
        assert benchmarker.results_file == BENCHMARK_FILE


# ---------------------------------------------------------------------------
# _score_faithfulness
# ---------------------------------------------------------------------------

class TestScoreFaithfulness:
    def _make_reranked(self, text):
        entry = {'text': text, 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        return [(entry, 0.9, 0.9)]

    def test_exact_overlap_returns_high_score(self, benchmarker):
        reranked = self._make_reranked('cats sleep sixteen hours per day')
        score = benchmarker._score_faithfulness('cats sleep sixteen hours per day', reranked)
        assert score > 0.5

    def test_no_overlap_returns_low_score(self, benchmarker):
        reranked = self._make_reranked('dogs bark all day long')
        score = benchmarker._score_faithfulness('cats sleep sixteen hours', reranked)
        assert score < 0.5

    def test_empty_response_returns_zero(self, benchmarker):
        reranked = self._make_reranked('cats sleep sixteen hours')
        score = benchmarker._score_faithfulness('', reranked)
        assert score == 0.0

    def test_score_between_0_and_1(self, benchmarker):
        reranked = self._make_reranked('cats sleep sixteen hours per day')
        score = benchmarker._score_faithfulness('cats are wonderful animals', reranked)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _score_relevancy
# ---------------------------------------------------------------------------

class TestScoreRelevancy:
    def test_same_words_returns_high(self, benchmarker):
        score = benchmarker._score_relevancy('how many hours cats sleep', 'cats sleep sixteen hours')
        assert score > 0.3

    def test_unrelated_returns_low(self, benchmarker):
        score = benchmarker._score_relevancy('how many hours cats sleep', 'dogs bark loudly')
        assert score < 0.3

    def test_empty_question_returns_zero(self, benchmarker):
        score = benchmarker._score_relevancy('', 'some response')
        assert score == 0.0

    def test_score_between_0_and_1(self, benchmarker):
        score = benchmarker._score_relevancy('what is the answer', 'the answer is forty two')
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _score_keyword_recall
# ---------------------------------------------------------------------------

class TestScoreKeywordRecall:
    def test_all_keywords_found(self, benchmarker):
        score = benchmarker._score_keyword_recall('cats sleep 16 hours a day', ['sleep', '16'])
        assert score == 1.0

    def test_no_keywords_found(self, benchmarker):
        score = benchmarker._score_keyword_recall('dogs bark', ['sleep', '16'])
        assert score == 0.0

    def test_partial_keywords_found(self, benchmarker):
        score = benchmarker._score_keyword_recall('cats sleep a lot', ['sleep', '16'])
        assert score == 0.5

    def test_empty_keywords_returns_one(self, benchmarker):
        score = benchmarker._score_keyword_recall('any response', [])
        assert score == 1.0

    def test_case_insensitive(self, benchmarker):
        score = benchmarker._score_keyword_recall('Cats Sleep 16 Hours', ['sleep', '16'])
        assert score == 1.0


# ---------------------------------------------------------------------------
# _score_context_relevance
# ---------------------------------------------------------------------------

class TestScoreContextRelevance:
    def test_empty_returns_zero(self, benchmarker):
        score = benchmarker._score_context_relevance([])
        assert score == 0.0

    def test_returns_mean_of_scores(self, benchmarker):
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        reranked = [(entry, 0.8, 0.8), (entry, 0.6, 0.6)]
        score = benchmarker._score_context_relevance(reranked)
        assert abs(score - 0.7) < 0.01

    def test_score_between_0_and_1(self, benchmarker):
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        reranked = [(entry, 0.5, 0.5)]
        score = benchmarker._score_context_relevance(reranked)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _read_results / _save_results
# ---------------------------------------------------------------------------

class TestReadSaveResults:
    def test_read_returns_empty_list_when_no_file(self, benchmarker):
        benchmarker.results_file = '/tmp/nonexistent_benchmark_test_xyz.json'
        result = benchmarker._read_results()
        assert result == []

    def test_save_and_read_round_trip(self, benchmarker, tmp_path):
        benchmarker.results_file = str(tmp_path / 'bench.json')
        data = [{'timestamp': '2024-01-01', 'summary': {'overall': 0.75}, 'results': []}]
        benchmarker._save_results(data)
        loaded = benchmarker._read_results()
        assert len(loaded) == 1
        assert loaded[0]['summary']['overall'] == 0.75


# ---------------------------------------------------------------------------
# _compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def test_returns_string(self, benchmarker):
        current  = {'faithfulness': 0.8, 'answer_relevancy': 0.7,
                    'keyword_recall': 0.9, 'context_relevance': 0.6, 'overall': 0.75}
        previous = {'faithfulness': 0.7, 'answer_relevancy': 0.6,
                    'keyword_recall': 0.8, 'context_relevance': 0.5, 'overall': 0.65}
        result = benchmarker._compare_runs(current, previous)
        assert isinstance(result, str)

    def test_shows_improvement_indicator(self, benchmarker):
        current  = {'faithfulness': 0.8, 'answer_relevancy': 0.7,
                    'keyword_recall': 0.9, 'context_relevance': 0.6, 'overall': 0.75}
        previous = {'faithfulness': 0.6, 'answer_relevancy': 0.5,
                    'keyword_recall': 0.7, 'context_relevance': 0.4, 'overall': 0.55}
        result = benchmarker._compare_runs(current, previous)
        assert '▲' in result

    def test_shows_decline_indicator(self, benchmarker):
        current  = {'faithfulness': 0.5, 'answer_relevancy': 0.4,
                    'keyword_recall': 0.6, 'context_relevance': 0.3, 'overall': 0.45}
        previous = {'faithfulness': 0.8, 'answer_relevancy': 0.7,
                    'keyword_recall': 0.9, 'context_relevance': 0.6, 'overall': 0.75}
        result = benchmarker._compare_runs(current, previous)
        assert '▼' in result


# ---------------------------------------------------------------------------
# run — integration with mocks
# ---------------------------------------------------------------------------

class TestRun:
    def _chat_mock(**kw):
        if kw.get('stream'):
            return [{'message': {'content': 'Cats sleep 16 hours a day.'}}]
        return {'message': {'content': 'Cats sleep 16 hours a day.'}}

    def test_run_returns_summary_dict(self, benchmarker, tmp_path):
        benchmarker.results_file = str(tmp_path / 'bench.json')
        test_cases = [
            {'question': 'How many hours do cats sleep?', 'expected_keywords': ['sleep', '16']}
        ]
        with patch('ollama.chat', side_effect=lambda **kw: (
            [{'message': {'content': 'Cats sleep 16 hours a day.'}}]
            if kw.get('stream') else {'message': {'content': 'Cats sleep 16 hours.'}}
        )):
            summary = benchmarker.run(test_cases)
        assert isinstance(summary, dict)
        assert 'overall' in summary

    def test_run_saves_to_file(self, benchmarker, tmp_path):
        benchmarker.results_file = str(tmp_path / 'bench.json')
        test_cases = [
            {'question': 'Do cats sleep a lot?', 'expected_keywords': ['sleep']}
        ]
        with patch('ollama.chat', side_effect=lambda **kw: (
            [{'message': {'content': 'Yes, cats sleep a lot.'}}]
            if kw.get('stream') else {'message': {'content': 'Yes, cats sleep.'}}
        )):
            benchmarker.run(test_cases)
        assert os.path.exists(benchmarker.results_file)

    def test_run_uses_default_test_cases(self, benchmarker, tmp_path):
        benchmarker.results_file = str(tmp_path / 'bench.json')
        with patch('ollama.chat', side_effect=lambda **kw: (
            [{'message': {'content': 'Cats sleep 16 hours a day.'}}]
            if kw.get('stream') else {'message': {'content': 'Cats sleep 16 hours.'}}
        )):
            summary = benchmarker.run()
        assert isinstance(summary, dict)
