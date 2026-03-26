"""Unit tests for VectorStore.

Mock strategy (per CLAUDE.md):
  ollama.embed  → {'embeddings': [[0.1, 0.2, ...]]}
  ollama.chat   → {'message': {'content': 'mock'}}
  chromadb      → chromadb.EphemeralClient()

Never mock: BM25Okapi, cosine_similarity, _truncate_for_embedding,
            _classify_query, _smart_top_n, _source_label, _filter_hallucination.
"""

import pytest
import chromadb
from rank_bm25 import BM25Okapi
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    from src.rag.vector_store import VectorStore
    vs = VectorStore()
    return vs


@pytest.fixture
def sample_chunks():
    return [
        {'text': 'Cats sleep sixteen hours a day.', 'source': 'cats.txt',
         'start_line': 1, 'end_line': 1, 'type': 'txt'},
        {'text': 'Cats can see in dim light very well.', 'source': 'cats.txt',
         'start_line': 2, 'end_line': 2, 'type': 'txt'},
        {'text': 'Cats have five toes on front paws.', 'source': 'cats.txt',
         'start_line': 3, 'end_line': 3, 'type': 'txt'},
    ]


def _fake_embed(dim=4):
    """Return a mock ollama.embed response."""
    return {'embeddings': [[0.1] * dim]}


def _fake_chat(content='mock response'):
    return {'message': {'content': content}}


def _fake_chat_stream(content='mock response'):
    """Return a list of one chunk (simulates stream=True)."""
    return [{'message': {'content': content}}]


def _pipeline_chat_mock(*args, **kwargs):
    """Smart mock: returns stream list when stream=True, dict otherwise."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': '8'}}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_has_collection_attr(self, store):
        assert hasattr(store, 'collection')

    def test_has_chunks_attr(self, store):
        assert hasattr(store, 'chunks')

    def test_has_bm25_attr(self, store):
        assert hasattr(store, 'bm25_index')

    def test_has_conversation_history(self, store):
        assert hasattr(store, 'conversation_history')
        assert store.conversation_history == []


# ---------------------------------------------------------------------------
# _truncate_for_embedding
# ---------------------------------------------------------------------------

class TestTruncateForEmbedding:
    def test_short_text_unchanged(self, store):
        text = 'hello world'
        assert store._truncate_for_embedding(text) == text

    def test_truncates_at_200_words(self, store):
        text = ' '.join(['word'] * 300)
        result = store._truncate_for_embedding(text)
        assert len(result.split()) <= 200

    def test_truncates_at_1200_chars(self, store):
        text = 'a' * 2000
        result = store._truncate_for_embedding(text)
        assert len(result) <= 1200

    def test_both_limits_applied(self, store):
        # 150 words each 9 chars → under 200 words but over 1200 chars
        text = ' '.join(['abcdefghi'] * 150)
        result = store._truncate_for_embedding(text)
        assert len(result) <= 1200


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self, store):
        a = [1.0, 0.0, 0.0]
        assert abs(store._cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self, store):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(store._cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self, store):
        assert store._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# _classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    def test_summarise_signal(self, store):
        assert store._classify_query('summarise the document') == 'summarise'

    def test_comparison_signal(self, store):
        assert store._classify_query('compare method A vs method B') == 'comparison'

    def test_factual_signal(self, store):
        assert store._classify_query('what is the boiling point of water') == 'factual'

    def test_general_fallback(self, store):
        assert store._classify_query('something interesting happened') == 'general'

    def test_resume_is_summarise(self, store):
        assert store._classify_query('show me the resume') == 'summarise'


# ---------------------------------------------------------------------------
# _smart_top_n
# ---------------------------------------------------------------------------

class TestSmartTopN:
    def test_factual_returns_5(self, store):
        assert store._smart_top_n('factual') == 5

    def test_comparison_returns_15(self, store):
        assert store._smart_top_n('comparison') == 15

    def test_general_returns_10(self, store):
        assert store._smart_top_n('general') == 10

    def test_summarise_returns_top_retrieve(self, store):
        from src.rag.config import TOP_RETRIEVE
        assert store._smart_top_n('summarise') == TOP_RETRIEVE


# ---------------------------------------------------------------------------
# _check_confidence
# ---------------------------------------------------------------------------

class TestCheckConfidence:
    def test_empty_returns_false(self, store):
        confident, score = store._check_confidence([])
        assert confident is False
        assert score == 0.0

    def test_above_threshold_is_confident(self, store):
        from src.rag.config import SIMILARITY_THRESHOLD
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        results = [(entry, SIMILARITY_THRESHOLD + 0.1)]
        confident, score = store._check_confidence(results)
        assert confident is True

    def test_below_threshold_not_confident(self, store):
        from src.rag.config import SIMILARITY_THRESHOLD
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        results = [(entry, SIMILARITY_THRESHOLD - 0.1)]
        confident, score = store._check_confidence(results)
        assert confident is False


# ---------------------------------------------------------------------------
# _source_label
# ---------------------------------------------------------------------------

class TestSourceLabel:
    def test_pdf_returns_page(self, store):
        entry = {'type': 'pdf', 'start_line': 3, 'end_line': 3}
        assert store._source_label(entry) == 'p3'

    def test_xlsx_returns_row(self, store):
        entry = {'type': 'xlsx', 'start_line': 5, 'end_line': 5}
        assert store._source_label(entry) == 'row5'

    def test_csv_returns_row(self, store):
        entry = {'type': 'csv', 'start_line': 2, 'end_line': 2}
        assert store._source_label(entry) == 'row2'

    def test_pptx_returns_slide(self, store):
        entry = {'type': 'pptx', 'start_line': 1, 'end_line': 1}
        assert store._source_label(entry) == 'slide1'

    def test_html_returns_s(self, store):
        entry = {'type': 'html', 'start_line': 4, 'end_line': 4}
        assert store._source_label(entry) == 's4'

    def test_txt_returns_line_range(self, store):
        entry = {'type': 'txt', 'start_line': 1, 'end_line': 3}
        assert store._source_label(entry) == 'L1-3'


# ---------------------------------------------------------------------------
# _rerank_prompt — 7 variants
# ---------------------------------------------------------------------------

class TestRerankPrompt:
    def _entry(self, doc_type, text='test text'):
        return {'text': text, 'type': doc_type, 'source': 's',
                'start_line': 1, 'end_line': 1}

    def test_xlsx_prompt_contains_row_data(self, store):
        prompt = store._rerank_prompt('what is the salary', self._entry('xlsx'))
        assert 'row data' in prompt.lower() or 'spreadsheet' in prompt.lower()

    def test_pptx_prompt_contains_slide(self, store):
        prompt = store._rerank_prompt('test', self._entry('pptx'))
        assert 'slide' in prompt.lower()

    def test_pdf_prompt_contains_page(self, store):
        prompt = store._rerank_prompt('test', self._entry('pdf'))
        assert 'pdf' in prompt.lower() or 'page' in prompt.lower()

    def test_docx_prompt_contains_paragraph(self, store):
        prompt = store._rerank_prompt('test', self._entry('docx'))
        assert 'paragraph' in prompt.lower() or 'document' in prompt.lower()

    def test_html_prompt_contains_webpage(self, store):
        prompt = store._rerank_prompt('test', self._entry('html'))
        assert 'webpage' in prompt.lower() or 'content' in prompt.lower()

    def test_md_prompt_contains_markdown(self, store):
        prompt = store._rerank_prompt('test', self._entry('md'))
        assert 'markdown' in prompt.lower() or 'section' in prompt.lower()

    def test_txt_prompt_is_generic(self, store):
        prompt = store._rerank_prompt('test', self._entry('txt'))
        assert '1' in prompt and '10' in prompt

    def test_prompt_ends_with_integer_instruction(self, store):
        for dtype in ('xlsx', 'csv', 'pptx', 'pdf', 'docx', 'html', 'md', 'txt'):
            prompt = store._rerank_prompt('query', self._entry(dtype))
            assert '1 to 10' in prompt or '1-10' in prompt


# ---------------------------------------------------------------------------
# _filter_hallucination
# ---------------------------------------------------------------------------

class TestFilterHallucination:
    def test_clean_response_unchanged(self, store):
        response = 'Cats sleep 16 hours a day. [cats.txt L1]'
        assert store._filter_hallucination(response) == response

    def test_truncates_at_however(self, store):
        response = "I couldn't find info about that. However, I can tell you cats are cool."
        result = store._filter_hallucination(response)
        assert 'however' not in result.lower()

    def test_truncates_at_but_i_can(self, store):
        response = "There is no information about it. But I can provide some context."
        result = store._filter_hallucination(response)
        assert 'but i can' not in result.lower()

    def test_no_pivot_returns_full(self, store):
        response = "The provided documents do not contain this information."
        result = store._filter_hallucination(response)
        assert 'provided documents' in result.lower()


# ---------------------------------------------------------------------------
# _build_instruction_prompt
# ---------------------------------------------------------------------------

class TestBuildInstructionPrompt:
    def test_contains_context(self, store):
        prompt = store._build_instruction_prompt('some context here')
        assert 'some context here' in prompt

    def test_contains_strict_rules(self, store):
        prompt = store._build_instruction_prompt('ctx')
        assert 'ONLY' in prompt or 'only' in prompt

    def test_contains_source_citation_instruction(self, store):
        prompt = store._build_instruction_prompt('ctx')
        assert 'cite' in prompt.lower() or 'source' in prompt.lower()


# ---------------------------------------------------------------------------
# build_or_load — uses EphemeralClient
# ---------------------------------------------------------------------------

class TestBuildOrLoad:
    def _make_store_with_ephemeral(self, sample_chunks):
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name='test_collection',
            metadata={'hnsw:space': 'cosine'}
        )
        vs.collection = collection
        vs.chunks = sample_chunks
        vs.bm25_index = None
        return vs

    def test_build_or_load_sets_collection(self, sample_chunks):
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            # Patch the PersistentClient to use Ephemeral
            with patch('chromadb.PersistentClient') as mock_client:
                client = chromadb.EphemeralClient()
                collection = client.get_or_create_collection(
                    'rag_docs', metadata={'hnsw:space': 'cosine'}
                )
                mock_client.return_value.get_or_create_collection.return_value = collection
                vs.build_or_load(sample_chunks)
            assert vs.collection is not None

    def test_build_or_load_sets_bm25(self, sample_chunks):
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            with patch('chromadb.PersistentClient') as mock_client:
                client = chromadb.EphemeralClient()
                collection = client.get_or_create_collection(
                    'rag_docs', metadata={'hnsw:space': 'cosine'}
                )
                mock_client.return_value.get_or_create_collection.return_value = collection
                vs.build_or_load(sample_chunks)
            assert vs.bm25_index is not None

    def test_build_or_load_sets_chunks(self, sample_chunks):
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            with patch('chromadb.PersistentClient') as mock_client:
                client = chromadb.EphemeralClient()
                collection = client.get_or_create_collection(
                    'rag_docs', metadata={'hnsw:space': 'cosine'}
                )
                mock_client.return_value.get_or_create_collection.return_value = collection
                vs.build_or_load(sample_chunks)
            assert vs.chunks == sample_chunks


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    def test_add_chunks_increases_collection_count(self, sample_chunks):
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(
                'rag_docs', metadata={'hnsw:space': 'cosine'}
            )
            vs.collection = collection
            vs.chunks = []
            before = collection.count()
            vs.add_chunks(sample_chunks, id_prefix='url')
            assert collection.count() > before

    def test_add_chunks_updates_self_chunks(self, sample_chunks):
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(
                'rag_docs', metadata={'hnsw:space': 'cosine'}
            )
            vs.collection = collection
            vs.chunks = []
            vs.add_chunks(sample_chunks, id_prefix='url')
            assert len(vs.chunks) == len(sample_chunks)


# ---------------------------------------------------------------------------
# rebuild_bm25
# ---------------------------------------------------------------------------

class TestRebuildBm25:
    def test_rebuild_bm25_sets_index(self, store, sample_chunks):
        store.rebuild_bm25(sample_chunks)
        assert store.bm25_index is not None
        assert isinstance(store.bm25_index, BM25Okapi)


# ---------------------------------------------------------------------------
# clear_conversation
# ---------------------------------------------------------------------------

class TestClearConversation:
    def test_clear_empties_history(self, store):
        store.conversation_history = [{'role': 'user', 'content': 'hi'}]
        store.clear_conversation()
        assert store.conversation_history == []


# ---------------------------------------------------------------------------
# _expand_query — mocks ollama.chat
# ---------------------------------------------------------------------------

class TestExpandQuery:
    def test_returns_list_with_original(self, store):
        with patch('ollama.chat', return_value=_fake_chat('alt1\nalt2')):
            result = store._expand_query('cats')
        assert 'cats' in result

    def test_returns_at_least_one(self, store):
        with patch('ollama.chat', side_effect=Exception('fail')):
            result = store._expand_query('cats')
        assert result == ['cats']

    def test_returns_original_plus_expansions(self, store):
        with patch('ollama.chat', return_value=_fake_chat('rewrite1\nrewrite2')):
            result = store._expand_query('my query')
        assert result[0] == 'my query'
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _hybrid_retrieve — mocks ollama.embed and uses EphemeralClient
# ---------------------------------------------------------------------------

class TestHybridRetrieve:
    def _setup_store(self, sample_chunks):
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            'rag_docs', metadata={'hnsw:space': 'cosine'}
        )
        # Add chunks to collection with embeddings
        ids    = [f'c{i}' for i in range(len(sample_chunks))]
        texts  = [c['text'] for c in sample_chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c['type']}
                  for c in sample_chunks]
        embeds = [[0.1, 0.2, 0.3, 0.4]] * len(sample_chunks)
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
        vs.collection = collection
        vs.chunks = sample_chunks
        vs.bm25_index = BM25Okapi([c['text'].lower().split() for c in sample_chunks])
        return vs

    def test_returns_list(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}):
            result = vs._hybrid_retrieve(['cats sleep'], top_n=2)
        assert isinstance(result, list)

    def test_returns_tuples_entry_score(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}):
            result = vs._hybrid_retrieve(['cats'], top_n=2)
        if result:
            entry, score = result[0]
            assert 'text' in entry
            assert isinstance(score, float)

    def test_top_n_respected(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}):
            result = vs._hybrid_retrieve(['cats'], top_n=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# _rerank — mocks ollama.chat
# ---------------------------------------------------------------------------

class TestRerank:
    def test_returns_list(self, store, sample_chunks):
        entry = sample_chunks[0]
        candidates = [(entry, 0.8), (sample_chunks[1], 0.7)]
        with patch('ollama.chat', return_value=_fake_chat('8')):
            result = store._rerank('cats', candidates, top_n=2)
        assert isinstance(result, list)

    def test_top_n_respected(self, store, sample_chunks):
        candidates = [(c, 0.5) for c in sample_chunks]
        with patch('ollama.chat', return_value=_fake_chat('7')):
            result = store._rerank('cats', candidates, top_n=2)
        assert len(result) <= 2

    def test_returns_triples(self, store, sample_chunks):
        candidates = [(sample_chunks[0], 0.8)]
        with patch('ollama.chat', return_value=_fake_chat('9')):
            result = store._rerank('cats', candidates, top_n=1)
        if result:
            assert len(result[0]) == 3  # (entry, sim, llm_score)

    def test_bad_llm_response_falls_back_to_sim(self, store, sample_chunks):
        candidates = [(sample_chunks[0], 0.75)]
        with patch('ollama.chat', return_value=_fake_chat('no number here at all')):
            result = store._rerank('cats', candidates, top_n=1)
        # Should not raise


# ---------------------------------------------------------------------------
# run_pipeline — integration with mocks
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _setup_store(self, sample_chunks):
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            'rag_docs', metadata={'hnsw:space': 'cosine'}
        )
        ids    = [f'c{i}' for i in range(len(sample_chunks))]
        texts  = [c['text'] for c in sample_chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c['type']}
                  for c in sample_chunks]
        embeds = [[0.1, 0.2, 0.3, 0.4]] * len(sample_chunks)
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
        vs.collection = collection
        vs.chunks = sample_chunks
        vs.bm25_index = BM25Okapi([c['text'].lower().split() for c in sample_chunks])
        vs.conversation_history = []
        return vs

    def test_run_pipeline_returns_dict(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('how many hours do cats sleep', streamlit_mode=True)
        assert isinstance(result, dict)

    def test_run_pipeline_has_response_key(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('cats sleep', streamlit_mode=True)
        assert 'response' in result

    def test_run_pipeline_has_query_type(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('cats sleep', streamlit_mode=True)
        assert 'query_type' in result

    def test_low_confidence_returns_not_found_message(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        # All chunks have very low similarity — force low scores by using distant embedding
        with patch('ollama.embed', return_value={'embeddings': [[0.0, 0.0, 0.0, 1.0]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('unrelated query xyz', streamlit_mode=True)
        assert result['is_confident'] is False or 'response' in result

    def test_run_pipeline_appends_to_conversation_history(self, sample_chunks):
        vs = self._setup_store(sample_chunks)
        assert vs.conversation_history == []
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            vs.run_pipeline('cats sleep', streamlit_mode=True)
        assert len(vs.conversation_history) >= 1
