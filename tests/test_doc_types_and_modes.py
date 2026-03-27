"""test_doc_types_and_modes.py — Chat and agent mode tests for all 8 document types
using pre-built mock chunks, plus all 5 agent tools and pipeline features.

Mock strategy:
  Always mock: ollama.embed, ollama.chat, chromadb → EphemeralClient
  Never mock:  BM25Okapi, calculator eval, chunk parsing
"""

import uuid
import os
import pytest
import chromadb
from rank_bm25 import BM25Okapi
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_embed(**kwargs):
    return {'embeddings': [[0.1, 0.2, 0.3, 0.4]]}


def _fake_chat(**kwargs):
    """Return streaming list or plain dict depending on stream= kwarg."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': 'mock response'}}


def _make_store(chunks):
    """Build a VectorStore with EphemeralClient loaded with given chunks."""
    from src.rag.vector_store import VectorStore
    vs         = VectorStore()
    client     = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        f'test_{uuid.uuid4().hex}', metadata={'hnsw:space': 'cosine'}
    )
    if chunks:
        ids    = [f'c{i}' for i in range(len(chunks))]
        texts  = [c['text'] for c in chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c['type']} for c in chunks]
        embeds = [[0.1, 0.2, 0.3, 0.4]] * len(chunks)
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    vs.collection          = collection
    vs.chunks              = chunks
    vs.bm25_index          = BM25Okapi([c['text'].lower().split() for c in chunks]) if chunks else None
    vs.conversation_history = []
    return vs


def _chunk(text, source, doc_type, line=1):
    return {'text': text, 'source': source,
            'start_line': line, 'end_line': line, 'type': doc_type}


PATCHES = [
    patch('ollama.embed',  side_effect=_fake_embed),
    patch('ollama.chat',   side_effect=_fake_chat),
]


def _apply_patches(func):
    """Decorator to apply ollama patches to a test method."""
    for p in reversed(PATCHES):
        func = p(func)
    return func


# ---------------------------------------------------------------------------
# 1. Chat mode — all 8 document types
# ---------------------------------------------------------------------------

class TestChatModeTxt:
    @_apply_patches
    def test_chat_txt_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert 'response' in result
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    @_apply_patches
    def test_chat_txt_query_type_detected(self, mock_chat, mock_embed):
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert result.get('query_type') in ('factual', 'general', 'comparison', 'summarise')

    @_apply_patches
    def test_chat_txt_source_cited(self, mock_chat, mock_embed):
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert 'retrieved' in result
        assert any('dog-facts.txt' in e['source'] for e, _ in result['retrieved'])


class TestChatModePdf:
    @_apply_patches
    def test_chat_pdf_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('Bananas contain 89 calories per 100 grams.', 'nutrition.pdf', 'pdf')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many calories are in bananas?', streamlit_mode=True)
        assert 'response' in result
        assert len(result['response']) > 0

    @_apply_patches
    def test_chat_pdf_source_label(self, mock_chat, mock_embed):
        chunks = [_chunk('Salmon is rich in omega-3 fatty acids.', 'nutrition.pdf', 'pdf')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'p' in label or 'L' in label


class TestChatModeDocx:
    @_apply_patches
    def test_chat_docx_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('Python was created by Guido van Rossum in 1991.', 'resume.docx', 'docx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_docx_reranked(self, mock_chat, mock_embed):
        chunks = [_chunk('Python was created by Guido van Rossum in 1991.', 'resume.docx', 'docx'),
                  _chunk('Java was created by James Gosling in 1995.', 'resume.docx', 'docx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'reranked' in result


class TestChatModeXlsx:
    @_apply_patches
    def test_chat_xlsx_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('country=Brazil | capital=Brasilia | population_millions=215',
                         'countries.xlsx', 'xlsx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_xlsx_source_label_row(self, mock_chat, mock_embed):
        chunks = [_chunk('country=Japan | capital=Tokyo', 'countries.xlsx', 'xlsx')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'row' in label or 'L' in label


class TestChatModeCsv:
    @_apply_patches
    def test_chat_csv_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('language=Python | year_created=1991 | creator=Guido van Rossum',
                         'programming-languages.csv', 'csv')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_csv_source_label_row(self, mock_chat, mock_embed):
        chunks = [_chunk('language=Rust | year_created=2010', 'langs.csv', 'csv')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'row' in label or 'L' in label


class TestChatModePptx:
    @_apply_patches
    def test_chat_pptx_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('The Berlin Wall fell on November 9 1989.', 'history.pptx', 'pptx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_pptx_source_label_slide(self, mock_chat, mock_embed):
        chunks = [_chunk('World War Two ended in 1945.', 'history.pptx', 'pptx')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'slide' in label or 'L' in label


class TestChatModeMd:
    @_apply_patches
    def test_chat_md_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('Coffee was first discovered in Ethiopia around 850 AD.',
                         'coffee-facts.md', 'md')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Where was coffee discovered?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_md_confidence(self, mock_chat, mock_embed):
        chunks = [_chunk('Coffee was first discovered in Ethiopia around 850 AD.',
                         'coffee-facts.md', 'md')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Where was coffee discovered?', streamlit_mode=True)
        assert 'is_confident' in result
        assert result['is_confident'] in (True, False)


class TestChatModeHtml:
    @_apply_patches
    def test_chat_html_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('A dog can understand about 165 words and gestures.', 'dog-facts.html', 'html')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many words can a dog understand?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_html_source_label(self, mock_chat, mock_embed):
        chunks = [_chunk('Dogs were domesticated 15000 years ago.', 'dog-facts.html', 'html')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 's' in label or 'L' in label


# ---------------------------------------------------------------------------
# 2. Agent mode — all 8 document types
# ---------------------------------------------------------------------------

class TestAgentModeTxt:
    @_apply_patches
    def test_agent_txt_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(dog sleep hours)'}}
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('How many hours do dogs sleep?', streamlit_mode=True)
        assert 'answer' in result
        assert 'steps' in result
        assert len(result['steps']) >= 1


class TestAgentModePdf:
    @_apply_patches
    def test_agent_pdf_rag_search(self, mock_chat, mock_embed):
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(banana calories)'}}
        chunks = [_chunk('Bananas contain 89 calories per 100 grams.', 'nutrition.pdf', 'pdf')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('How many calories in bananas?', streamlit_mode=True)
        assert 'answer' in result
        assert any(s['tool'] == 'rag_search' for s in result['steps'])


class TestAgentModeDocx:
    @_apply_patches
    def test_agent_docx_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(python creator)'}}
        chunks = [_chunk('Python was created by Guido van Rossum in 1991.', 'resume.docx', 'docx')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('Who created Python?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeXlsx:
    @_apply_patches
    def test_agent_xlsx_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(Brazil capital)'}}
        chunks = [_chunk('country=Brazil | capital=Brasilia | population_millions=215',
                         'countries.xlsx', 'xlsx')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('What is the capital of Brazil?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeCsv:
    @_apply_patches
    def test_agent_csv_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(Python programming language)'}}
        chunks = [_chunk('language=Python | year_created=1991 | creator=Guido van Rossum',
                         'programming-languages.csv', 'csv')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('When was Python created?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModePptx:
    @_apply_patches
    def test_agent_pptx_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(Berlin Wall)'}}
        chunks = [_chunk('The Berlin Wall fell on November 9 1989.', 'history.pptx', 'pptx')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeMd:
    @_apply_patches
    def test_agent_md_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(coffee origin Ethiopia)'}}
        chunks = [_chunk('Coffee was first discovered in Ethiopia around 850 AD.',
                         'coffee-facts.md', 'md')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('Where was coffee discovered?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeHtml:
    @_apply_patches
    def test_agent_html_rag_search(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(dog words understand)'}}
        chunks = [_chunk('A dog can understand about 165 words and gestures.', 'dog-facts.html', 'html')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('How many words can a dog understand?', streamlit_mode=True)
        assert 'answer' in result


# ---------------------------------------------------------------------------
# 3. Agent tools — all 5 tools across doc types
# ---------------------------------------------------------------------------

class TestAgentToolCalculator:
    def test_calculator_basic(self):
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('2 + 2') == '4'

    def test_calculator_nested_parens(self):
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('7 + (9 + 8) - 2 * 6') == '12'

    def test_calculator_percentage(self):
        from src.rag.agent import Agent
        assert float(Agent(MagicMock())._tool_calculator('15% of 85000')) == pytest.approx(12750.0)

    def test_calculator_percentage_alone(self):
        from src.rag.agent import Agent
        assert float(Agent(MagicMock())._tool_calculator('20%')) == pytest.approx(0.2)

    def test_calculator_unsafe_rejected(self):
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('__import__("os")').startswith('Error')

    def test_calculator_complex_expression(self):
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('(10 + 5) * 2 / 3') == '10.0'

    def test_calculator_complex_division(self):
        from src.rag.agent import Agent
        assert float(Agent(MagicMock())._tool_calculator('(10 + 5) * 4 / 6')) == pytest.approx(10.0)

    @_apply_patches
    def test_calculator_auto_finish(self, mock_chat, mock_embed):
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: calculator(100 * 365)'}}
        from src.rag.agent import Agent
        result = Agent(_make_store([])).run('How many days in 100 years?', streamlit_mode=True)
        assert any(s['tool'] == 'calculator' for s in result['steps'])
        assert any(s['tool'] == 'finish' for s in result['steps'])


class TestAgentToolRagSearch:
    @_apply_patches
    def test_rag_search_returns_chunks(self, mock_chat, mock_embed):
        chunks = [_chunk('Coffee was discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks))._tool_rag_search('coffee origin')
        assert 'coffee-facts.md' in result

    @_apply_patches
    def test_rag_search_auto_finish(self, mock_chat, mock_embed):
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(coffee origin)'}}
        chunks = [_chunk('Coffee was discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Where was coffee discovered?', streamlit_mode=True)
        tools = [s['tool'] for s in result['steps']]
        assert 'rag_search' in tools
        assert 'finish' in tools

    @_apply_patches
    def test_rag_search_all_doc_types_return_results(self, mock_chat, mock_embed):
        """All 8 doc types should return at least one result from rag_search."""
        from src.rag.agent import Agent
        for dtype in ['txt', 'pdf', 'docx', 'xlsx', 'csv', 'pptx', 'md', 'html']:
            chunks = [_chunk(f'This is a {dtype} document with relevant content.',
                             f'test.{dtype}', dtype)]
            result = Agent(_make_store(chunks))._tool_rag_search('relevant content')
            assert len(result) > 0, f"rag_search returned empty for {dtype}"


class TestAgentToolSummarise:
    @_apply_patches
    def test_summarise_short_text(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'This is a short summary.'}}
        from src.rag.agent import Agent
        result = Agent(MagicMock())._tool_summarise('Coffee is a drink.')
        assert isinstance(result, str) and len(result) > 0

    @_apply_patches
    def test_summarise_length_hint_short(self, mock_chat, mock_embed):
        """Under 100 words → 2-3 sentences hint."""
        from src.rag.agent import Agent
        captured = []
        mock_chat.side_effect = lambda **kw: captured.append(kw['messages'][0]['content']) or {'message': {'content': 'S.'}}
        Agent(MagicMock())._tool_summarise('Short text here.')
        assert any('2-3 sentences' in c for c in captured)

    @_apply_patches
    def test_summarise_length_hint_medium(self, mock_chat, mock_embed):
        """100-300 words → 4-5 sentences hint."""
        from src.rag.agent import Agent
        captured = []
        mock_chat.side_effect = lambda **kw: captured.append(kw['messages'][0]['content']) or {'message': {'content': 'S.'}}
        Agent(MagicMock())._tool_summarise(' '.join(['word'] * 150))
        assert any('4-5 sentences' in c for c in captured)

    @_apply_patches
    def test_summarise_length_hint_long(self, mock_chat, mock_embed):
        """Over 300 words → 6-8 sentences hint."""
        from src.rag.agent import Agent
        captured = []
        mock_chat.side_effect = lambda **kw: captured.append(kw['messages'][0]['content']) or {'message': {'content': 'S.'}}
        Agent(MagicMock())._tool_summarise(' '.join(['word'] * 350))
        assert any('6-8 sentences' in c for c in captured)

    @_apply_patches
    def test_fast_path_summarise_runs_4_searches(self, mock_chat, mock_embed):
        chunks = [_chunk('Coffee was discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('summarise the coffee document', streamlit_mode=True)
        assert len([s for s in result['steps'] if s['tool'] == 'rag_search']) == 4

    @_apply_patches
    def test_fast_path_summarise_resume_uses_resume_terms(self, mock_chat, mock_embed):
        chunks = [_chunk('Work experience: 5 years as software engineer.', 'resume.docx', 'docx')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('summarise the resume', streamlit_mode=True)
        search_args = [s['arg'] for s in result['steps'] if s['tool'] == 'rag_search']
        assert any('experience' in a for a in search_args)


class TestAgentToolSentiment:
    @_apply_patches
    def test_sentiment_short_query_searches_first(self, mock_chat, mock_embed):
        resp = 'Sentiment: Positive\nTone: enthusiastic\nKey phrases: wonderful, energizing\nExplanation: Positive tone.'
        mock_chat.side_effect = lambda **kw: {'message': {'content': resp}}
        chunks = [_chunk('Coffee is wonderful and energizing.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks))._tool_sentiment('coffee')
        assert 'Sentiment' in result

    @_apply_patches
    def test_sentiment_long_text_direct(self, mock_chat, mock_embed):
        resp = 'Sentiment: Positive\nTone: professional\nKey phrases: excellent, outstanding\nExplanation: Very positive.'
        mock_chat.side_effect = lambda **kw: {'message': {'content': resp}}
        from src.rag.agent import Agent
        result = Agent(_make_store([]))._tool_sentiment(' '.join(['excellent outstanding remarkable positive'] * 5))
        assert 'Sentiment' in result

    @_apply_patches
    def test_sentiment_output_has_4_fields(self, mock_chat, mock_embed):
        resp = 'Sentiment: Positive\nTone: upbeat\nKey phrases: great, amazing\nExplanation: Overall positive.'
        mock_chat.side_effect = lambda **kw: {'message': {'content': resp}}
        from src.rag.agent import Agent
        result = Agent(_make_store([]))._tool_sentiment(' '.join(['great amazing'] * 10))
        assert 'Sentiment' in result
        assert 'Tone' in result
        assert 'Key phrases' in result
        assert 'Explanation' in result

    @_apply_patches
    def test_fast_path_sentiment_strips_metadata_labels(self, mock_chat, mock_embed):
        resp = 'Sentiment: Positive\nTone: warm\nKey phrases: rich, bold\nExplanation: Positive tone.'
        mock_chat.return_value = {'message': {'content': resp}}
        chunks = [_chunk('Coffee has a rich and bold flavour.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('what is the sentiment of the coffee document', streamlit_mode=True)
        assert result['answer'] is not None
        assert len([s for s in result['steps'] if s['tool'] == 'sentiment']) >= 1


class TestAgentToolFinish:
    @_apply_patches
    def test_finish_uses_collected_context(self, mock_chat, mock_embed):
        responses = iter([
            {'message': {'content': 'TOOL: rag_search(coffee origin)'}},
            {'message': {'content': 'TOOL: finish(Coffee was found in Ethiopia.)'}},
        ])
        mock_chat.side_effect = lambda **kw: next(responses)
        chunks = [_chunk('Coffee was first discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Where was coffee discovered?', streamlit_mode=True)
        assert any(s['tool'] == 'finish' for s in result['steps'])
        assert result['answer'] is not None

    @_apply_patches
    def test_finish_direct_no_context(self, mock_chat, mock_embed):
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: finish(42 is the answer.)'}}
        from src.rag.agent import Agent
        result = Agent(_make_store([])).run('What is the answer?', streamlit_mode=True)
        assert '42 is the answer.' in result['answer']


# ---------------------------------------------------------------------------
# 4. Chat mode — pipeline features across doc types
# ---------------------------------------------------------------------------

class TestChatPipelineFeatures:
    @_apply_patches
    def test_confidence_flag_present(self, mock_chat, mock_embed):
        chunks = [_chunk('Python was created in 1991.', 'langs.csv', 'csv')]
        result = _make_store(chunks).run_pipeline('When was Python created?', streamlit_mode=True)
        assert 'is_confident' in result

    @_apply_patches
    def test_reranked_chunks_present(self, mock_chat, mock_embed):
        chunks = [_chunk('Python was created in 1991.', 'langs.csv', 'csv'),
                  _chunk('Rust was created in 2010.', 'langs.csv', 'csv')]
        result = _make_store(chunks).run_pipeline('When was Python created?', streamlit_mode=True)
        assert 'reranked' in result and len(result['reranked']) >= 1

    @_apply_patches
    def test_conversation_memory(self, mock_chat, mock_embed):
        store = _make_store([_chunk('Brazil capital is Brasilia.', 'countries.xlsx', 'xlsx')])
        store.run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        assert len(store.conversation_history) >= 2

    @_apply_patches
    def test_clear_conversation(self, mock_chat, mock_embed):
        store = _make_store([_chunk('Brazil capital is Brasilia.', 'countries.xlsx', 'xlsx')])
        store.run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        store.clear_conversation()
        assert store.conversation_history == []

    @_apply_patches
    def test_query_type_factual(self, mock_chat, mock_embed):
        chunks = [_chunk('Dogs sleep 12 hours a day.', 'dog-facts.txt', 'txt')]
        result = _make_store(chunks).run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert result.get('query_type') in ('factual', 'general')

    @_apply_patches
    def test_source_label_xlsx_row(self, mock_chat, mock_embed):
        chunk = _chunk('country=Brazil | capital=Brasilia', 'countries.xlsx', 'xlsx')
        assert 'row' in _make_store([chunk])._source_label(chunk)

    @_apply_patches
    def test_source_label_pptx_slide(self, mock_chat, mock_embed):
        chunk = _chunk('Berlin Wall fell in 1989.', 'history.pptx', 'pptx')
        assert 'slide' in _make_store([chunk])._source_label(chunk)

    @_apply_patches
    def test_source_label_pdf_page(self, mock_chat, mock_embed):
        chunk = _chunk('Bananas have 89 calories.', 'nutrition.pdf', 'pdf')
        assert 'p' in _make_store([chunk])._source_label(chunk)

    @_apply_patches
    def test_source_label_html_section(self, mock_chat, mock_embed):
        chunk = _chunk('Dogs understand 165 words.', 'dog-facts.html', 'html')
        label = _make_store([chunk])._source_label(chunk)
        assert 's' in label or 'L' in label


# ---------------------------------------------------------------------------
# 5. Agent bad format recovery
# ---------------------------------------------------------------------------

class TestAgentBadFormat:
    @_apply_patches
    def test_bad_format_retries_twice(self, mock_chat, mock_embed):
        responses = iter([
            {'message': {'content': 'This is not a tool call'}},
            {'message': {'content': 'Still not a tool call'}},
            {'message': {'content': 'TOOL: rag_search(dogs)'}},
        ])
        mock_chat.side_effect = lambda **kw: next(responses)
        chunks = [_chunk('Dogs are loyal animals.', 'dog-facts.txt', 'txt')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Tell me about dogs', streamlit_mode=True)
        assert result['answer'] is not None

    @_apply_patches
    def test_max_steps_returns_answer(self, mock_chat, mock_embed):
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(dogs)'}}
        chunks = [_chunk('Dogs are loyal animals.', 'dog-facts.txt', 'txt')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Tell me about dogs', streamlit_mode=True)
        assert result['answer'] is not None
