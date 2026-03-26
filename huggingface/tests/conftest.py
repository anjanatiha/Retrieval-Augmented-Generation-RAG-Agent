"""conftest.py — global fixtures for HF Space test suite.

Patches sentence-transformers and _llm_call globally so tests don't need
sentence_transformers installed locally and don't hit the HF Inference API.
"""

import hashlib
import random
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

FAKE_DIM = 384


class _FakeVector(list):
    def tolist(self):
        return list(self)


def _fake_embed(text, normalize_embeddings=True):
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(FAKE_DIM)]
    norm = sum(x**2 for x in vec) ** 0.5 or 1.0
    return _FakeVector(x / norm for x in vec)


def _mock_st_model():
    m = MagicMock()
    m.encode.side_effect = _fake_embed
    return m


@pytest.fixture(autouse=True)
def patch_st_model():
    """Patch sentence-transformers globally for every test."""
    with patch('src.rag.vector_store._get_st_model', return_value=_mock_st_model()):
        yield


@pytest.fixture(autouse=True)
def patch_llm_call(request):
    """Default _llm_call to return a safe mock string for every test.
    Tests in TestLlmCallFallback are excluded so they test the real function."""
    if request.node.cls and request.node.cls.__name__ == 'TestLlmCallFallback':
        yield   # no patch — let the real _llm_call run
        return
    with patch('src.rag.vector_store._llm_call', return_value="mock rewrite\nalternative rewrite"):
        yield


@pytest.fixture(autouse=True)
def unique_chroma_collection(monkeypatch):
    """Use a unique ChromaDB collection name per test to prevent cross-test state."""
    import uuid
    monkeypatch.setattr('src.rag.vector_store.CHROMA_COLLECTION', f'test_{uuid.uuid4().hex}')
    # Also patch config so new VectorStore instances pick up the unique name
    yield
