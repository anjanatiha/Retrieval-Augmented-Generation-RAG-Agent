"""config.py — constants only, no functions, no classes."""

import os

__all__ = [
    'EMBEDDING_MODEL', 'LANGUAGE_MODEL',
    'DOCS_ROOT', 'DOC_FOLDERS', 'EXT_TO_TYPE',
    'CHROMA_DIR', 'CHROMA_COLLECTION', 'LOG_FILE', 'BENCHMARK_FILE',
    'SIMILARITY_THRESHOLD', 'TOP_RETRIEVE', 'TOP_RERANK',
    'TXT_CHUNK_SIZE', 'TXT_CHUNK_OVERLAP', 'PDF_CHUNK_SENTENCES',
    'DOCX_CHUNK_PARAS', 'PPTX_CHUNK_SLIDES', 'HTML_CHUNK_SENTENCES',
]

EMBEDDING_MODEL      = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL       = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'

# Root docs folder
DOCS_ROOT            = './docs'

# Subfolders — canonical home for each type
DOC_FOLDERS = {
    'pdf':  os.path.join(DOCS_ROOT, 'pdfs'),
    'txt':  os.path.join(DOCS_ROOT, 'txts'),
    'docx': os.path.join(DOCS_ROOT, 'docx'),
    'xlsx': os.path.join(DOCS_ROOT, 'xlsx'),
    'pptx': os.path.join(DOCS_ROOT, 'pptx'),
    'csv':  os.path.join(DOCS_ROOT, 'csv'),
    'md':   os.path.join(DOCS_ROOT, 'md'),
    'html': os.path.join(DOCS_ROOT, 'html'),
}

# Maps every supported file extension → canonical type key
EXT_TO_TYPE = {
    '.pdf':      'pdf',
    '.txt':      'txt',
    '.docx':     'docx',
    '.doc':      'docx',
    '.xlsx':     'xlsx',
    '.xls':      'xlsx',
    '.pptx':     'pptx',
    '.ppt':      'pptx',
    '.csv':      'csv',
    '.md':       'md',
    '.markdown': 'md',
    '.html':     'html',
    '.htm':      'html',
}

CHROMA_DIR           = './chroma_db'
CHROMA_COLLECTION    = 'rag_docs'
LOG_FILE             = 'rag_logs.json'
BENCHMARK_FILE       = 'benchmark_results.json'
SIMILARITY_THRESHOLD = 0.40
TOP_RETRIEVE         = 20
TOP_RERANK           = 5
# TXT / MD: 1 line per chunk — original behaviour, do not change
TXT_CHUNK_SIZE       = 1
TXT_CHUNK_OVERLAP    = 0
# New format chunk sizes — tuned per type
PDF_CHUNK_SENTENCES  = 5
DOCX_CHUNK_PARAS     = 3
PPTX_CHUNK_SLIDES    = 1
HTML_CHUNK_SENTENCES = 5
