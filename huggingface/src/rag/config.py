"""config.py — constants only, no functions, no classes. HF Space version."""

__all__ = [
    'EMBEDDING_MODEL', 'LANGUAGE_MODEL',
    'CHROMA_COLLECTION',
    'EXT_TO_TYPE',
    'SIMILARITY_THRESHOLD', 'TOP_RETRIEVE', 'TOP_RERANK',
    'TXT_CHUNK_SIZE', 'TXT_CHUNK_OVERLAP', 'PDF_CHUNK_SENTENCES',
    'DOCX_CHUNK_PARAS', 'PPTX_CHUNK_SLIDES', 'HTML_CHUNK_SENTENCES',
]

# sentence-transformers model (runs locally in the Space)
EMBEDDING_MODEL      = 'BAAI/bge-base-en-v1.5'
# HF Serverless Inference API model
LANGUAGE_MODEL       = 'Qwen/Qwen2.5-3B-Instruct'

CHROMA_COLLECTION    = 'rag_docs'
SIMILARITY_THRESHOLD = 0.40
TOP_RETRIEVE         = 20
TOP_RERANK           = 5

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

# TXT / MD: 1 line per chunk
TXT_CHUNK_SIZE       = 1
TXT_CHUNK_OVERLAP    = 0
# Format chunk sizes — tuned per type
PDF_CHUNK_SENTENCES  = 5
DOCX_CHUNK_PARAS     = 3
PPTX_CHUNK_SLIDES    = 1
HTML_CHUNK_SENTENCES = 5
