"""document_loader.py — DocumentLoader class: owns all ingestion. HF Space version."""

import os
import re
import tempfile
from typing import Callable, List, Optional, Set
from urllib.parse import urlparse

from src.rag.config import (
    EXT_TO_TYPE,
    TXT_CHUNK_SIZE, TXT_CHUNK_OVERLAP,
    PDF_CHUNK_SENTENCES, DOCX_CHUNK_PARAS,
    PPTX_CHUNK_SLIDES, HTML_CHUNK_SENTENCES,
)
from src.rag import chunkers
from src.rag.url_utils import detect_url_type, build_source_name, extract_links

__all__ = ['DocumentLoader']


class DocumentLoader:
    """Owns all document ingestion — chunkers, URL fetching.

    All format-specific chunking logic lives in chunkers.py as stateless
    functions. This class owns the state (extension map, chunk sizes) and
    the orchestration: detecting document type, dispatching to chunkers,
    and fetching URLs.

    State:
        ext_to_type (dict): Maps file extension strings to canonical type keys.
        chunk_sizes (dict): All chunk-size constants keyed by config name.

    Public API:
        chunk_url(url)          — Fetch a URL and route to the correct chunker.
        _dispatch_chunker(info) — Pick and call the right chunker function.
    """

    def __init__(self) -> None:
        """Load config constants as instance vars so they are easy to override in tests."""
        self.ext_to_type = EXT_TO_TYPE
        self.chunk_sizes = {
            'txt_chunk_size':        TXT_CHUNK_SIZE,
            'txt_chunk_overlap':     TXT_CHUNK_OVERLAP,
            'pdf_chunk_sentences':   PDF_CHUNK_SENTENCES,
            'docx_chunk_paras':      DOCX_CHUNK_PARAS,
            'pptx_chunk_slides':     PPTX_CHUNK_SLIDES,
            'html_chunk_sentences':  HTML_CHUNK_SENTENCES,
        }

    # ------------------------------------------------------------------ Public

    def chunk_url(self, url: str) -> List[dict]:
        """Fetch a URL and produce chunks using the appropriate format handler.

        Type detection uses four priorities in strict order:
            1. Content-Type response header (most reliable).
            2. File extension in the URL path (strips query strings first).
            3. PDF magic bytes sniff (content[:4] == b'%PDF').
            4. Default to 'html' when no other signal is present.

        Args:
            url: Public HTTP/HTTPS URL to fetch.

        Returns:
            List of chunk dicts, each with keys: text, source, start_line,
            end_line, type. Returns an empty list on network error.
        """
        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed.")
            return []

        url = url.strip()
        try:
            resp = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }, allow_redirects=True, stream=True)
            resp.raise_for_status()
            content = resp.content
            text    = None
        except Exception as e:
            print(f"  [ERROR] Could not fetch URL: {e}")
            return []

        # ── Priority 1: exact Content-Type header match ──────────────
        content_type = resp.headers.get('Content-Type', '').lower().split(';')[0].strip()
        dtype        = None

        CONTENT_TYPE_MAP = {
            'application/pdf':                                          'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword':                                       'docx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/vnd.ms-excel':                                 'xlsx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            'application/vnd.ms-powerpoint':                            'pptx',
            'text/csv':                                                 'csv',
            'text/plain':                                               'txt',
            'text/markdown':                                            'md',
            'text/html':                                                'html',
            'application/xhtml+xml':                                    'html',
        }
        dtype = CONTENT_TYPE_MAP.get(content_type)

        # Fuzzy fallback for unusual MIME subtypes (e.g. "application/x-pdf")
        if dtype is None:
            if 'pdf'          in content_type: dtype = 'pdf'
            elif 'word'       in content_type: dtype = 'docx'
            elif 'excel'      in content_type or 'spreadsheet' in content_type: dtype = 'xlsx'
            elif 'powerpoint' in content_type or 'presentation' in content_type: dtype = 'pptx'
            elif 'csv'        in content_type: dtype = 'csv'

        # ── Priority 2: file extension in URL path ───────────────────
        if dtype is None:
            parsed_path = urlparse(url).path.lower()
            clean_path  = re.sub(r'[?#].*$', '', parsed_path)
            ext         = os.path.splitext(clean_path)[1]
            dtype       = self.ext_to_type.get(ext)

        # ── Priority 3: PDF magic bytes ───────────────────────────────
        if dtype is None and content[:4] == b'%PDF':
            dtype = 'pdf'

        # ── Priority 4: default to HTML ───────────────────────────────
        if dtype is None:
            dtype = 'html'

        # ── Build source label from URL ───────────────────────────────
        parsed      = urlparse(url)
        source_name = (parsed.netloc + parsed.path).rstrip('/')
        if parsed.query:
            source_name += '?' + parsed.query[:20] + ('...' if len(parsed.query) > 20 else '')
        if len(source_name) > 60:
            source_name = source_name[:57] + '...'
        if not source_name:
            source_name = url[:60]

        # ── Binary formats: write to temp file, chunk, delete ────────
        if dtype in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            suffix = {'pdf': '.pdf', 'docx': '.docx', 'xlsx': '.xlsx',
                      'pptx': '.pptx', 'xls': '.xls'}[dtype]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                file_info = {
                    'filepath':      tmp_path,
                    'filename':      source_name,
                    'detected_type': dtype,
                    'is_misplaced':  False,
                }
                result = self._dispatch_chunker(file_info)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return result

        try:
            text = content.decode(resp.encoding or 'utf-8', errors='replace')
        except Exception:
            text = content.decode('utf-8', errors='replace')

        if dtype == 'csv':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w',
                                             encoding='utf-8', errors='replace') as tmp:
                tmp.write(text)
                tmp_path = tmp.name
            try:
                result = chunkers.chunk_csv(tmp_path, source_name)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return result

        if dtype == 'txt':
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return [{'text': line, 'source': source_name, 'start_line': i + 1,
                     'end_line': i + 1, 'type': 'txt'} for i, line in enumerate(lines)]

        if dtype == 'md':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.md', mode='w',
                                             encoding='utf-8', errors='replace') as tmp:
                tmp.write(text)
                tmp_path = tmp.name
            try:
                result = chunkers.chunk_md(tmp_path, source_name,
                                           self.chunk_sizes['txt_chunk_size'],
                                           self.chunk_sizes['txt_chunk_overlap'])
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return result

        # ── HTML / webpage ────────────────────────────────────────────
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(text, 'html.parser').get_text(separator=' ', strip=True)
        except ImportError:
            text = re.sub(r'<[^>]+>', ' ', text)

        sents  = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        result = []
        for i in range(0, len(sents), HTML_CHUNK_SENTENCES):
            window = sents[i: i + HTML_CHUNK_SENTENCES]
            if window:
                result.append({'text': ' '.join(window), 'source': source_name,
                                'start_line': i + 1, 'end_line': i + len(window),
                                'type': 'html'})
        return result

    def chunk_url_recursive(
        self,
        url: str,
        depth: int = 1,
        max_pages: int = 10,
        allowed_types: Optional[Set[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[dict]:
        """Crawl a seed URL and all linked pages up to a given depth.

        Depth and max_pages are intentionally small on the HF Space (default 1/10)
        because the free-tier CPU is shared and slow. Users can still increase them
        from the UI if they choose.

        Args:
            url:               Seed URL to start crawling from.
            depth:             How many link-levels deep to follow.
            max_pages:         Maximum total pages/documents to fetch and index.
            allowed_types:     Set of type strings to index, or None for all.
            progress_callback: Optional callback(url, dtype, chunk_count) after each page.

        Returns:
            Flat list of all chunk dicts from every crawled URL.
        """
        visited: Set[str] = set()
        all_chunks: List[dict] = []

        self._crawl_url(
            url.strip(), depth, max_pages,
            allowed_types, visited, all_chunks, progress_callback,
        )

        print(f"\n  [CRAWL] Finished — {len(visited)} pages crawled, "
              f"{len(all_chunks)} chunks total")
        return all_chunks

    # ----------------------------------------------------------------- Private

    def _crawl_url(
        self,
        url: str,
        depth: int,
        max_pages: int,
        allowed_types: Optional[Set[str]],
        visited: Set[str],
        all_chunks: List[dict],
        progress_callback: Optional[Callable],
    ) -> None:
        """Recursively fetch a URL and follow links up to the given depth.

        See chunk_url_recursive() for full documentation. This internal helper
        mutates the shared visited set and all_chunks list rather than returning
        values, so progress is accumulated naturally across all recursive calls.

        Args:
            url:               Absolute URL to fetch.
            depth:             Remaining link levels to follow.
            max_pages:         Hard cap on total pages fetched.
            allowed_types:     If set, only index pages whose type is in this set.
            visited:           Shared set of already-fetched URLs (mutated in place).
            all_chunks:        Shared accumulator for all chunks (mutated in place).
            progress_callback: Optional callback after each page.
        """
        if url in visited or len(visited) >= max_pages:
            return

        visited.add(url)

        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed.")
            return

        try:
            response = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }, allow_redirects=True, stream=True)
            response.raise_for_status()
            content = response.content
        except Exception as fetch_error:
            print(f"  [CRAWL] Could not fetch: {url} — {fetch_error}")
            return

        content_type = response.headers.get('Content-Type', '')
        dtype        = detect_url_type(url, content, content_type, self.ext_to_type)
        source_name  = build_source_name(url)

        print(f"  [CRAWL] [{dtype.upper()}] {source_name}")

        # Decode text content for text-based formats
        text = None
        if dtype not in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            try:
                text = content.decode(response.encoding or 'utf-8', errors='replace')
            except Exception:
                text = content.decode('utf-8', errors='replace')

        # Only index if type passes the filter
        chunks: List[dict] = []
        if allowed_types is None or dtype in allowed_types:
            # Reuse chunk_url for a single page — avoids duplicating chunking logic
            chunks = self.chunk_url(url)
            all_chunks.extend(chunks)

        if progress_callback is not None:
            progress_callback(url, dtype, len(chunks))

        # Only recurse from HTML pages — documents have no crawlable links
        if dtype != 'html' or depth <= 0 or text is None:
            return

        links = extract_links(text, url, self.ext_to_type)
        for link in links:
            if len(visited) >= max_pages:
                break
            self._crawl_url(
                link, depth - 1, max_pages,
                allowed_types, visited, all_chunks, progress_callback,
            )

    def _dispatch_chunker(self, file_info: dict) -> List[dict]:
        """Route a file_info dict to the correct chunker function in chunkers.py.

        Args:
            file_info: Dict with keys filepath, filename, detected_type, is_misplaced.

        Returns:
            List of chunk dicts. Empty list when the type is unrecognised.
        """
        fp  = file_info['filepath']
        fn  = file_info['filename']
        ext = os.path.splitext(fn)[1].lower()
        t   = file_info['detected_type']

        if t == 'txt':
            return chunkers.chunk_txt(fp, fn,
                                      self.chunk_sizes['txt_chunk_size'],
                                      self.chunk_sizes['txt_chunk_overlap'])
        elif t == 'md':
            return chunkers.chunk_md(fp, fn,
                                     self.chunk_sizes['txt_chunk_size'],
                                     self.chunk_sizes['txt_chunk_overlap'])
        elif t == 'pdf':
            return chunkers.chunk_pdf(fp, fn,
                                      self.chunk_sizes['pdf_chunk_sentences'])
        elif t == 'docx':
            return chunkers.chunk_docx(fp, fn,
                                       self.chunk_sizes['docx_chunk_paras'])
        elif t == 'xlsx':
            if ext == '.xls':
                return chunkers.chunk_xls(fp, fn)
            elif ext == '.csv':
                return chunkers.chunk_csv(fp, fn)
            else:
                return chunkers.chunk_xlsx(fp, fn)
        elif t == 'csv':
            return chunkers.chunk_csv(fp, fn)
        elif t == 'pptx':
            return chunkers.chunk_pptx(fp, fn,
                                       self.chunk_sizes['pptx_chunk_slides'])
        elif t == 'html':
            return chunkers.chunk_html(fp, fn,
                                       self.chunk_sizes['html_chunk_sentences'])
        else:
            print(f"  [SKIP] No chunker for type '{t}' ({fn})")
            return []
