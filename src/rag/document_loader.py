"""document_loader.py — DocumentLoader class: owns all ingestion."""

import os
import re
import sys
import tempfile
from typing import List
from urllib.parse import urlparse

from src.rag.config import (
    DOCS_ROOT, DOC_FOLDERS, EXT_TO_TYPE,
    TXT_CHUNK_SIZE, TXT_CHUNK_OVERLAP,
    PDF_CHUNK_SENTENCES, DOCX_CHUNK_PARAS,
    PPTX_CHUNK_SLIDES, HTML_CHUNK_SENTENCES,
)
from src.rag import chunkers

__all__ = ['DocumentLoader']


class DocumentLoader:
    """Owns all document ingestion: reading files, parsing 9 formats, chunking, URL fetching.

    All format-specific chunking logic lives in chunkers.py as stateless
    functions. This class owns the state (folder paths, chunk sizes) and
    the orchestration: scanning files, dispatching to the right chunker,
    and fetching URLs.

    State:
        doc_folders:  dict mapping type → folder path (from config)
        ext_to_type:  dict mapping file extension → document type
        chunk_sizes:  dict of all chunk-size constants (txt, pdf, docx, pptx, html)

    Public API:
        ensure_folders()        — create ./docs subfolders if missing
        scan_all_files()        — find every file under DOCS_ROOT, flag misplaced ones
        chunk_all_documents()   — scan + dispatch all files → list of chunk dicts
        chunk_url(url)          — fetch a URL, detect type, return chunks
    """

    def __init__(self) -> None:
        """Bind config constants to instance state so they are easy to override in tests."""
        self.doc_folders = DOC_FOLDERS
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

    def ensure_folders(self) -> None:
        """Create DOCS_ROOT and all subfolders if they do not exist yet."""
        os.makedirs(DOCS_ROOT, exist_ok=True)
        for folder in self.doc_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"  Created folder: {folder}/")

    def scan_all_files(self) -> List[dict]:
        """Scan every subfolder under DOCS_ROOT and return a list of file info dicts.

        Files placed in the wrong subfolder are included but flagged as misplaced
        so they are still processed with a notice to the user.

        Returns:
            List of dicts with keys: filepath, filename, detected_type,
            found_in, canonical_dir, is_misplaced.
        """
        found = []
        for folder_type, folder_path in self.doc_folders.items():
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                ext          = os.path.splitext(fname)[1].lower()
                detected_type = self.ext_to_type.get(ext)
                if detected_type is None:
                    continue  # unsupported extension — skip silently
                filepath      = os.path.join(folder_path, fname)
                canonical_dir = self.doc_folders[detected_type]
                is_misplaced  = (folder_path != canonical_dir)
                found.append({
                    'filepath':      filepath,
                    'filename':      fname,
                    'detected_type': detected_type,
                    'found_in':      folder_path,
                    'canonical_dir': canonical_dir,
                    'is_misplaced':  is_misplaced,
                })
                if is_misplaced:
                    print(f"  [MISPLACED] '{fname}' found in '{folder_path}/' "
                          f"— detected as '{detected_type.upper()}', "
                          f"canonical folder is '{canonical_dir}/'. Processing anyway.")
        return found

    def chunk_all_documents(self) -> List[dict]:
        """Scan every file under ./docs/, route each to the correct chunker, and return all chunks.

        Files in the wrong folder are still processed (with a notice).
        Exits with code 1 if no supported documents are found.

        Returns:
            Flat list of all chunk dicts from every document.
        """
        print("\nLoading documents...")
        file_list = self.scan_all_files()

        if not file_list:
            print(f"\nNo supported documents found under '{DOCS_ROOT}/'")
            print("Supported types: PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, MD, HTML")
            print("Place files in the matching subfolder (or any subfolder — smart detection handles the rest).")
            sys.exit(1)

        all_chunks  = []
        type_counts = {}

        for file_info in file_list:
            t      = file_info['detected_type']
            chunks = self._dispatch_chunker(file_info)
            all_chunks.extend(chunks)
            type_counts[t] = type_counts.get(t, 0) + len(chunks)
            flag = " [MISPLACED — processed anyway]" if file_info['is_misplaced'] else ""
            print(f"  [{t.upper()}] '{file_info['filename']}': {len(chunks)} chunks{flag}")

        print("\n  Chunk summary by type:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t.upper():<8} {count} chunks")
        print(f"  Total: {len(all_chunks)} chunks\n")

        return all_chunks

    def chunk_url(self, url: str) -> List[dict]:
        """Fetch a URL and return chunks using the appropriate format handler.

        Type detection uses four priorities in strict order:
            1. Content-Type response header (most reliable).
            2. File extension in the URL path (strips query strings first).
            3. PDF magic bytes sniff (content[:4] == b'%PDF').
            4. Default to 'html' when no other signal is present.

        Args:
            url: Public HTTP/HTTPS URL to fetch.

        Returns:
            List of chunk dicts. Empty list on network error.
        """
        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed. pip install requests")
            return []

        url = url.strip()
        print(f"\n  Fetching URL: {url}")
        try:
            resp = requests.get(url, timeout=30, headers={
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

        # ── Priority 1: Content-Type header ──────────────────────────────
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

        # Fuzzy fallback for non-standard content-type values
        if dtype is None:
            if 'pdf'          in content_type: dtype = 'pdf'
            elif 'word'       in content_type: dtype = 'docx'
            elif 'excel'      in content_type or 'spreadsheet' in content_type: dtype = 'xlsx'
            elif 'powerpoint' in content_type or 'presentation' in content_type: dtype = 'pptx'
            elif 'csv'        in content_type: dtype = 'csv'

        # ── Priority 2: file extension in URL path ────────────────────────
        if dtype is None:
            parsed_path = urlparse(url).path.lower()
            clean_path  = re.sub(r'[?#].*$', '', parsed_path)
            ext         = os.path.splitext(clean_path)[1]
            dtype       = self.ext_to_type.get(ext)

        # ── Priority 3: PDF magic bytes ───────────────────────────────────
        if dtype is None and content[:4] == b'%PDF':
            dtype = 'pdf'

        # ── Priority 4: default to HTML ───────────────────────────────────
        if dtype is None:
            dtype = 'html'

        # ── Build source label from URL ───────────────────────────────────
        parsed      = urlparse(url)
        source_name = (parsed.netloc + parsed.path).rstrip('/')
        if parsed.query:
            source_name += '?' + parsed.query[:20] + ('...' if len(parsed.query) > 20 else '')
        if len(source_name) > 60:
            source_name = source_name[:57] + '...'
        if not source_name:
            source_name = url[:60]

        print(f"  Detected type: {dtype.upper()} (content-type: {content_type})")

        # ── Binary formats: write to temp file, chunk, delete ────────────
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
                chunks = self._dispatch_chunker(file_info)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if not chunks:
                print(f"  [WARNING] No chunks extracted from {dtype.upper()} at URL. "
                      f"File may be empty, password-protected, or corrupt.")
            return chunks

        # ── Text formats: decode bytes ────────────────────────────────────
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

        # ── HTML / webpage ────────────────────────────────────────────────
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

    # ----------------------------------------------------------------- Private

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
            # .xls uses xlrd (legacy binary format); .csv routed correctly if misplaced
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
