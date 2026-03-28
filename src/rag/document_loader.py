"""document_loader.py — DocumentLoader class: owns all ingestion."""

import os
import re
import sys
import tempfile
from typing import Callable, List, Optional, Set
from urllib.parse import urlparse

from src.rag.config import (
    DOCS_ROOT, DOC_FOLDERS, EXT_TO_TYPE,
    TXT_CHUNK_SIZE, TXT_CHUNK_OVERLAP,
    PDF_CHUNK_SENTENCES, DOCX_CHUNK_PARAS,
    PPTX_CHUNK_SLIDES, HTML_CHUNK_SENTENCES,
)
from src.rag import chunkers
from src.rag.url_utils import (
    detect_url_type, build_source_name, extract_links, url_matches_topic,
)

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
        ensure_folders()                          — create ./docs subfolders if missing
        scan_all_files()                          — find every file under DOCS_ROOT, flag misplaced ones
        chunk_all_documents()                     — scan + dispatch all files → list of chunk dicts
        chunk_directory(directory)                — chunk all files in any given folder (used for benchmarking)
        chunk_url(url)                            — fetch a single URL, detect type, return chunks
        chunk_url_recursive(url, depth,           — crawl a seed URL and all linked pages
                            max_pages,
                            allowed_types,
                            progress_callback)
    """

    def __init__(self) -> None:
        """Bind config constants to instance state so they are easy to override in tests."""
        self.docs_root   = DOCS_ROOT
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
        """Create docs_root and all subfolders if they do not exist yet."""
        os.makedirs(self.docs_root, exist_ok=True)
        for folder in self.doc_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"  Created folder: {folder}/")

    def scan_all_files(self) -> List[dict]:
        """Recursively scan all files under DOCS_ROOT and return a list of file info dicts.

        Walks the entire DOCS_ROOT tree at any depth, so a folder dropped anywhere
        under ./docs/ — with any mix of file types and any number of sublevels —
        will have all its files detected and processed.

        The relative path from DOCS_ROOT is stored as 'filename' so that two files
        with the same name in different subfolders can be told apart in source citations
        (e.g. 'project/data/q1.xlsx' instead of just 'q1.xlsx').

        Files not in their canonical type subfolder are flagged as misplaced but
        are still processed with a notice.

        Returns:
            List of dicts with keys: filepath, filename (relative path), detected_type,
            found_in, canonical_dir, is_misplaced.
        """
        found = []
        if not os.path.isdir(self.docs_root):
            return found

        for root, _, files in os.walk(self.docs_root):
            for fname in files:
                ext           = os.path.splitext(fname)[1].lower()
                detected_type = self.ext_to_type.get(ext)
                if detected_type is None:
                    continue  # unsupported extension — skip silently

                filepath      = os.path.join(root, fname)
                # Store path relative to docs_root so source citations are unique
                # across deeply nested folders (e.g. 'project/data/q1.xlsx')
                relative_path = os.path.relpath(filepath, self.docs_root)
                canonical_dir = self.doc_folders[detected_type]
                is_misplaced  = (root != canonical_dir)

                found.append({
                    'filepath':      filepath,
                    'filename':      relative_path,
                    'detected_type': detected_type,
                    'found_in':      root,
                    'canonical_dir': canonical_dir,
                    'is_misplaced':  is_misplaced,
                })
                if is_misplaced:
                    print(f"  [MISPLACED] '{relative_path}' "
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
            print(f"\nNo supported documents found under '{self.docs_root}/'")
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

    def chunk_directory(self, directory: str) -> List[dict]:
        """Chunk all supported files inside a given directory (flat, one level deep).

        Unlike chunk_all_documents() which is tied to the configured ./docs/ tree,
        this method accepts any arbitrary folder path. This is used by the benchmark
        runner to load sample documents from benchmark_docs/ without mixing them into
        the main document index.

        Files that have no supported extension are silently skipped.

        Args:
            directory: Absolute or relative path to the folder to scan.

        Returns:
            Flat list of chunk dicts from every supported file in the directory.
            Returns an empty list if the directory does not exist or has no files.
        """
        all_chunks = []

        if not os.path.isdir(directory):
            print(f"  [benchmark] Directory not found: {directory}")
            return all_chunks

        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)

            # Skip sub-directories — only process files at the top level
            if not os.path.isfile(filepath):
                continue

            extension     = os.path.splitext(filename)[1].lower()
            detected_type = self.ext_to_type.get(extension)

            # Skip file types we do not support
            if detected_type is None:
                continue

            file_info = {
                'filepath':      filepath,
                'filename':      filename,
                'detected_type': detected_type,
                'found_in':      directory,
                'canonical_dir': directory,
                'is_misplaced':  False,
            }

            chunks = self._dispatch_chunker(file_info)
            all_chunks.extend(chunks)
            print(f"  [benchmark/{detected_type.upper()}] '{filename}': {len(chunks)} chunks")

        return all_chunks

    def chunk_url(self, url: str) -> List[dict]:
        """Fetch a single URL and return chunks using the appropriate format handler.

        Type detection uses four priorities in strict order (via url_utils):
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
        except Exception as e:
            print(f"  [ERROR] Could not fetch URL: {e}")
            return []

        content_type = resp.headers.get('Content-Type', '')
        dtype        = detect_url_type(url, content, content_type, self.ext_to_type)
        source_name  = build_source_name(url)

        print(f"  Detected type: {dtype.upper()} (content-type: {content_type.split(';')[0].strip()})")

        # Decode text content for text-based formats
        text = None
        if dtype not in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            try:
                text = content.decode(resp.encoding or 'utf-8', errors='replace')
            except Exception:
                text = content.decode('utf-8', errors='replace')

        return self._chunk_content(content, text, dtype, source_name)

    def chunk_url_recursive(
        self,
        url: str,
        depth: int = 2,
        max_pages: int = 25,
        allowed_types: Optional[Set[str]] = None,
        topic_filter: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> List[dict]:
        """Crawl a seed URL and all linked pages up to a given depth.

        Follows links on HTML pages and chunks every discovered URL using the
        same type-detection and chunking pipeline as chunk_url(). Document URLs
        (PDF, DOCX, XLSX, etc.) are chunked but not recursed into — they do not
        contain links to follow.

        Args:
            url:               Seed URL to start crawling from.
            depth:             How many link-levels deep to follow (1 = direct
                               links only, 2 = links of links, etc.).
            max_pages:         Maximum total pages/documents to fetch and index.
            allowed_types:     Set of type strings to index, e.g. {'html', 'pdf'}.
                               If None, all detected types are indexed.
            topic_filter:      Optional keyword the URL path must contain to be
                               crawled (e.g. 'python', 'api'). Empty string means
                               no filter — all pages on the domain are crawled.
                               The seed URL itself is always crawled regardless.
            progress_callback: Optional function called after each page is fetched.
                               Signature: callback(url, detected_type, chunk_count).

        Returns:
            Flat list of all chunk dicts from every crawled URL.
        """
        visited: Set[str] = set()
        all_chunks: List[dict] = []

        self._crawl_url(
            url.strip(), depth, max_pages,
            allowed_types, topic_filter, visited, all_chunks, progress_callback,
            is_seed=True,
        )

        print(f"\n  [CRAWL] Finished — {len(visited)} pages crawled, "
              f"{len(all_chunks)} chunks total")
        return all_chunks

    # ----------------------------------------------------------------- Private

    def _chunk_content(
        self,
        content: bytes,
        text: Optional[str],
        dtype: str,
        source_name: str,
    ) -> List[dict]:
        """Route fetched URL content to the correct chunker based on detected type.

        Binary formats (pdf, docx, xlsx, pptx, xls) are written to a temporary
        file on disk so the existing file-based chunkers can read them. The
        temp file is always deleted after chunking, even if an error occurs.

        Text formats (csv, md) are also written to a temp file because the
        chunkers expect a filepath. Plain text and HTML are chunked directly
        from the decoded string without touching disk.

        Args:
            content:     Raw response bytes from the HTTP request.
            text:        Decoded string for text-based formats; None for binary.
            dtype:       Document type string ('pdf', 'html', 'txt', etc.).
            source_name: Short citation label built from the URL.

        Returns:
            List of chunk dicts. Empty list if chunking fails or type is unknown.
        """
        # Binary formats — write bytes to a temp file, chunk it, then delete
        binary_types = {'pdf', 'docx', 'xlsx', 'pptx', 'xls'}
        if dtype in binary_types:
            # Choose the right file extension so the chunker knows what format it is
            extension_map = {
                'pdf': '.pdf', 'docx': '.docx', 'xlsx': '.xlsx',
                'pptx': '.pptx', 'xls': '.xls',
            }
            suffix = extension_map[dtype]
            try:
                # Write to a named temp file that persists until we delete it
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                # Build a fake file_info dict so _dispatch_chunker can handle it
                file_info = {
                    'filepath':      tmp_path,
                    'filename':      source_name,
                    'detected_type': dtype,
                    'found_in':      '',
                    'canonical_dir': '',
                    'is_misplaced':  False,
                }
                return self._dispatch_chunker(file_info)
            except Exception as error:
                print(f"  [ERROR] Could not chunk {dtype.upper()} from URL: {error}")
                return []
            finally:
                # Always clean up the temp file — even if an exception was raised
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # CSV and Markdown — write decoded text to a temp file
        if dtype in ('csv', 'md'):
            suffix = '.csv' if dtype == 'csv' else '.md'
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False, mode='w', encoding='utf-8'
                ) as tmp:
                    tmp.write(text or '')
                    tmp_path = tmp.name

                file_info = {
                    'filepath':      tmp_path,
                    'filename':      source_name,
                    'detected_type': dtype,
                    'found_in':      '',
                    'canonical_dir': '',
                    'is_misplaced':  False,
                }
                return self._dispatch_chunker(file_info)
            except Exception as error:
                print(f"  [ERROR] Could not chunk {dtype.upper()} from URL: {error}")
                return []
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Plain text — chunk line by line directly from the decoded string
        if dtype == 'txt':
            return chunkers.chunk_txt_from_string(
                text or '', source_name,
                self.chunk_sizes['txt_chunk_size'],
                self.chunk_sizes['txt_chunk_overlap'],
            )

        # HTML / webpage — strip tags and chunk sentences directly from the string
        if dtype == 'html':
            return chunkers.chunk_html_from_string(
                text or '', source_name,
                self.chunk_sizes['html_chunk_sentences'],
            )

        # Unknown type — skip with a notice
        print(f"  [SKIP] No URL chunker for type '{dtype}' ({source_name})")
        return []

    def _crawl_url(
        self,
        url: str,
        depth: int,
        max_pages: int,
        allowed_types: Optional[Set[str]],
        topic_filter: str,
        visited: Set[str],
        all_chunks: List[dict],
        progress_callback: Optional[Callable],
        is_seed: bool = False,
    ) -> None:
        """Recursively fetch a URL and follow links up to the given depth.

        This is the internal engine behind chunk_url_recursive(). It mutates
        the shared `visited` set and `all_chunks` list so the caller can track
        progress across the full crawl without returning values at each level.

        Recursion stops when:
          - The URL has already been visited
          - The max_pages cap is reached
          - depth reaches 0
          - The URL does not match the topic filter (unless it is the seed URL)
          - The URL fails to fetch

        Document URLs (PDF, DOCX, etc.) are chunked but NOT recursed into —
        they do not contain HTML links to follow.

        Args:
            url:               Absolute URL to fetch.
            depth:             Remaining link levels to follow (decrements each call).
            max_pages:         Hard cap on total pages fetched across the whole crawl.
            allowed_types:     If set, only index pages whose detected type is in this set.
            topic_filter:      Keyword the URL path must contain. Ignored for the seed URL.
            visited:           Shared set of already-fetched URLs (mutated in place).
            all_chunks:        Shared list accumulating all chunk dicts (mutated in place).
            progress_callback: Optional callback called after each page is fetched.
                               Signature: callback(url, detected_type, chunk_count).
            is_seed:           True only for the very first (user-supplied) URL — the seed
                               is always fetched regardless of the topic filter.
        """
        # Stop if we have already visited this URL or hit the page cap
        if url in visited or len(visited) >= max_pages:
            return

        # Apply topic filter to all pages except the seed URL
        # (the seed is always crawled — the user explicitly chose it)
        if not is_seed and not url_matches_topic(url, topic_filter):
            return

        # Mark as visited before fetching to prevent parallel loops
        visited.add(url)

        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed. pip install requests")
            return

        # Fetch the page content
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }, allow_redirects=True, stream=True)
            response.raise_for_status()
            content = response.content
        except Exception as fetch_error:
            print(f"  [CRAWL] Could not fetch: {url} — {fetch_error}")
            return

        # Detect the document type using the same 4-priority pipeline as chunk_url()
        # Use the final URL after any HTTP redirects (e.g. example.com → www.example.com)
        # so that relative links on the page resolve correctly and domain checks work.
        final_url    = response.url if hasattr(response, 'url') else url
        content_type = response.headers.get('Content-Type', '')
        dtype        = detect_url_type(final_url, content, content_type, self.ext_to_type)
        source_name  = build_source_name(final_url)

        print(f"  [CRAWL] [{dtype.upper()}] {source_name}")

        # Decode text content for text-based formats
        text = None
        if dtype not in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            try:
                text = content.decode(response.encoding or 'utf-8', errors='replace')
            except Exception:
                text = content.decode('utf-8', errors='replace')

        # Only index this page if its type is in the allowed set (or no filter set)
        chunks: List[dict] = []
        if allowed_types is None or dtype in allowed_types:
            chunks = self._chunk_content(content, text, dtype, source_name)
            all_chunks.extend(chunks)

        # Call the progress callback so the UI can update its progress bar
        if progress_callback is not None:
            progress_callback(url, dtype, len(chunks))

        # Only follow links from HTML pages — PDFs and DOCX files have no links to crawl
        html_types = {'html'}
        if dtype not in html_types or depth <= 0 or text is None:
            return

        # Use final_url (post-redirect) as the base so relative links resolve correctly
        # and domain comparison uses the actual domain the browser landed on.
        links = extract_links(text, final_url, self.ext_to_type)

        # Recurse into each discovered link, reducing depth by one each level
        for link in links:
            if len(visited) >= max_pages:
                break
            self._crawl_url(
                link, depth - 1, max_pages,
                allowed_types, topic_filter, visited, all_chunks, progress_callback,
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
