"""logger.py — stateless log functions."""

import json
import os
from datetime import datetime
from typing import List

from src.rag.config import LOG_FILE

__all__ = ['log_interaction']


def log_interaction(query: str, qtype: str, chunks_used: int,
                    sim_scores: list, response: str) -> None:
    """Append one interaction entry to the JSON log file."""
    entry = {
        'timestamp':       datetime.now().isoformat(),
        'query':           query,
        'query_type':      qtype,
        'chunks_used':     chunks_used,
        'top_similarity':  round(sim_scores[0], 4) if sim_scores else 0,
        'avg_similarity':  round(sum(sim_scores)/len(sim_scores), 4) if sim_scores else 0,
        'response_length': len(response),
    }
    logs = _read_log()
    logs.append(entry)
    _write_log(logs)


def _read_log() -> List[dict]:
    """Read the log file, returning [] on any error."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def _write_log(entries: list) -> None:
    """Write entries to the log file as indented JSON."""
    with open(LOG_FILE, 'w') as f:
        json.dump(entries, f, indent=2)
