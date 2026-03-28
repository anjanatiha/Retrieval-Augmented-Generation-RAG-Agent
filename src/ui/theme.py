"""ui/theme.py — CSS + style constants. No functions, no classes."""

__all__ = ['CSS', 'BADGE_CLASSES', 'CONFIDENCE_BADGE', 'AVATAR']

CSS: str = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ── Base ────────────────────────────────────────────────────────────────── */
html,body,[class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #f8fbfd;
    color: #0d2b45;
}
.stApp { background: #f8fbfd; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.rag-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.85rem;
    font-weight: 600;
    color: #1565a0;
    letter-spacing: -.02em;
    padding-left: .6rem;
    border-left: 4px solid #1565a0;
    margin-bottom: .3rem;
}
.rag-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    color: #7aafc8;
    margin-bottom: 1.5rem;
    padding-left: .6rem;
}

/* ── Chunk cards ─────────────────────────────────────────────────────────── */
.chunk {
    background: #ffffff;
    border: 1px solid #b3d4e8;
    border-radius: 6px;
    padding: .5rem .7rem;
    margin: .25rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    color: #4a8fa8;
}
.cs  { color: #1565a0; font-weight: 600; }
.src { color: #4a9fc4; }

/* ── Agent step cards ────────────────────────────────────────────────────── */
.step {
    background: #eef7ee;
    border: 1px solid #aed4bb;
    border-left: 3px solid #43a047;
    border-radius: 6px;
    padding: .5rem .7rem;
    margin: .2rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    color: #2e7d32;
}

/* ── Agent tools reference panel ─────────────────────────────────────────── */
.tools-panel {
    background: #eaf4fd;
    border-left: 4px solid #1565a0;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.82rem;
    line-height: 1.7;
}

/* ── Badges ──────────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .65rem;
    font-weight: 600;
    padding: .2rem .5rem;
    border-radius: 4px;
    margin: .1rem;
    letter-spacing: .02em;
}
.b-fact { background: #daeaf4; color: #1565a0; }
.b-comp { background: #e3f0f9; color: #0d47a1; }
.b-gen  { background: #ede7f6; color: #512da8; }
.b-ok   { background: #daeaf4; color: #1565a0; }
.b-warn { background: #fff8e1; color: #e65100; }

/* ── Sidebar stats ───────────────────────────────────────────────────────── */
.stat {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    color: #7aafc8;
    padding: .28rem 0;
    border-bottom: 1px solid #daeaf4;
    display: flex;
    justify-content: space-between;
}
.sv { color: #1565a0; font-weight: 600; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #1565a0 !important;
    color: #ffffff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    transition: background .15s ease !important;
}
.stButton > button:hover {
    background: #0d47a1 !important;
}

/* ── Sidebar background ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #f4f9fc !important;
    border-right: 1px solid #b3d4e8 !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: #daeaf4 !important; }

/* ── Mode radio ──────────────────────────────────────────────────────────── */
.stRadio > div {
    background: #eaf4fd;
    border-radius: 8px;
    padding: .3rem .6rem;
    border: 1px solid #b3d4e8;
}

/* ── Chat bubbles ────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #daeaf4;
    border-radius: 10px;
    margin: .3rem 0;
}
[data-testid="stChatMessageContent"] p { margin: 0; line-height: 1.6; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #b3d4e8 !important;
    border-radius: 8px !important;
    margin-bottom: .5rem;
}

/* ── Chat input ──────────────────────────────────────────────────────────── */
.stChatInputContainer textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #ffffff !important;
    border: 1px solid #b3d4e8 !important;
    color: #0d2b45 !important;
    border-radius: 8px !important;
}
.stChatInputContainer textarea:focus,
[data-testid="stChatInputContainer"]:focus-within,
input:focus,
textarea:focus {
    border-color: #1565a0 !important;
    box-shadow: 0 0 0 2px rgba(21,101,160,.2) !important;
    outline: none !important;
}

/* ── Text inputs (URL box, number inputs) ────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #ffffff !important;
    border: 1px solid #b3d4e8 !important;
    border-radius: 6px !important;
    color: #0d2b45 !important;
}

/* ── Global focus ring ───────────────────────────────────────────────────── */
*:focus { outline-color: #1565a0 !important; }
:root   { --primary-color: #1565a0 !important; }
</style>
"""

BADGE_CLASSES: dict = {
    'factual':    'b-fact',
    'comparison': 'b-comp',
    'general':    'b-gen',
    'summarise':  'b-gen',
}

CONFIDENCE_BADGE: dict = {
    True:  'b-ok',
    False: 'b-warn',
}

AVATAR: dict = {
    'user':      '🧑',
    'assistant': '💬',
    'agent':     '🤖',
}
