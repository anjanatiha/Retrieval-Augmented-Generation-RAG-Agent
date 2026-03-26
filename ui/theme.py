"""ui/theme.py — CSS + style constants. No functions, no classes."""

__all__ = ['CSS', 'BADGE_CLASSES', 'CONFIDENCE_BADGE', 'AVATAR']

CSS: str = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#ffffff;color:#0d2b45}
.stApp{background:#ffffff}
.rag-title{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;color:#1565a0;letter-spacing:-.02em}
.rag-sub{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#7aafc8;margin-bottom:1.5rem}
.chunk{background:#ffffff;border:1px solid #b3d4e8;border-radius:6px;padding:.5rem .7rem;margin:.25rem 0;font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:#4a8fa8}
.cs{color:#1565a0;font-weight:600}.src{color:#4a9fc4}
.step{background:#e8f5e8;border:1px solid #aed4bb;border-radius:6px;padding:.5rem .7rem;margin:.2rem 0;font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:#2e7d32}
.badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.65rem;padding:.15rem .4rem;border-radius:3px;margin:.1rem}
.b-fact{background:#daeaf4;color:#1565a0}.b-comp{background:#e3f0f9;color:#0d47a1}.b-gen{background:#ede7f6;color:#512da8}
.b-ok{background:#daeaf4;color:#1565a0}.b-warn{background:#fff8e1;color:#f57f17}
.stat{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#7aafc8;padding:.25rem 0;border-bottom:1px solid #daeaf4;display:flex;justify-content:space-between}
.sv{color:#1565a0;font-weight:600}
.stButton>button{background:#1565a0!important;color:#ffffff!important;font-family:'IBM Plex Mono',monospace!important;font-weight:600!important;border:none!important;border-radius:6px!important}
[data-testid="stSidebar"]{background:#f4f9fc!important;border-right:1px solid #b3d4e8!important}
hr{border-color:#daeaf4!important}
.stRadio>div{background:#f4f9fc;border-radius:6px;padding:.3rem .5rem}
/* chat bubbles */
[data-testid="stChatMessage"]{background:#f4f9fc;border-radius:10px;margin:.3rem 0}
[data-testid="stChatMessageContent"] p{margin:0;line-height:1.6}
.stChatInputContainer textarea{font-family:'IBM Plex Mono',monospace!important;background:#f4f9fc!important;border:1px solid #b3d4e8!important;color:#0d2b45!important}
.stChatInputContainer textarea:focus{border-color:#1565a0!important;box-shadow:0 0 0 2px rgba(21,101,160,0.25)!important;outline:none!important}
.stChatInputContainer:focus-within{border-color:#1565a0!important;box-shadow:0 0 0 2px rgba(21,101,160,0.25)!important}
[data-testid="stChatInputContainer"]:focus-within{border-color:#1565a0!important;box-shadow:0 0 0 2px rgba(21,101,160,0.25)!important}
input:focus, textarea:focus{border-color:#1565a0!important;box-shadow:0 0 0 2px rgba(21,101,160,0.25)!important;outline:none!important}
*:focus{outline-color:#1565a0!important}
:root{--primary-color:#1565a0!important}
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
