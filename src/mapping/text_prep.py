"""
src/mapping/text_prep.py
--------------------------
ONE text-preparation path shared by both sides of the semantic comparison:
cluster/noise-unit behaviour narratives AND ATT&CK technique descriptions go
through `prepare_text` before they are embedded, so the two vectors live in
the same space under the same preprocessing. `PREP_VERSION` is part of the
embedding-cache key: change the preprocessing, bump the version, and cached
technique embeddings are rebuilt.
"""

import re
from typing import Any, Dict

PREP_VERSION = "prep-v1"
DEFAULT_MAX_CHARS = 2000        # ~ the useful window of 256-word-piece MiniLM-class models

# C0/C1 controls, zero-width and bidi-override characters (incl. U+202E RTLO).
_UNSAFE_RE = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏‪-‮⁦-⁩﻿]")
_CITATION_RE = re.compile(r"\(Citation:[^)]*\)")
_WS_RE = re.compile(r"\s+")


def prepare_text(text: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Deterministic cleaning for embedding input. NOT used for display."""
    if text is None:
        return ""
    t = str(text)
    t = _CITATION_RE.sub(" ", t)
    t = _UNSAFE_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t[:max_chars]


def technique_embedding_text(t: Dict[str, Any]) -> str:
    """The exact string embedded for one ATT&CK technique record."""
    name = t.get("full_name") or t.get("technique_name") or ""
    tactics = ", ".join(t.get("tactics") or []) or (t.get("tactic") or "")
    desc = prepare_text(t.get("description") or "", max_chars=1700)
    head = f"{name}."
    if tactics:
        head += f" Tactic: {tactics}."
    return prepare_text(f"{head} {desc}")


def has_unsafe_chars(text: Any) -> bool:
    """True if `text` contains control / zero-width / bidirectional-override characters
    (the same set `display_safe` makes visible). Used to flag file names such as RTLO tricks."""
    return bool(text) and _UNSAFE_RE.search(str(text)) is not None


def display_safe(text: Any, max_len: int = 0) -> str:
    """Make arbitrary log text safe to print to a terminal: bidirectional
    overrides / control characters are shown as visible \\uXXXX escapes
    instead of silently re-ordering or hiding what the analyst sees (an RTLO
    file name is itself a finding)."""
    if text is None:
        return ""
    out = []
    for ch in str(text):
        if _UNSAFE_RE.match(ch) and ch not in "\t":
            out.append("\\u%04x" % ord(ch))
        elif ch in "\r\n":
            out.append(" ")
        else:
            out.append(ch)
    s = "".join(out)
    if max_len and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s
