"""Shared I/O helpers: transparent gzip/text opening and text normalization.

See docs/ARCHITECTURE.md section 5.2 for the normalize() rules.
"""

from __future__ import annotations

import contextlib
import gzip
import re
from typing import Iterator, TextIO


@contextlib.contextmanager
def open_text(path: str) -> Iterator[TextIO]:
    """Open a log file for streaming text reads.

    Transparently handles ``.gz`` files. Returns a context manager yielding
    an iterable-of-lines file object.
    """
    if str(path).endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        f = open(path, "rt", encoding="utf-8", errors="replace")
    try:
        yield f
    finally:
        f.close()


_PATH_RE = re.compile(r"/\S+")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8}\b")
_DEC_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Normalize a message/line for use as a clustering template.

    Order (per ARCHITECTURE 5.2): strip paths, then hex numbers (0x-prefixed
    or bare 8-digit hex), then decimal numbers, collapse whitespace, strip.
    """
    s = _PATH_RE.sub("", s)
    s = _HEX_RE.sub("<HEX>", s)
    s = _DEC_RE.sub("<N>", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()
