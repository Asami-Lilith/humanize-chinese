#!/usr/bin/env python3
"""CiLin (Cilin) synonym expansion subsystem.

Harbin Institute of Technology CiLin extended synonym dictionary
(~40K words, offline). Opt-in via --cilin CLI flag.
"""

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from _humanize_data import (
    _CILIN_BLACKLIST, _CILIN_SOURCE_BLACKLIST, _USE_CILIN,
    _AI_PATTERN_BLACKLIST, ACADEMIC_BLACKLIST_CANDIDATES, NOVEL_BLACKLIST_CANDIDATES,
)

_CILIN_CACHE = None
_CILIN_FILE = os.path.join(SCRIPT_DIR, 'data/cilin_synonyms.json')


def _load_cilin():
    """Lazy-load filtered CiLin synonyms. Returns dict[word] -> list[candidate] or empty dict."""
    global _CILIN_CACHE
    if _CILIN_CACHE is not None:
        return _CILIN_CACHE
    if not os.path.exists(_CILIN_FILE):
        _CILIN_CACHE = {}
        return _CILIN_CACHE
    try:
        with open(_CILIN_FILE, 'r', encoding='utf-8') as f:
            _CILIN_CACHE = json.load(f)
    except (json.JSONDecodeError, OSError):
        _CILIN_CACHE = {}
    return _CILIN_CACHE


def expand_with_cilin(word, candidates, scene='general'):
    """Expand a candidate list with CiLin synonyms (filtered through blacklists).

    Only used when enabled via --cilin CLI flag. CiLin has ~40K words vs the
    hand-curated ~200 in WORD_SYNONYMS, so expansion gives much more variety —
    but CiLin's "synonym" relation is loose (taxonomic, not strictly substitutable)
    and contains archaic/idiomatic candidates. Always filter through scene blacklist.
    """
    cilin = _load_cilin()
    extras = cilin.get(word, [])
    if not extras:
        return candidates
    existing = set(candidates)
    filtered = []
    for c in extras:
        if c in existing:
            continue
        if c in _AI_PATTERN_BLACKLIST:
            continue
        if c in _CILIN_BLACKLIST:
            continue
        if scene == 'academic' and c in ACADEMIC_BLACKLIST_CANDIDATES:
            continue
        if scene == 'novel' and c in NOVEL_BLACKLIST_CANDIDATES:
            continue
        filtered.append(c)
        existing.add(c)
    return list(candidates) + filtered
