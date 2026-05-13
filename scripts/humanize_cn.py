#!/usr/bin/env python3
"""
Chinese AI Text Humanizer v2.0 — re-export shim.

This file is a backward-compatible re-export layer. All original imports
continue to work. Implementation lives in the _humanize_* sub-modules.

Sub-modules:
  _humanize_data.py      — Static lookup tables and config
  _humanize_cilin.py     — CiLin synonym expansion
  _humanize_strategies.py — Rewriting strategies (bigram, sentence rand, noise)
  _humanize_pipeline.py  — Text transformation pipeline (13 steps)
  _humanize_core.py      — Orchestration, best_of_n, CLI entry
"""

import os
import sys

# Ensure scripts/ is on path for sub-module imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from _humanize_core import (
    humanize,
    main,
    _estimate_source_aiscore,
    _compute_secondary_signal,
    _pick_lr_scene,
    _format_best_of_debug,
    DEFAULT_BEST_OF_N,
    DEFAULT_SECONDARY_WEIGHT,
    _USE_NOISE,
    _USE_STATS,
    _USE_CILIN,
)

from _humanize_data import (
    load_config,
    CONFIG,
    SCENES,
    WORD_SYNONYMS,
    PHRASE_REPLACEMENTS,
    PLAIN_REPLACEMENTS,
    REGEX_REPLACEMENTS,
    _AI_PATTERN_BLACKLIST,
    ACADEMIC_PRESERVE_WORDS,
    ACADEMIC_BLACKLIST_CANDIDATES,
    NOVEL_BLACKLIST_CANDIDATES,
    _CILIN_BLACKLIST,
    _CILIN_SOURCE_BLACKLIST,
    NOISE_EXPRESSIONS,
    NOISE_ACADEMIC_EXPRESSIONS,
    NOISE_ACADEMIC_CATEGORIES,
    _NARRATIVE_SAFE_CATEGORIES,
    _PARA_BOOST_ATTRIBUTION,
    _PARA_BOOST_SUBORDINATE,
    _PARA_BOOST_BARE_CONTINUATOR,
    _PARA_BOOST_REACTIONS,
    _PARA_INTERJECTION_NEUTRAL,
    _PARA_INTERJECTION_NOVEL,
    _LONGFORM_PARA_HEAD_MARKERS,
    _LONGFORM_STARTER_MARKERS,
    _filter_candidates_for_scene,
    _count_chinese_chars,
    pick_best_replacement,
    _compute_burstiness,
)

from _humanize_cilin import (
    _load_cilin,
    expand_with_cilin,
)

from _humanize_pipeline import (
    remove_three_part_structure,
    replace_phrases,
    merge_short_sentences,
    split_long_sentences,
    vary_paragraph_rhythm,
    reduce_punctuation,
    cap_transition_density,
    inject_sentence_particles,
    add_casual_expressions,
    shorten_paragraphs,
    diversify_vocabulary,
)

from _humanize_strategies import (
    reduce_high_freq_bigrams,
    _simple_synonym_pass,
    randomize_sentence_lengths,
    inject_noise_expressions,
    boost_para_cv_via_merge,
    boost_para_sent_len_cv,
    insert_short_interjection_paragraph,
    reduce_cross_para_3gram_repeat,
    _apply_longform_mutation_profile,
)

if __name__ == '__main__':
    main()
