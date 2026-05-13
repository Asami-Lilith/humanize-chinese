#!/usr/bin/env python3
"""Rewriting strategy engine for Chinese AI Text Humanizer v2.0.

Strategy 1: Low-frequency bigram injection (reduce_high_freq_bigrams)
Strategy 2: Sentence-length randomization (randomize_sentence_lengths)
Strategy 3: Noise expression injection (inject_noise_expressions)
Paragraph CV boosting, longform mutations, cross-paragraph trigram reduction.
"""

import sys
import re
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys

from _humanize_data import (
    _USE_NOISE, _USE_STATS, _USE_CILIN,
    WORD_SYNONYMS,
    _CILIN_BLACKLIST, _CILIN_SOURCE_BLACKLIST,
    _AI_PATTERN_BLACKLIST,
    ACADEMIC_BLACKLIST_CANDIDATES, NOVEL_BLACKLIST_CANDIDATES,
    _filter_candidates_for_scene,
    NOISE_EXPRESSIONS, NOISE_ACADEMIC_EXPRESSIONS, NOISE_ACADEMIC_CATEGORIES,
    _NARRATIVE_SAFE_CATEGORIES,
    _PARA_BOOST_ATTRIBUTION, _PARA_BOOST_SUBORDINATE, _PARA_BOOST_BARE_CONTINUATOR,
    _PARA_BOOST_REACTIONS,
    _LONGFORM_PARA_HEAD_MARKERS, _LONGFORM_STARTER_MARKERS,
    _PARA_INTERJECTION_NEUTRAL, _PARA_INTERJECTION_NOVEL,
    pick_best_replacement, _compute_burstiness,
)
from _humanize_cilin import expand_with_cilin, _load_cilin

try:
    from _text_utils import join_paragraphs, split_paragraphs
except ImportError:
    from scripts._text_utils import join_paragraphs, split_paragraphs

from _humanize_pipeline import _para_cv, vary_paragraph_rhythm
# Lazy imports: _humanize_data, _humanize_cilin, ngram_model (fallback chain)
_ngram_freq = None
_HUMANIZE_DATA = None

def _get_data():
    global _HUMANIZE_DATA
    if _HUMANIZE_DATA is None:
        try:
            import _humanize_data as _HUMANIZE_DATA
        except ImportError:
            import scripts._humanize_data as _HUMANIZE_DATA
    return _HUMANIZE_DATA

def _load_bigram_freq():
    """Load bigram frequencies from the n-gram frequency table."""
    try:
        from ngram_model import _load_freq
    except ImportError:
        try:
            from scripts.ngram_model import _load_freq
        except ImportError:
            return {}
    freq = _load_freq()
    return freq.get('bigrams', {})
def reduce_high_freq_bigrams(text, strength=0.3, scene='general'):
    """
    策略1: 扫描文本中的高频 bigram，尝试用低频同义替换降低可预测性。
    strength: 0-1，控制替换比例。
    scene: 'general' / 'academic' / 'social' —
      - academic: 跳过 ACADEMIC_PRESERVE_WORDS，候选过 ACADEMIC_BLACKLIST_CANDIDATES

    使用基于词的替换（非位置），避免长度变化导致的错位问题。
    """
    bigram_freq = _load_bigram_freq()
    if not bigram_freq:
        return _simple_synonym_pass(text, strength, scene=scene)

    chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chars) < 4:
        return text

    preserve = ACADEMIC_PRESERVE_WORDS if scene == 'academic' else set()

    # Step 1: Score each WORD_SYNONYMS word by its surrounding bigram frequency
    word_scores = []  # (word, total_bigram_freq, count_in_text)
    for word in WORD_SYNONYMS:
        if word in preserve:
            continue
        count = text.count(word)
        if count == 0:
            continue
        # Compute bigram frequency of this word's characters
        word_chars = re.findall(r'[\u4e00-\u9fff]', word)
        total_freq = 0
        for i in range(len(word_chars) - 1):
            bg = word_chars[i] + word_chars[i + 1]
            total_freq += bigram_freq.get(bg, 0)
        word_scores.append((word, total_freq, count))

    if not word_scores:
        return text

    # Step 2: Sort by bigram frequency (highest first)
    word_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Replace top N unique words (controlled by strength)
    n_replace = max(1, int(len(word_scores) * strength))
    replaced_words = set()

    for word, freq_score, count in word_scores[:n_replace]:
        if word in replaced_words:
            continue

        candidates = _filter_candidates_for_scene(word, WORD_SYNONYMS[word], scene)
        if _USE_CILIN:
            candidates = expand_with_cilin(word, candidates, scene)

        # Rank candidates by bigram frequency ascending (rarest first)
        ranked = []
        for candidate in candidates:
            cand_chars = re.findall(r'[\u4e00-\u9fff]', candidate)
            if not cand_chars:
                continue
            total_f = 0
            for i in range(len(cand_chars) - 1):
                total_f += bigram_freq.get(cand_chars[i] + cand_chars[i + 1], 0)
            ranked.append((candidate, total_f))
        if not ranked:
            continue
        ranked.sort(key=lambda x: x[1])

        # Pick strategy: NOT the rarest (too weird, e.g. 施用/拉高/本事),
        # but moderately rare — lower third by bigram frequency when possible.
        n_cand = len(ranked)
        if n_cand == 1:
            primary = ranked[0][0]
        elif n_cand == 2:
            primary = ranked[0][0]
        else:
            idx = min(max(1, n_cand // 3), n_cand - 2)
            primary = ranked[idx][0]

        # Partial replacement: don't replace EVERY occurrence of the word.
        # Replacing all creates NEW AI-pattern repetition (e.g. "系统"×6 → "架构"×6).
        # Keep some original occurrences + mix in alternative candidates for variation.
        SENTINEL = '\x00'

        def _protect(w):
            return SENTINEL.join(w) if len(w) > 1 else w

        occurrences = [m.start() for m in re.finditer(re.escape(word), text)]
        if not occurrences:
            continue
        # Replace ~60% of occurrences (min 1, always at least the first)
        n_replace_occ = max(1, int(len(occurrences) * 0.6))
        # Randomly select which occurrences to replace (deterministic via current seed)
        to_replace = set(random.sample(range(len(occurrences)), n_replace_occ))

        # Protection guard: skip occurrences inside domain terms.
        # Must be computed from current text because prior word replacements
        # shift character positions within this same reduce_high_freq_bigrams call.
        import _humanize_data as _hd
        if _hd._USE_PROTECT_FLAG and _hd._PROTECTION_SET:
            _blocked = set()
            for t in _hd._PROTECTION_SET:
                for m in re.finditer(re.escape(t), text):
                    for p in range(m.start(), m.end()):
                        _blocked.add(p)
            to_replace = {k for k in to_replace
                          if occurrences[k] not in _blocked}
            if not to_replace:
                continue

        # Pick alternative candidates for variety when multiple occurrences replaced
        # (avoid monotone repetition of single replacement)
        alt_candidates = [c for c, _ in ranked if c != primary] or [primary]

        # Capture original text for next-char lookups (text mutates inside loop)
        original_text = text
        ranked_alts = [c for c, _ in ranked]

        def _pick_safe(default, next_ch):
            """Avoid alts whose last char equals next_ch (would double).
            Falls back to default if no safe alt exists."""
            if not next_ch or default[-1:] != next_ch:
                return default
            for cand in ranked_alts:
                if cand and cand[-1] != next_ch:
                    return cand
            return default

        # Rebuild text by iterating occurrences back-to-front (avoid shifting positions)
        for k in reversed(range(len(occurrences))):
            pos = occurrences[k]
            if k not in to_replace:
                continue
            # Word-boundary doubling guard: check next char in source after the
            # word being replaced. If alt ends with that char, swap to a
            # non-doubling alt. Catches '能够以X' → '可以以X' / '系统的研究'
            # → '架构的的' family of bugs without removing the entry entirely.
            next_ch = original_text[pos + len(word):pos + len(word) + 1]
            # Cycle 54: left-context cross-boundary guard. '解决' inside
            # '了解决策' actually spans 了解|决策 (two distinct words);
            # replacing 解决 with 攻克 corrupts to '了攻克策'. Skip when
            # the word's leading char + prev char form a known 2-char word
            # AND the word's trailing char + next char also form a 2-char
            # word — that's the cross-boundary signature.
            prev_ch = original_text[pos - 1:pos] if pos > 0 else ''
            if word == '解决' and prev_ch == '了' and next_ch in '策心议定断':
                continue
            if word == '解决' and next_ch == '方':
                continue
            if word == '研究' and prev_ch == '本':
                continue
            # Compound-noun guard: '发展' acts as N1 in 'X的发展前景/态势/...'
            # — substituting to verb-form alts (推进/进展/推动) breaks the
            # NP. Skip when followed by a known compound noun suffix.
            if word == '发展':
                next_two = original_text[pos + len(word):pos + len(word) + 2]
                _np_suffixes = (
                    '前景', '前途', '态势', '趋势', '历程', '规律',
                    '方向', '格局', '局面', '动力', '空间', '潜力',
                    '阶段', '路径', '路线', '方式', '模式',
                )
                if next_two in _np_suffixes:
                    continue
                # Same-sentence repetition guard: '推动X长效发展' → '推动X长效推进'
                # gives 推动+推进 redundancy. Skip if 推 appears in prior 6
                # chars within same sentence.
                left_ctx = original_text[max(0, pos - 6):pos]
                if '推' in left_ctx and not any(c in '。！？' for c in left_ctx):
                    continue
            # 分析 in noun-modifier slot: '分析师' / '分析员' should not
            # become '解读师' / '剖析员' (not real words).
            if word == '分析':
                next_ch = original_text[pos + len(word):pos + len(word) + 1]
                if next_ch in '师员家者':
                    continue
            # Pick primary for first replaced occurrence, alternate for others
            if k == min(to_replace):
                replacement = _pick_safe(primary, next_ch)
            else:
                pick = random.choice([primary] + alt_candidates)
                replacement = _pick_safe(pick, next_ch)
            protected = _protect(replacement)
            text = text[:pos] + protected + text[pos + len(word):]

        replaced_words.add(word)

        # Also mark synonyms of the same word to avoid replacing the replacement
        for syn in candidates:
            if syn != primary and syn in WORD_SYNONYMS:
                replaced_words.add(syn)

    # Strip sentinels
    text = text.replace('\x00', '')

    return text


def _simple_synonym_pass(text, strength=0.3, scene='general'):
    """Fallback: replace a fraction of WORD_SYNONYMS matches randomly.

    scene: 'academic' filters PRESERVE words and BLACKLIST candidates.
    """
    preserve = ACADEMIC_PRESERVE_WORDS if scene == 'academic' else set()
    found = []
    for word in WORD_SYNONYMS:
        if word in preserve:
            continue
        start = 0
        while True:
            pos = text.find(word, start)
            if pos < 0:
                break
            found.append((word, pos))
            start = pos + len(word)
    if not found:
        return text
    n_replace = max(1, int(len(found) * strength))
    random.shuffle(found)
    replaced_positions = set()
    # Protection guard: compute blocked positions once before loop.
    import _humanize_data as _hd
    _blocked = set()
    if _hd._USE_PROTECT_FLAG and _hd._PROTECTION_SET:
        for t in _hd._PROTECTION_SET:
            for m in re.finditer(re.escape(t), text):
                for p in range(m.start(), m.end()):
                    _blocked.add(p)
    for word, pos in found[:n_replace]:
        if any(p in replaced_positions for p in range(pos, pos + len(word))):
            continue
        # Protection guard: skip if this occurrence falls inside a protected term
        if pos in _blocked:
            continue
        candidates = _filter_candidates_for_scene(word, WORD_SYNONYMS[word], scene)
        if not candidates:
            continue
        candidate = random.choice(candidates)
        text = text[:pos] + candidate + text[pos + len(word):]
        for p in range(pos, pos + len(candidate)):
            replaced_positions.add(p)
    return text
def _boost_one_paragraph_cv(para, target_cv):
    """Truncate the longest sentence at first comma if paragraph-internal
    sentence-length CV is below target. Reuses guards from
    randomize_sentence_lengths Strategy B."""
    cn_count = len(re.findall(r'[一-鿿]', para))
    if cn_count < 60:
        return para

    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            pairs.append([s, p])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])

    if len(pairs) < 3:
        return para

    lens = [len(re.findall(r'[一-鿿]', s)) for s, _ in pairs]
    valid = [(i, l) for i, l in enumerate(lens) if l >= 5]
    if len(valid) < 3:
        return para
    vl = [l for _, l in valid]
    m = sum(vl) / len(vl)
    if m == 0:
        return para
    var = sum((l - m) ** 2 for l in vl) / len(vl)
    cv = (var ** 0.5) / m

    if cv >= target_cv:
        return para

    long_idx = max(range(len(pairs)), key=lambda i: lens[i])
    long_s, long_p = pairs[long_idx]
    if lens[long_idx] < 18:
        return para

    comma_pos = long_s.find('，')
    if comma_pos < 0:
        return para
    first_part = long_s[:comma_pos]
    rest_part = long_s[comma_pos + 1:]
    if (len(re.findall(r'[一-鿿]', first_part)) < 8 or
            len(re.findall(r'[一-鿿]', rest_part)) < 8):
        return para

    first_stripped = first_part.lstrip()
    last_nl = first_part.rfind('\n')
    if last_nl >= 0:
        tail_cn = len(re.findall(r'[一-鿿]',
                                 first_part[last_nl + 1:]))
        if tail_cn < 3:
            return para

    if first_part.endswith(_PARA_BOOST_ATTRIBUTION):
        return para
    if first_stripped.startswith(_PARA_BOOST_SUBORDINATE):
        return para
    if rest_part.lstrip().startswith(_PARA_BOOST_BARE_CONTINUATOR):
        return para

    pairs[long_idx] = [first_part, '。']
    pairs.insert(long_idx + 1, [rest_part, long_p or '。'])
    return ''.join(s + p for s, p in pairs)


def _boost_one_para_via_merge(para, target_cv):
    """Merge a single pair of adjacent short-medium sentences with a comma
    to lift a uniform paragraph's internal sentence-length CV. Reuses the
    Strategy-A merge guards from randomize_sentence_lengths (reactions,
    paragraph-break boundary, total length cap)."""
    cn_count = len(re.findall(r'[一-鿿]', para))
    if cn_count < 60:
        return para

    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            pairs.append([s, p])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])

    if len(pairs) < 4:
        return para

    lens = [len(re.findall(r'[一-鿿]', s)) for s, _ in pairs]
    valid = [l for l in lens if l >= 5]
    if len(valid) < 3:
        return para
    m = sum(valid) / len(valid)
    if m == 0:
        return para
    var = sum((l - m) ** 2 for l in valid) / len(valid)
    cv = (var ** 0.5) / m

    if cv >= target_cv:
        return para

    # Find an adjacent pair both 5..25 chars whose merged length is <=60
    # (so we cross the medium→long boundary and lift CV without making
    # the merged sentence unwieldy).
    for i in range(len(pairs) - 1):
        l1, l2 = lens[i], lens[i + 1]
        if not (5 <= l1 <= 25 and 5 <= l2 <= 25):
            continue
        if l1 + l2 > 60:
            continue
        s1, _ = pairs[i]
        s2, p2 = pairs[i + 1]
        if (s1.strip() in _PARA_BOOST_REACTIONS or
                s2.strip() in _PARA_BOOST_REACTIONS):
            continue
        if '\n' in s2:
            continue
        merged = s1.rstrip() + '，' + s2.lstrip()
        pairs[i] = [merged, p2]
        pairs.pop(i + 1)
        break

    return ''.join(s + p for s, p in pairs)


def reduce_cross_para_3gram_repeat(text, max_replacements=4, scene='general',
                                   style=None, seed=None):
    """v5 P1.3 humanize counter-measure for cross_para_3gram_repeat
    (LR coef +2.24 on longform).

    Walks paragraphs, identifies 2-char words (CiLin keys) that appear
    in 2+ paragraphs, and replaces ONE occurrence in a later paragraph
    with a CiLin synonym. Replacing a 2-char word breaks two
    overlapping 3-grams, so even a handful of substitutions measurably
    drops the cross-paragraph trigram repetition ratio.

    Scene-aware via the same blacklists as expand_with_cilin
    (_AI_PATTERN_BLACKLIST / _CILIN_BLACKLIST / ACADEMIC_BLACKLIST_CANDIDATES
    / NOVEL_BLACKLIST_CANDIDATES). Skips when the scene/style filters
    yield no usable synonym.

    Prefers words in exactly 2 paragraphs (each replacement directly
    drops a repeat — words spanning 3+ paragraphs need more sub work
    to clear).
    """
    if seed is not None:
        random.seed(seed)

    cilin = _load_cilin()
    if not cilin:
        return text

    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return text

    cilin_keys = set(cilin.keys()) - _CILIN_SOURCE_BLACKLIST
    para_words = []
    for p in paragraphs:
        chars = re.findall(r'[一-鿿]', p)
        words_in_p = set()
        for i in range(len(chars) - 1):
            w = chars[i] + chars[i + 1]
            if w in cilin_keys:
                words_in_p.add(w)
        para_words.append(words_in_p)

    word_paras = {}
    for i, words in enumerate(para_words):
        for w in words:
            word_paras.setdefault(w, []).append(i)

    candidates = [(w, ps) for w, ps in word_paras.items() if len(ps) >= 2]
    if not candidates:
        return text

    # Prefer words appearing in fewer paragraphs (each replacement
    # there directly clears the repeat). Then random within tier.
    candidates.sort(key=lambda x: len(x[1]))
    # Shuffle within each tier of equal paragraph-count
    tier_buckets = {}
    for w, ps in candidates:
        tier_buckets.setdefault(len(ps), []).append((w, ps))
    for k in tier_buckets:
        random.shuffle(tier_buckets[k])
    ordered = []
    for k in sorted(tier_buckets):
        ordered.extend(tier_buckets[k])

    new_paragraphs = list(paragraphs)
    replaced = 0

    for word, para_indices in ordered:
        if replaced >= max_replacements:
            break
        synonyms = cilin.get(word, [])
        if not synonyms:
            continue
        filtered = []
        for c in synonyms:
            if c in _AI_PATTERN_BLACKLIST:
                continue
            if c in _CILIN_BLACKLIST:
                continue
            if scene == 'academic' and c in ACADEMIC_BLACKLIST_CANDIDATES:
                continue
            if (scene == 'novel' or style == 'novel') and \
                    c in NOVEL_BLACKLIST_CANDIDATES:
                continue
            filtered.append(c)
        if not filtered:
            continue
        synonym = random.choice(filtered)
        # Replace in the LAST occurrence paragraph (so the established
        # term lands in earlier paragraphs and the variation shows up
        # later — closer to how humans drift).
        last_idx = para_indices[-1]
        new_para = new_paragraphs[last_idx].replace(word, synonym, 1)
        if new_para != new_paragraphs[last_idx]:
            new_paragraphs[last_idx] = new_para
            replaced += 1

    return join_paragraphs(new_paragraphs)


def _strip_leading_marker_once(fragment, markers):
    s = fragment.lstrip()
    prefix = fragment[:len(fragment) - len(s)]
    for marker in sorted(markers, key=len, reverse=True):
        if s.startswith(marker):
            rest = s[len(marker):]
            if rest.startswith(('，', ',', '、', '：', ':')):
                rest = rest[1:]
            if len(re.findall(r'[一-鿿]', rest)) >= 12:
                return prefix + rest.lstrip()
    return fragment


def _longform_discourse_marker_diversity(text, seed=None):
    """Remove repeated paragraph-head discourse markers on long candidates."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return text

    seen = set()
    changed = 0
    result = []
    for p in paragraphs:
        stripped = p.lstrip()
        marker = None
        for m in sorted(_LONGFORM_PARA_HEAD_MARKERS, key=len, reverse=True):
            if stripped.startswith(m):
                marker = m
                break
        if marker and marker in seen and changed < 3 and random.random() < 0.8:
            new_p = _strip_leading_marker_once(p, (marker,))
            if new_p != p and new_p.strip():
                p = new_p
                changed += 1
        if marker:
            seen.add(marker)
        result.append(p)

    return join_paragraphs(result)


def _longform_merge_one_sentence_pair(para):
    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip():
            pairs.append([parts[i], parts[i + 1]])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])
    if len(pairs) < 3:
        return para

    for i in range(len(pairs) - 1):
        s1, _ = pairs[i]
        s2, p2 = pairs[i + 1]
        l1 = len(re.findall(r'[一-鿿]', s1))
        l2 = len(re.findall(r'[一-鿿]', s2))
        if not (8 <= l1 <= 28 and 8 <= l2 <= 32 and l1 + l2 <= 62):
            continue
        if s2.lstrip().startswith(_PARA_BOOST_BARE_CONTINUATOR):
            continue
        pairs[i] = [s1.rstrip() + '，' + s2.lstrip(), p2]
        pairs.pop(i + 1)
        return ''.join(s + p for s, p in pairs)
    return para


def _longform_split_one_comma_clause(para):
    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip():
            pairs.append([parts[i], parts[i + 1]])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])
    if len(pairs) < 2:
        return para

    for i, (sent, punct) in enumerate(pairs):
        if len(re.findall(r'[一-鿿]', sent)) < 34:
            continue
        for m in re.finditer(r'[，,]', sent):
            left = sent[:m.start()]
            right = sent[m.end():]
            if (len(re.findall(r'[一-鿿]', left)) >= 12 and
                    len(re.findall(r'[一-鿿]', right)) >= 14 and
                    not right.lstrip().startswith(_PARA_BOOST_BARE_CONTINUATOR)):
                pairs[i] = [left.rstrip() + '。' + right.lstrip(), punct]
                return ''.join(s + p for s, p in pairs)
    return para


def _longform_paragraph_punct_drift(text, seed=None):
    """Create mild paragraph-to-paragraph punctuation rhythm drift."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return text

    result = []
    changed = 0
    start = random.randrange(2)
    for idx, p in enumerate(paragraphs):
        new_p = p
        if changed < 3 and len(re.findall(r'[一-鿿]', p)) >= 70:
            if (idx + start) % 2 == 0:
                new_p = _longform_split_one_comma_clause(p)
            else:
                new_p = _longform_merge_one_sentence_pair(p)
            if new_p != p and new_p.strip():
                changed += 1
        result.append(new_p)
    return join_paragraphs(result)


def _longform_paragraph_length_cv_micro_adjust(text, seed=None):
    """Single guarded merge/split pass when paragraph lengths are too uniform."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 5:
        return text
    cv = _para_cv(paragraphs)
    if cv is not None and cv >= 0.48:
        return text
    adjusted = vary_paragraph_rhythm(text)
    if len(split_paragraphs(adjusted)) < len(paragraphs) - 1:
        return text
    return adjusted


def _longform_starter_entropy_boost(text, seed=None):
    """Reduce repeated safe transition starters without inventing new wording."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return text

    starter_counts = {}
    for sent in re.split(r'[。！？!?；;\n]+', text):
        chars = re.findall(r'[一-鿿]', sent)
        if len(chars) >= 2:
            key = ''.join(chars[:2])
            starter_counts[key] = starter_counts.get(key, 0) + 1
    repeated = {k for k, v in starter_counts.items() if v >= 2}
    if not repeated:
        return text

    changed = 0

    def strip_sentence(m):
        nonlocal changed
        boundary, body = m.group(1), m.group(2)
        chars = re.findall(r'[一-鿿]', body)
        key = ''.join(chars[:2]) if len(chars) >= 2 else ''
        if key not in repeated or changed >= 3 or random.random() >= 0.7:
            return m.group(0)
        new_body = _strip_leading_marker_once(body, _LONGFORM_STARTER_MARKERS)
        if new_body != body and new_body.strip():
            changed += 1
            return boundary + new_body
        return m.group(0)

    result = []
    pattern = re.compile(r'(^|[。！？!?；;\n])([^。！？!?；;\n]+)')
    for p in paragraphs:
        result.append(pattern.sub(strip_sentence, p))
    return join_paragraphs(result)


def _apply_longform_mutation_profile(text, mutation_seed=None, scene='general',
                                     style=None):
    """Candidate-only longform mutations for best-of-n exploration."""
    before_paras = len(split_paragraphs(text))
    if before_paras < 3:
        return text

    try:
        from ngram_model import compute_lr_score
    except ImportError:
        try:
            from scripts.ngram_model import compute_lr_score
        except ImportError:
            compute_lr_score = None

    def lr_score(candidate):
        if compute_lr_score is None:
            return None
        lr = compute_lr_score(candidate, scene='longform')
        return lr['score'] if lr else None

    def structurally_safe(candidate):
        after_paras = split_paragraphs(candidate)
        if any(not p.strip() for p in after_paras):
            return False
        return len(after_paras) >= before_paras - 2

    current = text
    current_score = lr_score(current)
    steps = (
        lambda t: _longform_discourse_marker_diversity(
            t, seed=None if mutation_seed is None else mutation_seed + 23),
        lambda t: _longform_paragraph_punct_drift(
            t, seed=None if mutation_seed is None else mutation_seed + 37),
        lambda t: _longform_paragraph_length_cv_micro_adjust(
            t, seed=None if mutation_seed is None else mutation_seed + 41),
        lambda t: _longform_starter_entropy_boost(
            t, seed=None if mutation_seed is None else mutation_seed + 53),
    )

    for step in steps:
        candidate = step(current)
        if candidate == current or not structurally_safe(candidate):
            continue
        candidate_score = lr_score(candidate)
        if (current_score is not None and candidate_score is not None and
                candidate_score > current_score):
            continue
        current = candidate
        if candidate_score is not None:
            current_score = candidate_score

    return current if structurally_safe(current) else text


def insert_short_interjection_paragraph(text, target_cv=0.50, style=None,
                                        seed=None):
    """v5 P1.2 humanize counter-measure for paragraph_length_cv (d=-1.49).

    For multi-paragraph text whose paragraph-length CV is below target,
    insert a single short interjection paragraph (~20-22 cn chars) AFTER
    one of the longer existing paragraphs (top quartile by length).
    The interjection sharply lifts paragraph-length variance without
    restructuring existing paragraphs (cycle 28 lesson: split/merge of
    existing paragraphs has persistently negative ROI; this function
    only adds, never restructures).

    Two pools, picked by style:
      - novel  : narrative beats (atmosphere / action / dialogue gap)
      - other  : reflective neutral-formal sentences

    Skips:
      - Single-paragraph text
      - Text already varied (CV >= target)
      - When adjacent paragraph is a markdown header / list / bold
        subheader (would split a structural pair)
    """
    if seed is not None:
        random.seed(seed)

    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return text

    lens = [len(re.findall(r'[一-鿿]', p)) for p in paragraphs]
    valid_pairs = [(i, l) for i, l in enumerate(lens) if l >= 20]
    if len(valid_pairs) < 3:
        return text
    valid_lens = [l for _, l in valid_pairs]
    m = sum(valid_lens) / len(valid_lens)
    if m == 0:
        return text
    var = sum((l - m) ** 2 for l in valid_lens) / len(valid_lens)
    cv = (var ** 0.5) / m

    if cv >= target_cv:
        return text

    sorted_pairs = sorted(valid_pairs, key=lambda x: -x[1])
    top_count = max(2, len(valid_pairs) // 4)
    top_indices = [i for i, _ in sorted_pairs[:top_count]]

    insert_after = random.choice(top_indices)
    next_idx = insert_after + 1
    if next_idx < len(paragraphs):
        next_lstrip = paragraphs[next_idx].lstrip()
        if (next_lstrip.startswith('#') or next_lstrip.startswith('- ') or
                next_lstrip.startswith('* ') or
                (next_lstrip.startswith('**') and
                 next_lstrip.rstrip().endswith('**'))):
            return text

    pool = _PARA_INTERJECTION_NOVEL if style == 'novel' \
        else _PARA_INTERJECTION_NEUTRAL
    interjection = random.choice(pool)

    new_paragraphs = list(paragraphs)
    new_paragraphs.insert(next_idx, interjection)
    return join_paragraphs(new_paragraphs)


def boost_para_cv_via_merge(text, target_cv=0.40):
    """v5 P1 humanize counter-measure (merge variant).

    Walks paragraphs and, for any whose internal sentence-length CV is
    below target, merges a single pair of adjacent short-medium
    sentences with a comma. This removes one period (counter to the
    truncation variant in boost_para_sent_len_cv that adds one) so
    the punct_density LR contribution doesn't cancel the para-CV
    contribution, and the merged sentence typically clears the
    medium→long threshold (sent_len_long_frac coef in the longform LR
    is -0.44, so producing more longs helps).
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        return text
    return join_paragraphs(_boost_one_para_via_merge(p, target_cv)
                           for p in paragraphs)


def boost_para_sent_len_cv(text, target_cv=0.40):
    """v5 P1 humanize counter-measure for stat_low_para_sent_len_cv (d=-2.08).

    For each paragraph (>=60 cn chars, >=3 sentences) where internal
    sentence-length CV is below target, truncate the longest sentence
    at its first comma so the paragraph contains at least one short
    sentence among its mediums. Single pass — does not iterate.

    Skips short paragraphs and applies the same guards as
    randomize_sentence_lengths Strategy B (attribution verbs, subordinate
    clause heads, bare causative continuators, paragraph-break tail).
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        # Single-paragraph text — signal doesn't apply.
        return text
    return join_paragraphs(_boost_one_paragraph_cv(p, target_cv)
                           for p in paragraphs)


def randomize_sentence_lengths(text, aggressive=False, seed=None):
    """
    策略2: 刻意制造不均匀的句子长度分布。
    - 随机选 20% 的短句保持极短
    - 随机选 10% 的句子通过合并拉长
    - 制造"短-长-短-长-特长-短"的节奏
    """
    if seed is not None:
        random.seed(seed)

    # Split into sentences preserving punctuation
    parts = re.split(r'([。！？])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            sentences.append((s, p))
    # Handle trailing text
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append((parts[-1], ''))

    if len(sentences) < 4:
        return text

    merge_rate = 0.15 if not aggressive else 0.25
    truncate_rate = 0.15 if not aggressive else 0.25

    result = []
    i = 0
    while i < len(sentences):
        s, p = sentences[i]
        cn_len = len(re.findall(r'[\u4e00-\u9fff]', s))

        # Strategy A: merge short adjacent sentences into a long one
        if (i + 1 < len(sentences) and random.random() < merge_rate):
            s2, p2 = sentences[i + 1]
            cn_len2 = len(re.findall(r'[\u4e00-\u9fff]', s2))
            # Don't merge if adjacent sentence is a known reaction phrase (cycle 22
            # bug fix — short reactions inserted by `insert_short_reactions` were
            # being silently merged back, collapsing the short_frac signal).
            _reactions = (
                '的确', '确实如此', '颇有道理', '不无道理', '事出有因',
                '耐人寻味', '值得深思', '让人深思', '可见一斑', '有一定道理',
                '各有道理', '各有说法', '难以一概', '难以断言', '说来话长',
                '一言难尽',
            )
            s_stripped = s.strip()
            s2_stripped = s2.strip()
            # Paragraph boundary: split by [。！？] preserves \n\n as leading
            # whitespace on the next sentence. Merging would .lstrip() the
            # \n\n away and collapse two paragraphs into one — discourse
            # structure loss (Petalses issue #5).
            if '\n' in s2 or s_stripped in _reactions or s2_stripped in _reactions:
                pass
            elif cn_len + cn_len2 < 100:
                merged = s.rstrip() + '，' + s2.lstrip()
                result.append(merged + p2)
                i += 2
                continue

        # Strategy B: truncate longer sentences to their first clause (creates short punchy sentences)
        if cn_len > 20 and cn_len < 50 and random.random() < truncate_rate:
            # Truncate to first clause (split at first comma), keep rest as next sentence
            comma_pos = s.find('，')
            if comma_pos > 5 and comma_pos < len(s) - 5:
                first_part = s[:comma_pos]
                first_stripped = first_part.lstrip()
                # Guard 0: don't truncate when the first_part fragment after the
                # last paragraph break is too short. The [。！？] split doesn't
                # respect \n\n, so a segment can span "## header\n\n现在，X..."
                # — truncating yields "## header\n\n现在。X..." stranding a
                # 2-char fragment after the section header.
                last_nl = first_part.rfind('\n')
                if last_nl >= 0:
                    tail_cn = len(re.findall(r'[一-鿿]',
                                             first_part[last_nl + 1:]))
                    if tail_cn < 3:
                        result.append(s + p)
                        i += 1
                        continue
                # Guard 1: skip if first part ends in an attribution/reporting verb.
                # Otherwise "X 指出，" becomes "X 指出。" + bare clause — broken grammar.
                _attribution_suffixes = (
                    '指出', '表明', '认为', '揭示', '发现', '显示', '提出',
                    '说', '称', '讲', '强调', '主张', '断言',
                )
                if first_part.endswith(_attribution_suffixes):
                    result.append(s + p)
                    i += 1
                    continue
                # Guard 2: skip if first part is a subordinate clause (starts with
                # 随着/鉴于/为了/由于/尽管/虽然/如果 etc.). Splitting at comma would
                # leave a fragment that can't stand alone: "随着X的发展。Y" is broken.
                _subordinate_prefixes = (
                    '随着', '鉴于', '为了', '由于', '尽管', '虽然',
                    '如果', '假如', '若是', '倘若', '要是', '即便', '纵然',
                    '除了', '除非', '只要', '只有', '无论', '不管',
                    '当', '每当', '一旦',
                    # cycle 201: 面对X / 处在X = context introducer that needs
                    # a main clause. Splitting at comma leaves a fragment.
                    # ('在' kept out — too broad; handled by suffix guard below)
                    '面对', '处在',
                )
                if first_stripped.startswith(_subordinate_prefixes):
                    result.append(s + p)
                    i += 1
                    continue
                # cycle 201: context-introducer SUFFIXES that need a main
                # clause (covers "在X的背景下" cycle-190 alts: "...这种局面，"
                # "...这个情境里，" "...之中，"). Catches the "在" case
                # without blocking all "在..." sentences.
                _context_suffixes = (
                    '这种局面', '这种情况', '这个情境里', '这种背景下',
                    '之中', '的背景下',
                )
                if first_part.endswith(_context_suffixes):
                    result.append(s + p)
                    i += 1
                    continue
                # Guard 3: skip if next clause starts with a bare causative
                # verb (使/使得/导致/造成 etc.) OR a continuation marker
                # (同时/此外/另外/更/不仅/而且/进而/继而/充分/进一步/同样).
                # These all assume the prior clause's subject/context — splitting
                # creates fragment "X。同时Y。" which reads as orphaned.
                # cycle 206 (sway 标点符号奇怪): added 同时/充分/进一步 etc.
                # Audit on workplace example showed pattern "工作效率，同时也Y，
                # 充分体现Z" splitting into 3 short sentences with multiple
                # paragraph-end periods — sway flagged as awkward.
                _bare_continuators = (
                    '使得', '使', '导致', '引起', '造成', '致使',
                    '同时', '同样', '此外', '另外', '更', '不仅', '而且',
                    '进而', '继而', '充分', '进一步', '同时也',
                )
                # Modal/aux continuators: only block when first_part is a
                # short bare NP (no main verb). Long first_parts with their
                # own verb can stand alone, so allow truncation there.
                _modal_continuators = (
                    '能够', '能', '可以', '可', '将会', '将',
                    '亦可', '亦', '也将', '也能', '也可',
                )
                rest_after_comma = s[comma_pos + 1:].lstrip()
                if rest_after_comma.startswith(_bare_continuators):
                    result.append(s + p)
                    i += 1
                    continue
                if rest_after_comma.startswith(_modal_continuators):
                    # Heuristic: if first_part has a main verb, allow split.
                    # Bare NP (subject-only) creates a fragment.
                    first_cn = len(re.findall(r'[一-鿿]', first_part))
                    _verb_markers = (
                        '是', '有', '做', '用', '把', '让', '使', '给',
                        '提', '推', '完', '实', '达', '形', '构', '反',
                        '显', '表', '维', '保', '改', '优', '调', '处',
                        '通过', '运用', '采用', '成为', '需要', '获得',
                    )
                    if first_cn < 12 and not any(m in first_part for m in _verb_markers):
                        result.append(s + p)
                        i += 1
                        continue
                rest_part = s[comma_pos + 1:]
                result.append(first_part + p)
                # Push the rest as a new "sentence" to be processed
                if rest_part.strip():
                    result.append(rest_part + '。')
                i += 1
                continue

        result.append(s + p)
        i += 1

    return ''.join(result)
def _dialogue_density_local(text):
    """Fraction of chars inside Chinese dialogue quotes. AI novels use a
    mix of curly U+201C/D (“”), corner U+300C/D (「」), and ASCII pairs
    (which some models output instead). Threshold 0.08 flags narrative."""
    n = 0
    for pat in (r'“[^“”]{3,}?”', r'「[^「」]{3,}?」'):
        for m in re.findall(pat, text):
            n += len(m)
    # ASCII " pairs: split on ", odd-indexed segments are inside quotes
    parts = text.split('"')
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            if len(parts[i]) >= 3:
                n += len(parts[i])
    return n / max(1, len(text))


# Narrative-safe subset of NOISE_EXPRESSIONS categories. filler/personal/
# transition_casual inject 1st-person author voice or oral fillers that
# break 3rd-person fiction register.
def inject_noise_expressions(text, density=0.15, style='general'):
    """
    策略3: 在句子间或句中适当位置插入噪声表达。
    density: 大约每多少句插入一个（0.15 ≈ 每 6-7 句一个）
    style: general / academic
    """
    # cycle 152: when style='general' but the text has 2+ markdown
    # headers (academic survey / technical article), the 'filler' /
    # 'transition_casual' / 'personal' categories from NOISE_EXPRESSIONS
    # ('当然了' / '坦白讲' / '不瞒你说' etc.) read off-register inside
    # formal prose. Promote to the academic noise subset, which keeps
    # only hedging / self_correction / uncertainty.
    if style == 'general':
        n_md_headers = sum(1 for line in text.split('\n')
                           if re.match(r'^\s*#{1,6}\s', line))
        if n_md_headers >= 2:
            style = 'academic'

    if style == 'academic':
        categories = NOISE_ACADEMIC_CATEGORIES
        expressions = NOISE_ACADEMIC_EXPRESSIONS
    else:
        categories = list(NOISE_EXPRESSIONS.keys())
        expressions = NOISE_EXPRESSIONS
        # Narrative guard: if text is dialogue-heavy, drop categories that
        # break 3rd-person voice (filler/personal/transition_casual).
        if _dialogue_density_local(text) >= 0.08:
            categories = [c for c in categories if c in _NARRATIVE_SAFE_CATEGORIES]
            if not categories:
                return text

    # Split into sentences
    parts = re.split(r'([。！？])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            sentences.append([s, p])
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append([parts[-1], ''])

    if len(sentences) < 3:
        return text

    # Track expressions already injected in this run. Re-injecting the same
    # phrase ("\u5f80\u6df1\u4e86\u8bb2" / "\u5e73\u5fc3\u800c\u8bba") three times in one sample reads as a
    # tic, which detect_cn flags as repetitive and a human reviewer flags as
    # robot-style.
    used = set()

    # cycle 203 (sway directive \u8bed\u53e5\u901a\u987a\u4f18\u5148): track which paragraphs already
    # had a noise injection. Multiple injections per paragraph create
    # "\u5728\u6211\u770b\u6765\uff0cX\u3002\u6ce8\u610f\uff0cY\u3002\u8bf4\u5230\u5e95\uff0cZ" robotic chains. Hard cap = 1
    # injection per paragraph. Identifies paragraph by the cumulative \n\n
    # count in text up to the sentence position.
    para_injected = {}

    injected = 0
    cum_text = ''
    for i in range(len(sentences)):
        s_text = sentences[i][0]
        s_punct = sentences[i][1] or ''
        # cycle 203: track cumulative text to identify current paragraph
        # (paragraph = chunk between \n\n breaks). Update at top so all
        # `continue` branches keep para_idx in sync.
        para_idx = cum_text.count('\n\n')
        cum_text += s_text + s_punct
        # Skip the last sentence (avoid orphaned expressions)
        if i >= len(sentences) - 1:
            continue
        # Skip very short sentences
        if len(re.findall(r'[\u4e00-\u9fff]', s_text)) < 8:
            continue
        # Skip sentences that contain dialogue quotes. Injecting a noise
        # expression into a quoted line puts narrator filler inside a
        # character's mouth \u2014 awkward and breaks dialogue flow.
        if '"' in s_text or '\u201c' in s_text or '\u201d' in s_text or '\u300c' in s_text or '\u300d' in s_text:
            continue
        # Cycle 57/58: skip sentences that start with markdown structural
        # markers (# heading / - * bullet / **bold** subheader / 1. 2.
        # numbered list). Injecting '\u4e0d\u7792\u4f60\u8bf4\uff0c' before '#### 2.2 ...' or
        # '\u5728\u6211\u770b\u6765\uff0c**3. \u54c1\u724c\u5efa\u8bbe\uff1a\u6587\u5316\u2026**' corrupts the structural marker.
        # Cycle 58 widens the **-prefix check from "starts AND ends with **"
        # (pure bold subheader) to just "starts with **" \u2014 covers hybrid
        # forms like '**1. \u8d44\u6e90\u74f6\u9888\uff1a** \u9ad8\u5e76\u53d1\u610f\u5473\u7740\u2026' that the cycle 57
        # check missed (audit found 34 longform samples with this pattern).
        s_lstripped = s_text.lstrip()
        # cycle 203 (sway directive 语句通顺优先): skip if sentence already
        # starts with a SHORT transition marker. These come from
        # patterns_cn.json replacements (值得注意的是→注意, 综上所述→总之,
        # 其次→另外/此外, etc.). Stacking noise on top reads as
        # "在我看来，注意，X..." — multiple transitions piled up, robotic.
        # Trade: drops some LR-favorable noise, accepted per sway directive.
        _existing_transitions = (
            '注意，', '特别说一下，', '要提醒的是，', '总之，', '说到底，',
            '简单讲，', '归结起来，', '另外，', '此外，', '还有，',
            '可以看到，', '很明显，', '你会发现，',
            '一开始，', '最初，', '起头，', '先说，',
            '接着，', '然后，', '再就是，', '最后说一点，',
            # Standard discourse connectors: stacking noise before these
            # creates "顺着这个思路，然而，X" double-connector reads.
            '然而，', '但是，', '不过，', '可是，', '因此，', '所以，',
            '因而，', '而且，', '同时，', '不仅，', '相反，', '反之，',
        )
        if s_lstripped.startswith(_existing_transitions):
            continue
        # cycle 203: per-paragraph injection cap = 1 (sway 语句通顺优先).
        # Skip if this paragraph already had an injection — prevents
        # "在我看来，X。注意，Y" cross-sentence stacking.
        if para_injected.get(para_idx, 0) >= 1:
            continue
        # cycle 203 sub: also skip if the same paragraph (the one we're
        # in, or that the current sentence will land in) already contains
        # any of the existing-transition markers (from replacements). This
        # catches "注意，X。" + "在我看来，Y" same-paragraph stacking
        # where 注意 came from values_注意的是 replacement, not noise.
        # Build the paragraph slice: all sentences sharing this paragraph.
        para_slice = ''
        running_para = 0
        for j in range(len(sentences)):
            if running_para == para_idx:
                para_slice += sentences[j][0] + (sentences[j][1] or '')
            running_para += (sentences[j][0] + (sentences[j][1] or '')).count('\n\n')
            if running_para > para_idx:
                break
        if any(t in para_slice for t in _existing_transitions):
            continue
        if s_lstripped.startswith('#') or s_lstripped.startswith('- ') or s_lstripped.startswith('* '):
            continue
        if s_lstripped.startswith('**'):
            continue
        if re.match(r'^\d+[.\u3002\uff0e)\uff09]', s_lstripped):
            continue
        # Chinese numbered section headers: "\u4e00\u3001X" / "\uff08\u4e00\uff09X" \u2014 common
        # in long-form Chinese essays. Don't inject noise before them.
        if re.match(r'^[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001,\uff0c]', s_lstripped):
            continue
        if re.match(r'^[\uff08(][\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\uff09)]', s_lstripped):
            continue
        # Title/heading guard: standalone line without terminal punctuation
        # (ends in non-\u3002\uff01\uff1f and is followed by \n\n) \u2014 usually a title.
        # "\u4ece\u7a0b\u5e8f\u5458\u8f6c\u4ea7\u54c1\u7ecf\u7406\uff0c\u7b2c\u4e00\u5e74\u5b66\u5230\u7684\u4e09\u4ef6\u4e8b" \u2192 skip noise injection.
        s_trimmed = s_text.rstrip()
        if s_trimmed and s_trimmed[-1] not in '\u3002\uff01\uff1f.!?':
            next_text = sentences[i + 1][0] if i + 1 < len(sentences) else ''
            if next_text.startswith('\n\n') or next_text.startswith('\n'):
                continue
        if random.random() > density:
            continue

        cat = random.choice(categories)
        expr_list = expressions.get(cat, [])
        if not expr_list:
            continue
        avail = [e for e in expr_list if e not in used]
        if not avail:
            avail = expr_list  # fallback when category exhausted
        expr = random.choice(avail)
        used.add(expr)

        s, p = sentences[i]

        # Preserve leading whitespace (\n\n paragraph breaks) — sentences
        # that start a new paragraph have \n\n at their head (artifact of
        # the [。！？] split). .lstrip() would eat those and collapse
        # paragraph structure.
        leading_ws_len = len(s) - len(s.lstrip())
        leading = s[:leading_ws_len]
        s_body = s[leading_ws_len:]

        # Decide insertion position
        if cat in ('hedging', 'filler', 'personal', 'transition_casual'):
            # Insert at sentence beginning (after any paragraph break)
            s = leading + expr + '，' + s_body
        elif cat in ('self_correction', 'uncertainty'):
            # Insert mid-sentence at a comma
            comma_pos = s_body.find('，')
            if comma_pos > 3:
                s = leading + s_body[:comma_pos + 1] + expr + '，' + s_body[comma_pos + 1:]
            else:
                s = leading + expr + '，' + s_body

        sentences[i] = [s, p]
        injected += 1
        # cycle 203: bump per-paragraph counter for cap enforcement
        para_injected[para_idx] = para_injected.get(para_idx, 0) + 1

    return ''.join(s + p for s, p in sentences)
