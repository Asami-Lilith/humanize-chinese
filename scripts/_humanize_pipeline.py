#!/usr/bin/env python3
"""Text transformation pipeline for Chinese AI Text Humanizer v2.0.

Thirteen-step rewriting pipeline: remove_three_part_structure, replace_phrases,
merge_short_sentences, split_long_sentences, vary_paragraph_rhythm,
reduce_punctuation, cap_transition_density, inject_sentence_particles,
add_casual_expressions, shorten_paragraphs, diversify_vocabulary.
"""

import sys
import re
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from _text_utils import join_paragraphs, split_paragraphs
except ImportError:
    from scripts._text_utils import join_paragraphs, split_paragraphs

from _humanize_data import (
    _USE_NOISE, _USE_STATS, _USE_CILIN,
    SCENES, CONFIG, PATTERNS_FILE,  
    PHRASE_REPLACEMENTS, PLAIN_REPLACEMENTS, REGEX_REPLACEMENTS,
    WORD_SYNONYMS, _CILIN_BLACKLIST, _CILIN_SOURCE_BLACKLIST,
    _AI_PATTERN_BLACKLIST,
    ACADEMIC_BLACKLIST_CANDIDATES, NOVEL_BLACKLIST_CANDIDATES,
    _filter_candidates_for_scene,
    _PARA_BOOST_ATTRIBUTION, _PARA_BOOST_SUBORDINATE, _PARA_BOOST_BARE_CONTINUATOR,
    pick_best_replacement, _compute_burstiness,
)
# Module-level imports resolved at call time via try/except chain

def remove_three_part_structure(text):
    """Remove 首先/其次/最后, 第一/第二/第三 patterns"""
    # Don't just delete — replace with natural transitions
    replacements = [
        (r'首先[，,]\s*', ''),
        (r'其次[，,]\s*', lambda m: random.choice(['另外，', '此外，', ''])),
        (r'最后[，,]\s*', lambda m: random.choice(['还有，', ''])),  # cycle 208: drop 最后说一点 (awkward in essays)
        (r'第一[，,、]\s*', ''),
        (r'第二[，,、]\s*', lambda m: random.choice(['接着，', '然后，', ''])),
        (r'第三[，,、]\s*', lambda m: random.choice(['还有，', '再就是，', ''])),
        (r'第[四五六七八九][，,、]\s*', lambda m: random.choice(['另外，', ''])),
        (r'其一[，,、]\s*', ''),
        (r'其二[，,、]\s*', lambda m: random.choice(['另外，', ''])),
        (r'其三[，,、]\s*', lambda m: random.choice(['还有，', ''])),
    ]
    
    for pattern, repl in replacements:
        if callable(repl):
            text = re.sub(pattern, repl, text)
        else:
            text = re.sub(pattern, repl, text)
    
    return text
def replace_phrases(text, casualness=0.3):
    """Replace AI phrases with natural alternatives (context-aware)"""
    # Apply regex replacements FIRST (per-sentence, max 1 regex replacement per sentence)
    # Split by sentence-ending punctuation to handle multiple templates in same line
    parts = re.split(r'([。！？\n])', text)
    rebuilt = []
    for part in parts:
        replaced = False
        for pattern, alternatives in REGEX_REPLACEMENTS.items():
            if replaced:
                break
            if isinstance(alternatives, str):
                alternatives = [alternatives]
            try:
                match = re.search(pattern, part)
                if match:
                    replacement = random.choice(alternatives)
                    expanded = match.expand(replacement)
                    part = part[:match.start()] + expanded + part[match.end():]
                    replaced = True
            except re.error:
                pass
        rebuilt.append(part)
    text = ''.join(rebuilt)
    
    # Then plain replacements, sorted by length (longest first) to avoid partial matches
    sorted_phrases = sorted(PLAIN_REPLACEMENTS.keys(), key=len, reverse=True)
    
    for phrase in sorted_phrases:
        alternatives = PLAIN_REPLACEMENTS[phrase]
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        
        if phrase in text:
            # Filter out alternatives that contain the phrase as a substring —
            # those cause infinite re-match loops (e.g. 相反 -> 相反地 reinserts
            # 相反). Without this, slow-path bug: cycle 2 HC3 500 hang, cycle 13
            # longform benchmark kill on samples 85/86/133/144 (all had 相反).
            safe_alts = [alt for alt in alternatives if phrase not in alt]
            if not safe_alts:
                continue
            # Dedupe replacement choices for this phrase. pick_best_replacement
            # is deterministic on perplexity, so when the same phrase occurs
            # multiple times in a long sample it gets rewritten to the same
            # alternative every iteration ('可能引起' x4-5 in audit). Track
            # which alts have been used and prefer unused ones; fall back to
            # the full safe list once exhausted.
            used = set()
            replacement = pick_best_replacement(text, phrase, safe_alts)
            text = text.replace(phrase, replacement, 1)
            used.add(replacement)
            while phrase in text:
                avail = [a for a in safe_alts if a not in used]
                if not avail:
                    # Cycle exhausted — clear `used` so the next round
                    # rotates through the alts again instead of falling
                    # back to a single deterministic pick. Without this
                    # reset, sample 38 of the longform corpus rewrites
                    # 9 occurrences of '然后' as 6×'随后' + '接着' + '之后'
                    # + '随后' instead of an even distribution.
                    used.clear()
                    avail = safe_alts
                replacement = pick_best_replacement(text, phrase, avail)
                text = text.replace(phrase, replacement, 1)
                used.add(replacement)

    return text
def merge_short_sentences(text, min_len=8):
    """Merge overly short consecutive sentences, with burstiness guard."""
    # Measure burstiness before restructuring
    burst_before = _compute_burstiness(text)

    sentences = re.split(r'([。！？])', text)
    if len(sentences) < 4:
        return text
    
    result = []
    i = 0
    while i < len(sentences) - 1:
        sent = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        # Check if this and next sentence are both short
        next_sent = sentences[i + 2] if i + 2 < len(sentences) else ''
        
        if len(sent.strip()) < min_len and len(next_sent.strip()) < min_len and next_sent.strip():
            # Don't merge across paragraph boundaries — \n\n leading
            # next_sent would be stripped by .strip(), collapsing paragraphs.
            if '\n' in sent or '\n' in next_sent:
                result.append(sent + punct)
                i += 2
            else:
                # Merge with comma
                merged = sent.strip() + '，' + next_sent.strip()
                next_punct = sentences[i + 3] if i + 3 < len(sentences) else '。'
                result.append(merged + next_punct)
                i += 4
        else:
            result.append(sent + punct)
            i += 2
    
    # Handle remaining
    while i < len(sentences):
        result.append(sentences[i])
        i += 1
    
    new_text = ''.join(result)

    # Burstiness guard: if merging made text more uniform, revert
    if burst_before is not None:
        burst_after = _compute_burstiness(new_text)
        if burst_after is not None and burst_after < burst_before * 0.8:
            return text  # revert — merging reduced burstiness too much

    return new_text
def split_long_sentences(text, max_len=80):
    """Split overly long sentences at natural breakpoints, with burstiness guard."""
    burst_before = _compute_burstiness(text)

    sentences = re.split(r'([。！？])', text)
    result = []
    
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        chinese_len = len(re.findall(r'[\u4e00-\u9fff]', sent))
        
        if chinese_len > max_len:
            # Find natural split points: 但是/不过/然而/同时/而且
            split_points = [
                (m.start(), m.group()) for m in
                re.finditer(r'[，,](但是|不过|然而|同时|而且|所以|因此|另外)', sent)
            ]

            def _tail_too_short(part):
                # Skip splits that would strand a tiny fragment after the most
                # recent paragraph/line break. Sentences split by [。！？] can
                # span "## header\n\nX，Y" so a comma-split produces broken
                # "## header\n\nX。Y" output.
                last_nl = part.rfind('\n')
                if last_nl < 0:
                    return False
                tail_cn = len(re.findall(r'[一-鿿]', part[last_nl + 1:]))
                return tail_cn < 3

            if split_points:
                # Split at the most central point
                mid = len(sent) // 2
                best = min(split_points, key=lambda x: abs(x[0] - mid))
                part1 = sent[:best[0]]
                part2 = sent[best[0]+1:]  # Skip the comma
                if _tail_too_short(part1):
                    result.append(sent + punct)
                else:
                    result.append(part1 + '。' + part2 + punct)
            else:
                # Split at a comma near the middle. Filter commas whose
                # following clause starts with a bare causative verb
                # (使得/导致/etc.) — splitting there yields "X。使得Y" which
                # strands a subject-less verb.
                _bare_continuators = (
                    '使得', '使', '导致', '引起', '造成', '致使',
                )
                def _safe_comma(idx):
                    rest = sent[idx + 1:].lstrip()
                    return not rest.startswith(_bare_continuators)
                commas = [m.start() for m in re.finditer(r'[，,]', sent)
                          if _safe_comma(m.start())]
                if commas:
                    mid = len(sent) // 2
                    best_comma = min(commas, key=lambda x: abs(x - mid))
                    part1 = sent[:best_comma]
                    part2 = sent[best_comma+1:]
                    if _tail_too_short(part1):
                        result.append(sent + punct)
                    else:
                        result.append(part1 + '。' + part2 + punct)
                else:
                    result.append(sent + punct)
        else:
            result.append(sent + punct)
    
    # Handle remaining
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1])
    
    new_text = ''.join(result)

    # Burstiness guard: if splitting made text more uniform, revert
    if burst_before is not None:
        burst_after = _compute_burstiness(new_text)
        if burst_after is not None and burst_after < burst_before * 0.8:
            return text

    return new_text
def _para_cv(paragraphs):
    """Helper: compute paragraph-length CV over valid (>=20 cn) paragraphs."""
    cn_lens = [len(re.findall(r'[一-鿿]', p)) for p in paragraphs]
    valid_lens = [l for l in cn_lens if l >= 20]
    if len(valid_lens) < 3:
        return None
    m = sum(valid_lens) / len(valid_lens)
    if m == 0:
        return None
    var = sum((l - m) ** 2 for l in valid_lens) / len(valid_lens)
    return (var ** 0.5) / m
def vary_paragraph_rhythm(text):
    """Break uniform paragraph lengths by merging or splitting"""
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return text

    # v5 P1.2 guard (cycle 143): if paragraph-length CV is already
    # adequate (>=0.40, near human distribution), skip merge/split.
    # cycle 142 found that further structural tweaks on already-varied
    # paragraphs push the distribution back toward uniform — a stuck
    # academic sample went from CV 0.405 to 0.320 after the full
    # pipeline because a long paragraph got split, averaging the
    # distribution down.
    cv_initial = _para_cv(paragraphs)
    if cv_initial is not None and cv_initial >= 0.40:
        return text

    lengths = [len(p) for p in paragraphs]
    avg_len = sum(lengths) / len(lengths) if lengths else 100

    def _is_md_header(p):
        # Markdown headers ('# ', '## ', '### ' …), bullets, bold section
        # subheaders, numbered list items, and dialogue lines are
        # deliberately short structural paragraphs; merging them collapses
        # document structure (sample 63 of longform corpus: ## headers
        # lost; cycle-44 audit: bold subheaders + numbered list items;
        # cycle-46 audit: novel sample 1323 had two dialogue paragraphs
        # like '"嗯，我很喜欢。"' merged into one block, losing the
        # turn-by-turn formatting).
        s = p.lstrip()
        if s.startswith('#') or s.startswith('- ') or s.startswith('* '):
            return True
        if s.startswith('**') and s.rstrip().endswith('**'):
            return True
        if re.match(r'^\d+[.。．)）]', s):
            return True
        # Chinese numbered section headers: "一、X" / "（一）X" / "(一)X"
        # Common in long-form Chinese essays (long_blog 一、 二、 三、)
        if re.match(r'^[一二三四五六七八九十]+[、,，]', s):
            return True
        if re.match(r'^[（(][一二三四五六七八九十]+[）)]', s):
            return True
        # Dialogue line (Chinese / Western quotes / Japanese 「」)
        if s and s[0] in '"“「':
            return True
        return False

    # cycle 226 N-2d: enumeration markers AS THEY SURVIVE the upstream
    # remove_three_part_structure (strips 首先/其次/最后; replaces 其次→
    # 另外/此外/'') and replace_phrases (此外→还有/再说/加之, 综上所述→
    # 总之/说到底/简单讲). The AI long-form pattern is "each enumerator
    # gets its own paragraph" — so the surviving paragraph heads are what
    # we want to merge into the previous block to break that rhythm.
    _ENUM_PARA_HEADS = (
        '综上', '此外', '另外', '总之', '总的来说', '总而言之', '再者',
        '还有', '接着', '然后', '再就是', '加之', '再说', '说到底', '简单讲',
    )

    def _starts_enum(p):
        s = p.lstrip()
        for m in _ENUM_PARA_HEADS:
            if s.startswith(m + '，') or s.startswith(m + ',') or s.startswith(m + '、'):
                return True
        return False

    result = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]

        # Enum-marker preferential merge (N-2d): if the NEXT paragraph starts
        # with a surviving enumeration head, merge into current with combined
        # length cap to avoid mega-paragraphs. Length cap 2.5x avg keeps the
        # merged block within human plausible range.
        if (i + 1 < len(paragraphs) and
            _starts_enum(paragraphs[i + 1]) and
            not _is_md_header(para) and
            not _is_md_header(paragraphs[i + 1]) and
            len(para) + len(paragraphs[i + 1]) < avg_len * 2.5 and
            random.random() < 0.45):
            merged = para + '\n' + paragraphs[i + 1]
            result.append(merged)
            i += 2
            continue

        # Randomly merge short adjacent paragraphs (skip markdown headers /
        # bullet items — those are deliberately short structural markers).
        if (i + 1 < len(paragraphs) and
            len(para) < avg_len * 0.6 and
            len(paragraphs[i + 1]) < avg_len * 0.6 and
            not _is_md_header(para) and
            not _is_md_header(paragraphs[i + 1]) and
            random.random() < 0.4):
            merged = para + '\n' + paragraphs[i + 1]
            result.append(merged)
            i += 2
            continue
        
        # Split long paragraphs
        if len(para) > avg_len * 1.5:
            sentences = re.split(r'([。！？])', para)
            mid = len(sentences) // 2
            # Ensure we split at a sentence boundary (every other element is punctuation)
            if mid % 2 == 1:
                mid -= 1
            part1 = ''.join(sentences[:mid])
            part2 = ''.join(sentences[mid:])
            if part1.strip() and part2.strip():
                result.append(part1.strip())
                result.append(part2.strip())
                i += 1
                continue
        
        result.append(para)
        i += 1

    # cycle 219 N-2b: fallback to push paragraph CV up for AI uniform
    # texts. When the main loop's 0.6/1.5 thresholds didn't fire (all
    # paragraphs in narrow band), CV stays low. To INCREASE variance,
    # we merge two adjacent paragraphs whose combined length would be
    # > 1.5x current average — this creates a long outlier on the
    # right tail. Only fires on real long text (>= 8 paragraphs) to
    # avoid distorting medium-length 5-paragraph compositions where
    # merging 2/5 over-shifts CV past human distribution.
    cv_after = _para_cv(result)
    if cv_after is not None and cv_after < 0.35 and len(result) >= 8:
        cn_lens = [len(re.findall(r'[一-鿿]', p)) for p in result]
        valid_lens = [l for l in cn_lens if l >= 20]
        avg_cn = sum(valid_lens) / len(valid_lens) if valid_lens else 100
        # Find adjacent pair whose combined length is the most above
        # average (creates a long outlier when merged).
        best = None
        for k in range(len(result) - 1):
            if _is_md_header(result[k]) or _is_md_header(result[k + 1]):
                continue
            if cn_lens[k] < 20 or cn_lens[k + 1] < 20:
                continue
            combined = cn_lens[k] + cn_lens[k + 1]
            # Want combined > 1.5x avg to push variance.
            if combined > avg_cn * 1.5:
                excess = combined - avg_cn
                if best is None or excess > best[0]:
                    best = (excess, k)
        if best is not None and random.random() < 0.6:
            k = best[1]
            merged = result[k] + '\n' + result[k + 1]
            result = result[:k] + [merged] + result[k + 2:]

    return join_paragraphs(result)
def reduce_punctuation(text):
    """Reduce excessive punctuation intelligently"""
    # Replace some semicolons with commas or periods
    parts = text.split('；')
    if len(parts) > 3:
        result_parts = [parts[0]]
        for i, part in enumerate(parts[1:], 1):
            # Alternate between comma and period
            if i % 2 == 0:
                result_parts.append('。' + part.lstrip())
            else:
                result_parts.append('，' + part)
        text = ''.join(result_parts)
    
    # Limit consecutive em dashes
    text = re.sub(r'——', '—', text)
    
    return text
def cap_transition_density(text, target=6.0):
    """Drop clause-initial transition phrases until density <= target.

    Runs AFTER all other humanize passes. Keeps transitions that are
    low-density already; removes excess probabilistically. Detect threshold
    fires at density > 8 per 1000 chars, so target 6 gives margin.
    """
    try:
        from ngram_model import _TRANSITION_PHRASES
    except ImportError:
        from scripts.ngram_model import _TRANSITION_PHRASES

    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cn_chars < 100:
        return text

    hits = sum(text.count(p) for p in _TRANSITION_PHRASES)
    density = hits / cn_chars * 1000
    if density <= target:
        return text

    remove_prob = min(0.9, (density - target) / density)

    for phrase in sorted(_TRANSITION_PHRASES, key=len, reverse=True):
        esc = re.escape(phrase)
        pattern = re.compile(r'(^|[。！？\n])(' + esc + r')([，,、])?')

        def sub(m):
            if random.random() < remove_prob:
                return m.group(1)
            return m.group(0)

        text = pattern.sub(sub, text)

    return text
def inject_sentence_particles(text, rate=0.15):
    """Append casual sentence-ending particles (吧/嘛/呗) to random statements.

    Intended for casual/social/chat scenes only. Skips questions/exclamations
    (already tonal) and sentences already ending in a particle. Short sentences
    skipped (too brittle), very long ones skipped (feels forced).
    """
    parts = re.split(r'([。！？])', text)
    particles = ['吧', '嘛', '呗']
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ''
        if punct in '！？':
            continue
        cn = sum(1 for c in sent if '\u4e00' <= c <= '\u9fff')
        if cn < 6 or cn > 40:
            continue
        rstripped = sent.rstrip()
        if rstripped and rstripped[-1] in '吧嘛呗呢啊哦嗯哈的了':
            continue
        if random.random() < rate:
            parts[i] = rstripped + random.choice(particles)
    return ''.join(parts)
def add_casual_expressions(text, casualness=0.3):
    """Inject casual/human expressions"""
    if casualness < 0.2:
        return text
    
    casual_openers = ['说实话', '其实', '确实', '讲真', '坦白说']
    casual_transitions = ['话说回来', '说到这个', '不过呢', '但是吧']
    casual_endings = ['就是这么回事', '差不多就这样', '大概就这些']
    
    sentences = re.split(r'([。！？])', text)
    result = []
    added = 0
    total = len(sentences) // 2
    max_additions = max(1, int(total * casualness * 0.3))
    
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        if added < max_additions and random.random() < casualness * 0.2:
            if i == 0:
                opener = random.choice(casual_openers)
                sent = opener + '，' + sent
            elif i > total:
                transition = random.choice(casual_transitions)
                sent = transition + '，' + sent
            added += 1
        
        result.append(sent + punct)
    
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1])
    
    return ''.join(result)
def shorten_paragraphs(text, max_length=150):
    """Break long paragraphs for social/chat scenes"""
    paragraphs = split_paragraphs(text)
    result = []
    
    for para in paragraphs:
        if len(para) > max_length:
            sentences = re.split(r'([。！？])', para)
            chunks = []
            current = ''
            
            for i in range(0, len(sentences) - 1, 2):
                sent = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
                if len(current) + len(sent) > max_length and current:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current += sent
            
            if current.strip():
                chunks.append(current.strip())
            
            result.extend(chunks)
        else:
            result.append(para)
    
    return join_paragraphs(result)
def diversify_vocabulary(text):
    """Reduce word repetition by using synonyms"""
    # Common overused words and their alternatives
    diversity_map = {
        '进行': ['做', '开展', '实施', '推进'],
        '实现': ['达到', '做到', '完成'],
        '提供': ['给出', '带来'],  # Cycle 63: dropped 拿出 (see WORD_SYNONYMS comment)
        '具有': ['有', '拥有', '带有'],
        # cycle 252: dropped '深入' — adjacency cascade "进一步深入" → "深入深入"
        # leaves no alt; effectively skip 进一步 in diversify_vocabulary path.
        # '进一步': drop entry (was ['深入'])
        '不断': ['持续', '一直', '始终'],
        # '有效' skipped: attributive/adj usage (有效证件) breaks with verb substitutes
        '积极': ['主动', '热心'],
        '促进': ['推动', '带动'],
        '加强': ['强化', '增强'],
        '提高': ['提升', '增加'],
        # cycle 164: dropped '重要' — same compound-breakage as
        # WORD_SYNONYMS upstream (重要性 → 核心性, 至关重要 → 至关核心
        # both broken).
    }
    
    for word, alts in diversity_map.items():
        count = text.count(word)
        if count > 2:
            # Protection guard: skip if word always inside protected terms
            # Computed from current text — words may be modified by prior loop iterations.
            import _humanize_data as _hd
            if _hd._USE_PROTECT_FLAG and _hd._PROTECTION_SET:
                _blocked = set()
                for t in _hd._PROTECTION_SET:
                    for m in re.finditer(re.escape(t), text):
                        for p in range(m.start(), m.end()):
                            _blocked.add(p)
                occ_positions = [m.start() for m in re.finditer(re.escape(word), text)]
                if occ_positions and all(p in _blocked for p in occ_positions):
                    continue
            # Keep first occurrence, replace subsequent
            first = True
            parts = text.split(word)
            result = [parts[0]]
            for part in parts[1:]:
                if first:
                    result.append(word)
                    first = False
                else:
                    result.append(random.choice(alts))
                result.append(part)
            text = ''.join(result)
    
    return text
