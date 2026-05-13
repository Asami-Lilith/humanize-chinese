#!/usr/bin/env python3
"""Core orchestration and CLI entry for Chinese AI Text Humanizer v2.0.

Main humanize() function, best_of_n search, scoring helpers, and CLI main().
"""

import sys
import re
import random
import os
import argparse
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level flags — toggled by CLI arguments
_USE_NOISE = True
_USE_STATS = True
_USE_CILIN = False

# Protection flags — domain-term protection via DomainWordsDict
_PROTECT_ENABLED = False
_USE_PROTECT_FLAG = False
_PROTECTION_SET = set()

_ACADEMIC_LR_MARKERS = (
    '本研究', '研究表明', '理论意义', '实践价值',
    '研究发现', '研究结果', '研究对象', '研究方法',
    '文献综述', '实证分析', '理论框架', '学术价值',
    '现实意义', '实践意义', '变量', '样本', '模型', '假设',
)

_LONGFORM_LR_CN_CHAR_THRESHOLD = 1500

# ── Lazy imports (resolved at first use via try/except chain) ──

def _count_chinese_chars(text):
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

try:
    from ngram_model import analyze_text as ngram_analyze
except ImportError:
    try:
        from scripts.ngram_model import analyze_text as ngram_analyze
    except ImportError:
        ngram_analyze = None

try:
    from _text_utils import join_paragraphs, split_paragraphs
except ImportError:
    from scripts._text_utils import join_paragraphs, split_paragraphs

from _humanize_data import (
    CONFIG, load_config, SCENES, WORD_SYNONYMS,
    PHRASE_REPLACEMENTS, PLAIN_REPLACEMENTS, REGEX_REPLACEMENTS,
    _AI_PATTERN_BLACKLIST,
    ACADEMIC_PRESERVE_WORDS, ACADEMIC_BLACKLIST_CANDIDATES,
    NOVEL_BLACKLIST_CANDIDATES,
    NOISE_EXPRESSIONS, NOISE_ACADEMIC_EXPRESSIONS,
    NOISE_ACADEMIC_CATEGORIES,
    _NARRATIVE_SAFE_CATEGORIES,
    _PARA_BOOST_ATTRIBUTION, _PARA_BOOST_SUBORDINATE,
    _PARA_BOOST_BARE_CONTINUATOR, _PARA_BOOST_REACTIONS,
    _PARA_INTERJECTION_NEUTRAL, _PARA_INTERJECTION_NOVEL,
    _LONGFORM_PARA_HEAD_MARKERS, _LONGFORM_STARTER_MARKERS,
    _CILIN_BLACKLIST, _CILIN_SOURCE_BLACKLIST,
    _filter_candidates_for_scene, _count_chinese_chars as _data_count_chinese_chars,
    pick_best_replacement, _compute_burstiness,
)
from _humanize_cilin import expand_with_cilin
from _humanize_pipeline import (
    remove_three_part_structure, replace_phrases,
    merge_short_sentences, split_long_sentences,
    vary_paragraph_rhythm, reduce_punctuation,
    cap_transition_density, inject_sentence_particles,
    add_casual_expressions, shorten_paragraphs,
    diversify_vocabulary,
)
from _humanize_strategies import (
    reduce_high_freq_bigrams, randomize_sentence_lengths,
    inject_noise_expressions, _apply_longform_mutation_profile,
    boost_para_cv_via_merge, boost_para_sent_len_cv,
    insert_short_interjection_paragraph,
    reduce_cross_para_3gram_repeat,
    _simple_synonym_pass,
)


DEFAULT_BEST_OF_N = 20
DEFAULT_SECONDARY_WEIGHT = 0.2


def _estimate_source_aiscore(text):
    """Quick pre-detect of how AI-like the input is. Returns 0-100 score or None."""
    try:
        from detect_cn import detect_patterns, calculate_score
    except ImportError:
        try:
            from scripts.detect_cn import detect_patterns, calculate_score
        except ImportError:
            return None
    try:
        issues, metrics = detect_patterns(text)
        return calculate_score(issues, metrics)
    except Exception:
        return None


def _clamp_0_100(value):
    return max(0.0, min(100.0, float(value)))


def _norm_linear(value, low, high, invert=False):
    if value is None or high == low:
        return 0.0
    raw = (float(value) - low) / (high - low)
    if invert:
        raw = 1.0 - raw
    return _clamp_0_100(raw * 100.0)


def _starter_entropy(text, width=2):
    sentences = [s.strip() for s in re.split(r'[。！？!?；;\n]+', text) if s.strip()]
    starters = {}
    total = 0
    for sent in sentences:
        chars = re.findall(r'[\u4e00-\u9fff]', sent)
        if len(chars) < width:
            continue
        key = ''.join(chars[:width])
        starters[key] = starters.get(key, 0) + 1
        total += 1
    if total < 5:
        return 0.0
    entropy = 0.0
    for count in starters.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _secondary_signal_details(text):
    """Return auxiliary best-of-n AI-likeness score and raw/capped features.

    These are deliberately not LR calibration inputs. They reuse already
    implemented but capped/disabled signals to sway candidate ranking only.
    """
    if not text or ngram_analyze is None:
        return {
            'score': 0.0,
            'bino': 0.0,
            'curv': 0.0,
            'mattr': 0.0,
            'starter_h': 0.0,
            'bino_s': 0.0,
            'curv_s': 0.0,
            'mattr_s': 0.0,
            'starter_s': 0.0,
        }
    try:
        analysis = ngram_analyze(text)
    except Exception:
        return {
            'score': 0.0,
            'bino': 0.0,
            'curv': 0.0,
            'mattr': 0.0,
            'starter_h': 0.0,
            'bino_s': 0.0,
            'curv_s': 0.0,
            'mattr_s': 0.0,
            'starter_s': 0.0,
        }

    bino = (analysis.get('bino') or {}).get('mean_lp_diff') or 0.0
    curv = (analysis.get('curv') or {}).get('curvature_mean') or 0.0
    mattr = analysis.get('char_mattr') or 0.0
    starter_h = _starter_entropy(text, width=2)

    # Direction: higher score means more AI-like. Binoculars diff is less
    # negative on HC3 ChatGPT; curvature is higher; MATTR and starter entropy
    # are lower when wording/openers are more repetitive.
    bino_s = _norm_linear(bino, -4.6, -2.2)
    curv_s = _norm_linear(curv, 0.0, 1.2)
    mattr_s = _norm_linear(mattr, 0.50, 0.72, invert=True)
    starter_s = _norm_linear(starter_h, 1.2, 2.4, invert=True)

    score = (
        0.35 * bino_s +
        0.25 * curv_s +
        0.25 * mattr_s +
        0.15 * starter_s
    )
    return {
        'score': round(_clamp_0_100(score), 2),
        'bino': round(float(bino), 4),
        'curv': round(float(curv), 4),
        'mattr': round(float(mattr), 4),
        'starter_h': round(float(starter_h), 4),
        'bino_s': round(bino_s, 2),
        'curv_s': round(curv_s, 2),
        'mattr_s': round(mattr_s, 2),
        'starter_s': round(starter_s, 2),
    }
def _compute_secondary_signal(text):
    return _secondary_signal_details(text)['score']
def _pick_lr_scene(text):
    """Pick the LR scorer used to rank best-of-n candidates."""
    academic_hits = sum(1 for marker in _ACADEMIC_LR_MARKERS if marker in text)
    if _count_chinese_chars(text) >= _LONGFORM_LR_CN_CHAR_THRESHOLD:
        return 'longform'
    if academic_hits >= 2:
        return 'academic'
    return 'general'
def _format_best_of_debug(seed, scene_picked, lr_scores, secondary, rank_score,
                          fused_score, top_contribs):
    top = ', '.join(f'{name}={value:+.2f}' for name, value in top_contribs[:3])
    return (
        f'best_of_n seed={seed} scene_picked={scene_picked} '
        f'LR_general={lr_scores.get("general", "NA")} '
        f'LR_academic={lr_scores.get("academic", "NA")} '
        f'LR_longform={lr_scores.get("longform", "NA")} '
        f'secondary={secondary["score"]} '
        f'[bino={secondary["bino"]} curv={secondary["curv"]} '
        f'mattr={secondary["mattr"]} starter_h={secondary["starter_h"]}] '
        f'rank={rank_score:.2f} fused={fused_score} top_3_contributions=[{top}]'
    )
def humanize(text, scene='general', aggressive=False, seed=None, best_of_n=DEFAULT_BEST_OF_N,
             style=None, debug_best_of_n=False, score_mode='lr',
             secondary_weight=DEFAULT_SECONDARY_WEIGHT, protect=False):
    """Apply all humanization transformations in order.

    Graduated intensity based on source AI-score (pre-detect):
      - score < 15 (conservative): only phrase replacement + punctuation cleanup
      - score 15-39 (moderate): + restructure + lighter bigram substitution
      - score >= 40 (full): entire pipeline including noise injection
    Aggressive flag forces 'full' tier.

    best_of_n: if set to an integer, runs humanize N times with different seeds
    and returns the output that scores lowest on the scene-aware LR ensemble
    (requires scripts/lr_coef_*.json). Useful when minimizing LR score matters
    more than latency.

    Rationale: HC3 benchmark showed that full pipeline on already-clean text
    (source score < 15) adds spurious AI patterns (段落均匀/熵低) via noise
    injection, sometimes INCREASING detected score. Tiered intensity avoids this.
    """
    if best_of_n and best_of_n > 1:
        try:
            from ngram_model import compute_lr_score
        except ImportError:
            from scripts.ngram_model import compute_lr_score
        if score_mode not in ('lr', 'fused', 'lr+rule'):
            raise ValueError('score_mode must be one of: lr, fused, lr+rule')
        detect_for_rule = None
        if score_mode in ('fused', 'lr+rule') or debug_best_of_n:
            try:
                from detect_cn import calculate_score, detect_patterns
            except ImportError:
                from scripts.detect_cn import calculate_score, detect_patterns
            detect_for_rule = (calculate_score, detect_patterns)
        base_seed = seed if seed is not None else 42
        candidates = []
        for i in range(best_of_n):
            s = base_seed + i
            out = humanize(text, scene=scene, aggressive=aggressive,
                           seed=s, best_of_n=None, style=style,
                           protect=protect)
            lr_scene = _pick_lr_scene(out)
            if lr_scene == 'longform':
                out = _apply_longform_mutation_profile(
                    out, mutation_seed=s, scene=scene, style=style)
                lr_scene = _pick_lr_scene(out)
            lr = compute_lr_score(out, scene=lr_scene)
            score = lr['score'] if lr else 50
            rule_score = 0
            if detect_for_rule:
                calculate_score, detect_patterns = detect_for_rule
                issues, metrics = detect_patterns(out)
                rule_score = calculate_score(issues, metrics)
            fused = round(0.8 * score + 0.2 * rule_score)
            secondary = _secondary_signal_details(out)
            if score_mode == 'fused':
                rank_score = fused + secondary_weight * secondary['score']
                rank_tiebreak = score
            elif score_mode == 'lr+rule':
                rank_score = score + secondary_weight * secondary['score']
                rank_tiebreak = rule_score
            else:
                rank_score = score + secondary_weight * secondary['score']
                rank_tiebreak = 0
            if debug_best_of_n:
                lr_scores = {}
                for debug_scene in ('general', 'academic', 'longform'):
                    debug_lr = compute_lr_score(out, scene=debug_scene)
                    lr_scores[debug_scene] = debug_lr['score'] if debug_lr else 'NA'
                top_contribs = lr.get('top_contributions', []) if lr else []
                print(_format_best_of_debug(s, lr_scene, lr_scores, secondary,
                                            rank_score, fused, top_contribs),
                      file=sys.stderr)
            candidates.append((rank_score, rank_tiebreak, s, out))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][3]

    if seed is not None:
        random.seed(seed)

    config = SCENES.get(scene, SCENES['general'])
    casualness = config.get('casualness', 0.3)
    if aggressive:
        casualness = min(1.0, casualness + 0.3)

    source_score = _estimate_source_aiscore(text)
    # Tier thresholds calibrated on HC3-Chinese: most naturally-written ChatGPT
    # scores 5-25 on detect_cn. Full pipeline on very-clean input (< 5) adds
    # spurious noise. Moderate tier skips noise/sentence-randomization but keeps
    # everything else. Trade picks up most of the full-tier gains with fewer regressions.
    if aggressive or source_score is None or source_score >= 25:
        tier = 'full'
    elif source_score >= 5:
        tier = 'moderate'
    else:
        tier = 'conservative'

    # ── Protection: build term set for guard injection ──
    # Guards rebuild blocked positions from current text each check
    # because prior passes (restructure, merge, split) shift character positions.
    _protection_set = set()
    if protect or _PROTECT_ENABLED:
        try:
            from _humanize_protect import get_layer as _get_protect_layer
        except ImportError:
            _get_protect_layer = None
        if _get_protect_layer:
            _layer = _get_protect_layer()
            if _layer.is_ready():
                _domains = _layer.detect_domains(text, top_n=5)
                if _domains:
                    _domain_names = [d for d, _ in _domains]
                    _protection_set = _layer.extract_protected_terms(
                        text, _domain_names)

    import _humanize_data
    _humanize_data._PROTECTION_SET = _protection_set
    _humanize_data._USE_PROTECT_FLAG = bool(_protection_set)

    # Pass 1: Structure cleanup — always run (safe, targeted)
    text = remove_three_part_structure(text)
    text = replace_phrases(text, casualness)

    # Pass 2: Deep sentence restructuring — all tiers (with moderate strength in conservative)
    try:
        from restructure_cn import deep_restructure
    except ImportError:
        try:
            from scripts.restructure_cn import deep_restructure
        except ImportError:
            deep_restructure = None
    if deep_restructure:
        # Conservative keeps restructure but with aggressive=False to be gentler
        text = deep_restructure(text, aggressive=aggressive, scene=scene)

    # Pass 2b: Sentence merge/split
    if config.get('merge_short', False):
        text = merge_short_sentences(text)
    if config.get('split_long', False):
        text = split_long_sentences(text)

    # Pass 3: Rhythm and variety — diversify all tiers, rhythm only moderate+
    text = reduce_punctuation(text)
    text = diversify_vocabulary(text)
    if tier != 'conservative' and config.get('rhythm_variation', False):
        text = vary_paragraph_rhythm(text)

    # Pass 4: Scene-specific — only at full tier
    if tier == 'full':
        if config.get('add_casual', False) or aggressive:
            text = add_casual_expressions(text, casualness)
            # Sentence-end particles (吧/嘛/呗) — cycle 14 tried but caused xhs regression
            # (seed=42: 53 → 59). Random state shift + downstream interaction. Parked.
        if config.get('shorten_paragraphs', False):
            text = shorten_paragraphs(text)

    # ── Perplexity-boosting strategies — tier-gated ──
    # Bigram substitution active in moderate+full (safe, targeted)
    if tier != 'conservative':
        bigram_strength = 0.5 if aggressive else 0.3
        if tier == 'moderate':
            bigram_strength *= 0.6
        # Route bigram substitution through the novel-register filter when
        # --style novel is active. NOVEL_BLACKLIST_CANDIDATES strips the
        # overtly colloquial / book-Chinese substitutes ('搞'/'拉高'/'业已'/
        # '早就') that break narrative register, while keeping
        # ('察觉'/'识破') that academic mode rejects.
        bigram_scene = 'novel' if style == 'novel' else scene
        text = reduce_high_freq_bigrams(text, strength=bigram_strength, scene=bigram_scene)

    # Noise + sentence randomization only at full tier — these are the operations
    # that on HC3 sometimes added spurious AI patterns to already-clean text.
    if tier == 'full' and _USE_NOISE:
        noise_density = 0.25 if aggressive else 0.15
        # Novel/fiction register: noise injection (regardless of expression
        # subset) frequently lands on prepositional or vocative sentence heads
        # ('作为...' / '人物名+verb') and reads as awkward. Lean on word
        # substitutions + transition cap + paraphrase replacement for delta
        # in novel mode instead.
        if style != 'novel':
            # Cycle 104: route academic scene through NOISE_ACADEMIC_EXPRESSIONS
            # subset (hedging / self_correction / uncertainty). Cycle 54 tried
            # this and lost -2 academic hero, but cycles 76-101 since cleaned
            # the pool of self-defeating entries — second attempt with the
            # tighter pool. Audit found 20+ filler / transition_casual /
            # personal injections in academic samples ('不瞒你说' / '说到底' /
            # '讲真' / '约莫' / '估摸着') that read off-register.
            noise_style = 'academic' if scene == 'academic' else 'general'
            text = inject_noise_expressions(text, density=noise_density, style=noise_style)
        text = randomize_sentence_lengths(text, aggressive=aggressive, seed=seed)

    # v5 P1 humanize counter-measure for stat_low_para_sent_len_cv. The
    # truncation variant (boost_para_sent_len_cv) was shelved because
    # adding a period bumps punct_density and cancels the para-CV win.
    # The merge variant lifts a uniform paragraph by combining two
    # adjacent short-medium sentences with a comma — removing one
    # period, often pushing the merged sentence over the long threshold,
    # both of which point LR away from AI. n=20 sweep at target=0.40
    # showed avg LR delta -0.95 with zero regressions.
    text = boost_para_cv_via_merge(text)

    # v5 P1.2 humanize counter-measure for paragraph_length_cv (LR coef
    # -1.99 on longform). For multi-paragraph text whose paragraph
    # length CV is below 0.60, insert a single 22-24 cn-char reflection
    # paragraph after one of the longer existing paragraphs. Skipped
    # for novel style (narrative paragraphs differ; reflective
    # interjections read off-register). n=30 by-genre sweep:
    #   novel    skipped 10/10 ✓
    #   academic fired 4/10, LR delta 0.00 (neutral)
    #   news     fired 10/10, LR avg -2.10 (3 down / 1 up / 6 same)
    text = insert_short_interjection_paragraph(text, target_cv=0.60,
                                               style=style, seed=seed)

    # v5 P1.3 humanize counter-measure for cross_para_3gram_repeat (LR
    # coef +2.24 on longform). Replaces a few CiLin-known 2-char words
    # that recur across paragraphs with scene-filtered synonyms,
    # breaking the cross-paragraph trigram repetition. n=20 sweep at
    # max_replacements=4: fired 20/20, LR delta avg -1.65, zero
    # regressions.
    text = reduce_cross_para_3gram_repeat(text, max_replacements=4,
                                          scene=scene, style=style,
                                          seed=seed)

    # Final transition cap — AI overuses 首先/然而/此外/因此 etc, detect fires
    # density > 8/1000 chars. Cap at 6 to leave margin. Preserves text that's
    # already under the threshold.
    # Long-form (novel/blog) humans use far fewer transitions (d=0.92 gap vs
    # AI). Drop cap target on long text so novel humanize approaches human 2.4
    # density instead of staying at AI's 4.4 baseline.
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    trans_target = 3.0 if cn_chars >= 1500 else 6.0
    text = cap_transition_density(text, target=trans_target)

    # Novel/fiction register: strip overused AI-style intensifiers.
    # Spot-check on 20 \u7384\u5e7b samples showed \u300c\u5341\u5206/\u975e\u5e38/\u6781\u5176/\u683c\u5916/\u6781\u4e3a/\u6781\u5ea6/
    # \u5c24\u4e3a/\u9887\u4e3a\u300d+ adj appears ~25-28 times per 20-sample batch as an AI
    # mannerism. Negative lookaheads exclude the two false positives we
    # observed: '\u5341\u5206\u949f' (time noun) and '\u975e\u5e38\u89c4' (adv prefix).
    # Skip '\u65e0\u6bd4' (\u53e5\u5c3e idiomatic, deletion would break clauses) and
    # '\u76f8\u5f53' (quantifier, '\u76f8\u5f53\u591a/\u76f8\u5f53\u957f' \u2260 intensifier).
    if style == 'novel':
        text = re.sub(r'\u5341\u5206(?![\u949f\u4e4b])', '', text)
        text = re.sub(r'\u975e\u5e38(?![\u89c4])', '', text)
        text = re.sub(r'\u6781\u5176', '', text)
        text = re.sub(r'\u683c\u5916', '', text)
        text = re.sub(r'\u6781\u4e3a', '', text)
        text = re.sub(r'\u6781\u5ea6', '', text)
        text = re.sub(r'\u5c24\u4e3a', '', text)
        text = re.sub(r'\u9887\u4e3a', '', text)

    # Clean up artifacts
    text = re.sub(r'[，,]{2,}', '，', text)  # Remove double commas
    text = re.sub(r'[。]{2,}', '。', text)    # Remove double periods
    text = re.sub(r'\n{3,}', '\n\n', text)    # Normalize newlines
    text = re.sub(r'，。', '。', text)          # Remove comma before period
    text = re.sub(r'。，', '。', text)          # Remove period before comma
    text = re.sub(r'(版本(?:显著|明显|可观))，(提升了)', r'\1\2', text)
    
    # ── Final verification loop (stats-optimized) ──
    # If perplexity is still too low, do a targeted second pass on worst sentences
    if _USE_STATS and ngram_analyze:
        stats = ngram_analyze(text)
        ppl = stats.get('perplexity', 0)
        # Threshold: if perplexity is in the "too smooth" zone, try to improve.
        # D-5 (cycle 31): raised 200 → 350 to cover the typical humanized-output
        # perplexity range (~250-300) where indicators still fire.
        if 0 < ppl < 350 and len(text) >= 100:
            sentences = re.split(r'([。！？])', text)
            # Score each sentence
            sent_scores = []
            for i in range(0, len(sentences) - 1, 2):
                s = sentences[i]
                if len(s.strip()) < 5:
                    continue
                s_stats = ngram_analyze(s)
                sent_scores.append((i, s_stats.get('perplexity', 0)))
            
            if sent_scores:
                # Sort by perplexity ascending (worst = most predictable first)
                sent_scores.sort(key=lambda x: x[1])
                # Try to improve the worst 20% (at most 5 sentences)
                n_fix = min(5, max(1, len(sent_scores) // 5))
                
                # Use a different random seed for the second pass
                if seed is not None:
                    random.seed(seed + 1)
                
                for idx, _ in sent_scores[:n_fix]:
                    sent = sentences[idx]
                    # Try each replacement on this sentence
                    sorted_phrases = sorted(PLAIN_REPLACEMENTS.keys(), key=len, reverse=True)
                    for phrase in sorted_phrases:
                        if phrase in sent:
                            alternatives = PLAIN_REPLACEMENTS[phrase]
                            if isinstance(alternatives, str):
                                alternatives = [alternatives]
                            best = pick_best_replacement(sent, phrase, alternatives)
                            sentences[idx] = sent.replace(phrase, best, 1)
                            break  # one fix per sentence to avoid over-rewriting
                
                text = ''.join(sentences)

    
    return text.strip()
def main():
    parser = argparse.ArgumentParser(description='中文 AI 文本人性化 v2.0')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--scene', default='general',
                       choices=['general', 'social', 'tech', 'formal', 'chat'],
                       help='场景 (default: general)')
    parser.add_argument('--style', help='写作风格 (调用 style_cn.py)')
    parser.add_argument('-a', '--aggressive', action='store_true', help='激进模式')
    parser.add_argument('--seed', type=int, help='随机种子（可复现）')
    parser.add_argument('--best-of-n', type=int, default=DEFAULT_BEST_OF_N, metavar='N',
                        help=f'运行 N 次 humanize 取 LR 分数最低的那次（默认 {DEFAULT_BEST_OF_N}，N 倍延迟，0 关闭）')
    parser.add_argument('--debug-best-of-n', action='store_true',
                       help='打印 best-of-n 每个候选的 LR scene、分数和主要贡献（stderr）')
    parser.add_argument('--score-mode', default='lr', choices=['lr', 'fused', 'lr+rule'],
                       help='best-of-n 排序方式：lr=scene-aware LR；fused=0.8*LR+0.2*rule；lr+rule=LR 优先、rule 打破平局')
    parser.add_argument('--secondary-weight', type=float, default=DEFAULT_SECONDARY_WEIGHT,
                       help=f'best-of-n secondary signal 权重（默认 {DEFAULT_SECONDARY_WEIGHT}，0 关闭）')
    parser.add_argument('--no-stats', action='store_true',
                       help='跳过统计优化（困惑度反馈），回退到纯规则替换')
    parser.add_argument('--no-noise', action='store_true',
                       help='跳过噪声策略（句长随机化 + 噪声表达插入）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（= --no-stats --no-noise），只跑短语替换 + 结构清理')
    parser.add_argument('--cilin', action='store_true',
                       help='用 CiLin 同义词词林扩展候选（~40K 词 vs 手工 200 词）')
    parser.add_argument('--protect', action='store_true',
                       help='启用领域术语保护（DomainWordsDict），专业术语不被改写')

    args = parser.parse_args()

    # Toggle stats optimization
    global _USE_STATS
    _USE_STATS = not (args.no_stats or args.quick)
    import _humanize_data
    _humanize_data._USE_STATS = _USE_STATS

    # Toggle noise strategies
    global _USE_NOISE
    _USE_NOISE = not (args.no_noise or args.quick)
    _humanize_data._USE_NOISE = _USE_NOISE

    # Toggle CiLin expansion
    global _USE_CILIN
    _USE_CILIN = args.cilin
    _humanize_data._USE_CILIN = _USE_CILIN

    # Toggle protection
    global _PROTECT_ENABLED
    _PROTECT_ENABLED = args.protect
    
    # Read input
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f'错误: 文件未找到 {args.file}', file=sys.stderr)
            sys.exit(1)
    else:
        text = sys.stdin.read()
    
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)
    
    # Humanize
    result = humanize(text, args.scene, args.aggressive, args.seed,
                       best_of_n=args.best_of_n, style=args.style,
                       debug_best_of_n=args.debug_best_of_n,
                       score_mode=args.score_mode,
                       secondary_weight=args.secondary_weight,
                       protect=args.protect)
    
    # Apply style if specified
    if args.style:
        try:
            from style_cn import apply_style
        except ImportError:
            try:
                from scripts.style_cn import apply_style
            except ImportError:
                apply_style = None

        if apply_style:
            result = apply_style(result, args.style, humanize_first=False, seed=args.seed)
        else:
            print('警告: 未找到风格转换模块', file=sys.stderr)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        style_info = f' (风格: {args.style})' if args.style else ''
        scene_info = f' (场景: {args.scene})'
        print(f'✓ 已保存到 {args.output}{scene_info}{style_info}')
    else:
        print(result)

if __name__ == '__main__':
    main()
