#!/usr/bin/env python3
"""N-1d diagnostic: scan humanize output for in-sentence character-ngram repetition.

A repeated 2-char window inside one sentence often signals a substitution-cascade
bug, e.g. cycle 213's "推 长期推动" — the synonym replacement reintroduced the
same character.

Filters out legitimate Chinese repetition patterns (一X一X, X来X去, etc.) and
keeps only candidates likely to be substitution artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))

SENT_SPLIT = re.compile(r'[。！？!?\n]+')

# Legitimate-repeat patterns we should NOT flag.
LEGIT_PATTERNS = [
    re.compile(r'一[一-鿿]一[一-鿿]'),  # 一步一步
    re.compile(r'[一-鿿]来[一-鿿]去'),  # 跑来跑去
    re.compile(r'[一-鿿]前[一-鿿]后'),  # 思前想后
    re.compile(r'[一-鿿]+又[一-鿿]+'),  # 又X又Y form
    re.compile(r'越[一-鿿]越[一-鿿]'),  # 越来越好
    re.compile(r'[一-鿿]([一-鿿])\1'),  # AA-doubled like 慢慢
]

# Tokens that occur frequently and benignly: discourse markers, very common verbs.
# If a bigram is one of these AND appears 2x, it's usually fine — too noisy.
COMMON_BIGRAMS = {
    '我们', '他们', '可以', '我的', '你的', '一个', '一种', '一些', '这个', '这些',
    '那个', '那些', '什么', '怎么', '因为', '所以', '但是', '如果', '虽然', '因此',
    '然后', '现在', '已经', '正在', '可能', '应该', '需要', '能够', '通过', '关于',
    '对于', '由于', '因为', '所以', '不是', '就是', '还有', '或者', '比如', '这样',
    '那样', '其实', '其中', '其他', '自己', '大家', '一直', '总是', '从来', '从未',
    '没有', '不过', '而且', '并且', '以及', '而是', '虽然', '尽管', '只是', '只有',
    '只要', '为了', '在于', '说明', '表示', '认为', '觉得', '感觉', '希望', '想要',
    '了一', '一下', '了解', '不仅', '在这', '这种', '一种', '研究', '工作', '问题',
    '时候', '时间', '方面', '方法', '方式', '过程', '阶段', '形成', '出现', '存在',
    '发展', '提高', '增加', '减少', '改变', '影响', '促进', '推动', '实现', '完成',
}


def is_legit(sent: str, bigram: str) -> bool:
    """Bigram appears multiple times — but inside one of the OK patterns?"""
    for pat in LEGIT_PATTERNS:
        if pat.search(sent):
            # crude: if any legit pattern fires anywhere in sent, skip
            return True
    return False


def find_in_sentence_repeats(text: str, min_count: int = 2,
                             max_gap: int = 20):
    """Yield (sent, bigram, count, gap) for each close-range in-sentence repeat.

    Only flags bigrams whose two occurrences are within `max_gap` chars of
    each other (close-range = more likely substitution-cascade bug than
    topical repetition).
    """
    for raw_sent in SENT_SPLIT.split(text):
        sent = raw_sent.strip()
        if len(sent) < 10:
            continue
        # Track positions of every Chinese-bigram in the original sentence.
        # We use sentence positions (not char-only filtered) so the gap is real.
        pos = {}  # bigram -> list of indices
        for i in range(len(sent) - 1):
            a, b = sent[i], sent[i+1]
            if not ('一' <= a <= '鿿' and '一' <= b <= '鿿'):
                continue
            pos.setdefault(a + b, []).append(i)
        for bg, idxs in pos.items():
            if len(idxs) < min_count:
                continue
            if bg in COMMON_BIGRAMS:
                continue
            if is_legit(sent, bg):
                continue
            # Find smallest gap between any pair
            min_g = min(idxs[i+1] - idxs[i] for i in range(len(idxs) - 1))
            if min_g > max_gap:
                continue
            yield sent, bg, len(idxs), min_g


def humanize_file(path: str) -> str:
    """Run humanize_cn on a file and return stdout."""
    env = dict(os.environ)
    env.setdefault('PYTHONHASHSEED', '0')
    out = subprocess.run(
        [sys.executable, 'scripts/humanize_cn.py', path, '--seed', '42'],
        cwd=str(REPO), capture_output=True, text=True, env=env, timeout=120,
    )
    return out.stdout


def humanize_text(text: str) -> str:
    """Run humanize_cn on stdin text (no file arg → reads stdin)."""
    env = dict(os.environ)
    env.setdefault('PYTHONHASHSEED', '0')
    out = subprocess.run(
        [sys.executable, 'scripts/humanize_cn.py', '--seed', '42'],
        cwd=str(REPO), capture_output=True, text=True, env=env,
        input=text, timeout=180,
    )
    return out.stdout


def collect_repeats(text: str, min_count: int = 2, max_gap: int = 20):
    """Collect a set of (sent, bigram) keys for diff comparison."""
    return {
        (sent, bg) for sent, bg, _, _ in find_in_sentence_repeats(text, min_count, max_gap)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--examples', action='store_true', help='Scan examples/sample_*.txt')
    ap.add_argument('--hc3', type=int, default=0,
                    help='Also scan first N HC3 ChatGPT samples (humanized)')
    ap.add_argument('--longform', type=int, default=0,
                    help='Scan N AI longform corpus samples starting at --longform-start')
    ap.add_argument('--longform-start', type=int, default=0,
                    help='Index to start longform scan from (default 0)')
    ap.add_argument('--min-count', type=int, default=2)
    ap.add_argument('--diff', action='store_true',
                    help='Only flag bigrams whose count went UP after humanize')
    ap.add_argument('-o', '--output', default='audit/in_sentence_repeats.json')
    args = ap.parse_args()

    findings = []
    diff_findings = []

    def process(label: str, src_text: str, out_text: str):
        for sent, bg, c, gap in find_in_sentence_repeats(out_text, args.min_count):
            findings.append({
                'source': label, 'sent': sent, 'bigram': bg,
                'count': c, 'gap': gap,
            })
        if args.diff:
            src_bigrams = Counter(
                bg for _, bg, _, _ in find_in_sentence_repeats(src_text, args.min_count)
            )
            out_bigrams = Counter(
                bg for _, bg, _, _ in find_in_sentence_repeats(out_text, args.min_count)
            )
            for bg, c_out in out_bigrams.items():
                if c_out > src_bigrams.get(bg, 0):
                    # find one example sent in out
                    for sent, b, cnt, gap in find_in_sentence_repeats(out_text, args.min_count):
                        if b == bg:
                            diff_findings.append({
                                'source': label, 'sent': sent, 'bigram': bg,
                                'count': cnt, 'gap': gap,
                                'src_count': src_bigrams.get(bg, 0),
                                'out_count': c_out,
                            })
                            break

    if args.examples:
        for name in ('sample_academic.txt', 'sample_general.txt',
                     'sample_social.txt', 'sample_long_blog.txt'):
            p = REPO / 'examples' / name
            if not p.exists():
                continue
            print(f'humanize {name}...', file=sys.stderr)
            src = p.read_text()
            out = humanize_file(str(p))
            process(name, src, out)

    if args.longform > 0:
        # Longform corpus lives one level above repo too
        lf_path = REPO.parent.parent / 'data' / 'ai_longform_corpus.jsonl'
        if not lf_path.exists():
            lf_path = REPO / 'data' / 'ai_longform_corpus.jsonl'
        if lf_path.exists():
            with lf_path.open() as f:
                for i, line in enumerate(f):
                    if i < args.longform_start:
                        continue
                    if i >= args.longform_start + args.longform:
                        break
                    rec = json.loads(line)
                    text = rec.get('text') or rec.get('content') or ''
                    if len(text) < 200:
                        continue
                    genre = rec.get('genre', '?')
                    print(f'humanize longform {i} ({genre})...', file=sys.stderr)
                    out = humanize_text(text)
                    process(f'lf:{i}:{genre}', text, out)

    if args.hc3 > 0:
        # HC3 lives one level above repo (claudeclaw/humanize/data/)
        hc3_path = REPO.parent.parent / 'data' / 'hc3_chinese_all.jsonl'
        if not hc3_path.exists():
            hc3_path = REPO / 'data' / 'hc3_chinese_all.jsonl'
        if hc3_path.exists():
            with hc3_path.open() as f:
                for i, line in enumerate(f):
                    if i >= args.hc3:
                        break
                    rec = json.loads(line)
                    answers = rec.get('chatgpt_answers') or []
                    if not answers:
                        continue
                    text = answers[0]
                    if len(text) < 80:
                        continue
                    print(f'humanize hc3 {i}...', file=sys.stderr)
                    out = humanize_text(text)
                    process(f'hc3:{i}', text, out)

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        'count': len(findings),
        'samples_with_findings': len(set(f['source'] for f in findings)),
        'findings': findings,
        'diff_count': len(diff_findings),
        'diff_findings': diff_findings,
    }, ensure_ascii=False, indent=2))

    # Group by bigram for actionable summary
    bg_counter = Counter(f['bigram'] for f in findings)
    print(f'\nTotal findings: {len(findings)}', file=sys.stderr)
    if args.diff:
        diff_counter = Counter(f['bigram'] for f in diff_findings)
        print(f'Diff findings (humanize-introduced): {len(diff_findings)}',
              file=sys.stderr)
        if diff_counter:
            print('Top humanize-introduced bigrams:', file=sys.stderr)
            for bg, n in diff_counter.most_common(20):
                print(f'  {bg}: {n}', file=sys.stderr)
    else:
        print(f'Top repeated bigrams:', file=sys.stderr)
        for bg, n in bg_counter.most_common(20):
            print(f'  {bg}: {n}', file=sys.stderr)
    print(f'Output: {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
