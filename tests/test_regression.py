import os
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault('PYTHONHASHSEED', '0')

from _text_utils import split_paragraphs  # noqa: E402
from humanize_cn import (  # noqa: E402
    _apply_longform_mutation_profile,
    _compute_secondary_signal,
    _pick_lr_scene,
    humanize,
)
from detect_cn import calculate_score, detect_patterns  # noqa: E402
from ngram_model import compute_lr_score  # noqa: E402


# Floors are upper bounds (lower fused score = better humanization). Set to
# current PYTHONHASHSEED=0 / seed=42 / best_of_n=10 measurement plus a small
# safety margin so unintended regressions trip the gate. Tighten as quality
# improves; current target is v5.0.0 baseline (academic 28 / general 38 /
# social 31 / long_blog 39).
HERO_FLOORS = {
    'sample_academic.txt': 50,
    'sample_general.txt': 45,
    'sample_social.txt': 30,
    'sample_long_blog.txt': 46,
}


def fused_score(text):
    issues, metrics = detect_patterns(text)
    rule_score = calculate_score(issues, metrics)
    lr_result = compute_lr_score(text)
    if lr_result is None:
        return rule_score
    return round(0.2 * rule_score + 0.8 * lr_result['score'])


class RegressionTests(unittest.TestCase):
    def hero_texts(self):
        for name in HERO_FLOORS:
            yield name, (ROOT / 'examples' / name).read_text(encoding='utf-8')

    def test_hero_floors(self):
        for name, text in self.hero_texts():
            with self.subTest(name=name):
                rewritten = humanize(text, seed=42)
                self.assertLessEqual(fused_score(rewritten), HERO_FLOORS[name])

    def test_paragraph_preservation(self):
        # vary_paragraph_rhythm and insert_short_interjection_paragraph can
        # legitimately merge or insert paragraphs to break uniform rhythm
        # (v5 path para_sent_len_cv signal). Allow up to 2 paragraphs lost
        # before flagging as a content-eating bug.
        for name, text in self.hero_texts():
            with self.subTest(name=name):
                rewritten = humanize(text, seed=42)
                before = len(split_paragraphs(text))
                after = len(split_paragraphs(rewritten))
                self.assertGreaterEqual(after, before - 2,
                                        msg=f'{name}: {before} -> {after} paragraphs')

    def test_determinism(self):
        text = (ROOT / 'examples' / 'sample_general.txt').read_text(encoding='utf-8')
        outputs = [humanize(text, seed=42) for _ in range(3)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_crlf_handling(self):
        text = '第一段内容比较长，用来验证段落切分。\r\n\r\n第二段内容也比较长，用来验证 CRLF。'
        self.assertEqual(len(split_paragraphs(text)), 2)
        humanize(text, seed=42)

    def test_empty_input(self):
        for text in ['', '   ', '\n\n\n']:
            with self.subTest(repr=repr(text)):
                humanize(text, seed=42)

    def test_pick_lr_scene(self):
        academic = '本研究说明了一个问题。研究表明，这个变量会影响样本。'
        longform = '普通文本' * 400
        general = '普通文本，只有一个研究表明 marker。'
        self.assertEqual(_pick_lr_scene(academic), 'academic')
        self.assertEqual(_pick_lr_scene(longform), 'longform')
        self.assertEqual(_pick_lr_scene(general), 'general')

    def test_longform_mutations_safe(self):
        text = (ROOT / 'examples' / 'sample_long_blog.txt').read_text(encoding='utf-8')
        candidate = humanize(text, seed=44, best_of_n=None)
        mutated = _apply_longform_mutation_profile(candidate, mutation_seed=44)

        before = len(split_paragraphs(candidate))
        after_paragraphs = split_paragraphs(mutated)

        self.assertTrue(all(p.strip() for p in after_paragraphs))
        self.assertGreaterEqual(len(after_paragraphs), before - 2)

    def test_secondary_signal(self):
        empty_score = _compute_secondary_signal('')
        self.assertGreaterEqual(empty_score, 0)
        self.assertLessEqual(empty_score, 100)

        hc3_path = ROOT.parents[1] / 'data' / 'hc3_chinese_all.jsonl'
        if not hc3_path.exists():
            self.skipTest('HC3-Chinese dataset not available')

        evals_dir = ROOT / 'evals'
        if str(evals_dir) not in sys.path:
            sys.path.insert(0, str(evals_dir))
        from run_hc3_benchmark import load_hc3  # noqa: E402

        samples = load_hc3(str(hc3_path), n=12, seed=42)
        human_scores = [_compute_secondary_signal(s['human_answer']) for s in samples]
        ai_scores = [_compute_secondary_signal(s['chatgpt_answer']) for s in samples]
        human_avg = sum(human_scores) / len(human_scores)
        ai_avg = sum(ai_scores) / len(ai_scores)
        self.assertNotEqual(round(human_avg, 2), round(ai_avg, 2))
        self.assertGreater(ai_avg, human_avg)


if __name__ == '__main__':
    unittest.main()
