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
from humanize_cn import humanize  # noqa: E402
from detect_cn import calculate_score, detect_patterns  # noqa: E402
from ngram_model import compute_lr_score  # noqa: E402


HERO_FLOORS = {
    'sample_academic.txt': 30,
    'sample_general.txt': 42,
    'sample_social.txt': 42,
    'sample_long_blog.txt': 20,
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
        for name, text in self.hero_texts():
            with self.subTest(name=name):
                rewritten = humanize(text, seed=42)
                self.assertEqual(len(split_paragraphs(rewritten)), len(split_paragraphs(text)))

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


if __name__ == '__main__':
    unittest.main()
