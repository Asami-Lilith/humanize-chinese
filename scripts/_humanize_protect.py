#!/usr/bin/env python3
"""
Protection Layer for Humanize-Chinese
基于 DomainWordsDict 专业词库的保护层，防止专业术语被改写。

加载流程（两次尝试）：
  1. 优先从 scripts/data/DomainWordsDict/*.json 加载（预编译缓存，469 ms / 68 domains）
  2. 回退从 thirdparty/DomainWordsDict/*.txt 加载

扫描方式：
  每个文本位置对长度 2-10 的子串做 bisect 二分查找，
  在排序后的 [(term, domain, weight)] 列表中查找精确匹配。
  复杂度 O(text_len × 9 × log N)。

使用方式:
  1. humanize_cn.py 通过 --protect 标志自动调用本模块
  2. 也可独立使用:  from _humanize_protect import ProtectLayer

数据来源: https://github.com/liuhuanyong/DomainWordsDict
"""

import os
import re
import json
import bisect
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_THIRDPARTY_DIR = os.path.join(_SCRIPT_DIR, '..', 'thirdparty', 'DomainWordsDict')
_JSON_CACHE_DIR = os.path.join(_SCRIPT_DIR, 'data', 'DomainWordsDict')
DOMAIN_DICT_DIR = os.environ.get('DOMAIN_DICT_DIR', _THIRDPARTY_DIR)

_NOISE_DOMAINS = {
    '人物名称', '地点名称', '诗词歌赋', '民间习俗', '世界宗教',
    '新番动漫', '网络游戏', '网络文学', '网络用语', '休闲活动',
    '音乐歌曲', '电影影视', '敏感用词', '组织机构',
    '市场购物', '手机数码', '美容美发', '家居装饰', '餐饮食品',
    '纺织服装', '广告传媒', '办公文教', '人力招聘',
    '航空航天', '安全工程', '期货期权',
}

MIN_WEIGHT_FOR_INDEX = 3
MIN_WEIGHT_FOR_PROTECT = 5

MIN_TERM_LEN = 2
MAX_TERM_LEN = 10

MIN_DOMAIN_SCORE_RATIO = 0.0035
_ASCII_RE = re.compile(r'^[a-zA-Z0-9\s\.\,\-\'\"\(\)\&\/\#]+$')


def _is_noise_term(term):
    if len(term) < MIN_TERM_LEN:
        return True
    if len(term) > MAX_TERM_LEN:
        return True
    if _ASCII_RE.match(term):
        return True
    return False


class ProtectLayer:
    """保护层：领域检测 + 术语保护

    数据从预编译 JSON 缓存加载（scripts/data/DomainWordsDict/），
    文本扫描使用 bisect 二分查找在排序后的术语列表中精确匹配。
    """

    def __init__(self, domain_dir=DOMAIN_DICT_DIR):
        self.domain_dir = domain_dir
        self._sorted_terms = []    # [(term, domain, weight)] 按 term 排序
        self.domain_terms = defaultdict(list)
        self._loaded = False
        self._protected_domains = None
        self._term_count = 0
        self._last_scan_result = None
        self._last_scan_text = None

    def _load_from_json_cache(self):
        if not os.path.isdir(_JSON_CACHE_DIR):
            return False

        raw = defaultdict(dict)
        for fname in os.listdir(_JSON_CACHE_DIR):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(_JSON_CACHE_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            domain_cn = data.get('domain_cn', '')
            if domain_cn in _NOISE_DOMAINS:
                continue
            if not data.get('sorted'):
                continue
            for term, weight in data.get('terms', []):
                raw[term][domain_cn] = max(raw[term].get(domain_cn, 0), weight)

        if not raw:
            return False

        for term, domain_dict in raw.items():
            for domain, weight in domain_dict.items():
                self._sorted_terms.append((term, domain, weight))
                self.domain_terms[domain].append((term, weight))

        self._sorted_terms.sort(key=lambda x: x[0])
        for domain in self.domain_terms:
            self.domain_terms[domain].sort(key=lambda x: x[1], reverse=True)
        self._term_count = len(raw)
        return True

    def _load_from_txt(self):
        if not os.path.isdir(self.domain_dir):
            return False

        term_index = defaultdict(dict)
        for fname in os.listdir(self.domain_dir):
            if not fname.endswith('.txt'):
                continue
            domain = fname.replace('.txt', '')
            if domain in _NOISE_DOMAINS:
                continue

            fpath = os.path.join(self.domain_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) < 2:
                            continue
                        term = parts[0].strip()
                        if _is_noise_term(term):
                            continue
                        try:
                            weight = int(parts[1].strip())
                        except ValueError:
                            continue
                        if weight < MIN_WEIGHT_FOR_INDEX:
                            continue
                        term_index[term][domain] = max(
                            term_index[term].get(domain, 0), weight)
                        self.domain_terms[domain].append((term, weight))
            except Exception:
                continue

        if not term_index:
            return False

        for domain in self.domain_terms:
            self.domain_terms[domain].sort(key=lambda x: x[1], reverse=True)

        for term, domain_dict in term_index.items():
            for domain, weight in domain_dict.items():
                self._sorted_terms.append((term, domain, weight))
        self._sorted_terms.sort(key=lambda x: x[0])
        self._term_count = len(term_index)
        return True

    def load(self):
        if self._loaded:
            return

        if self._load_from_json_cache():
            self._loaded = True
            return

        if self._load_from_txt():
            self._loaded = True
            return

    def is_ready(self):
        if not self._loaded:
            self.load()
        return self._term_count > 0

    def _scan_text(self, text):
        """单遍扫描文本，使用 bisect 二分查找精确匹配术语。

        Returns:
            list[tuple[str, str, int, int]]:
                (term, domain, weight, start_pos)
        """
        matched = []
        n = len(text)
        st = self._sorted_terms
        for start in range(n):
            for length in range(MIN_TERM_LEN, MAX_TERM_LEN + 1):
                end = start + length
                if end > n:
                    break
                chunk = text[start:end]
                pos = bisect.bisect_left(st, (chunk, '', -1))
                if pos < len(st) and st[pos][0] == chunk:
                    while pos < len(st) and st[pos][0] == chunk:
                        term, domain, weight = st[pos]
                        matched.append((term, domain, weight, start))
                        pos += 1
        return matched

    def detect_domains(self, text, top_n=3):
        if not self._loaded:
            self.load()

        scan_result = self._scan_text(text)
        self._last_scan_result = scan_result
        self._last_scan_text = text

        domain_scores = defaultdict(float)
        text_len = len(text)
        for term, domain, weight, start in scan_result:
            domain_scores[domain] += 1.0 / weight

        if not domain_scores:
            return []

        norm = max(text_len, 1)
        ranked = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = [(d, round(s / norm, 6)) for d, s in ranked]
        ranked = [(d, s) for d, s in ranked if s >= MIN_DOMAIN_SCORE_RATIO]
        return ranked[:top_n]

    def extract_protected_terms(self, text, domains_or_top_n=3,
                                 min_weight=MIN_WEIGHT_FOR_PROTECT):
        if not self._loaded:
            self.load()

        if isinstance(domains_or_top_n, int):
            detected = self.detect_domains(text, top_n=domains_or_top_n)
            domains = [d for d, _ in detected]
        else:
            if self._last_scan_result is None or self._last_scan_text != text:
                self._scan_text(text)
                self._last_scan_text = text
            domains = domains_or_top_n

        self._protected_domains = domains

        domain_set = set(domains)
        seen_terms = set()
        protected = set()
        for term, domain, weight, start in self._last_scan_result:
            if domain in domain_set and weight >= min_weight:
                if term not in seen_terms:
                    seen_terms.add(term)
                    protected.add(term)

        return protected


_default_layer = None


def get_layer(domain_dir=DOMAIN_DICT_DIR):
    global _default_layer
    if _default_layer is None:
        _default_layer = ProtectLayer(domain_dir)
        _default_layer.load()
    return _default_layer
