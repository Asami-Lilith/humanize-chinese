#!/usr/bin/env python3
"""
Convert DomainWordsDict .txt files → per-domain sorted .json cache.

Each output JSON: {"domain_cn": "...", "sorted": true, "terms": [["term", weight], ...]}
Sorted by term (Python unicode order), compatible with bisect binary search.

Usage:
    python _domain_dict_convert.py <DomainWordsDict_txt_dir>

Example:
    python _domain_dict_convert.py path/to/DomainWordsDict-master/data

Data source: https://github.com/liuhuanyong/DomainWordsDict

You can also run this from the main entry-point:
    python humanize_cn.py --build-dict-cache <DomainWordsDict_txt_dir>
"""

import os
import json
import re
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'DomainWordsDict')

MIN_TERM_LEN = 2
MAX_TERM_LEN = 10
MIN_WEIGHT_FOR_INDEX = 3
_ASCII_RE = re.compile(r'^[a-zA-Z0-9\s\.\,\-\'\"\(\)\&\/\#]+$')


def _is_noise_term(term):
    if len(term) < MIN_TERM_LEN:
        return True
    if len(term) > MAX_TERM_LEN:
        return True
    if _ASCII_RE.match(term):
        return True
    return False


DOMAIN_ENGLISH = {
    '安全工程': 'safety_engineering',
    '办公文教': 'office_education',
    '材料包装': 'materials_packaging',
    '餐饮食品': 'food_beverage',
    '船舶工程': 'marine_engineering',
    '地产开发': 'real_estate',
    '地点名称': 'place_names',
    '地理测绘': 'geography_surveying',
    '电力电气': 'electrical_power',
    '电影影视': 'film_television',
    '电子工程': 'electronic_engineering',
    '动植生物': 'biology_zoology',
    '法律诉讼': 'legal_litigation',
    '纺织服装': 'textile_apparel',
    '钢铁冶金': 'metallurgy',
    '工业设计': 'industrial_design',
    '古代历史': 'ancient_history',
    '管理科学': 'management_science',
    '广告传媒': 'advertising_media',
    '汉语言学': 'chinese_linguistics',
    '航空航天': 'aerospace',
    '化学化工': 'chemistry_chemical',
    '环境科学': 'environmental_science',
    '机械工程': 'mechanical_engineering',
    '计算机业': 'computer_industry',
    '家居装饰': 'home_decor',
    '建筑装潢': 'architecture_decoration',
    '交通运输': 'transportation',
    '教育教学': 'education_teaching',
    '金融财经': 'finance_economics',
    '军事情报': 'military_intelligence',
    '考古挖掘': 'archaeology',
    '矿业勘探': 'mining_exploration',
    '旅游交通': 'travel_tourism',
    '美容美发': 'beauty_hair',
    '民间习俗': 'folk_customs',
    '敏感用词': 'sensitive_words',
    '农林牧渔': 'agriculture_forestry',
    '期货期权': 'futures_options',
    '汽车行业': 'automotive_industry',
    '人力招聘': 'hr_recruitment',
    '人文政治': 'humanities_politics',
    '人物名称': 'person_names',
    '社会科学': 'social_sciences',
    '诗词歌赋': 'poetry_lyrics',
    '世界哲学': 'world_philosophy',
    '世界宗教': 'world_religion',
    '市场购物': 'shopping_retail',
    '手机数码': 'mobile_digital',
    '书法艺术': 'calligraphy_art',
    '数学科学': 'mathematics',
    '水利工程': 'water_conservancy',
    '体育运动': 'sports',
    '天文科学': 'astronomy',
    '通信工程': 'telecommunications',
    '土木工程': 'civil_engineering',
    '外语学习': 'foreign_languages',
    '网络文学': 'web_literature',
    '网络用语': 'internet_slang',
    '网络游戏': 'online_gaming',
    '文学名著': 'literary_classics',
    '物理科学': 'physics',
    '新番动漫': 'anime_cartoon',
    '休闲活动': 'leisure_activities',
    '医药医学': 'medical_pharma',
    '音乐歌曲': 'music_songs',
    '印刷印染': 'printing_dyeing',
    '组织机构': 'organizations',
}


def convert(src_dir):
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f'error: source dir not found: {src_dir}')
        sys.exit(1)

    total_terms = 0
    total_domains = 0

    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith('.txt'):
            continue

        domain_cn = fname.replace('.txt', '')
        domain_en = DOMAIN_ENGLISH.get(domain_cn, domain_cn)

        fpath = os.path.join(src_dir, fname)
        terms = []
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
                    terms.append((term, weight))
        except Exception as e:
            print(f'  WARN: {fname}: {e}')
            continue

        terms.sort(key=lambda x: x[0])

        out_path = os.path.join(DATA_DIR, f'{domain_en}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'domain_cn': domain_cn,
                'sorted': True,
                'terms': terms,
            }, f, ensure_ascii=False, separators=(',', ':'))

        total_domains += 1
        total_terms += len(terms)
        print(f'  {domain_cn:<16s} -> {domain_en:<30s}  ({len(terms):>5d} terms)')

    print(f'\nDone: {total_domains} domains, {total_terms} terms')
    print(f'Output: {DATA_DIR}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python _domain_dict_convert.py <DomainWordsDict_txt_dir>')
        print()
        print('Example:')
        print('  python _domain_dict_convert.py path/to/DomainWordsDict-master/data')
        sys.exit(1)
    convert(sys.argv[1])