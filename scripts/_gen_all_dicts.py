#!/usr/bin/env python3
"""
One-shot: regenerate ALL DomainWordsDict JSON files from source .txt files.

Steps:
  1. Convert 69 .txt → 69 sorted per-domain .json
  2. Build mini_dict.json from 9 key technical domains
  3. Verify bisect compatibility

Usage:
    python _gen_all_dicts.py <DomainWordsDict_txt_dir>

Example:
    python _gen_all_dicts.py d:/working/.../DomainWordsDict-master/data
"""

import os, sys, json, bisect, re

_ASCII_RE = re.compile(r'^[a-zA-Z0-9\s\.\,\-\'\"\(\)\&\/\#]+$')
MIN_TERM_LEN = 2
MAX_TERM_LEN = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'DomainWordsDict')
MINI_DICT_PATH = os.path.join(SCRIPT_DIR, 'data', 'mini_dict.json')

DOMAIN_ENGLISH = {
    '安全工程': 'safety_engineering', '办公文教': 'office_education',
    '材料包装': 'materials_packaging', '餐饮食品': 'food_beverage',
    '船舶工程': 'marine_engineering', '地产开发': 'real_estate',
    '地点名称': 'place_names', '地理测绘': 'geography_surveying',
    '电力电气': 'electrical_power', '电影影视': 'film_television',
    '电子工程': 'electronic_engineering', '动植生物': 'biology_zoology',
    '法律诉讼': 'legal_litigation', '纺织服装': 'textile_apparel',
    '钢铁冶金': 'metallurgy', '工业设计': 'industrial_design',
    '古代历史': 'ancient_history', '管理科学': 'management_science',
    '广告传媒': 'advertising_media', '汉语言学': 'chinese_linguistics',
    '航空航天': 'aerospace', '化学化工': 'chemistry_chemical',
    '环境科学': 'environmental_science', '机械工程': 'mechanical_engineering',
    '计算机业': 'computer_industry', '家居装饰': 'home_decor',
    '建筑装潢': 'architecture_decoration', '交通运输': 'transportation',
    '教育教学': 'education_teaching', '金融财经': 'finance_economics',
    '军事情报': 'military_intelligence', '考古挖掘': 'archaeology',
    '矿业勘探': 'mining_exploration', '旅游交通': 'travel_tourism',
    '美容美发': 'beauty_hair', '民间习俗': 'folk_customs',
    '敏感用词': 'sensitive_words', '农林牧渔': 'agriculture_forestry',
    '期货期权': 'futures_options', '汽车行业': 'automotive_industry',
    '人力招聘': 'hr_recruitment', '人文政治': 'humanities_politics',
    '人物名称': 'person_names', '社会科学': 'social_sciences',
    '诗词歌赋': 'poetry_lyrics', '世界哲学': 'world_philosophy',
    '世界宗教': 'world_religion', '市场购物': 'shopping_retail',
    '手机数码': 'mobile_digital', '书法艺术': 'calligraphy_art',
    '数学科学': 'mathematics', '水利工程': 'water_conservancy',
    '体育运动': 'sports', '天文科学': 'astronomy',
    '通信工程': 'telecommunications', '土木工程': 'civil_engineering',
    '外语学习': 'foreign_languages', '网络文学': 'web_literature',
    '网络用语': 'internet_slang', '网络游戏': 'online_gaming',
    '文学名著': 'literary_classics', '物理科学': 'physics',
    '新番动漫': 'anime_cartoon', '休闲活动': 'leisure_activities',
    '医药医学': 'medical_pharma', '音乐歌曲': 'music_songs',
    '印刷印染': 'printing_dyeing', '组织机构': 'organizations',
}

# 9 key domains for built-in mini dictionary
MINI_DOMAINS = [
    '计算机业', '土木工程', '建筑装潢', '电力电气', '机械工程',
    '法律诉讼', '数学科学', '化学化工', '环境科学',
]


def is_noise(term):
    if len(term) < MIN_TERM_LEN or len(term) > MAX_TERM_LEN:
        return True
    if _ASCII_RE.match(term):
        return True
    return False


def convert_one(src_dir, fname, domain_en):
    """Read txt, deduplicate, sort, write JSON."""
    domain_cn = fname.replace('.txt', '')
    fpath = os.path.join(src_dir, fname)
    seen = {}  # term → max weight (dedup)
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
                if is_noise(term):
                    continue
                try:
                    weight = int(parts[1].strip())
                except ValueError:
                    continue
                seen[term] = max(seen.get(term, 0), weight)
    except Exception as e:
        print(f'  WARN: {fname}: {e}')
        return None, 0

    terms = sorted(seen.items(), key=lambda x: x[0])

    out_path = os.path.join(DATA_DIR, f'{domain_en}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'domain_cn': domain_cn,
            'sorted': True,
            'terms': [[t, w] for t, w in terms],
        }, f, ensure_ascii=False, separators=(',', ':'))

    # Verify sort
    term_list = [t[0] for t in terms]
    assert term_list == sorted(term_list), f'{fname}: NOT SORTED!'

    return term_list, len(terms)


def build_mini(mini_terms):
    """Build mini_dict.json from collected terms across 9 domains."""
    all_terms = set()
    for domain_cn, terms in mini_terms.items():
        all_terms.update(terms)

    sorted_terms = sorted(all_terms)
    with open(MINI_DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted_terms, f, ensure_ascii=False, separators=(',', ':'))

    assert sorted_terms == sorted(sorted_terms)
    return sorted_terms


def main():
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <DomainWordsDict_txt_dir>')
        sys.exit(1)

    src_dir = sys.argv[1]
    if not os.path.isdir(src_dir):
        print(f'error: {src_dir} not found')
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Convert all txt → JSON
    print('=== Step 1: Convert txt → JSON ===')
    total_terms = 0
    domain_count = 0
    mini_terms = {}

    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith('.txt'):
            continue
        domain_cn = fname.replace('.txt', '')
        domain_en = DOMAIN_ENGLISH.get(domain_cn, domain_cn)
        if not domain_en:
            domain_en = domain_cn

        terms, count = convert_one(src_dir, fname, domain_en)
        if terms is None:
            continue

        domain_count += 1
        total_terms += count
        print(f'  {domain_cn:<14s} → {domain_en:<30s} {count:>6d} terms')

        if domain_cn in MINI_DOMAINS:
            mini_terms[domain_cn] = terms

    print(f'\n  {domain_count} domains, {total_terms} total terms')

    # Step 2: Build mini_dict.json
    print('\n=== Step 2: Build mini_dict.json ===')
    mini = build_mini(mini_terms)
    size_kb = os.path.getsize(MINI_DICT_PATH) / 1024
    domains_str = ', '.join(MINI_DOMAINS)
    print(f'  {len(mini)} unique terms from [{domains_str}]')
    print(f'  {size_kb:.1f} KB')

    # Step 3: Verify bisect
    print('\n=== Step 3: Bisect verification ===')
    for _domain_cn in sorted(mini_terms.keys())[:3]:
        en = DOMAIN_ENGLISH.get(_domain_cn, _domain_cn)
        path = os.path.join(DATA_DIR, f'{en}.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        term_list = [t[0] for t in data['terms']]
        if term_list:
            mid = term_list[len(term_list) // 2]
            idx = bisect.bisect_left(term_list, mid)
            ok = idx < len(term_list) and term_list[idx] == mid
            print(f'  {_domain_cn}: bisect("{mid}") → {"OK" if ok else "FAIL"}')

    # Mini dict bisect
    sample_terms = ['作用域', '仿真分析', '测试方法', '广告机', '编辑机']
    for t in sample_terms:
        idx = bisect.bisect_left(mini, t)
        found = idx < len(mini) and mini[idx] == t
        print(f'  mini_dict bisect("{t}") → {"FOUND" if found else "MISS"}')

    print(f'\nDone! Output: {DATA_DIR}')
    print(f'       Mini: {MINI_DICT_PATH}')


if __name__ == '__main__':
    main()
