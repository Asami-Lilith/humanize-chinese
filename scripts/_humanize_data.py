#!/usr/bin/env python3
"""Data tables for Chinese AI Text Humanizer v2.0.

All static lookup tables — WORD_SYNONYMS, CiLin blacklists, noise expressions,
scene configs, paragraph boost data, longform markers, etc.
No logic, no imports beyond stdlib.
"""

import os
import json
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy ngram imports for utility functions
_ngram_freq = None

def _ngram_analyze():
    try:
        from ngram_model import analyze_text as _f
    except ImportError:
        try:
            from scripts.ngram_model import analyze_text as _f
        except ImportError:
            _f = None
    return _f
# Module-level flags — shared across all sub-modules
_USE_NOISE = True
_USE_STATS = True
_USE_CILIN = False

# Protection flags — domain-term protection via DomainWordsDict
_USE_PROTECT_FLAG = False
_PROTECTION_SET = set()


def _count_chinese_chars(text):
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')


PATTERNS_FILE = os.path.join(SCRIPT_DIR, 'data/patterns_cn.json')

def load_config():
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

CONFIG = load_config()

# ─── Replacement Mappings ───

PHRASE_REPLACEMENTS = CONFIG['replacements'] if CONFIG else {
    '值得注意的是': ['注意', '要提醒的是', '特别说一下'],
    '综上所述': ['总之', '说到底', '简单讲'],
    '不难发现': ['可以看到', '很明显'],
    '总而言之': ['总之', '总的来说'],
    '与此同时': ['同时', '这时候'],
    '赋能': ['帮助', '提升', '支持'],
    '闭环': ['完整流程', '全链路'],
    '助力': ['帮助', '支持'],
}

# Regex-based replacements (key is regex pattern)
_REGEX_REPLACEMENTS = {}
PLAIN_REPLACEMENTS = {}

for key, val in PHRASE_REPLACEMENTS.items():
    # Check if key contains regex special chars suggesting it's a pattern
    if any(c in key for c in ['.*', '.+', '[', '(', '|', '\\']):
        _REGEX_REPLACEMENTS[key] = val
    else:
        PLAIN_REPLACEMENTS[key] = val

# Sort regex replacements by key length descending (longer patterns first)
REGEX_REPLACEMENTS = dict(sorted(_REGEX_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True))
SCENES = {
    'general': {
        'casualness': 0.3,
        'merge_short': True,
        'split_long': True,
        'rhythm_variation': True,
    },
    'social': {
        'casualness': 0.7,
        'merge_short': True,
        'split_long': True,
        'shorten_paragraphs': True,
        'add_casual': True,
        'rhythm_variation': True,
    },
    'tech': {
        'casualness': 0.3,
        'merge_short': True,
        'split_long': True,
        'keep_technical': True,
        'rhythm_variation': True,
    },
    'formal': {
        'casualness': 0.1,
        'merge_short': True,
        'split_long': True,
        'reduce_rhetoric': True,
        'rhythm_variation': True,
    },
    'chat': {
        'casualness': 0.8,
        'merge_short': True,
        'split_long': True,
        'shorten_paragraphs': True,
        'add_casual': True,
        'rhythm_variation': True,
    },
}
WORD_SYNONYMS = {
    # ── 逻辑连接 / 转折 ──
    # Cycle 95: dropped '所以' (logic_connectors w=7 self-defeat).
    '因此': ['因而', '为此', '故而'],
    # Cycle 97: dropped '不过' from both — logic_connectors w=7 self-defeat.
    '然而': ['但', '可是', '只是'],
    # Cycle 98: dropped '然而' (logic_connectors w=7 self-defeat — replacing
    # 但是 with 然而 just trades one detected connector for another).
    '但是': ['可是', '只是'],
    '虽然': ['尽管', '即便', '就算', '纵然'],
    # Cycle 96: dropped '因此' (logic_connectors w=7 self-defeat — replacing
    # 所以 with 因此 just trades one detected connector for another).
    '所以': ['因而', '故而', '于是'],
    '而且': ['并且', '况且', '何况', '再说'],
    '或者': ['要么', '抑或', '或是', '还是'],
    '如果': ['倘若', '假如', '若是', '要是'],
    '因为': ['由于', '缘于', '出于', '鉴于'],
    '尽管': ['虽然', '即便', '纵使', '就算'],
    # ── 动词 / 行为 ──
    '能够': ['可以', '得以', '足以', '有能力'],
    '进行': ['开展', '实施', '做', '搞'],
    '实现': ['达成', '做到', '完成', '办到'],
    '提高': ['提升', '增强', '改善', '拉高'],
    # cycle 160: dropped 演进 — fixed term '发展中国家' becomes
    # '演进中国家' which reads broken (developing country, not
    # evolving). Other 发展 contexts can substitute via 推进/进展/推动.
    '发展': ['推进', '进展', '推动'],
    # '影响' removed: the idiom slot 「在 X 影响下」 is high-frequency in
    # both academic and 玄幻 register, and every candidate breaks it —
    # '波及'/'左右' are verb-only ('在...左右下' / '在...波及下' are
    # ungrammatical), '触动' is instantaneous-emotional ('在...触动下'
    # reads as 在...刺激下 but awkward), only '冲击' fits the slot. Same
    # ambiguity as the historical removals of '存在' / '有效' / '发现'.
    # cycle 160: dropped 考察 — '研究表明' commonly substituted to
    # '考察表明', which reads off-register (考察 = inspection visit).
    # Audit found in deepseek academic sample.
    # cycle 205: dropped '审视' — "本研究" → "本审视" broken
    # (审视 = critically examine, "本审视" reads as "this examination")
    '研究': ['探究', '钻研'],
    '表明': ['显示', '说明', '反映', '揭示'],
    '认为': ['觉得', '以为', '判断', '主张'],
    '需要': ['有必要', '须', '要', '得'],
    '使用': ['运用', '采用', '用', '动用'],
    '具有': ['带有', '拥有', '含有', '具备'],
    '导致': ['引发', '造成', '招致', '引起'],
    # Cycle 63: dropped '拿出' (physical/colloquial register).
    # Cycle 65: dropped '供给' — '供给' carries an economics-supply sense
    # (goods/resources), not the conceptual '提供 解释/思路/借鉴' sense.
    # Audit on 170 samples found 76 humanize-introduced 供给 cases across
    # all genres ("无法供给清晰的推理路径" / "供给代码示例" / "供给精神
    # 食粮" / "供给一面思考的镜子"). Added '给予' (grant/give) which works
    # in abstract conceptual contexts.
    # cycle 214: dropped 呈上 — overly formal "respectfully present",
    # in tech/business context "呈上聊天功能" / "呈上食物" reads off.
    # Already in _CILIN_BLACKLIST but WORD_SYNONYMS path bypassed.
    '提供': ['给出', '给予'],
    '分析': ['剖析', '解读'],  # cycle 205: drop 审视; heartbeat: drop 拆解
    '促进': ['推动', '助推', '带动', '催动'],
    '利用': ['借用', '运用', '动用', '凭借'],
    '建立': ['搭建', '构筑', '组建', '创设'],
    '引起': ['招来', '激起', '触发', '挑起'],
    '采取': ['采用', '动用', '使出', '施行'],
    '包括': ['涵盖', '囊括', '含', '包含'],
    '产生': ['催生', '引出', '萌生', '冒出'],
    '增加': ['添加', '追加', '扩充', '加大'],
    '减少': ['缩减', '削减', '降低', '裁减'],
    '保持': ['维持', '持续'],
    # cycle 229: dropped '破解' — fits "破解难题/谜团" but reads aggressive on
    # generic "解决具体问题" ("什么破解具体的问题" landed in long_blog audit).
    '解决': ['化解', '处置', '攻克'],
    '改变': ['改动', '扭转', '调整', '变化'],
    '选择': ['挑选', '选定', '选用'],
    '支持': ['撑持', '扶持', '支撑'],
    '组成': ['构成', '拼成', '组合', '凑成'],
    '形成': ['催生', '铸成', '生成', '酿成'],
    '获得': ['取得', '赢得', '得到', '揽获'],
    # cycle 164: dropped '确定' — substring matches inside 确定性 (37 hits)
    # and 不确定性 (30 hits) which are technical noun terms; substitution
    # produces broken '锁定性' / '明确性' / '不敲定性' etc. Same family of
    # bug as the historical removals of '发现' / '存在' / '有效'.
    # '发现' removed: substring inside the 4-char idiom 案发现场 gets
    # corrupted into '案察觉场'/'案觉察场'/'案识破场' when the word-level
    # substitution crosses the idiom boundary. Same family of bug as '存在'
    # / '有效' below — without proper word-boundary tagging the safe move
    # is to drop the entry. Lost LR delta is small ('发现' is mostly used
    # as a finite verb where surrounding 2-char windows already vary).
    '推动': ['驱动', '助推', '催动', '拉动'],
    '加强': ['强化', '增强', '夯实', '巩固'],
    # Cycle 78: dropped '彰显' / '凸显' — both are in detect_cn's
    # ai_high_freq_words pattern (weight 6), so injecting them as humanize
    # alts for '体现' raises the AI score (self-defeating, same family
    # as cycles 76/77). Added '反映' which is a synonym in the same
    # semantic neighborhood without being an AI-flagged term.
    '体现': ['映射', '折射', '反映'],
    '满足': ['达到', '契合', '符合', '迎合'],
    # '存在' removed: substring matches across word boundaries like 留存+在
    # → 留存有 which breaks the 留存 compound. Too error-prone without
    # word-boundary awareness.
    '属于': ['归属', '算是', '属', '归入'],
    '考虑': ['斟酌', '权衡', '琢磨', '思量'],
    # cycle 164: dropped '处理' — substring matches inside 处理器 (12 hits
    # in longform corpus, technical noun); substitution produces broken
    # '处置器' / '打理器' / '应对器'. Same as 确定/发现/存在 above.
    '参与': ['加入', '介入', '参加', '投身'],
    '创造': ['缔造', '开创', '营造', '打造'],
    '描述': ['刻画', '勾勒', '叙述', '描绘'],
    '强调': ['着重', '突出', '力陈', '重申'],
    '反映': ['映射', '折射', '体现', '呈现'],
    '应用': ['运用', '采用', '使用', '施用'],
    '结合': ['融合', '配合', '糅合', '衔接'],
    '关注': ['留意', '聚焦', '在意', '着眼'],
    '涉及': ['牵涉', '关乎', '触及', '波及'],
    '依据': ['按照', '参照', '凭', '根据'],
    # Cycle 61: dropped '取用' (informal/archaic 'fetch and use').
    # Cycle 62: dropped '引用' too — '引用' means 'cite/quote/reference',
    # not 'adopt/employ'. Same audit found 27 hits where '采用' was
    # substituted with '引用' in formal contexts ("引用对抗学习" / "引用
    # 先进的5纳米制程" / "引用复式教学法") — clear semantic error: a method
    # is adopted, not cited.
    '采用': ['选用', '沿用'],
    # ── 副词 / 程度 ──
    '目前': ['眼下', '当前', '现阶段', '如今'],
    # Cycle 80: dropped '与此同时' — it is in detect_cn's mechanical_connectors
    # pattern (weight 10), so substituting '同时' with '与此同时' raises the
    # AI score (self-defeating). Pool 4→3.
    # Cycle 80 dropped '与此同时'. Cycle 94 swap '此外'/'另外'
    # (logic_connectors w=7 self-defeat) for '同样' / '一并' (clean).
    '同时': ['并且', '同样', '一并'],
    '通过': ['借助', '凭借', '经由', '依靠'],
    '根据': ['按照', '依据', '参照', '依照'],
    # '有效' removed: word is often adjectival (有效证件/有效身份/有效期),
    # and every alternative (管用/奏效/见效/起作用) is a verb/predicate that
    # breaks attributive usage (奏效身份证件). Would need word-level POS
    # tagging to handle safely.
    '基于': ['立足于', '依托', '以…为基础', '仰赖'],
    '对于': ['针对', '就', '关于', '面对'],
    '非常': ['极其', '十分', '很', '格外'],
    # cycle 214: dropped 业已 — archaic ("已经" classical alt). In modern
    # informational text "工作功底业已落后" reads stilted. Kept for novel
    # via NOVEL_BLACKLIST_CANDIDATES exclusion (but no longer in default).
    '已经': ['早已', '已', '早就'],
    '完全': ['彻底', '全然', '纯粹', '压根'],
    '不断': ['持续', '始终', '一再', '反复'],
    '逐渐': ['渐渐', '慢慢', '一步步', '日渐'],
    # '最要紧' alt removed: when source is '最主要', substitution gives
    # '最最要紧' (doubled-最 across word boundary).
    '主要': ['核心', '关键', '首要'],
    '一般': ['通常', '往常', '照例', '大抵'],
    '大量': ['海量', '大批', '众多', '成堆的'],
    # cycle 203: dropped '更', '再' — "更进一步" → "更更" / "更再" broken;
    # "再X" reads as repetition (wrong meaning, 进一步 implies progression).
    # cycle 252: dropped '深入' — when source has "进一步深入" adjacency, sub
    # gives "深入深入" (lf:42 academic audit). 继续 is safe single alt.
    '进一步': ['继续'],
    '充分': ['尽情', '透彻', '淋漓', '饱满'],
    '直接': ['径直', '当面', '立刻', '干脆'],
    # cycle 164: '特别' alts trimmed to '尤其' only — '格外'/'极'/'分外'
    # all break inside 特别是 (56 hits in longform corpus, common
    # transition) producing '格外是'/'极是'/'分外是' which read as
    # ungrammatical. '尤其' is the one alt that survives the substring
    # collision: '特别是' → '尤其是' is a valid rewrite.
    '特别': ['尤其'],
    '一定': ['某种', '相当', '一些', '多少'],
    '必须': ['得', '务必', '非得', '须'],
    # cycle 214: dropped 兴许 — archaic dialect ("perhaps" 北方话/古风).
    # "事情兴许并不如表面所示" 读起来古旧。
    '可能': ['也许', '或许', '大概'],
    # ── 名词 / 概念 ──
    # cycle 164: dropped '重要' — substring matches inside 重要性 (28 hits)
    # and 至关重要 (16 hits) and 重要性 → 核心性 / 要紧性 / 紧要性 is
    # broken (none of those are standard Chinese nouns), 至关重要 → 至关
    # 核心 also breaks the fixed idiom. The earlier '关键' alt was already
    # dropped here (cycle ~57) for doubled-关; the remaining alts have the
    # same compound-breakage bug just less visibly.
    # Cycle 60: dropped '醒目' (visually striking, not degree adverb).
    # Cycle 66: dropped '突出' too — 突出 is verb/adjective ('stick out /
    # prominent') and doesn't work as a degree adverb. Audit found 19
    # adverb-position substitutions where it produced register/semantic
    # mismatch ('突出下降' / '突出高于' / '突出提升'). Replaced with '大幅',
    # which works as adverb of degree (118 hits in human news corpus).
    # '突出' is kept in '强调' alts where it functions as V (突出重要性).
    # cycle 202: dropped '大幅' — adverb-only, "显著进展" → "大幅进展"
    # awkward (大幅 only modifies verbs of change like 提升/下降, not nouns).
    # B-3 long_blog audit: "版本可观/明显，提升了..." shows this slot is
    # too brittle after sentence restructuring.
    # '显著': [],
    # cycle 214: dropped 症结 — too narrow (medical "crux/critical
    # bottleneck"), 破解症结 doesn't compose grammatically (症结 needs
    # 解决 / 找到, not 破解). 难题 / 麻烦 cover most contexts.
    '问题': ['难题', '麻烦'],
    # cycle 203: dropped '层面' — "多方面" → "多层面" sub broken.
    # Also dropping 维度: "多方面" → "多维度" lands in detect_cn's
    # empty_grand_words list (self-defeat). 领域 alone doesn't carry
    # the adverbial sense of 多方面, so the whole entry retires.
    # '方面': ['维度', '领域'],
    '情况': ['状况', '形势', '境况', '局面'],
    '特点': ['特征', '属性', '标志', '特色'],
    # Cycle 71: dropped '招数' — colloquial 'trick / move' (martial-arts
    # connotation), wrong register for '方法' (systematic approach). Audit
    # found 16 humanize-introduced 招数 in news/blog ("教学招数" / "学习
    # 招数" / "教育招数论" / "工作招数" / "冲洗招数"). 招数 was already
    # blacklisted for academic, so this drop only affects general/social/
    # novel where it was firing inappropriately.
    '方法': ['办法', '手段', '途径'],
    '过程': ['历程', '进程', '流程', '经过'],
    '结果': ['成果', '产物', '结局'],
    '条件': ['前提', '条件', '要件', '门槛'],
    '作用': ['功用', '效用', '效能', '功能'],
    '内容': ['要素', '成分', '要点', '素材'],
    '程度': ['幅度', '力度', '地步', '深浅'],
    '原因': ['缘由', '根源', '起因', '来由'],
    '目标': ['目的', '指向', '靶心', '方向'],
    # cycle 214: dropped 层次 — organizational/hierarchical sense, not
    # numerical. "雌激素水平 → 雌激素层次" semantically broken (hormone
    # has level/水平/水准, not hierarchy/层次).
    '水平': ['档次', '高度', '水准'],
    '范围': ['领域', '地带', '区间', '覆盖面'],
    '趋势': ['走向', '苗头', '势头', '倾向'],
    # cycle 208: dropped '实力' — "沟通能力" → "沟通实力" wrong (cycle 205
    # blocked from cilin but WORD_SYNONYMS path was missed). 实力 = strength,
    # 能力 = capability — different concepts.
    # cycle 214: dropped 功底 — too narrow (skill foundation in art/craft).
    # B-3 long_blog audit: 才干 also reads off in product prose
    # ("沟通才干"). No safe broad synonym remains.
    # '能力': [],
    '优势': ['长处', '强项', '亮点', '好处'],
    '资源': ['物资', '储备', '要素'],
    # '场景' alt removed: when source is '市场环境', substitution gives
    # '市场场景' (doubled-场 across word boundary).
    # Cycle 79: dropped '生态' — it is in detect_cn's empty_grand_words
    # pattern (weight 12, the highest). Substituting '环境' with '生态'
    # produces AI-buzzword uses ('AI生态' / '教育生态') that the detector
    # immediately flags. Added '局面' / '情境' as clean alts in the same
    # semantic neighborhood without doubled-char boundary issues.
    '环境': ['氛围', '背景', '局面', '情境'],
    '系统': ['体系', '架构', '框架'],
    '策略': ['路线', '方案', '对策', '路子'],
}
_AI_PATTERN_BLACKLIST = {
    # empty_grand_words
    '赋能', '闭环', '智慧时代', '数字化转型', '生态', '愿景', '顶层设计',
    '协同增效', '降本增效', '打通壁垒', '深度融合', '创新驱动', '全方位',
    '多维度', '系统性',
    # ai_high_freq_words
    '助力', '彰显', '凸显', '焕发', '深度剖析', '加持', '赛道', '破圈',
    '出圈', '颠覆', '革新', '底层逻辑', '抓手', '链路', '触达', '心智',
    '沉淀', '对齐', '拉通', '复盘', '迭代',
}


# Words that should NOT be substituted at all in academic context.
# These are core academic vocabulary; mechanical substitution ("研究"→"探究" etc.)
# degrades readability without reducing AIGC detection score.
ACADEMIC_PRESERVE_WORDS = {
    '研究', '分析', '发现', '指出', '表明', '认为', '显示', '揭示',
    '系统', '方法', '结果', '数据', '效果', '作用', '问题', '目标',
    '应用', '提高', '能力', '影响', '过程', '条件',
}

# Candidates that are too colloquial / archaic / informal for academic writing.
# When scene='academic', these will be filtered out of the synonym candidate pool
# before picking. If only a blacklisted candidate remains, the original word is kept.
ACADEMIC_BLACKLIST_CANDIDATES = {
    # 动词 - 过于口语或古语
    '施用', '拉高', '搞', '弄', '整', '做', '做过', '搞定', '摆平',
    '挑', '琢磨', '思量', '打理', '料理', '撑持', '揽获', '敲定',
    '识破', '觉察', '察觉', '看出', '拆解', '宛若',
    # 名词/形容词 - 口语化
    '本事', '家底', '本钱', '档次', '段位', '地带', '招数', '打法',
    '麻烦', '症结', '亮点', '好处', '苗头', '势头', '门槛',
    '成堆的', '最要紧的', '海量',
    # 程度词 - 口语
    '压根', '干脆', '径直', '当面', '兴许', '估摸着', '约莫', '大抵',
    '早就', '业已',
    # 架构/框架 对 "系统" - 过于泛化
    '架构', '框架',
    # 探究/剖析/审视 对 "研究/分析" - 虽然偶尔可用但大规模替换破坏学术调性
    '探究', '剖析',
    # 连接词口语化
    '缘于', '缘由', '来由',
    # 因果/序列连词 - 在 academic 里 '于是' 倾向 sequential temporal sense
    # ('then …'), 不像 '因此 / 因而' 那样表示 logical inference. Cycle 64
    # audit found 12 academic samples with '于是 解释 / 于是 削弱 / 于是
    # 及时干预' 等 logical 上下文里被误用. 保留给 general/novel scene.
    '于是',
}


# Novel/fiction register: a subset of ACADEMIC_BLACKLIST_CANDIDATES still
# applies to 3rd-person 玄幻/武侠/小说 prose, but several entries are
# narrative-friendly verbs ('察觉'/'识破') that academic writing rejects yet
# read naturally in fiction. Carve those out so novel mode keeps useful
# perplexity-boosting substitutes while still stripping colloquial ones
# ('搞'/'拉高'/'业已') that break narrative register.
NOVEL_BLACKLIST_CANDIDATES = ACADEMIC_BLACKLIST_CANDIDATES - {
    # Action/perception verbs that fiction uses freely
    '觉察', '察觉', '识破', '看出', '拆解',
    # 海量/眼下 are 武侠/玄幻 idioms ("海量灵气" / "眼下危机")
    '海量', '眼下',
    # 古风 register friendly
    '宛若',
    # Investigation verbs OK in narrative ("探究秘境奥秘")
    '探究', '剖析',
}

def _filter_candidates_for_scene(word, candidates, scene):
    """过滤不适合场景的候选词。返回过滤后的列表，若全被过滤则返回原列表。

    Always filters _AI_PATTERN_BLACKLIST (candidates that trigger detect_cn itself).
    Additionally filters ACADEMIC_BLACKLIST_CANDIDATES when scene='academic',
    or NOVEL_BLACKLIST_CANDIDATES when scene='novel'.
    """
    filtered = [
        c for c in candidates
        if c not in _AI_PATTERN_BLACKLIST and c not in _CILIN_BLACKLIST
    ]
    if scene == 'academic':
        filtered = [c for c in filtered if c not in ACADEMIC_BLACKLIST_CANDIDATES]
    elif scene == 'novel':
        filtered = [c for c in filtered if c not in NOVEL_BLACKLIST_CANDIDATES]
    return filtered if filtered else candidates
_CILIN_BLACKLIST = {
    # Archaic / 文言 — "conscript/order-around" tone for 使用/应用
    '使唤', '使役', '役使', '差遣', '驱使',
    # Mismatched POS (noun / noun-phrase for adjective 重要)
    '严重性', '要紧性', '关键性', '基本点', '国本',
    # Domain-mismatched (upward-numerical for 发展)
    '上扬', '上移', '上进', '升华',
    # 发展 alts: 前行/前进 = literal motion, "X的发展前景" → "X的前行前景" broken
    '前行', '前进',
    # Archaic / classical for 系统
    '板眼', '伦次', '条贯', '战线',
    # Overly colloquial / butcher-y for 分析
    '剖解', '解构',
    # Redundant / unnatural
    '显要', '要害', '紧要',
    # cycle 150 quality audit additions —— cilin synonyms that broke
    # semantics in real bn=10 humanize output across academic / novel
    # / review samples. each entry: source word → bad synonym observed.
    # Poetic / descriptive for 最高 ("highest" — quantitative)
    '万丈', '亭亭', '凌云', '参天', '摩天', '高高的',
    # Wrong scale / register for 团队 ("team")
    '团伙', '集团',
    # Technical / wrong-POS for 核心 ("core")
    '主从', '为主',
    # Assembly / event mismatch for 会议 ("meeting")
    '集会',
    # Specific-context for 完成 ("complete")
    '交卷', '到位', '姣好', '完了', '完事',
    # Wrong meaning for 问题 ("problem/issue")
    '主焦点', '事端', '关节', '关子',
    # Wrong meaning for 进行 ("conduct")
    '前进', '行进',
    # POS / meaning mismatch found in cycle 150 quality audit
    '容许',  # replaces 可能 — verb instead of modal
    '呈上',  # replaces 提供 — overly formal "submit upward"
    # cycle 186: cilin 领域 alts that mean physical land, wrong for
    # abstract domain — 教育领域 → 教育土地/园地/国土/圈子/天地 broken
    '土地', '园地', '国土', '圈子', '天地',
    # cycle 195: broken alts surfaced in README humanize 输出 audit
    '念书',  # 学习 alt — "深度学习" → "深度念书" semantically wrong
    '攻读',  # 学习 alt — only "study academically", off in "深度学习"
    '学学',  # 学习 alt — broken (just repeated char)
    '修业',  # 学习 alt — archaic ("study at school")
    '上学',  # 学习 alt — only "go to school", off in tech contexts
    '就学',  # 学习 alt — same as 上学
    '肥力',  # 精力 alt — 肥力 means soil fertility (土壤肥力)
    '个私',  # 个人 alt — regional/dialect, off in formal text
    '人家',  # 个人 alt — pronoun "she/he/they", semantic shift
    '匹夫',  # 个人 alt — archaic "common person"
    '一发',  # 更加 alt — archaic, "一发充实" reads broken
    '事体',  # 工作/事情 alt — regional dialect, off in formal text
    '本性',  # 个性 alt — "个性化" → "本性化" broken (本性 ≈ nature)
    '天性',  # 个性 alt — "个性化" → "天性化" broken
    '生性',  # 个性 alt — "个性化" → "生性化" broken
    '秉性',  # 个性 alt — same broken pattern
    '赋性',  # 个性 alt — same broken pattern
    '擘画',  # 规划/计划 alt — archaic, off in modern Chinese
    '宏图',  # 规划/计划 alt — "任务规划" → "任务宏图" wrong (宏图 = grand vision)
    '圈圈',  # 层面/局面/范畴 alt — wrong meaning ("circle")
    '框框',  # 层面/范畴 alt — wrong meaning ("frame")
    '局面',  # 层面 alt — "各个层面" → "各个局面" awkward
    '对头',  # 正确/科学 alt — colloquial "correct/foe", semantic shift
    '不利',  # 正确/科学 alt — opposite meaning ("unfavorable")!
    '不易',  # 正确/科学 alt — unrelated ("not easy")
    '得法',  # 正确/科学 alt — narrow ("appropriate method")
    '上头',  # 方面 alt — body part ("top of head")
    '恰切',  # 适应 alt — "自适应" → "自恰切" broken
    '出发点',  # 角度 alt — "从角度" → "从出发点" register-narrow
    '动用',  # 应用/使用 alt — "应用" → "动用" implies mobilizing resources
    '深浅',  # 深度 alt — "深度学习" → "深浅学习" broken
    '纵深',  # 深度 alt — military register, off
    '穿越',  # 通过 alt — "通过" → "穿越" wrong (穿越 = traverse)
    '穿过',  # 通过 alt — same wrong meaning
    '越过',  # 通过 alt — same wrong meaning
    '适于',  # 适应 alt — "自适应" → "自适于" broken
    '升任',  # 提升 alt — only "promote in rank"
    '升官',  # 提升 alt — same job-promotion narrow
    '升迁',  # 提升 alt — same job-promotion narrow
    '提干',  # 提升 alt — same, military/cadre register
    '咱家',  # 个人 alt — colloquial regional ("us/me"), wrong meaning
    '助长',  # 推动 alt — implies negative ("AI 推动教育" → "AI 助长教育" wrong, 助长 = abet/encourage-bad)
    '事理',  # 道理 alt — archaic register, off in modern Chinese
    '理路',  # 道理 alt — same archaic
    '所以然',  # 道理 alt — too philosophical, off in modern Chinese
    '技巧',  # 技术 alt — narrow "skill", off in tech contexts
    '招术',  # 技术 alt — wuxia register, very off
    '规模',  # 层面/范畴 alt — wrong dimension ("scale" not "aspect")
    '升格',  # 提升 alt — "upgrade to higher class", off in skill/effort contexts
    '升级',  # 提升 alt — software/version register, off in many contexts
    '数目字',  # 数字 alt — "数字化" → "数目字化" broken (数目字 = numerical figure)
    # cycle 203 (sway 语句通顺优先 directive): more broken alts surfaced
    '兼具',  # 具有 alt — narrow "include both", "兼具广阔前景" broken
    '由此',  # 通过 alt — connector word, "由此各方合力" broken (loses 通过 means "via")
    '稿子',  # 规划/计划 alt — colloquial "draft", off in formal "任务稿子"
    '不错',  # 科学 alt — informal compliment, "践行不错的时间管理" broken
    '正值',  # 正在 alt — only with time periods (正值春季), broken in "正值推动"
    '条理',  # 系统 alt — "智能评估系统" → "智能评估条理" broken (条理 = orderliness)
    '功用',  # 意义/作用 alt — narrow "function", "意义" → "功用" register-mismatched
    # cycle 205 (sway 语义不通畅 directive 续):
    '世界',  # 领域 alt — "教育领域" → "教育世界" semantic shift (世界 = world)
    '实力',  # 能力 alt — "沟通能力" → "沟通实力" wrong (能力 = capability, 实力 = strength)
    '体系',  # 系统 alt — "智能评估系统" → "智能评估体系" register-mismatched
    '审美',  # 审视 alt — "审视" → "审美" totally wrong meaning (aesthetic judgment)
    '琢磨',  # 研究 alt — informal "ponder", off in formal contexts
    '作用',  # 意义 alt — "真正意义上" → "真正作用上" broken (作用 = function, 意义 = meaning/significance)
    '力量',  # 意义/能力 alt — "真正意义上" → "真正力量上" broken
    '功力',  # 意义 alt — "真正意义上" → "真正功力上" broken (功力 = 内力 wuxia)
    '功效',  # 意义 alt — "真正意义上" → "真正功效上" broken
    '功能',  # 意义 alt — "真正意义上" → "真正功能上" broken (function not meaning)
    '今朝',  # 现在 alt — archaic poetic register ("今朝有酒今朝醉"), off in modern prose
    '目下',  # 目前 alt — archaic ("at present" classical Chinese), sway flagged msg 2198
    '手上',  # 目前 alt — colloquial "in hand", off in formal/academic
    '时下',  # 目前 alt — narrow ("nowadays" trend-context), off in research register
    # cycle 208 (sway 整理 README sweep):
    '于今',  # 现在 alt — archaic, "于今" 不像现代汉语
    '今日',  # 现在 alt — slightly poetic, off in modern prose ("今日X" 报纸 register)
    '今昔',  # 现在 alt — comparative "now and then", different meaning
    '参酌',  # 研究 alt — archaic "consult and consider", off in modern formal
    '掂量',  # 研究 alt — colloquial "weigh up"
    '揣摩',  # 研究 alt — narrow "ponder/figure out"
    '斟酌',  # 研究 alt — narrow "deliberate carefully", off in technical research
    '切磋',  # 研究 alt — narrow "exchange skills" (martial arts/scholarly)
    '技艺',  # 技术 alt — narrow "art/craft", off in tech contexts
    '技能',  # 技术 alt — narrow "skill", off when 技术 means "technology"
    '反过来看',  # noise/transition alt — odd opener mid-essay
    '说到这里',  # noise/transition alt — narrative voice, off in essay
    '人为',  # 人工 alt — "人工智能" → "人为智能" broken (人为=man-made, conceptually different)
    '人造',  # 人工 alt — same; "人造智能" reads as "fake AI"
    '力士',  # 人工 alt — totally different ("strongman")
    '人力',  # 人工 alt — "人工智能" → "人力智能" broken (人力 = manpower)
    '教养',  # 教育 alt — "教育教学" → "教养教学" broken (教养=upbringing/manners)
    '教化',  # 教育 alt — moralistic tone, off in modern AI/tech context
    '感化',  # 教育 alt — moralistic, off
    '启蒙',  # 教育 alt — narrow ("enlighten" beginner level)
    '教诲',  # 教育 alt — moralistic ("teaching/admonition"), off
    '教导',  # 教育 alt — narrow ("guide/instruct"), off in 教育领域
    '力促',  # 推动 alt — archaic ("forcefully promote")
    '方略',  # 规划/计划 alt — military/strategic, off in "任务规划" → "任务方略"
    '透过',  # 通过 alt — physical "penetrate through", off in 通过合力 context
    '末了',  # 最后 alt — colloquial dialect
    '末后',  # 最后 alt — archaic
    '末尾',  # 最后 alt — physical position, off in temporal context
    '尾子',  # 最后 alt — colloquial
    '尾声',  # 最后 alt — narrow ("finale" of event/work)
    '鹏程',  # 前景 alt — mythological "Peng's flight", way too poetic
    '奔头儿',  # 前景 alt — colloquial dialect ("something to look forward to")
    '乌纱',  # 前程 alt — archaic "official's hat", career-narrow
    '乌纱帽',  # 前程 alt — same
    '功名',  # 前程 alt — imperial-exam era register
    '前程',  # 前景 alt — career-path slot, breaks "广阔的发展前景" idiom
    '前途',  # 前景 alt — career-track slot, "广阔的发展前途" reads off in 现代 prose
    '兼备',  # 具有 alt — "具有X" → "兼备X" requires plural object
    '议会',  # 会议 alt — "parliament", totally different ("此次议会" 错)
    '集会',  # 会议 alt — narrow ("rally"), off in business meeting context
    '治理',  # 管理 alt — "governance", domain-shift from "manage"
    '治本',  # 管理 alt — narrow medical idiom ("treat root cause")
    '治治',  # 管理 alt — colloquial duplicate ("punish/teach a lesson")
    '管事',  # 管理 alt — narrow ("be in charge of trifles"), colloquial
    '贯彻',  # 实现 alt — narrow ("carry through policy"), "技术实现" → "技术贯彻" 错
    '落实',  # 实现 alt — narrow ("implement policy"), "技术实现" → "技术落实" 错
    '装具',  # 设备 alt — military equipment, "厨房设备" → "厨房装具" 错
    '两样',  # 不同 alt — colloquial, "上百种不同" → "上百两样" 病句
    '释疑',  # 解释 alt — classical, "希望找到一种解释" → "找到一种释疑" 古文
    '训诂',  # 解释 alt — narrow (textual exegesis), 古文 register
    '层系',  # 层次 alt — geological layer, semantic shift
    '兴许',  # 可能 alt — archaic dialect ("perhaps" 北方话), reads stilted
    '业经',  # 已经 alt — formal/legal classical
    '著录',  # 记录 alt — narrow (catalog/formally record), "聊天记录" → "聊天著录" 错
    '笔录',  # 记录 alt — narrow (deposition), domain-shift from generic record
    '记要',  # 记录 alt — minute-taking, narrow
    '主焦点',  # 症结 alt — non-word
    '关子',  # 症结 alt — narrow ("punchline of joke")
    '各别',  # 不同 alt — colloquial "individually different", "上百各别" 错
    '唯恐',  # 可能 alt — archaic "for fear that", "事情可能" → "事情唯恐" 错
    '例外',  # 不同 alt — "exception", "上百种不同" → "上百例外" 病句
    '下狠心',  # 决定 alt — colloquial idiom "make up mind", "决定您" → "下狠心您" 错
    '主宰',  # 决定 alt — too strong ("rule over")
    '了得',  # 决定 alt — different meaning ("remarkable")
    '仲裁',  # 决定 alt — legal/formal arbitration
    '公决',  # 决定 alt — legal/formal public decision
    '公断',  # 决定 alt — legal/formal public arbitration
    '品目',  # 种类 alt — narrow ("article entries" in catalog)
    '档级',  # 种类 alt — narrow ("rank/grade")
    # cycle 216 longform audit additions:
    '惨遭',  # 面临 alt — "suffer tragically" wrong tone for "面临挑战"
    '屡遭',  # 面临 alt — narrow (repeatedly suffer)
    '倍受',  # 面临 alt — only fits 关注/重视 (positive), "倍受挑战" 错
    '备受',  # 面临 alt — same constraint
    '未遭',  # 面临 alt — archaic
    # cycle 218 longform inject_noise + cilin audit additions:
    '筋肉',  # 肌肉 alt — Japanese-derived, off in modern Chinese
    '年华',  # 时间 alt — poetic ("youth"), "恢复年华" 错
    '光阴',  # 时间 alt — poetic, off in tech/business
    '岁时',  # 时间 alt — archaic
    '年光',  # 时间 alt — poetic
    '上马',  # 开始 alt — colloquial "start project (大跃进 era)"
    '伊始',  # 开始 alt — formal "at the start", off in casual
    '先声',  # 开始 alt — narrow ("first signs/prelude")
    '城池',  # 城市 alt — ancient walled city
    '城邑',  # 城市 alt — archaic
    '地市',  # 城市 alt — gov-policy "city-prefecture"
    '大哥大',  # 手机 alt — 90s mobile phone
    '无绳机',  # 手机 alt — cordless landline phone
    '固化',  # 稳定 alt — "solidify" wrong meaning
    '安乐',  # 稳定 alt — "peaceful/comfortable"
    '原则性',  # 稳定 alt — "principled" wrong slot
    # cycle 220 quality cleanup:
    '不无',  # 具有/具备/拥有 alt — literary double-negative, "不无广阔" 错
    '万顷',  # 广阔 alt — ancient land measure (万顷土地)
    '周边',  # 广阔 alt — "peripheral", wrong slot for 广阔
    '周遍',  # 广阔 alt — archaic
    '宏阔',  # 广阔 alt — formal/literary
    '能事',  # 能力 alt — narrow ("things one can do well")
    '能耐',  # 能力 alt — colloquial "skill/ability"
    '身手',  # 能力 alt — narrow ("agility/martial skill")
    '意志',  # 旨在 alt — "willpower" not "purpose", "意志提高" 错
    '心意',  # 旨在 alt — "intention" but not "aim", same slot mismatch
    '意旨',  # 旨在 alt — archaic "imperial decree"
    '旨意',  # 旨在 alt — same archaic
    '法旨',  # 旨在 alt — Buddhist/imperial decree
    '拍卖',  # 处理 alt — "auction" totally different domain
    '处分',  # 处理 alt — narrow disciplinary action
    '上座',  # 首席 alt — "seat of honor", wrong for executive title
    '上位',  # 首席 alt — narrow ("upper position")
    '剖示',  # 展示 alt — non-word/very rare
    '兆示',  # 展示 alt — narrow archaic ("portend")
    '呈示',  # 展示 alt — formal/legal narrow
    '试点县',  # 县城 alt — "pilot county" gov-policy specific
    '版纳',  # 县城 alt — actual place name (西双版纳), nonsense as alt
    '京都',  # 北京 alt — Kyoto / archaic capital
    '上京',  # 北京 alt — archaic
    '京华',  # 北京 alt — poetic
    '京城',  # 北京 alt — slightly archaic, ok in some contexts but mostly off
    '京师',  # 北京 alt — imperial-era term
    '凤城',  # 北京 alt — poetic name for capitals
    '中标',  # 成功 alt — "win bid" commercial
    '交卷',  # 成功 alt — "submit exam paper"
    '姣好',  # 成功 alt — "beautiful" not "successful"
    '完了',  # 成功 alt — "finished" not "succeeded"
    '到位',  # 成功 alt — narrow ("in place"), often wrong slot
    '作派',  # 主义 alt — "mannerism/style" wrong, "存在主义" → "存在作派" 错
    '官气',  # 主义 alt — "bureaucratic air"
    '架子',  # 主义 alt — "framework/airs"
    '作风',  # 主义 alt — "style" sometimes ok but breaks 主义 idioms
    '犯得上',  # 值得 alt — colloquial "worth doing" 北方话
    '犯得着',  # 值得 alt — same
    '其时',  # 当时 alt — archaic "at that time"
    '讲堂',  # 教室 alt — formal/grand "lecture hall"
    '归于',  # 归属 alt — preposition rather than noun, "归于的" 怪
    # cycle 221 academic 5-sample audit additions:
    '胁从',  # 威胁 alt — "be coerced into" (legal term)
    '威慑',  # 威胁 alt — narrow ("intimidate")
    '威逼',  # 威胁 alt — colloquial coercion
    '胁迫',  # 威胁 alt — legal coercion
    '体贴',  # 关注 alt — personal "considerate"
    '关怀',  # 关注 alt — soft "show concern (caring)"
    '关爱',  # 关注 alt — "love & care"
    '眷顾',  # 关注 alt — formal/literary "favor"
    '求战',  # 挑战 alt — military "ask for battle"
    '包罗',  # 包括 alt — formal "encompass all"
    '席卷',  # 包括 alt — "sweep over"
    '揽括',  # 包括 alt — "monopolize/include all"
    '强攻',  # 攻击 alt — military
    '抢攻',  # 攻击 alt — sports/military
    '挨斗',  # 攻击 alt — political-era "denounced"
    '掊击',  # 攻击 alt — archaic
    '反响',  # 影响 alt — "echo/response", narrow
    '反射',  # 影响 alt — physics
    # 反应 NOT blocked: high-frequency word, blocking costs HC3 avg -0.5
    # without clear fluency win in informational text. Keep alt available.
    '反馈',  # 影响 alt — narrow technical
    '想当然',  # 影响 alt — different ("take for granted")
    '感应',  # 影响 alt — physics/spiritual
    '安好',  # 安全 alt — 古文 "well-being"
    '安康',  # 安全 alt — 古文 "health"
    '安然',  # 安全 alt — adverb-like "safely"
    '康宁',  # 安全 alt — 古文 "peace"
    '无恙',  # 安全 alt — 古文 "no harm"
    '音信',  # 信息 alt — 古文 "tidings"
    '音尘',  # 信息 alt — 古文
    '音息',  # 信息 alt — 古文
    '音讯',  # 信息 alt — 古文
    '音问',  # 信息 alt — 古文
    '中坚',  # 核心 alt — narrow ("backbone force")
    '争辩',  # 理论 alt — different ("argue")
    '争鸣',  # 理论 alt — narrow ("contend")
    '反驳',  # 理论 alt — different ("rebut")
    '回驳',  # 理论 alt — same
    '声辩',  # 理论 alt — narrow ("plead")
    '仰承',  # 利用/凭借 → ??? chain — deferential "accept respectfully"
    '拼杀',  # 攻击 alt — "fight to death", off in academic
    # cycle 222 news/blog/review audit additions:
    '构造',  # 布局 alt — "structure" wrong slot, "战略布局" → "战略构造" 错
    '师资',  # 老师 alt — collective noun, "一名师资" 错
    '园丁',  # 老师 alt — metaphor, off in factual text
    '产物',  # 结果 alt — "product", semantic shift
    '下场',  # 结果 alt — narrow ("downfall")
    '下文',  # 结果 alt — narrow ("subsequent passage")
    '了局',  # 结果 alt — archaic
    '分晓',  # 结果 alt — narrow ("decisive outcome")
    '名堂',  # 结果 alt — colloquial ("trick/explanation")
    '部署',  # 配置 alt — "deploy" military/IT slot
    '嬉水',  # 游戏 alt — "play in water" totally wrong
    '嬉戏',  # 游戏 alt — narrow "play"
    '一日游',  # 游戏 alt — "one-day tour"
    '休闲游',  # 游戏 alt — "leisure tour"
    '好耍',  # 游戏 alt — colloquial "fun"
    '差一点',  # 几乎 alt — "almost (didn't)" different meaning
    '差点儿',  # 几乎 alt — colloquial 北方话
    '常备',  # 日常 alt — "regular/ready" wrong slot
    '万般',  # 日常 alt — "all kinds" wrong slot
    '一般性',  # 日常 alt — "general" wrong slot
    '处事',  # 处理 alt — substring collision with 处理器, "处理器" → "处事器" 错
    '措置',  # 处理 alt — formal/archaic
    '凌厉',  # 强烈 alt — narrow ("sharp/fierce")
    '凶猛',  # 强烈 alt — narrow ("fierce")
    '利害',  # 强烈 alt — narrow ("intense/clever")
    '剧烈',  # 强烈 alt — narrow ("violent")
    '凭依',  # 利用 alt — formal/literary
    '从新',  # 重新 alt — colloquial 北方话
    '再也',  # 重新 alt — adverb-only
    '再行',  # 重新 alt — formal
    '双重',  # 重新 alt — wrong meaning ("double")
    '允当',  # 适合 alt — formal/archaic
    '切合',  # 适合 alt — narrow ("fit closely")
    '切当',  # 适合 alt — formal/archaic
    '合乎',  # 适合 alt — formal pre-noun
    '合宜',  # 适合 alt — formal/archaic
    '搭架子',  # 布局 alt — colloquial "set up framework"
    '份额',  # 重量 alt — "share/portion" wrong
    '净重',  # 重量 alt — narrow ("net weight")
    '千粒重',  # 重量 alt — agricultural specific
    '毛重',  # 重量 alt — narrow ("gross weight")
    '咋舌',  # 惊叹 alt — "tongue-tied" wrong
    '奇异',  # 惊叹 alt — "strange/peculiar" semantic shift
    '奇怪',  # 惊叹 alt — "strange"
    '希罕',  # 惊叹 alt — "rare/curious"
    '惊呆',  # 惊叹 alt — narrow "stunned"
    # cycle 230 long_blog audit additions:
    '上心',  # 专注 alt — verb "take to heart"; "上心于X" 不通 (上心 doesn't take 于)
    '寻常',  # 通常 alt — literary "ordinary"; "数据寻常用于" reads off-register
    # cycle 231 general/workplace audit additions:
    '推向',  # 推动 alt — needs directional object; "推向教育的大趋势" 不通
             # (推向 expects target/level: "推向更高水平"; not abstract "X的大趋势")
    # cycle 232 long_blog seed=1 audit additions — bad alts in 变化 family:
    '变卦',  # 变化 alt — narrow ("renege on agreement"); "动态变卦" 不通
    '事变',  # 变化 alt — historical event ("七七事变"); wrong slot for generic 变化
    '变故',  # 变化 alt — "mishap/misfortune" too negative for generic context
    '变型',  # 变化 alt — likely typo of 变形, narrow material-science slot
    '切变',  # 变化 alt — physics term ("shear"), wrong slot for generic 变化
    # cycle 234 long_blog seed=7 audit additions — narrow 实现/创造 cilin alts:
    '促成',  # 实现 alt — "facilitate/bring about", off for "技术实现" → "技术促成"
    '兑现',  # 实现 alt — "fulfill (a promise)", narrow; "技术兑现" 不通
    '创设',  # 创造 alt — "set up institution"; "创设产品" off (institutions/laws fit, products don't)
    # cycle 235 general seed=7 audit additions — narrow 学生 + modal-mismatched alts:
    '学员',  # 学生 alt — "trainee" narrow, off for generic "学生" in education context
    '学童',  # 学生 alt — "young pupil" narrow age slot
    '桃李',  # 学生 alt — metaphor "students/disciples" idiom-only
    '可知',  # 能够/亦可 alt — modal mismatch ("it can be known"), not "can do"
    # cycle 236 long_blog seed=42 audit additions:
    '保险',  # 确保/保证 alt — "insurance" connotation; "保险产品能够" reads "insurance product"
    '一对一', # 一定/相当 alt — completely wrong meaning ("one-on-one"); "一对一要具备" 不通
    # cycle 237 longform sample audit — 首先/其次 family bad alts:
    '处女',  # 首先/状元/首度 alt — "maiden/virgin"; off in any modern context
    '头条',  # 首先/状元/首度 alt — "headline"; wrong slot
    '头版',  # 首先/首度 alt — "front page"; wrong slot
    '排头',  # 首先/首度 alt — "front of line"; wrong slot
    '伯仲',  # 其次 alt — "brothers/peers"; wrong slot
    '老二',  # 其次 alt — colloquial "second son"; vulgar register
    '第二性', # 其次 alt — Beauvoir's book "The Second Sex"; specific cultural reference
    # cycle 238 systematic word-family scan — 形成/帮助/需要/降低/考虑 wrong-slot alts:
    # 形成 family (all alts mean "achieve/in-place/mutate"; none fits "form/take shape"):
    '做到',  # 形成 alt — "achieve"; "形成共识" → "做到共识" 不通 (also in WORD_SYNONYMS 实现 alts, that path unaffected)
    '变异',  # 形成 alt — biology "mutation"; wrong slot
    '善变',  # 形成 alt — character trait "fickle"; wrong POS
    '多变',  # 形成 alt — adjective "changeable"; wrong POS
    # 帮助 family wrong-meaning alts:
    '佑助',  # 帮助 alt — literary "bless and help"; archaic
    '匡助',  # 帮助 alt — formal "assist (the upright)"; archaic register
    '匡扶',  # 帮助 alt — "support (the righteous)"; archaic political register
    '受助',  # 帮助 alt — OPPOSITE direction! "receive help"
    '增援',  # 帮助 alt — military "reinforce"; wrong slot
    # 需要 family wrong-slot alts:
    '内需',  # 需要 alt — "domestic demand" (economic noun); wrong slot
    '特需',  # 需要 alt — "special needs" (medical/service noun); wrong slot
    '索要',  # 需要 alt — "demand insistently"; wrong tone
    # 降低 family financial-narrow alts:
    '下挫',  # 降低 alt — financial "drop sharply"; narrow
    '下滑',  # 降低 alt — financial/sports "decline"; narrow
    '下跌',  # 降低 alt — financial "fall"; narrow
    '低落',  # 降低 alt — emotional "in low spirits"; wrong slot
    # 考虑 family wrong-meaning alts:
    '合计',  # 考虑 alt — "calculate/sum up"; wrong meaning
    # cycle 239 systematic word-family scan continued — 收集/工作/问题/方法/关系/特点:
    '募集',  # 收集 alt — "raise funds"; narrow financial
    '筹募',  # 收集 alt — "fundraise"; narrow financial
    '收载',  # 收集 alt — "include/contain (in book)"; wrong slot
    '综采',  # 收集 alt — "synthesize and extract"; archaic
    '专职',  # 工作 alt — "professional/full-time" noun; wrong slot
    '作业',  # 工作 alt — "homework/operations"; different concept
    '事故',  # 问题 alt — "accident/incident"; wrong slot for generic 问题
    '抓挠',  # 方法 alt — colloquial "scratch/lousy way"; wrong meaning
    '具结',  # 关系 alt — legal "guarantee in writing"; narrow
    '风味',  # 特点 alt — "flavor"; food-only
    # cycle 240 systematic adjective family scan — wrong-meaning/colloquial alts:
    # 深入 family:
    '中肯',  # 深入 alt — "to the point"; different concept
    '刻骨',  # 深入 alt — "deeply engraved"; emotional only
    # 明显 family:
    '扎眼',  # 明显 alt — "eye-catching" colloquial
    '明摆着', # 明显 alt — colloquial register
    # 清晰 family:
    '丁是丁', # 清晰 alt — "clear-cut" colloquial idiom
    '清丽',  # 清晰 alt — "clear and beautiful"; wrong slot
    # 复杂 family:
    '繁体',  # 复杂 alt — "traditional Chinese characters"; wrong slot
    # 稳定 family:
    '一贯',  # 稳定 alt — adverb "consistently"; wrong POS
    '原则性', # 稳定 alt — formal noun "principled"; wrong POS
    # 快速 family:
    '不会儿', # 快速 alt — colloquial 北方话 "in a moment"
    '劈手',  # 快速 alt — archaic "swiftly with hand"
    '快当',  # 快速 alt — colloquial "quick/efficient"
    # 准确 family:
    '准儿',  # 准确 alt — colloquial "for sure"
    '纯正',  # 准确 alt — "pure/authentic"; wrong slot
    # 有效 family:
    '中用',  # 有效 alt — colloquial "useful"
    '合用',  # 有效 alt — narrow "fit for use"
    # 严格 family:
    '严词',  # 严格 alt — narrow noun "stern words"
    # 强大 family:
    '强压',  # 强大 alt — "suppress strongly"; wrong slot
    # cycle 241 systematic domain noun scan — wrong-slot/narrow alts:
    # 经济 family (cost-effective ≠ economy):
    '上算',  # 经济 alt — "cost-effective" colloquial
    '划得来', # 经济 alt — "worthwhile" colloquial
    '占便宜', # 经济 alt — "take advantage" colloquial
    # 文化 family:
    '双文明', # 文化 alt — specific term ("two civilizations")
    # 产业 family (whole family means "family property", wrong slot for industry):
    '家业',  # 产业 alt — "family business/property"
    '家事',  # 产业 alt — "family affairs"
    '家产',  # 产业 alt — "family assets"
    '家底',  # 产业 alt — "family savings"
    '家当',  # 产业 alt — "household belongings"
    '家私',  # 产业 alt — "household goods"
    # 行业 family:
    '本行',  # 行业 alt — narrow "one's own line of work"
    '正业',  # 行业 alt — idiom "proper occupation"
    # 企业 family (specific store types ≠ enterprise):
    '代销店', # 企业 alt — "consignment store"
    '供销社', # 企业 alt — "supply and marketing cooperative"
    '信用社', # 企业 alt — "credit union"
    '合作社', # 企业 alt — "cooperative"
    '商厦',  # 企业 alt — "commercial building"
    # 政府 family:
    '内阁',  # 政府 alt — narrow "cabinet"
    # 国家 family (archaic poetic):
    '江山',  # 国家 alt — "rivers and mountains"; poetic
    '社稷',  # 国家 alt — "altars to gods of soil/grain"; archaic
    # 世界 family:
    '世道',  # 世界 alt — "ways of the world"; archaic
    '世风',  # 世界 alt — "social mores"; archaic
    '中外',  # 世界 alt — "Chinese and foreign"; wrong slot
    '五洲',  # 世界 alt — "five continents"; archaic
    # cycle 242: 频率 fixes "时间管理频率" → "时间管理效率"; cascade
    # protected via 轻重 _CILIN_SOURCE_BLACKLIST (idiom 轻重缓急) and
    # 生气 blacklist (精力 alt — primarily means "anger", off in vitality slot).
    '频率',  # 效率 alt — different meaning ("frequency")
    '生气',  # 精力 alt — primarily means "anger"; "时间和生气" 不通
    # cycle 243 longform blog audit additions:
    '上上',  # 可以 alt — colloquial "tops/very good"; modal mismatch ("上上在最大限度地" 不通)
    '分红',  # 分配 alt — financial "dividend"; "资源分红" 不通
    '不等',  # 不同 alt — "unequal" different concept; "不等的服务节点" 不通
    # cycle 244 longform blog audit additions:
    '放眼',  # 纵目/极目 alt — verb form; "在放眼全球" 不通 (放眼 needs object)
    '之际',  # 关键/关头 alt — archaic "at the moment of"; "之际介于我们" 不通
    '介于',  # 在于 alt — "between"; wrong meaning, "之际介于我们" wrong concept
    # cycle 245 long_blog seed=7 audit additions:
    '历历',  # 清晰 alt — archaic "vividly" (历历在目 idiom-only); "历历的产品构想" 不通
    # 创-family narrow alts (after cycle 234 blocked 创设, fallback to these still narrow):
    '创始',  # 创造 alt — "found/initiate"; narrow ("创始更具价值的产品" wrong)
    '创办',  # 创造 alt — "start (business)"; narrow
    '创立',  # 创造 alt — "establish"; narrow
    '创导',  # 创造 alt — formal "advocate"; rare
    # cycle 246 academic carbon trading audit additions:
    '企图',  # 作用 alt — "attempt/scheme"; "发挥重要企图" 不通
    # 手段 family — 一手/伎俩/心数/心眼/手眼 all colloquial/derogatory/idiom:
    '一手',  # 手段 alt — colloquial "trick/one hand"
    '伎俩',  # 手段 alt — negative "trick/scheme"
    '心数',  # 手段 alt — wrong slot ("mind/wits")
    '心眼',  # 手段 alt — wrong slot ("mind/heart")
    '手眼',  # 手段 alt — idiom-only ("手眼通天")
    # cycle 247 不-family + 反而 family scan:
    '是的',  # 不易/正确 alt — particle "yes"; wrong slot
    '倒转',  # 反而 alt — "reverse direction"; off-meaning
    '反是',  # 反而 alt — archaic
    # long_blog audit (post-d3dc2ea):
    # 检点 (alt of 上心/专注/只顾/在意/查点 etc) means "examine/scrutinize",
    # not "focus on" — "我常常只顾于" → "我常常检点于" misreads.
    '检点',
    # 使得 (alt of 驱动/有效/管用/可行/...) means "cause/make happen", not
    # "drive" — "数据驱动的决策" → "数据使得的决策" 不通.
    '使得',
    # 何等 (alt of 如何/什么/怎样/...) is exclamatory ("how/what a"), wrong
    # POS for interrogative slot — "如何攻克" → "何等攻克" 不通.
    '何等',
    # social/general 病句 audit (post-codex review):
    # 个人 cilin alts include 斯人 (literary "this person" — modern 不通);
    # 我们 cilin alts include 咱俩 (dual "the two of us" — wrong number),
    # 吾侪/吾辈/我辈 (all 文言 plurals), 俺们 (regional). Block them.
    '斯人',  # 个人 alt — 文言 "斯人" never modern Chinese
    '咱俩',  # 我们 alt — dual ("the two of us"), wrong for plural 我们
    '吾侪',  # 我们 alt — 文言
    '吾辈',  # 我们 alt — 文言
    '我辈',  # 我们 alt — 文言
    '俺们',  # 我们 alt — 北方方言, register mismatch in standard prose
    # 科学 cilin alts include 无误 ("error-free") — different concept;
    # 不易 family already source-blocked but 无误 leaks via 科学 path.
    '无误',  # 科学 alt — "error-free", different from "scientific/systematic"
    # 真正 cilin alts include 实际 ("actually" adv) — POS clash with attributive
    # use ("真正重要" → "实际重要" 不通). Other alts (一是一/委实/实在/实打实) keep.
    '实际',  # 真正 alt — POS clash in attributive contexts ("真正重要")
    # 毫无疑问 cilin alts include 大势所趋 ("inevitable trend") — totally
    # different meaning. 毫无疑问 = "no doubt"; 大势所趋 = "general/inevitable".
    '大势所趋',  # 毫无疑问 alt — meaning mismatch
    # general audit (post-d3dc2ea + e94e7b7):
    # 能够 cilin alts (亦可/可知) are both 文言. "能够 X" → "亦可 X" / "可知 X"
    # leaks 文言 register into general/academic prose. Block both alts; source
    # is also added to _CILIN_SOURCE_BLACKLIST below.
    '亦可',  # 能够 / 可知 alt — 文言, register slip in modern prose
    '可知',  # 能够 / 亦可 alt — 文言 ("可知道")
    # b4 hero candidate audit — substitutions appearing in dramatic-drop AI samples:
    # 可巧 (alt of 适时/及时/刚巧) means "happens to coincide", not "in a timely
    # manner" — "可巧调整" 不通.
    '可巧',
    # 接轨 (alt of 继续/延续/接续) means "integrate/connect with"; "接轨坚持"
    # / "接轨定投" misreads as integration.
    '接轨',
    # 报恩 (alt of 回报) means "repay kindness", not "return/feedback" —
    # "给我们最好的报恩" wrong concept.
    '报恩',
    # B-3 long_blog mutation audit: CiLin alternates that read broken in
    # modern product/blog prose ("在意于什么样攻克", "这一涉世",
    # "这一年不休是").
    '在意',
    '涉世',
    '不休',
    # "利用数据" is fine, but "利用频率" is a common false slot when replacing
    # 使用 inside product-metrics prose.
    '利用',
    # "顺着这个构思" is an off-slot replacement for 思路 in discourse markers.
    '构思',
    '小心',
    '不住',
    '笔触',
    '应用',
    '掌管',
    '关键',
    '可观',
    '保管',
    '才干',
    '什么样',
    '创制',
    '此即',
    '打算',
    '只顾',
    '调动',
    '在心',
    '何以',
    '什么',
    '当真',
    # heartbeat audit: 尤其 (alt of 更加) needs comparison context. "尤其充实"
    # / "迎来尤其充实和有意义的人生" (social hero) reads off; should be
    # intensifier "更加充实".
    '尤其',
    # 何如 (alt of 如何/什么/怎样) is 文言 "how about" — wrong slot for direct
    # interrogative; "何如解决" 不通.
    '何如',
    # 主体 (alt of 核心/主脑) means "main body/subject" not "essence" —
    # "产品决策的主体" misreads (long_blog audit).
    '主体',
    # 为重 (also alt of 核心) — adverbial "considered-important" doesn't fit
    # noun slot "决策的核心"; cilin fallback after 主体 still wrong.
    '为重',
    # 为主 (also alt of 核心) — adverbial "principally"; same slot mismatch.
    '为主',
    # 主从 (also alt of 核心) — relational "principal vs subordinate";
    # not a noun substitute for 核心 essence slot.
    '主从',
    # cycle 251: 一时 (alt of 时代/时期/时日/一代/一世) means "momentarily/for
    # a moment" — semantically opposite to era/period/lifetime. "技术快速
    # 推进、全球化深入推进的一时" (lf:80 audit, source had 时代) is broken;
    # should stay "时代". Multi-source mistake in cilin grouping.
    '一时',
}


# Source-side blacklist: 2-char cilin keys whose substitution produces
# broken Chinese — either because they're almost always part of longer
# compounds (substring-collision) or because their cilin alts shift
# meaning even in standalone position. Block at the source (skip these
# as replacement targets in reduce_cross_para_3gram_repeat).
#
# cycle 191: '不了' — X不了 negative-potential compound (受不了/少不了/
# 免不了/做不了…), 不了 → 不息/不停 breaks compound (少不息).
# cycle 192: empirical audit of 10 high-freq function words. Each line
# below = source word + the broken sample observed in test:
_CILIN_SOURCE_BLACKLIST = {
    '不了',  # 少不了 → 少不息
    '不是',  # 不是教师 → 纰缪教师 (alts: 不对/偏向/纰缪 — meaning shift)
    '一下',  # 想一下 → 想一瞬 (alts 一刹那/一瞬 too dramatic)
    '一些',  # 带一些礼物 → 带好几礼物 (好几 needs 个 measure word)
    '不要',  # 不要担心 → 并非担心 (并非 is statement-of-fact, not directive)
    '就是',  # 就是这样 → 即使这样 (即使 = "even if", needs main clause)
    '不能',  # 不能解决 → 未能解决 ("can't" → "didn't succeed", semantic shift)
    '什么',  # 什么东西 → 咋样东西 (咋样 colloquial + register-mismatch)
    '只是',  # 只是开始 → 单单开始 (单单 modifies things, not actions)
    # Idiom-anchor nouns: substituting these breaks fixed compounds even when
    # the alt is grammatically valid. "发展前景"→"发展未来" reads off.
    '前景',  # 发展前景 / 应用前景 / 推进前景 — idiom-fixed
    '前途',  # 发展前途 / 学术前途 — idiom-fixed
    # Adverbial-compound anchors: 方面 cilin alts (上头/上面/地方/方位/方向)
    # all break "多方面" / "各方面" idiomatic compounds.
    '方面',  # 多方面/各方面 → 多地方/各上面 — broken
    # Educational vocabulary anchors: 教学 cilin alts (上书/任课/执教/主讲)
    # are role-specific or archaic, all break the generic noun slot.
    '教学',  # 教育教学 → 教育上书 — archaic ("submit memorial")
    # Substring-collision anchors: 2-char keys frequently embedded in
    # larger compounds where substitution corrupts the parent. cycle 214
    # audit found "说不定" → "说不安" via 不定 → 不安 substring sub.
    '不定',  # inside 说不定 / 拿不定 / 一定不定; alts (不安/动乱) corrupt parent
    '末日',  # 后期/晚期/期末 alts shift "doomsday" → "later period"
    '后期',  # 后期/晚期/期末 cluster — same shift
    # cycle 235: 不容 cilin alts (不肯/回绝/拒绝/推却/推辞/闭门羹) all mean
    # "refuse/decline" — wrong for 不容忽视/不容置疑 (which means "doesn't
    # tolerate/permit"). Source blacklist since ALL alts are wrong-meaning.
    '不容',
    # cycle 242: 轻重 cilin alts (份量/份额/净重/分寸/分量/千粒重) all are
    # weight/measure terms; none fit the idiom 轻重缓急 (priority/important).
    # Source blacklist since substitution always breaks the idiom.
    '轻重',
    # cycle 242: 精力 cilin alts (元气/活力/生命力/生机/生气/肥力) all in
    # vitality family; "时间和精力" is set idiom, any substitution lands
    # on awkward "时间和生气/生机/活力". Block source.
    '精力',
    # cycle 243: 可以 cilin alts (上上/上佳/上好/不含糊/不离儿/不赖) all are
    # colloquial "good/fine"; none works as modal "可以" (can/may). Block source.
    '可以',
    # cycle 246: 温室 cilin alts (花房/保暖棚/大棚/暖房/暖棚/温棚) all are
    # physical greenhouse types; "温室气体" is fixed scientific term, ANY
    # substitution breaks it ("花房气体" → "flower-house gas" 错). Block source.
    '温室',
    # cycle 247: 不易 cilin alts (不利/不错/对头/得法/无误/是的) — 不利 is
    # OPPOSITE meaning, others wrong slot. Most already individually
    # blacklisted; source block to be exhaustive.
    '不易',
    # cycle 247: 不堪 alts (受不了/吃不住/吃不消/哪堪/架不住/禁不住) all
    # colloquial; 不堪 fixed term ("不堪重负/不堪入目/不堪一击") — substitution
    # breaks 4-char idioms.
    '不堪',
    # cycle 247: 进而 alts (一发/尤为/尤其/愈加/愈发/愈来愈) all degree adverbs;
    # 进而 is sequential connector ("furthermore/then"), different concept.
    '进而',
    # long_blog audit: substring collision. cilin '品蓝' → '藏蓝' fires inside
    # "产品蓝图" → "产藏蓝图" because regex matches '品蓝' substring across
    # word boundary (产品|蓝图). Block source — color noun never wanted in
    # AIGC humanize anyway.
    '品蓝',
    # general audit: 能够 cilin alts (亦可/可知) are both 文言. Source blacklist
    # since neither alt ever fits modern modal "能够 X" slot.
    '能够',
}


NOISE_EXPRESSIONS = {
    'hedging': ['说实话', '坦白讲', '客观地说', '实事求是地讲', '平心而论',
                '老实说', '不夸张地说', '公正地看'],
    'self_correction': ['或者说', '准确地讲', '换个角度看', '严格来说',
                        '更确切地说', '往深了讲', '细想一下'],
    # cycle 183 dropped '或许' — in detect_cn HEDGING_PHRASES, injection
    # increases hedging_language count, self-defeat (cycle 77 family).
    'uncertainty': ['大概', '差不多', '似乎', '多少有些',
                    '约莫', '估摸着', '八成'],
    # Cycle 77: dropped '换句话说' — it is in detect_cn's ai_high_freq_words
    # pattern, so injecting it raises the AI score (self-defeating).
    # cycle 208: trimmed — '话说回来'/'反过来看'/'说到这里'/'回过头看' all
    # narrative-voice openers that read as off-register in essay/factual text.
    # Kept '再往下想'/'顺着这个思路' which work in analytical contexts.
    'transition_casual': ['再往下想', '顺着这个思路'],
    # cycle 195: trimmed 8 → 3 — removed register-mismatched fillers
    # (怎么说呢/不瞒你说/你别说/讲真/这么说吧) that read very colloquial /
    # internet-slangy. They land in formal/business/academic text and
    # break fluency. Kept '其实/说到底/当然了' which fit most registers.
    'filler': ['当然了', '其实', '说到底'],
    # Cycle 55: dropped 5 entries that appear 0 times in 2.5M chars of
    # human Chinese (news + novel corpora) — '依我之见 / 以我的经验 /
    # 在我的理解里 / 就我所知 / 我个人倾向于'. These read as AI-style
    # stilted hedges in any register (academic / general / social), not
    # just academic. '我觉得' and '在我看来' kept (105 + 4 hits in human
    # corpus, idiomatic).
    'personal': ['我觉得', '在我看来'],
}

# Academic-safe categories (no oral fillers or personal opinions)
NOISE_ACADEMIC_CATEGORIES = ['hedging', 'self_correction', 'uncertainty']
NOISE_ACADEMIC_EXPRESSIONS = {
    # cycle 157: pool expanded from 4 → 7 each. Cycle 154 bn=10 academic
    # dropped from +15 (with casual-filler injection) to +10.5 (with this
    # formal-only pool). More formal candidates give random.choice more
    # variety, raising the chance of hitting LR-favorable phrasing.
    'hedging': ['客观地说', '实事求是地讲', '平心而论', '公正地看',
                '从客观角度看', '理性而言', '客观看待'],
    'self_correction': ['准确地讲', '严格来说', '更确切地说', '往深了讲',
                        '细究而论', '准确而言', '严谨地说'],
    # Cycle 77: dropped '在一定程度上' from this academic uncertainty pool too
    # (sister fix to cycle 76 in academic_cn). It is in detect_cn's hedging_
    # language and ai_high_freq_words patterns; injecting it raises the AI
    # score. Pool 5→4.
    # cycle 183 dropped '或许' from academic uncertainty too — sister
    # fix to general pool. Same detect_cn HEDGING_PHRASES self-defeat.
    'uncertainty': ['大致', '似乎', '多少',
                    '大体', '约莫', '大体上'],
}
_PARA_BOOST_ATTRIBUTION = (
    '指出', '表明', '认为', '揭示', '发现', '显示', '提出',
    '说', '称', '讲', '强调', '主张', '断言',
)
_PARA_BOOST_SUBORDINATE = (
    '随着', '鉴于', '为了', '由于', '尽管', '虽然',
    '如果', '假如', '若是', '倘若', '要是', '即便', '纵然',
    '除了', '除非', '只要', '只有', '无论', '不管',
    '当', '每当', '一旦',
)
_PARA_BOOST_BARE_CONTINUATOR = (
    '使得', '使', '导致', '引起', '造成', '致使',
)


_PARA_BOOST_REACTIONS = (
    '的确', '确实如此', '颇有道理', '不无道理',
    '有一定道理', '各有道理', '各有说法', '值得深思',
)


_LONGFORM_PARA_HEAD_MARKERS = (
    '首先', '其次', '再次', '最后', '然后', '接下来', '与此同时',
    '此外', '另外', '除此之外', '具体而言', '具体来说', '具体地说',
    '一方面', '另一方面', '总的来说', '总而言之', '综上所述',
    '因此', '所以', '由此', '进而', '从而', '基于此',
    '然而', '不过', '事实上', '实际上',
)

_LONGFORM_STARTER_MARKERS = (
    '同时', '此外', '另外', '因此', '所以', '然而', '不过',
    '事实上', '实际上', '具体来说', '具体而言', '总的来说',
    '换言之', '简而言之', '需要注意的是', '值得注意的是',
)


_PARA_INTERJECTION_NEUTRAL = (
    # cycle 195: trimmed 8 → 3 — removed 5 academic-only interjections
    # (此点尚需 / 此种情形 / 相关因素 / 若进一步 / 仔细推敲) that read
    # contemplative-academic when injected mid-text in informational /
    # workplace / general samples. Kept 3 entries that fit informational
    # registers (common-saying or "另一种角度" framing). Loses some pool
    # variety; bn=10 still has 3 distinct picks per pass.
    '事情可能并不如表面所示那般简单，需要更细致地审视。',
    '若从更多角度去考虑，结论恐怕会有不少不同之处。',
    '换个角度去看也成立，问题的另一面同样不容忽视。',
)


# Narrative-voice variants for novel style — character-internal / group
# beats only. Setting-specific lines (time-of-day, indoor / outdoor,
# weather) are deliberately excluded so the inserted paragraph doesn't
# contradict the surrounding scene state. Each is >=20 cn chars to pass
# the >=20 paragraph filter used by compute_paragraph_length_cv.
_PARA_INTERJECTION_NOVEL = (
    '众人都不约而同地陷入了一阵短暂的压抑沉默。',
    '他抬起头来，目光缓缓扫过众人脸上的神色一遍。',
    '他转过头去，目光在某处停留了片刻又缓缓移开。',
    '时间仿佛在这一刻悄然凝固住了，没有人开口说话。',
    '他心中暗暗思量了一阵子，事情似乎并不那么简单。',
    '气氛变得有些紧张了起来，众人之间默然不语好一会。',
    '他皱了皱眉头，似乎在心里反复斟酌着什么内容不解。',
    '他眯起了眼，神色之中流露出一种难以言喻的情绪。',
)


_NARRATIVE_SAFE_CATEGORIES = ['hedging', 'uncertainty', 'self_correction']


def pick_best_replacement(sentence, old, candidates):
    if not _USE_STATS or not candidates or len(candidates) <= 1:
        return random.choice(candidates) if candidates else ''
    try:
        from ngram_model import compute_perplexity
    except ImportError:
        from scripts.ngram_model import compute_perplexity
    scored = []
    for candidate in candidates:
        new_sentence = sentence.replace(old, candidate, 1)
        ppl_result = compute_perplexity(new_sentence, window_size=0)
        scored.append((candidate, ppl_result.get('perplexity', 0)))
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n <= 2:
        return scored[-1][0]
    return scored[n - 2][0]


def _compute_burstiness(text):
    if not _USE_STATS:
        return None
    ngram_analyze = _ngram_analyze()
    if not ngram_analyze:
        return None
    stats = ngram_analyze(text)
    return stats.get('burstiness', None)
