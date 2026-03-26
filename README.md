# humanize-chinese

Detect and humanize AI-generated Chinese text. Makes robotic AI writing natural and undetectable.

[![ClawHub](https://img.shields.io/badge/clawhub-humanize--chinese-blue)](https://clawhub.com/skills/humanize-chinese)

## Features

- **20+ detection categories** with weighted 0-100 scoring
- **Sentence-level analysis** — find the most AI-like sentences
- **Context-aware replacement** — regex patterns + plain text, longest-first
- **Sentence restructuring** — merge short, split long, vary rhythm
- **Vocabulary diversification** — reduce word repetition
- **7 writing styles** — casual, zhihu, xiaohongshu, wechat, academic, literary, weibo
- **🎓 Academic paper mode** — specialized AIGC detection & rewriting for Chinese academic papers (CNKI/VIP/Wanfang)
- **External config** — all patterns in `patterns_cn.json`
- **Pure Python** — no dependencies

## Install

```bash
clawhub install humanize-chinese
```

Or clone:

```bash
git clone https://github.com/voidborne-d/humanize-chinese.git
```

## Quick Start

```bash
# Detect AI patterns
python scripts/detect_cn.py text.txt
python scripts/detect_cn.py text.txt -v    # verbose with worst sentences
python scripts/detect_cn.py text.txt -s    # score only: "72/100 (high)"

# Humanize
python scripts/humanize_cn.py text.txt -o clean.txt
python scripts/humanize_cn.py text.txt --scene tech -a   # aggressive mode

# Style transform
python scripts/style_cn.py text.txt --style xiaohongshu

# Compare before/after
python scripts/compare_cn.py text.txt --scene tech -a

# 🎓 Academic paper AIGC reduction
python scripts/academic_cn.py paper.txt                    # detect + rewrite
python scripts/academic_cn.py paper.txt --detect-only      # detect only
python scripts/academic_cn.py paper.txt -o clean.txt       # rewrite and save
python scripts/academic_cn.py paper.txt -o clean.txt -a    # aggressive mode
python scripts/academic_cn.py paper.txt -o clean.txt --compare  # show before/after
```

## Scoring

| Score | Level | Meaning |
|-------|-------|---------|
| 0-24  | LOW | Likely human-written |
| 25-49 | MEDIUM | Some AI signals |
| 50-74 | HIGH | Probably AI-generated |
| 75-100 | VERY HIGH | Almost certainly AI |

## Detection Example

```
AI 评分: 100/100 [████████████████████] VERY HIGH
字符: 381 | 句子: 14 | 段落: 5
信息熵: 8.29 | 情感密度: 0.00%
问题总数: 25

🔴 三段式套路 (2)
   首先，值得注意的是...其次...最后
🔴 机械连接词 (9)
   值得注意的是, 综上所述, 总而言之...
🔴 空洞宏大词 (8)
   赋能, 闭环, 数字化转型...
🟠 AI 高频词 (3)
   助力, 彰显, 颠覆
🟠 模板句式 (2)
   随着...的不断发展, 在当今...时代
```

## Humanization Result

```
═══ 对比结果 ═══
原文:   100/100 [████████████████████] VERY_HIGH
改写后:  25/100 [█████░░░░░░░░░░░░░░░] MEDIUM
✅ 降低了 75 分
```

## Writing Styles

| Style | Name | Best For |
|-------|------|----------|
| `casual` | 口语化 | Social media, messaging |
| `zhihu` | 知乎 | Q&A, analysis |
| `xiaohongshu` | 小红书 | Reviews, lifestyle |
| `wechat` | 公众号 | Newsletters, articles |
| `academic` | 学术 | Papers, reports |
| `literary` | 文艺 | Creative writing |
| `weibo` | 微博 | Short posts |

---

## 🎓 学术论文 AIGC 降重工具

**专门针对中文学术论文场景**，帮助降低知网（CNKI）、维普（VIP）、万方等平台的 AIGC 检测率。

### 功能特点

- **10 个检测维度**：AI 学术措辞、被动句式、段落结构、连接词密度、同义匮乏、引用整合、数据论述、过度列举、结论圆满度、语气确定性
- **120+ 条学术替换词库**：将模板化学术表达替换为更自然的学术写法
- **保持学术严谨性**：改写不会降格为口语，保留论文该有的严谨与规范
- **学术犹豫语注入**（hedging language）：增加"可能""在一定程度上"等学术留白
- **作者主体性增强**：将"研究表明"替换为"笔者认为""本研究发现"
- **结论局限性补充**：检测并补充缺失的局限性讨论

### 使用方法

```bash
# 检测学术论文 AI 痕迹
python scripts/academic_cn.py paper.txt

# 改写并输出
python scripts/academic_cn.py paper.txt -o clean_paper.txt

# 仅检测，不改写
python scripts/academic_cn.py paper.txt --detect-only

# 激进模式（改写更多）
python scripts/academic_cn.py paper.txt -o clean.txt --aggressive

# 对比模式（显示改写前后评分变化）
python scripts/academic_cn.py paper.txt -o clean.txt --compare
```

### 检测维度说明

| 维度 | 说明 | 示例 |
|------|------|------|
| 🔴 AI 学术措辞 | AI 生成的典型学术套话 | "本文旨在""具有重要意义""进行了深入分析" |
| 🟠 被动句式过度 | 过多使用被动表达 | "被广泛应用""被认为是""被视为" |
| 🟠 段落结构整齐 | 每段长度、结构过于一致 | 每段都是"总述+3 分论点+小结" |
| 🟠 连接词密集 | 逻辑连接词使用频率异常 | "此外""综上所述""值得注意的是" |
| 🟡 同义表达匮乏 | 同一词汇反复出现 | "研究"出现 8 次 |
| 🟡 引用整合度低 | 引用方式机械化 | 每个引用都是"XX（2020）指出..." |
| 🟡 数据论述模板化 | 描述数据的方式单一 | "从表中可以看出""数据显示" |
| 🟡 过度列举 | 频繁使用编号列举 | 动辄①②③④ |
| 🟠 结论过于圆满 | 缺少局限性讨论 | 只说"成效显著"不说不足 |
| 🟠 语气过于确定 | 缺少学术犹豫表达 | "必然""毫无疑问""势必" |

### 改写前后对比

**改写前**（AIGC 评分: 100/100 VERY HIGH）：
> 随着社会的不断发展，人工智能技术在教育领域的应用引起了广泛关注。本文旨在探讨人工智能对高等教育教学模式的影响，具有重要的理论意义和实践价值。研究表明，人工智能技术已被广泛应用于课堂教学、学生评估和个性化学习等多个方面，发挥着重要作用。

**改写后**（AIGC 评分: 26/100 MEDIUM，降低 74 分）：
> 近年来社会变迁加速，人工智能技术在教育领域的应用逐渐进入研究者的视野。本文尝试探讨人工智能对高等教育教学模式的影响，兼具理论探索与实践参考的双重价值。前人研究发现，人工智能技术已广泛用于课堂教学、学生评估和个性化学习等多个方面，扮演着不可或缺的角色。

### 关于知网/维普 AIGC 检测

知网 AMLC（学术不端文献检测系统）和维普 AIGC 检测系统主要通过以下特征识别 AI 生成文本：

1. **词汇层面**：AI 倾向于使用模板化的学术表达，如"具有重要意义""进行了深入分析"
2. **句法层面**：被动句式过多、逻辑连接词密度异常高
3. **结构层面**：段落长度高度一致、"总-分-总"结构机械重复
4. **语义层面**：语气过于确定、缺少学术犹豫语、结论过于完美
5. **统计层面**：低信息熵、同义词使用匮乏、句长变异系数低

本工具的检测维度覆盖了以上所有层面，改写策略也针对性地处理每个特征。

### 注意事项

⚠️ 本工具仅做文本层面的风格调整，**不改变论文的学术内容和论证逻辑**。建议改写后仔细通读全文，确保：
- 论述逻辑未被打乱
- 专业术语未被误改
- 引用格式保持正确
- 整体可读性符合要求

---

## Customization

Edit `scripts/patterns_cn.json` to add/modify detection patterns, replacement alternatives, and scoring weights. The `academic_patterns` section contains all academic-specific patterns and replacements.

## License

MIT
