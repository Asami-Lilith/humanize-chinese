# /academic — 学术论文 AIGC 降重（v5.0）

Reduce AIGC detection score for academic papers. Targets CNKI (知网), VIP (维普), Wanfang (万方).

v5.0: **scene-aware academic LR** (在中文学术语料 +HC3 academic 子集训练，coef 文件 `lr_coef_academic.json`), **10 学术检测维度**, **165 replacement patterns** (含 academic-tone 过渡词), **双评分对比** (学术专用 + 通用 detect_cn fused)。

## Usage

The user provides academic text or a file path. Run academic-specific rewriting.

## Steps

1. Save the input:
   ```bash
   cat > /tmp/academic_input.txt << 'ACAD_EOF'
   [user's text here]
   ACAD_EOF
   ```

2. Run academic rewriting with comparison:
   ```bash
   $SKILL_DIR/humanize academic /tmp/academic_input.txt -o /tmp/academic_output.txt --compare
   # or:
   python $SKILL_DIR/scripts/academic_cn.py /tmp/academic_input.txt -o /tmp/academic_output.txt --compare
   ```

3. If score is still above 40, try aggressive mode:
   ```bash
   $SKILL_DIR/humanize academic /tmp/academic_input.txt -o /tmp/academic_output.txt -a --compare
   ```

4. Use `--quick` for fast mode (18× speed on 10k-char text):
   ```bash
   $SKILL_DIR/humanize academic /tmp/academic_input.txt -o /tmp/academic_output.txt --quick
   ```

5. Show the user:
   - Dual score: 学术专用评分 (11 维) + 通用评分 (同 detect_cn) 两个原分 → 改写后
   - The rewritten text
   - Remind them to review: check terminology accuracy and citation format

## Target Scores (v5.0 fused)

| Score | Level | Action |
|-------|-------|--------|
| 0-29  | LOW / 低 | ✅ Ready to submit |
| 30-49 | MEDIUM / 中 | ⚠️ Manual polish suggested |
| 50+   | HIGH / 高 | Try `-a` (aggressive) and review manually |

**Note v5.0**: fused 评分基于 scene-aware academic LR + rule + stat ensemble。Hero 学术样本 100 → 35 (-65 分)。学术文本对 detector 信号最饱和（论文格式 + 术语密度 + 引用模板），单纯词替换收益有限，best-of-10 + cilin 同义词扩展 + 段落级 CV 反制三者叠加。

## What It Does

- Replaces AI academic clichés with scholar-tone alternatives
  - "本文旨在" → "本研究聚焦于" / "本文尝试"
  - "研究表明" → "前人研究发现" / "相关研究揭示"
  - "被广泛应用" → "得到较多运用"
  - "近年来" → "过去数年间" / "此前数年"
  - "首先/其次/最后" → "其一/其二/末了" (等)
  - "因此" → "故而" / "由此" / "据此"
- Injects hedging language ("可能", "在一定程度上", "初步来看")
- Breaks uniform paragraph structure and long-sentence uniformity
- Inserts short neutral reactions at paragraph ends
- Adds author subjectivity ("笔者倾向于认为")
- Applies 40 paraphrase templates (e.g., 通过对X的分析→围绕X展开分析)

## Example

```
/academic 本文旨在探讨人工智能对高等教育教学模式的影响，具有重要的理论意义和实践价值。研究表明，人工智能技术已被广泛应用于课堂教学、学生评估和个性化学习等多个方面。
```
