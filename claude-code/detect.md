# /detect — 检测中文文本的 AI 痕迹（v5.0）

Detect AI-generated patterns in Chinese text. Fused 评分 (rule × 0.2 + LR × 0.8), 0-100。20+ rule dimensions + 8 HC3-calibrated statistical features (sentence-length CV, short-sentence fraction, comma density, perplexity, GLTR rank buckets, DivEye skew/kurt) + scene-aware LR (general / academic / longform 三路) + 段落级信号 (paragraph length CV / 段内句长 CV / 跨段 trigram 重复)。

## Usage

The user provides Chinese text (directly or as a file path). Run detection and report results.

## Steps

1. If the user provided a file path, use it directly. Otherwise, save the text to a temp file first:
   ```bash
   cat > /tmp/detect_input.txt << 'DETECT_EOF'
   [user's text here]
   DETECT_EOF
   ```

2. Run detection with verbose mode (use unified CLI if available, else script):
   ```bash
   $SKILL_DIR/humanize detect /tmp/detect_input.txt -v
   # 长文本/小说显式 scene:
   $SKILL_DIR/humanize detect /tmp/detect_input.txt --scene novel -v
   # 按长度自动切（≥1500 字走 longform LR）:
   $SKILL_DIR/humanize detect /tmp/detect_input.txt --scene auto -v
   # 学术论文显式 opt-in scene:
   $SKILL_DIR/humanize detect /tmp/detect_input.txt --scene academic -v
   # 直调脚本:
   python $SKILL_DIR/scripts/detect_cn.py /tmp/detect_input.txt -v
   ```

3. Report the results clearly:
   - Overall score and level (LOW/MEDIUM/HIGH/VERY HIGH)
   - Top suspicious sentences
   - Key AI patterns found (rule-based + statistical indicators)

## Score Reference

| Score | Level | Meaning |
|-------|-------|---------|
| 0–24  | 🟢 LOW | Reads like human-written |
| 25–49 | 🟡 MEDIUM | Some AI traces |
| 50–74 | 🟠 HIGH | Likely AI-generated |
| 75–100 | 🔴 VERY HIGH | Almost certainly AI |

## Statistical Indicators (v5.0)

校准基于 HC3-Chinese 300+300 短样本 + longform 170 长样本：

| Indicator | Cohen's d | Description |
|-----------|-----------|-------------|
| `para_sent_len_cv_avg` | -2.08 | 段内句长 CV (v5 长文本最强信号) |
| `paragraph_length_cv` | -1.49 | 段落长度 CV (人类段长方差大) |
| `stat_low_sentence_length_cv` | 1.22 | AI writes formulaic 15-25 char sentences |
| `stat_low_short_sentence_fraction` | 1.21 | Humans write short sentences; AI rarely |
| `cross_para_3gram_repeat` | +1.13 | 跨段 trigram 重复 (AI 容易复用短语) |
| `stat_low_perplexity` | 0.47 | Low character-level trigram perplexity |
| `stat_high_top10_bucket` | 0.44 | AI picks top-probability characters |
| `stat_low_comma_density` | 0.47 | AI writes longer uninterrupted clauses |
| `stat_low_surprisal_skew` | 0.41 | DivEye feature |

## Example

```
/detect 综上所述，人工智能技术在教育领域具有重要的应用价值和广阔的发展前景。
```
