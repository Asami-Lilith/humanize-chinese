# /humanize — 去除中文文本的 AI 痕迹（v5.0）

Rewrite Chinese text to remove AI fingerprints. Detect → Rewrite (best-of-N) → Verify.

v5.0 uses **scene-aware LR humanize loss** (general / academic / longform), **best-of-N humanize** (默认 N=10 取最低 LR), **165 replacement patterns**, **40+ paraphrase templates**, **段落级反制** (paragraph length CV / 跨段 trigram 重复), 加 **CiLin 同义词词林** 38873 with collision blacklist。

## Usage

The user provides Chinese text (directly or as a file path). Run the full pipeline: detect, rewrite, compare.

## Steps

1. Save the input text:
   ```bash
   cat > /tmp/humanize_input.txt << 'HUM_EOF'
   [user's text here]
   HUM_EOF
   ```

2. Run compare (detect + rewrite + score comparison in one step):
   ```bash
   $SKILL_DIR/humanize compare /tmp/humanize_input.txt -a -o /tmp/humanize_output.txt
   # or:
   python $SKILL_DIR/scripts/compare_cn.py /tmp/humanize_input.txt -a -o /tmp/humanize_output.txt
   ```

3. Show the user:
   - Original score → Rewritten score (target < 50 for general, < 40 for academic)
   - The rewritten text
   - Key changes made

## Options

- Default mode (best-of-10) 大多数场景足够
- `--best-of-n N` 自调 N (more = better but slower; N=1 等于单次 humanize)
- `-a` (aggressive) 强制 full-pipeline
- `--style xiaohongshu` / `--style novel` / `--style zhihu` etc. for platform-specific rewrites
- `--scene novel` 长文本/小说场景 (≥500 字推荐)
- `--quick` 跳统计优化 + best-of (18× 速度，单次 humanize)
- `--cilin` 启用 CiLin 同义词词林扩展 (38,873 words, 含 49 entry collision blacklist)

## Available Styles

casual, zhihu, xiaohongshu, wechat, academic, literary, weibo, **novel**（长篇叙事专属）

## Target Scores (v5.0 fused)

| Input type | Good output score |
|------------|------------------|
| 刻板 AI 样板文 (论文模板/八股) | < 35 (LOW, from 90+) |
| Natural ChatGPT | 5-15 (LOW, from 15-25) |
| Academic paper | < 35 (LOW, academic-specific + generic 双低) |
| 长篇博客/小说 (≥1500 字) | ~41 (MEDIUM, from 90+) |

## Example

```
/humanize 本文旨在探讨人工智能对高等教育教学模式的影响，具有重要的理论意义和实践价值。
```
