# N-1d 同句重复词 diagnostic — 2026-05-04

`audit/scan_in_sentence_repeats.py` 扫描 humanize 输出，找同句内 close-range
（gap ≤ 20 chars）的字 bigram 重复。`--diff` 模式只 surface humanize 引入的新
重复（src_count → out_count 增加的）。

## 设计

- 句切：`[。！？!?\n]+`
- 字 bigram 滑动窗口（仅 CJK 区间 `一-鿿`）
- 默认 max_gap=20 chars（剔除话题性长跨度重复）
- COMMON_BIGRAMS 白名单 70+ 词（我们/可以/通过/...）
- LEGIT_PATTERNS：一X一X / X来X去 / X前X后 / AA-doubled / 越X越X
- `--diff`：仅留 humanize 输出 bigram count > 源文本 bigram count 的项

## 数据

输入：4 个 hero examples（academic/general/social/long_blog）+ HC3 100 短问答

| 源 | raw findings | diff findings (humanize 引入) |
|---|---:|---:|
| sample_academic.txt | 1 | 0 |
| sample_general.txt | 1 | 0 |
| sample_social.txt | 1 | 0 |
| sample_long_blog.txt | 19 | 4 |
| HC3 100 | 0 | 0 |
| **合计** | **22** | **4** |

HC3 短问答（chatgpt_answers，~100-400 chars）几乎不会触发该信号——文本太短、
单句重复模式在 prose 长文本里更常见。

## 4 条 humanize 引入候选（全部 sample_long_blog.txt）

1. **`市场×2 gap=12`** ★ 实际 actionable
   - 源（line 17）：「异常激烈。为了在市场中突围」（句号分隔）
   - humanize 输出：「异常激烈，为了在市场中突围」（合并成同一句）
   - 触发：`randomize_sentence_lengths` 或类似 merge 把两句合并，
     原本跨句的 `市场` 落进同一句产生 in-sentence 重复

2. **`开发×2 / 发团×2 gap=19`** — 源 line 9 已有 3x 开发团队/设计团队 平行结构，
   humanize 输出基本保留（前缀 transition 替换）。源风格而非新引入。

3. **`产品×2 gap=8`** — 源 line 15 「产品经理必须在产品生命周期」原本 2x，
   humanize 改写为更紧凑表达 → 距离从源中较大 gap 缩到 gap=8。边界 case。

## 结论

humanize 引入 in-sentence 重复的实际 rate 极低：
- 100 HC3 短文本：0
- 4 hero 长文本：1 真 actionable（市场 sentence-merge）

cycle 230-247 共 +110 cilin alts + 8 source blacklist 工作显然把 substitution-
cascade 类同句重复 bug 已基本清干净。N-1d 主要发现的是 sentence-merge artifacts，
不是 cilin/word_synonyms 替换 cascade。

## 可行后续

- **不 fix**（推荐）：1 个候选不够触发改 merge 逻辑的 candidate-flip 风险，hero
  极度敏感（academic 47/floor 50，long_blog 45/floor 46）。
- **fix path 备案**：`randomize_sentence_lengths` strategy A merge 时 check
  combined sentence 是否产生 close-range 字 bigram 重复，若是则 skip 该 merge。
  建议 floor headroom 更宽时再做。
- **infrastructure**：诊断工具落盘 `audit/scan_in_sentence_repeats.py`，可定期
  跑做 regression / 监控。
