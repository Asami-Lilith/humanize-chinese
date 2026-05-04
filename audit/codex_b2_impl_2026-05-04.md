# B-2 Phase A Feasibility Report

Date: 2026-05-04

## 1. Summary

- Start HEAD: `b1ba820`
- End HEAD: `b1ba820` (no implementation commit; Phase A returned NO-GO)
- Phase A: completed
- Phase B: skipped
- Phase C: skipped
- Phase D: existing regression + HC3 sanity run only; no new social route test because `_pick_lr_scene` was not changed

Decision: **INFEASIBLE / NO-GO** for `lr_coef_social.json` with the currently available corpora.

Reason: the only numerically large subset is HC3 `open_qa` / `nlpcc_dbqa`, but it is Q&A / explanatory register, not traceable social-media human writing. Strict social-marker samples are far below the requested 100/100 stop threshold, and the human side is especially weak.

## 2. Phase A — Feasibility

### Available Corpus Check

Data was found under `/Users/mac/claudeclaw/humanize/data`.

Corpus row counts:

- `hc3_chinese_all.jsonl`: 12,853 rows
- `human_misc_corpus.jsonl`: 234 rows
- `m4_zh_ood.jsonl`: 400 rows
- `cudrt_zh_ood.jsonl`: 400 rows
- `human_news_corpus.jsonl`: 500 rows
- `human_novel_corpus.jsonl`: 221 rows
- `human_news_multipara_corpus.jsonl`: 2,324 rows

No corpus has explicit `xiaohongshu` / `weibo` / `douyin` source metadata.

### HC3 Short Q&A Subset

`open_qa`:

- Paired rows with both AI and human answer >= 20 CN chars: 2,972
- Paired rows where both sides are 50-300 CN chars: 2,016
- Paired rows where both sides have at least one broad marker: 82
- AI answers >= 20 CN chars: 3,783; marker >= 1: 288; marker >= 2: 13
- Human answers >= 20 CN chars: 6,900; marker >= 1: 916; marker >= 2: 68

`nlpcc_dbqa`:

- Paired rows with both AI and human answer >= 20 CN chars: 1,074
- Paired rows where both sides are 50-300 CN chars: 363
- Paired rows where both sides have at least one broad marker: 2
- AI answers >= 20 CN chars: 3,887; marker >= 1: 34; marker >= 2: 0
- Human answers >= 20 CN chars: 1,101; marker >= 1: 5; marker >= 2: 0

Interpretation: HC3 provides enough short paired Q&A examples, but not enough social-style examples. The marker hits are dominated by generic words such as `分享` / `真的` / `推荐`, often in explanatory or advice answers rather than social-media posts.

### Human Misc / OOD Subset

`human_misc_corpus.jsonl`:

- Rows: 234
- Median CN length: 1,152.5
- 50-500 CN chars: 32
- Marker >= 1: 68
- Marker >= 2: 10
- Broad short candidates with marker and no strong formal marker: 5

`m4_zh_ood.jsonl` AI:

- Rows: 400
- Median CN length: 170
- 50-500 CN chars: 398
- Marker >= 1: 33
- Marker >= 2: 1
- Broad short candidates with marker and no strong formal marker: 28

`cudrt_zh_ood.jsonl` AI:

- Rows: 400
- Median CN length: 1,063
- 50-500 CN chars: 43
- Marker >= 1: 112
- Marker >= 2: 15
- Broad short candidates with marker and no strong formal marker: 2

Interpretation: the AI side can supply a small number of casual/advice-like samples, but the human side has only 5 plausible short broad candidates and 10 marker>=2 examples total. This is below both the 200/200 meaningful threshold and the explicit 100/100 stop threshold.

### Style Match Against `sample_social.txt`

Current `examples/sample_social.txt` is not actually xiaohongshu/weibo-style social copy. It is a formal AI-ish time-management essay with phrases like:

- `在当今快节奏的生活中`
- `值得注意的是`
- `首先` / `其次` / `此外` / `综上所述`
- `关键抓手`
- `数字化转型`
- `赋能`
- `降本增效`
- `核心竞争力`

The HC3 short Q&A subset matches general Q&A better than social-media post style. `human_misc_corpus.jsonl` is mostly long CUDRT/M4 text and does not provide enough short, traceable social human examples. Therefore a social LR trained from these sources would mostly learn Q&A/formal-vs-AI artifacts, not social-friendly human features.

## 3. Phase B — Training

Skipped. No `scripts/lr_coef_social.json` was created.

No existing coefficient file was modified:

- `scripts/lr_coef_cn.json`: untouched
- `scripts/lr_coef_academic.json`: untouched
- `scripts/lr_coef_longform.json`: untouched

## 4. Phase C — Scene Routing

Skipped. `_pick_lr_scene` was not changed.

The proposed marker rule (`>= 2` social markers -> `social`, priority `academic > social > longform > general`) should wait until a real social LR exists. Routing text to a missing or low-quality social coefficient file would add behavior without a trustworthy scorer.

## 5. Tests + HC3 + Hero

Existing regression:

- `python3 -m unittest -v`
- Result: 8 tests passed

HC3 sanity:

- `PYTHONHASHSEED=0 python3 evals/run_hc3_benchmark.py --n 50 -o audit/hc3_50_b2_phase_a.json`
- Human vs AI correct separation: 92.0%
- Human mean: 17.0
- ChatGPT mean: 69.6
- Gap: 52.6
- Average humanize drop: 34.1
- Paragraph preservation: 98.0%
- Grammar defects introduced: 0

Hero metrics, current HEAD:

- `sample_academic.txt`: fused 49, picked `academic`, paragraphs 3 -> 3
- `sample_general.txt`: fused 36, picked `general`, paragraphs 4 -> 6
- `sample_social.txt`: fused 24, picked `general`, paragraphs 5 -> 3
- `sample_long_blog.txt`: fused 42, picked `longform`, paragraphs 16 -> 21

Before/after note: no implementation was performed after the NO-GO decision, so hero scores remain the current baseline.

## 6. Risks

- Training on HC3 short Q&A would create a mislabeled "social" LR that is actually Q&A-oriented.
- Training on `human_misc_corpus.jsonl` would overfit a tiny and long-form-biased human set.
- Broad markers such as `分享` / `真的` / `推荐` are too ambiguous; they appear in formal news, Q&A, and AI advice text.
- Adding social routing without a trustworthy `lr_coef_social.json` could silently route real inputs to the general fallback or to a weak model later.
- The current `sample_social.txt` is not a clean social-style validation target, so using it as the main acceptance signal would be misleading.
