# Codex B-1 Implementation Report — 2026-05-04

## 1. Summary

- Start HEAD: `4b4e1cc`
- End implementation HEAD: `926a37a`
- End report HEAD: this report commit
- Commits: 2
  - `926a37a Add secondary best-of-n ranking signal`
  - report commit: `Document B-1 secondary ranking results`
- Calibration files changed: none (`scripts/lr_coef_*.json` untouched)
- New heavy dependencies: none

## 2. Phase A — Secondary Features

Selected 4 secondary signals for best-of-n ranking only:

- `bino_lp_diff`: from `analyze_text()['bino']['mean_lp_diff']`. It is implemented and currently disabled as a rule indicator; HC3 spot checks show ChatGPT tends to have less-negative values than human answers.
- `curv_mean`: from `analyze_text()['curv']['curvature_mean']`. Implemented Fast-DetectGPT-style signal, also disabled in rule caps; higher values are treated as more AI-like.
- Character MATTR: from `analyze_text()['char_mattr']`. Implemented zero-LLM lexical diversity metric; lower values are treated as more repetitive / AI-like.
- Starter entropy: implemented in `scripts/humanize_cn.py` as a zero-LLM sentence opener entropy over 2-char Chinese starters, based on the prototype idea in `evals/v5_calibrate.py`. Lower entropy is treated as more repetitive / AI-like.

Not selected:

- `gltr_top10_frac`, `gltr_top100_frac`, `entropy_cv`, `uni_tri_ratio`: already participate in the LR feature vector, so they were not useful as independent secondary rank signals.

## 3. Phase B — Ranking Logic

Default weight: `SECONDARY_WEIGHT = 0.2`.

Rationale: keep LR as the primary ordering signal while allowing up to about 20 rank points of sway from capped/disabled auxiliary features. This matches the temporary transition tolerance while limiting candidate flips when LR has a clear winner.

Implementation:

- Added `_secondary_signal_details(text)` and `_compute_secondary_signal(text)`.
- `_compute_secondary_signal(text)` returns a 0-100 AI-like score; empty input returns a valid 0.
- Best-of-n ranking now uses:
  - `rank_score = LR_or_fused_primary + secondary_weight * secondary`
  - `rank_tiebreak` preserves the previous score-mode tie behavior.
- Added CLI flag:
  - `--secondary-weight FLOAT`
  - default `0.2`
  - `0` restores previous ranking behavior.

## 4. Phase C — Debug Logger

`--debug-best-of-n` now prints the secondary score and raw subfeatures.

Sample:

```text
best_of_n seed=42 scene_picked=general LR_general=39 LR_academic=3 LR_longform=7 secondary=60.0 [bino=-1.3233 curv=1.7317 mattr=0.7714 starter_h=3.7004] rank=51.00 fused=38 top_3_contributions=[sent_len_cv=-1.87, perplexity=+1.40, gltr_top10_frac=-1.13]
```

## 5. Phase D — Tests

New test:

- `tests/test_regression.py::RegressionTests::test_secondary_signal`
  - validates empty input range
  - on local HC3 data, validates HC3 human average and ChatGPT average differ, with ChatGPT higher
  - skips the HC3 subcheck if the dataset is unavailable

Regression results:

```text
PYTHONHASHSEED=0 python3 -m unittest discover tests
Ran 7 tests in 17.362s
OK
```

Pre-commit sanity:

```text
PYTHONHASHSEED=0 python3 evals/run_hc3_benchmark.py --n 50
correct=92.0% gap=52.6 paragraph_preserved=98.0% grammar_defects=0
```

## 6. Hero Before/After

`PYTHONHASHSEED=0`, `seed=42`, `best_of_n=10`, fused score.

| sample | before (`4b4e1cc`) | after (`926a37a`) | floor | status |
|---|---:|---:|---:|---|
| `sample_academic.txt` | 50 | 50 | 50 | pass |
| `sample_general.txt` | 38 | 38 | 45 | pass |
| `sample_social.txt` | 23 | 23 | 30 | pass |
| `sample_long_blog.txt` | 42 | 42 | 46 | pass |

The default secondary weight did not change the selected hero candidates. A direct comparison of `secondary_weight=0` vs `0.2` was identical for all 4 hero outputs.

## 7. HC3 100 Before/After

Before (`4b4e1cc`, from task context):

- correct: 95%
- gap: 55
- paragraph retained: 98%
- grammar defects: 0

After (`926a37a`):

```text
PYTHONHASHSEED=0 python3 evals/run_hc3_benchmark.py --n 100
correct=95.0% gap=55.0 paragraph_preserved=98.0% grammar_defects=0
```

Ship gate status:

- correct >= 75%: pass
- gap >= 14.8: pass
- paragraph preserved >= 95%: pass
- grammar defects: pass

## 8. Visual Spot-Check

The 4 hero outputs were manually spot-checked after implementation.

Most jarring residual phrases:

- academic/general: `各维度地评估`, `可观提升`, `才干`
- long_blog: `产品经理得利用数据`, `顺着这个思路。从加强沟通能力`

These were not introduced by B-1 ranking: with `secondary_weight=0` and `0.2`, all 4 hero outputs are byte-identical. No B-1-specific rollback was applied.

## 9. Risks

- The current `0.2` weight is conservative enough that it did not move the 4 hero candidates. It may only affect samples where LR candidates are close or where secondary scores are sharply different.
- `bino_lp_diff` often saturates above the current normalization high bound on humanized hero text, so it contributes a stable penalty rather than fine-grained separation there.
- Because this is not retrained calibration, the secondary blend is heuristic. If future general samples show HC3-human false positives, first knob should be `--secondary-weight 0.1` or `0`.
