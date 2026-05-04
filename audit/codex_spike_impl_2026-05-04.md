# Scene-aware best_of_n + contribution logger spike

## Summary

- Start HEAD: `a11c078`
- End HEAD: `d3dc2ea`
- Commits: 1 production commit (`humanize: route best-of scoring`)
- Phase A: done
- Phase B: done
- Phase C: done, opt-in only (`--score-mode`, default remains `lr`)

## Phase Status

| phase | status | key code |
| --- | --- | --- |
| A: scene-aware LR candidate ranking | done | `scripts/humanize_cn.py::_pick_lr_scene`, `humanize()` best_of_n loop |
| A: `compute_lr_score(scene=...)` verification | done | `scripts/ngram_model.py::compute_lr_score` already supported `scene`; no calibration files changed |
| B: debug-only contribution logger | done | `scripts/humanize_cn.py::_format_best_of_debug`, CLI `--debug-best-of-n`; prints to stderr only |
| C: multi-objective option | done | CLI `--score-mode {lr,fused,lr+rule}`; default `lr` |
| regression coverage | done | `tests/test_regression.py::test_pick_lr_scene` |

## Hero Before/After

`PYTHONHASHSEED=0`, `seed=42`, `best_of_n=10`. Before values are the provided `a11c078` baseline.

| sample | before (`a11c078`) | after (`d3dc2ea`) | floor | status |
| --- | ---: | ---: | ---: | --- |
| `sample_academic.txt` | 47 | 50 | 50 | pass |
| `sample_general.txt` | 39 | 39 | 45 | pass |
| `sample_social.txt` | 24 | 24 | 30 | pass |
| `sample_long_blog.txt` | 40 | 40 | 46 | pass |

After details:

| sample | route picked | general LR | picked LR | fused |
| --- | --- | ---: | ---: | ---: |
| `sample_academic.txt` | academic | 54 | 4 | 50 |
| `sample_general.txt` | general | 40 | 40 | 39 |
| `sample_social.txt` | general | 21 | 21 | 24 |
| `sample_long_blog.txt` | longform | 37 | 88 | 40 |

## HC3 100 Before/After

| run | correct_pct | gap | paragraph retention | grammar |
| --- | ---: | ---: | ---: | ---: |
| before (`a11c078`) | 95.0% | 55.0 | 98.0% | 0 |
| after (`d3dc2ea`) | 95.0% | 55.0 | 98.0% | 0 |

After command:

```bash
PYTHONHASHSEED=0 python3 evals/run_hc3_benchmark.py --n 100 --output audit/hc3_100_spike_after.json
```

## Scene Routing Stats

HC3 100 selected-candidate routing:

| route | selected samples |
| --- | ---: |
| general | 100 |
| academic | 0 |
| longform | 0 |

Auxiliary all-candidate distribution from the same 100 samples, 10 candidates each:

| route | candidates |
| --- | ---: |
| general | 1000 |
| academic | 0 |
| longform | 0 |

## Spot-check Audit

Hero outputs were generated into `audit/sample_*_after.txt` and reviewed manually.

Newly introduced jarring sentences: none found. The following awkward sentences are present in both `a11c078` and after outputs, so they are not introduced by this spike:

- `渗透到这些数字化工具让我们得以更加高效地进行任务规划和进度追踪。`
- `在制定产藏蓝图时，我与团队一起深入探究目标用户群体的需求和痛点，并将这些发现转化为具体的产品优势。`
- `数据使得的决策，能够帮助产品经理更客观地评估产品表现，并发现潜在的麻烦。`

## Validation

Pre-commit checks for `d3dc2ea`:

```bash
PYTHONHASHSEED=0 python3 -m unittest discover tests
PYTHONHASHSEED=0 python3 evals/run_hc3_benchmark.py --n 50
```

Results:

- Unit tests: 6 passed
- HC3 n=50: correct 92.0%, gap 52.6, paragraph retention 98.0%, grammar defects 0

Debug logger smoke test:

```bash
PYTHONHASHSEED=0 python3 scripts/humanize_cn.py examples/sample_general.txt --seed 42 --best-of-n 2 --debug-best-of-n
```

Confirmed candidate lines include `seed`, `scene_picked`, `LR_general`, `LR_academic`, `LR_longform`, `fused`, and `top_3_contributions`, while normal output remains stdout-only unless the flag is used.

## Risk Notes

Highest-risk change: scene-aware ranking in `scripts/humanize_cn.py::humanize()`.

Reason: `_pick_lr_scene()` is intentionally simple and substring-based. It works for the current spike and is separately tested, but longform routing can select by `lr_coef_longform.json` even when the general fused score is the production floor metric. The long_blog hero passes, but it is the best target for reviewer spot-checking.

Suggested review:

- Re-read `sample_academic_after.txt` and `sample_long_blog_after.txt`.
- Try a few real academic and >1500-char blog inputs with `--debug-best-of-n` to confirm the selected scene and top contributions look sane.
