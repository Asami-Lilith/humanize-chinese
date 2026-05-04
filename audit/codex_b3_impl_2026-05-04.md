# Codex B-3 Longform Mutation Profile Audit

Date: 2026-05-04

## 1. Summary

Start HEAD: `0360190`

End HEAD: `e75fe7d`

Commits:

- `8c1193d` Add longform candidate mutation profile
- `662cecd` Test longform mutation safety
- `56abc18` Blacklist unsafe longform rewrites
- `e75fe7d` Tighten synonym safety guards

No push / tag / release.

## 2. Phase A: Mutations Selected

Active longform-only profile uses 4 mutation families:

- `paragraph_punct_drift`: paragraph-to-paragraph punctuation rhythm drift via guarded one-clause split / one sentence-pair merge.
- `discourse_marker_diversity`: removes repeated paragraph-head discourse markers instead of adding new wording.
- `paragraph_length_cv_micro_adjust`: one guarded `vary_paragraph_rhythm()` pass only when paragraph CV is low.
- `starter_entropy_boost`: removes repeated safe transition starters; does not invent new sentence openers.

`cross_para_3gram_pruning_intensified` was implemented in the first pass but removed from the active profile after long-blog spot-check exposed bad CiLin substitutions (`在意于`, `涉世`, `不休`, etc.). The existing global `reduce_cross_para_3gram_repeat(max_replacements=4)` remains.

## 3. Phase B: Longform-Only Routing

Best-of candidate generation now routes only candidates classified by `_pick_lr_scene(out) == 'longform'` through `_apply_longform_mutation_profile(...)`.

Candidate count is unchanged: `best_of_n=10` still evaluates 10 candidates. Mutation seed is derived from candidate seed `s`.

Each mutation step is accepted only if:

- no empty paragraphs are introduced;
- paragraph count remains `>= before - 2`;
- longform LR does not get worse when LR scorer is available.

## 4. Phase C: Tests

Added:

- `tests/test_regression.py::test_longform_mutations_safe`

Final test result:

```text
PYTHONHASHSEED=0 python3 -m unittest tests/test_regression.py
Ran 8 tests in 20.590s
OK
```

## 5. Hero Before/After

Mode: default weight + scene-aware + secondary inert (`PYTHONHASHSEED=0`, `seed=42`, `best_of_n=10`).

| sample | before (`0360190`) | after (`e75fe7d`) | floor | status |
| --- | ---: | ---: | ---: | --- |
| `sample_academic.txt` | 50 | 49 | 50 | pass |
| `sample_general.txt` | 38 | 36 | 45 | pass |
| `sample_social.txt` | 23 | 24 | 30 | pass |
| `sample_long_blog.txt` | 42 | 42 | 46 | pass |

Long-blog final detail: rule `52`, general LR `39`, fused `42`, longform LR `94`.

## 6. HC3 100 Before/After

Before from task context:

- separation: `95.0%`
- gap: `55.0`
- paragraph preservation: `98.0%`
- grammar defects: `0`

After final HEAD:

- separation: `95.0%`
- human avg: `15.8`
- ChatGPT avg: `70.8`
- gap: `55.0`
- average reduction: `33.2`
- lowered samples: `86/100`
- paragraph preservation: `98.0%`
- length ratio: `1.011`
- duplicate-clause samples: `7`
- grammar defects: `0 in 0 samples`

Command:

```bash
PYTHONHASHSEED=0 python3 evals/run_hc3_benchmark.py --n 100 --seed 42 -o /tmp/hc3_100_b3_final2.json
```

## 7. 病句 Trip Wire

Final command used the original blacklist plus newly observed long-blog bad replacements:

```bash
PYTHONHASHSEED=0 python3 scripts/humanize_cn.py examples/sample_long_blog.txt --seed 42 > /tmp/lb.txt
grep -nE "产藏|检点|使得的|何等|渗透到|斯人|咱俩|无误|节本|本事|亦可|可知|在意于|小心于|在心于|在心|涉世|不休|不住是|不停是|利用频率|应用频率|顺着这个构思|顺着这个笔触|顺着这个思绪|掌管领域|保管领域|用户的关键|版本可观|版本明显|版本显著，|沟通才干|什么样攻克|什么解决|何以解决|此即|创制|产品打算|只顾于|调动产品" /tmp/lb.txt
```

Result: no matches.

## 8. 观感 Spot-Check

Four hero outputs were read manually.

Most jarring remaining lines:

- `sample_long_blog.txt`: "对我来说，也就是说，我一定要调整自己的沟通风格..." is redundant but grammatical.
- `sample_long_blog.txt`: ending "可行性不错" is a little template-like.
- `sample_social.txt`: "迎来尤其充实和有意义的人生" is slightly stiff.

No remaining spot-check item was traced to active B-3 mutation. The mutation-introduced synonym failures found during implementation were blacklisted or the mutation path was backed out.

## 9. Risk Record

- The intensified cross-paragraph synonym mutation is unsafe for long-blog prose; it exposed off-slot CiLin / synonym replacements. It is not active in final B-3.
- Active B-3 mutations are conservative and LR-guarded, so score impact is modest. Long-blog fused score stayed `42`; longform LR ended at `94`, not the hoped-for `45-65`.
- Several broader synonym safety guards were added because spot-check showed non-B-3 global rewrite paths could still surface bad wording when candidate choice changed.
- No calibration `.json` files were changed.
