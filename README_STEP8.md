# Step 8 — reference initialization and family-fold sensitivity

Extract beside the existing `main.py`; retain the `pebr_robust8` folder and
`PEBR_STEP8_MANIFEST.json` beside `pebr_step8_robustness.py`.

```powershell
& "C:\Users\Sabir\AppData\Local\Programs\Python\Python310\python.exe" .\pebr_step8_robustness.py --project-root .
```

The program resolves your inspected, completed Step 7 and its unchanged
Step 6/5/3/2 lineage. It writes a new short workspace under
`outputs/revision_rule_lab/robustness8/s8_<identity>/` and prints its location.
Send `PEBR_step8_inspection.zip` after completion, or after a block. Rerun the
same command to resume saved reference/action/model checkpoints.

For a cache/design check without any new fitting or selection, optionally use
`--mode preflight`. A preflight success is not a completed sensitivity study.

## Scientific design

Both full corpora use K=8, 10, 12 and final NMF seeds 0–4. The known first
fresh mask bank 3711 stays fixed. The rule, cap 1/3, lambda 1, masks, counts,
vocabulary, encoder and scoring/transfer operations remain unchanged.

| Case | Reference seed | Family-fold seed | Change |
| --- | ---: | ---: | --- |
| default | 0 | 2710 | Reuse the original first-fresh-bank fits |
| r1 | 1 | 2710 | Predictor initialization only |
| r2 | 2 | 2710 | Predictor initialization only |
| f3710 | 0 | 3710 | Family folds, donor eligibility and predictor fitting partitions |
| f4710 | 0 | 4710 | Family folds, donor eligibility and predictor fitting partitions |

Each case compares NMF, ordinary similarity sharing, matched-amount sharing,
the same-score similarity-prefix control and complete revised PEBR. The
matched control matches EACH receiver's realized mass and contributing donor
count. NPMI, C_v, TD and five-seed topic stability remain visible.

Before new experiments, the four-policy selector must exactly reproduce the
saved K12 actions and transferred representations for BOTH collections. It
uses the same frozen evaluator/search/transfer functions. Cached visible-query
embeddings must reconstruct the old masked and full donor lists exactly.
Missing or changed caches block the program; there is no encoder call,
download, automatic fallback or adjustment to obtain a preferred outcome.

The package also exports `transportation_clarification/`. It distinguishes an
empty selected action from zero transfer because selected donors are unavailable.
The original bus anchor illustrates the latter. A separate explicit empty-action
case is selected in source order without using protected outcomes; all controls
are retained. This corrects interpretation only and does not alter the rule.

## Workload and interpretation

- 120 new fold-excluded reference NMF fits.
- 420 new final topic-model fits; 150 unique existing fits reused.
- 750 case-fit records in the reporting grid, including reuse across cases.
  These are NOT 750 independent fitted models.
- 150 case-policy summaries and 360 paired metric contrasts.

Reference seeds and folds are separate sensitivity axes. Final-model SDs are
reported inside each setting. Do not pool the cases as independent corpora or
claim a confidence interval from their SDs. The fold test changes both
reference fitting and donor eligibility, so it cannot isolate just one of
those mechanisms. All reversals are saved.

The run does not change your old `src`, `main.py`, configurations, models or
experiment outputs. It performs new NMF fitting only in this isolated
workspace. It does not retrain the neural baselines or redo protected-word
audits. Timing files describe this invocation and may include resumed work;
the combined four-policy selection time is not standalone PEBR runtime.

## Remaining implementation after review

Genuine K10 BTM/kNNTM/GloCOM runs are still needed for a complete all-method
K8/10/12 benchmark. Baseline budget adequacy still requires review. Retained
corpus-size, cap/lambda and runtime figures must use the revised rule. The
final closeout will map every main/supplementary asset to its verified run;
only then will the complete results package be labelled ready for manuscript
revision. A successful Step 8 execution does not guarantee favorable rankings.
