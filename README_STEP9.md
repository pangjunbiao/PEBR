# Step 9 — complete revised-rule results and publication assets

Extract beside your existing `main.py`, retaining `pebr_complete9`,
`s9_accepted_step8.json` and `PEBR_STEP9_MANIFEST.json` beside the script.

```powershell
& "C:\Users\Sabir\AppData\Local\Programs\Python\Python310\python.exe" .\pebr_step9_complete.py --project-root .
```

This is a substantial resumable run. It reads the inspected Step 8 and its
frozen predecessors and writes a separate short workspace under
`outputs/revision_rule_lab/completion9/s9_<identity>/`. Existing project
source, experiments, checkpoints and figures remain unchanged. There are no
downloads, package upgrades or outcome-based choices during the run.

The final inspection ZIP is `PEBR_step9_inspection.zip`; its exact location is
printed, including after a block. Send that ZIP for final review. Rerunning the
same command reuses verified phase outputs and resumes incomplete work.

## What this completes

| Phase | Work |
| --- | --- |
| Benchmarks | Actual K10 BTM, kNNTM and GloCOM fits, both corpora and five seeds; 30 new fits |
| Budget review | Continue all neural K8/10/12 streams from 200 to 400 epochs, preserving settings and optimizer/RNG state; 60 new continuations |
| Parameters | Fixed cap/lambda sweeps, five final seeds, both corpora; 40 new fits and 20 verified shared fits |
| Collection size | Five nested family-aware ladders at targets 100/200/400/600, both corpora, K12, five seeds and five policies; 1,000 final fits and 200 reference fits |
| Runtime | One fresh-cache, standalone PEBR pipeline per corpus; includes audited encoder loading/encoding, retrieval, five reference fits, selection and final NMF/original-post inference |
| Auxiliary retrieval | Recompute the retained supplementary category audit from 150 current fitted representations, including new K10/400-epoch baselines; no new fitting and no direct-MiniLM row |
| Assets | Main/supplementary LaTeX tables, vector PDF/PNG plots, numerical source tables and a provenance ledger |

The main neural benchmark budget is **400 epochs for every neural method,
corpus and K**, fixed before this run's results. Both 200- and 400-epoch
results remain visible. This tests training-budget sensitivity, not optimal
hyperparameter tuning. BTM remains at 1,000 iterations. NMF settings remain
unchanged. PEBR is not modified: the default remains cap1/3 and lambda1.

New optimizer checkpoints are temporary. After decoder/membership replay and
compact inference export are verified, only the optimizer checkpoints created
inside this new workspace are pruned. The compact inference parameters, topic
scores, original-post memberships, loss histories and exact source identities
remain. Original checkpoints are never pruned or replaced.

## Corpus-size protocol

Sampling uses entire original text families in fixed nested orders. Each
sample refits minimum document frequency 2, excludes and records zero-count
representations, and retains original family-fold assignments for its posts.
Actual retained sizes and vocabularies are saved in the manifests. Previously
retained posts remain nested as the nominal target increases. Donors and
predictive fitting rows are restricted to the sample; no unsampled post supplies
donor evidence. Frozen pretrained encoders remain common to all methods.

Both coherence metrics use the same unaugmented full-collection evaluation
reference across sizes. This is a controlled subsampling analysis, not a new
held-out-corpus evaluation. The five overlapping ladders are not independent
corpora. Size-plot SD bars are across five ladder means, each averaging five
paired final-seed changes from NMF; they are not confidence intervals.

## Parameter and runtime scope

Parameter sweeps use the known first fresh selection bank3711 and five final
seeds. Only one parameter changes at a time. The default and lambda0 reuse
exact matching current-rule fits. No setting is selected from the sweep.

Runtime is measured on one new isolated attempt per corpus, final seed0.
It times standalone PEBR rather than the earlier combined control loop.
Encoder loading includes backend verification probes. Topic-quality metrics,
post-run verification and file export are excluded from timed computation.
Incomplete runtime attempts are not reused as fresh timings. These are
hardware-specific observations, not a speedup claim or uncertainty estimate.

## Optional phase commands

Use `--phase preflight` to check the existing inputs/dependencies without new
training. A successful preflight does not complete Step 9. It also verifies the pinned original primary CSV needed for the auxiliary annotation audit, without using labels for training.

The same script also accepts `--phase benchmarks`, `--phase parameters`,
`--phase sizes`, `--phase runtime`, `--phase retrieval`, and `--phase assets`. These share the same
workspace; `assets` requires all five preceding result phases. The default
`all` runs the full plan in order.

## Outputs and final status

`publication_assets/main/` contains the predefined K12 benchmark, controlled
policies, seed0 component tables, parameter tables and traceable transportation
evidence. `publication_assets/supplement/` contains the other K values,
neural-budget checks, repeated selection effects, protected-word audits,
reference/fold sensitivity, updated within-collection category retrieval, case recurrence and runtime. Scientific plots are
in `publication_assets/figures/`. Full-precision tables are in `source_data/`.

The main PEBR benchmark uses the predefined first fresh bank3711 with five
final fits. Four-bank selection variation remains a separate supporting
analysis; reused NMF fits are never counted as extra independent experiments.

The auxiliary retrieval audit keeps the original 368-query category protocol,
budgets 5/10/20 and family/fold exclusions. It reads labels only after training,
uses original-post representations, and retains every unfavorable result.
Paired 95% intervals use 2,000 query-family resamples within categories after
averaging fitted seeds. This is within-collection evaluation, not evidence of
generalization to unseen complaints or improved passenger services.

The bus anchor is correctly described as unavailable-donor fallback. Explicit
abstention is illustrated separately by source row508, a news-like aircraft
safety report rather than a direct passenger complaint. The unsuccessful case
and all adverse numerical results are retained. Cases illustrate mechanisms;
they do not establish incident causes or passenger-service improvements.

`results_ready_for_manuscript_revision=true` is written only when every
required result phase and the publication assets are complete. Even then the
inspection ZIP must be reviewed before updating the manuscript. The script
never edits your manuscript and never substitutes old-method figures for new
results. `asset_manifest.json` maps assets to their numerical sources and
verified predecessor runs.

LaTeX fragments require `booktabs`, `graphicx`, `array`, `tabularx` and `textcomp`. Tables are fragments for your IEEE manuscript or supplement; no LaTeX installation is required to run the Python experiments. PDF and PNG scientific figures are generated directly by Matplotlib. `main_results_tables.tex`, `supplement_results.tex` and `FIGURE_CAPTIONS.md` identify the supplied assets. This package contains no synthetic test results for use in the paper.
