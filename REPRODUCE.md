# Reproduce the manuscript results

## Validation completed

| Check | Result | Scope |
| --- | --- | --- |
| Windows preparation smoke test | 97 tests passed and both 32-post English samples prepared | Fresh English input downloads, preprocessing and candidate preparation |
| REALM-inspired comparison | 20 fits completed and the four-row summary matched the accepted CSV byte for byte | Five seeds with the original project's verified saved inputs |
| Local input audit | 464 checksums matched and 122 additional entries were present | Audited files on the original machine |
| Step 9 publication replay | All 106 assets generated. 101 matched byte for byte. Five PDF hashes differed and the corresponding PNGs matched byte for byte | Exports from saved numerical inputs. Figure PDF differences were recorded by file hash |

The REALM-inspired summary SHA-256 is
`485cf3e718e160b84c7b89537d85d18a31ae08d091cf3ab9d1f5f8bd8cf90593`.
A full raw-data-to-results run in a new checkout remains to be verified.

## Local check on Windows

Extract the candidate into `Downloads\PEBR_Release_Candidate`. Open the
original PyCharm project terminal, at the folder containing the original
`main.py` and `outputs`. Use the same interpreter as the accepted runs.

```powershell
$bundle = Join-Path $env:USERPROFILE 'Downloads\PEBR_Release_Candidate'
$work = (Get-Location).Path
$py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = (Get-Command python -ErrorAction Stop).Source }
& $py (Join-Path $bundle 'verify_release.py')
if ($LASTEXITCODE -ne 0) { throw 'Release checksum check failed.' }
$env:OMP_NUM_THREADS = '1'
& $py (Join-Path $bundle 'pebr_chinese_published_baselines\run_benchmark.py') --project-root $work --mode preflight
if ($LASTEXITCODE -ne 0) { throw 'Chinese baseline preflight failed.' }
& $py (Join-Path $bundle 'pebr_ntm_step1\run_pebr_ntm.py') --project-root $work --mode preflight
if ($LASTEXITCODE -ne 0) { throw 'ProdLDA preflight failed.' }
```

The two preflights use the staged source with the original project's existing
inputs. They create preflight records and perform no new fitting. This checks
the recovered source versions on the laptop. The clean-checkout test is a
separate milestone.

## Clean Windows preparation smoke test

Extract the updated ZIP into a new `Downloads\PEBR_Clean_Test` folder.
Use this copy for the smoke test so the runner starts without data, assets,
caches or outputs from the original PyCharm project.

```powershell
$clean = Join-Path $env:USERPROFILE 'Downloads\PEBR_Clean_Test\PEBR_Release_Candidate'
$py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = (Get-Command python -ErrorAction Stop).Source }
Set-Location -LiteralPath $clean -ErrorAction Stop
& $py .\verify_release.py
if ($LASTEXITCODE -ne 0) { throw 'Release checksum check failed.' }
& $py -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$smokePy = Join-Path $clean '.venv\Scripts\python.exe'
& $smokePy -m pip install -r requirements.txt numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 tokenizers==0.19.1 onnxruntime==1.22.1
if ($LASTEXITCODE -ne 0) { throw 'Smoke-test dependency installation failed.' }
& $smokePy .\main.py --config configs/experiments/corpus_smoke.yaml --mode run --device cpu --no-install
if ($LASTEXITCODE -ne 0) { throw 'Corpus preparation smoke test failed.' }
```

The preparation runner downloads and verifies the two English input files and
the frozen encoder assets. It prepares small samples and logs `COLLECTION COMPLETE`
for `primary_smoke` and `external_smoke`. This check exercises input acquisition
and preprocessing on CPU. Model fitting and paper-result comparisons are separate
checks. The smoke test creates local `data`, `assets`, `cache`, `logs` and `outputs`
directories, which are excluded from Git.

## Accepted runs

| Group | Completed run | Fits |
| --- | --- | ---: |
| Core Step 9 | `s9_f9c03fbcc688` | See table-specific run maps |
| English published baselines | `full_a78f98dc4b66` | 15 |
| Chinese published baselines | `full_5d8ff968911c` | 15 |
| REALM-inspired control | `full_20261106T034651308273Z` | 20 |
| ProdLDA | `full_7bd73b016645` | 30 |

The extension summaries under `reference_results/` match the reviewed
archives. The core CSVs are selected aggregate references. Case-level
data and the underlying models are separate inputs. Stage 4 belongs to
development provenance. Stage 5 provides the manuscript's fresh-bank runs.

## Replay the verified comparison and publication exports

Run these from the repository root on the Windows machine with the accepted
project. Use the interpreter from that project's accepted run for the export
step. The publication exporter uses NumPy 1.26.4, SciPy 1.13.1 and Matplotlib
3.9.2. The first command writes a small audit report, the comparison writes
new fits under this checkout, and the publication replay writes to a separate
folder under Downloads.

```powershell
$release = (Get-Location).Path
$project = Join-Path $env:USERPROFILE 'PycharmProjects\proposed'
$py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'

$audit = Join-Path $env:TEMP ('pebr_full_reproduction_' + $stamp + '.json')
& $py -B .\tools\audit_inputs.py --project-root $project --out $audit
if ($LASTEXITCODE -ne 0) { throw "Review the audit at $audit" }

$smokePy = Join-Path $release '.venv\Scripts\python.exe'
& $smokePy -m pip install -r .\pebr_marginal_likelihood_benchmark\requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Benchmark dependency installation failed.' }
$receiptPath = Join-Path $project 'outputs\pebr_extensions\marginal_likelihood_retrieval\full_20261106T034651308273Z\run_manifest.json'
$receipt = Get-Content -LiteralPath $receiptPath -Raw -ErrorAction Stop | ConvertFrom-Json
if ($receipt.status -ne 'complete' -or $receipt.fits -ne 20) { throw 'Accepted comparison receipt is incomplete.' }
$inputs = $receipt.inputs
& $smokePy .\pebr_marginal_likelihood_benchmark\run_benchmark.py `
    --project-root $release --mode full `
    --data-root $inputs.prepared_corpus --mask-bank $inputs.mask_bank `
    --query-cache (Split-Path -Parent $inputs.visible_query_cache) `
    --reference-cache (Split-Path -Parent $inputs.reference_fits.'0') `
    --pebr-actions $inputs.default_pebr_actions
if ($LASTEXITCODE -ne 0) { throw 'REALM-inspired comparison failed.' }

$out = Join-Path $env:USERPROFILE ('Downloads\PEBR_Publication_Replay_' + $stamp)
& $py -B .\tools\pebr_rebuild_publication_assets.py --project-root $project --release-root $release --out $out
if ($LASTEXITCODE -ne 0) { throw "Review the publication replay in $out" }
```

Compare the new `summary.csv` from the comparison with
`reference_results/realm_inspired/summary.csv`. The publication check writes
`publication_replay_report.json` in its destination. It regenerates the Step 9
exports and preserves the saved original files. It uses the original Windows
project for the saved loss histories referenced by the archived results.

## Manuscript result map

| Manuscript item | Archived producer or input |
| --- | --- |
| Main Table I and Supplement S1 | Frozen Step 2 preparation and original corpus configuration |
| Main Table II and Supplement S2–S3 benchmark tables | Step 9 benchmark grid and neural budget check, plus both published-baseline runners for EnCOT, FASTopic and NeuroMax at K=12 |
| Main Table III and Chinese policy table in S3 | Fixed-cap PEBR and its controls, with the K=12 mask-3711 replay. The primary one-donor rows use the separate REALM-inspired benchmark |
| Supplement S3.1 | ProdLDA with original, matched-sharing and PEBR targets, five final seeds on both corpora |
| Main Tables IV–V and Supplement S4–S5 | Fresh-selection summaries and Step 8 reference/fold checks |
| Main Table VI, Fig. 1 and Supplement S7.1 | Step 9 parameter and nested-size runs |
| Main Table VII and Supplement S6–S7.3 | Protected-word audit, transport evidence ledgers, original-post topic readout and recurrence |
| Supplement S8 | Step 9 runtime phase and recorded hardware |

The final borrowing policy is identified as `preserve_forward_fixed_cap`. In
the historical audit tree, `proposed` identifies a different rule. Archive
the accepted frozen source and the final runner revisions separately from the
mutable project root. The independently reviewed Step 9 workspace was
`outputs/revision_rule_lab/completion9/s9_f9c03fbcc688` and used the Step 8
replay at `outputs/revision_rule_lab/robustness8/s8_eeadcb7a250a`. The stage 2 receipts identify all four final extension runs.

## Resolve inputs and permissions

| Input | Recorded source and placement | Prepared size |
| --- | --- | --- |
| Transport-grievance | [Source project](https://github.com/sohampoddar26/indian-transport-snam). The configuration expects `data/raw/indian_transport/grievance.csv` | 619 posts, 1,296 terms |
| Pooled Chinese transport | Local `data/raw/chinese_transport/` with `beijing.csv`, `shanghai.csv` and `xiamen.csv` in that order. Confirm the authoritative origin, release permission and access route | 2,053 posts, 5,002 terms |

For each corpus, record the citation, revision or access date, file SHA-256,
encoding, source-row order, text column, preparation configuration, and terms
of access and redistribution. Include the vocabulary, split and input hashes
in the release manifest. Use the approved access route for raw text and any
text-derived artifacts. Publish synthetic fixtures and cleared aggregate
outputs in the repository. Record pinned upstream revisions, input checksums and
download instructions for external encoders, embeddings and baseline sources.
Select the software license with the authors or institution. Dataset and
third-party rights are separate from the project software license.

## Recorded experiment commands

These entry points are recorded in the archived runner documentation. Use
them from an assembled release only after the input paths, predecessor
manifests and environment have been tested in a new checkout. The old audit
stages consume existing historical artifacts. For a public clean clone we must
provide a supported way to generate each dependency or obtain a verified,
shareable input bundle. Pin the accepted run directories explicitly wherever
a runner provides that option. Run the historical training commands in a
separate project workspace because they write phase reports into its output
tree.

### Historical provenance and core experiments

```powershell
& $py .\pebr_step1_audit.py --project-root .
& $py .\pebr_step1_closeout.py --project-root .
& $py .\pebr_step2_prepare.py --project-root .
& $py .\pebr_step3_run.py --project-root . --mode compare
& $py .\pebr_export_lab_inputs.py --project-root .
& $py .\pebr_revised_rule.py --project-root . --mode candidate
& $py .\pebr_validate_revised_selection.py --project-root .
& $py .\pebr_step6_protected_words.py --project-root .
& $py .\pebr_step7_audit.py --project-root .
& $py .\pebr_step8_robustness.py --project-root .
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $py .\pebr_step9_complete.py --project-root . --phase preflight
& $py .\pebr_step9_complete.py --project-root . --phase all
```

Step 9 also exposes `benchmarks`, `parameters`, `sizes`, `runtime`,
`retrieval` and `assets` as `--phase` options. The runtime table records the
original machine. Compare the procedure and backend on a new machine, with
elapsed times reported as new measurements.

### Final extensions

Use the interpreter and device recorded in each accepted run receipt.
The archived command lines below use each runner's default device. Confirm
those defaults against the accepted run before freezing the public commands.

```powershell
& $py .\pebr_published_baseline_benchmark\run_benchmark.py --project-root . --mode preflight
& $py .\pebr_published_baseline_benchmark\run_benchmark.py --project-root . --mode full
& $py .\pebr_chinese_published_baselines\run_benchmark.py --project-root . --mode preflight
& $py .\pebr_chinese_published_baselines\run_benchmark.py --project-root . --mode full
& $py .\pebr_marginal_likelihood_benchmark\run_benchmark.py --project-root . --mode preflight
& $py .\pebr_marginal_likelihood_benchmark\run_benchmark.py --project-root . --mode full
& $py .\pebr_ntm_step1\run_pebr_ntm.py --project-root . --mode preflight
& $py .\pebr_ntm_step1\run_pebr_ntm.py --project-root . --mode full
```

All four extension runners also accept `--mode smoke`. Use smoke outputs
to check installation and execution. Verify paper results against the full
runs and their accepted configurations.

## Clean-checkout release gate

Provision an environment pinned to the producing manifests. The archived
Windows runs used Python 3.10.9 and PyTorch 2.7.0+cu128. The extension runs
recorded CPU execution. The core historical chain also uses pinned assets and
validated predecessor artifacts. Start with the accepted inputs and explicitly
record the acquisition or bootstrap route for a new user.

The four native GloCOM/kNNTM Python files are external source dependencies.
Use the commit and per-file digests in release_provenance.json for exact
acquisition, then match the required project-relative paths. Keep separately
managed data, weights and archived intermediate outputs out of the repository.

Confirm source and access terms for the pooled Chinese data before publication.
Rebuild and compare each claimed result family, including figure inputs,
with the accepted references using documented numerical tolerances.
Record the actual checks performed. Then prepare the GitHub release.
