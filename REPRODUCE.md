# Reproduce the PEBR results

The commands below are for Windows PowerShell. Start in the checkout folder
containing `main.py`. Run the numbered sections in order for the route you
need. Each route states which inputs it expects. See [DATASETS.md](DATASETS.md)
for corpus placement and checksums.

| Route | Inputs | Validation |
| --- | --- | --- |
| Fresh checkout | Git checkout, Python 3.10 and network access to the pinned English sources | English preparation smoke test passed with 97 tests and two 32-post samples |
| Replay from archived inputs | Accepted project artifacts from the original machine | REALM-inspired comparison reproduced all 20 fits and the exact summary CSV. Step 9 regenerated all 106 publication assets |
| Full historical workflow | Authorized corpora, historical assets and checkpoints, accepted stage receipts, and the producing numerical environment | Commands below identify each stage. A fresh raw-data-to-paper run remains to be verified |

## 1. Verify the checkout and prepare English smoke-test inputs

Use a fresh checkout for this test. The following installs dependencies
into a virtual environment inside the checkout.

```powershell
$release = (Get-Location).Path
$python310 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
if (-not (Test-Path -LiteralPath $python310)) {
    $python310 = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-PEBR {
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "PEBR command failed with exit code $LASTEXITCODE" }
}

$py = $python310
Invoke-PEBR -B .\verify_release.py
Invoke-PEBR -m venv .venv
$smokePy = Join-Path $release '.venv\Scripts\python.exe'
$py = $smokePy
Invoke-PEBR -m pip install -r requirements.txt numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 tokenizers==0.19.1 onnxruntime==1.22.1
Invoke-PEBR .\main.py --config configs/experiments/corpus_smoke.yaml --mode run --device cpu --no-install
```

The preparation logs should show `COLLECTION COMPLETE` for `primary_smoke`
and `external_smoke`. The runner downloads the two pinned English source
files and the English encoder assets if needed. This exercise checks
preparation. It uses sample sizes of 32 posts and produces no manuscript fits.

## 2. Locate and audit the accepted local inputs

Run this in the same PowerShell session. Enter the original project
directory containing `main.py` and the accepted `outputs` when prompted.
For the manuscript's archived Windows runs the interpreter was Python
3.10.9. If that interpreter is elsewhere, set `$python310` to its full path.

```powershell
$project = (Resolve-Path -LiteralPath (Read-Host 'Original PEBR project folder') -ErrorAction Stop).Path
$py = $python310
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$audit = Join-Path $env:TEMP ('pebr_full_reproduction_' + $stamp + '.json')
Invoke-PEBR -B (Join-Path $release 'tools\audit_inputs.py') --project-root $project --out $audit
```

The archived audit passed with 464 matching checksums and 122 additional
entries present. A missing or changed entry means the corresponding
prerequisite needs to be restored before using the replay commands.

## 3. Replay the verified REALM-inspired comparison and Step 9 exports

### REALM-inspired comparison

This runs all five final seeds for the four comparison methods. It reads the
accepted mask bank, reference cache, corpus and actions from the receipt and
writes new results under the checkout's `outputs/pebr_extensions/`.

```powershell
$py = $smokePy
Invoke-PEBR -m pip install -r (Join-Path $release 'pebr_marginal_likelihood_benchmark\requirements.txt')
$receiptPath = Join-Path $project 'outputs\pebr_extensions\marginal_likelihood_retrieval\full_20261106T034651308273Z\run_manifest.json'
$receipt = Get-Content -LiteralPath $receiptPath -Raw -ErrorAction Stop | ConvertFrom-Json
if ($receipt.status -ne 'complete' -or $receipt.fits -ne 20) { throw 'Accepted comparison receipt is incomplete.' }
$inputs = $receipt.inputs
Invoke-PEBR (Join-Path $release 'pebr_marginal_likelihood_benchmark\run_benchmark.py') `
    --project-root $release --mode full `
    --data-root $inputs.prepared_corpus --mask-bank $inputs.mask_bank `
    --query-cache (Split-Path -Parent $inputs.visible_query_cache) `
    --reference-cache (Split-Path -Parent $inputs.reference_fits.'0') `
    --pebr-actions $inputs.default_pebr_actions
```

Compare the generated `summary.csv` with
[`reference_results/realm_inspired/summary.csv`](reference_results/realm_inspired/summary.csv).
Both files had SHA-256
`485cf3e718e160b84c7b89537d85d18a31ae08d091cf3ab9d1f5f8bd8cf90593`
in the verified Windows replay.

### Step 9 tables and figures

The archived interpreter needs NumPy 1.26.4, SciPy 1.13.1 and Matplotlib
3.9.2 for this export. The script reads the accepted numerical outputs
and original saved loss histories and writes into a new directory.

```powershell
$py = $python310
$out = Join-Path $env:USERPROFILE ('Downloads\PEBR_Publication_Replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
Invoke-PEBR -B (Join-Path $release 'tools\pebr_rebuild_publication_assets.py') --project-root $project --release-root $release --out $out
```

Inspect `publication_replay_report.json` in the new output directory.
The verified run generated 106 of 106 assets. Of these, 101 matched byte for
byte. The five differing files were PDFs, and their companion PNGs matched
byte for byte.

## 4. Run the historical manuscript stages

Use an isolated working project with the authorized raw inputs, prepared
collections, historical baseline checkpoints, encoder assets and receipts.
Check the archived file paths before replaying a relocated project. These
commands use the producing interpreter and write reports and new results
into the working project's output tree. The original archive used PyTorch
2.7.0+cu128 with the numerical versions recorded by the stage receipts.
The CUDA-dependent stages require a compatible CUDA environment.

The historical audit and closeout consume completed earlier project runs.
The later stage gates verify specific predecessors and their artifacts. A
checkout with only the source files or raw CSVs cannot supply those inputs.
See [release_provenance.json](release_provenance.json) for the pinned
GloCOM and kNNTM source revision, expected paths and checksums. See
[DATASETS.md](DATASETS.md) for the current Chinese data access status.

First set the working path. Use the same PowerShell session as Section 1
so that `Invoke-PEBR`, `$release` and `$python310` are available.

```powershell
$work = (Resolve-Path -LiteralPath (Read-Host 'Isolated PEBR working project folder') -ErrorAction Stop).Path
if ($work -eq $project -or $work -eq $release) { throw 'Choose a separate working project.' }
$py = $python310
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
```

Run each block in order, checking its output before moving to the next.
The revision sequence uses the fresh-selection stage after the fixed-cap
candidate stage. Development-only Step 4 is outside the manuscript workflow.

The historical source pipelines can be run separately when the working
project has its original configurations and input assets. The Chinese
configuration requires the approved CSVs, the fixed stopword file and
the encoder manifest. These entry points create new run IDs. The
manuscript gates still require the accepted predecessor identities.

```powershell
Invoke-PEBR (Join-Path $work 'main.py') --config configs/experiments/corpus_preparation.yaml --mode run
Invoke-PEBR (Join-Path $work 'main.py') --config configs/experiments/full_study.yaml --mode run
Invoke-PEBR (Join-Path $work 'main.py') --config configs/chinese/full_study.yaml --mode check
Invoke-PEBR (Join-Path $work 'main.py') --config configs/chinese/full_study.yaml --mode run
```

### 4.1 Audit and freeze the historical inputs

```powershell
Invoke-PEBR (Join-Path $release 'pebr_step1_audit.py') --project-root $work
Invoke-PEBR (Join-Path $release 'pebr_step1_closeout.py') --project-root $work
Invoke-PEBR (Join-Path $release 'pebr_step2_prepare.py') --project-root $work
```

### 4.2 Run the default-bank comparison and revised fixed-cap candidate

```powershell
Invoke-PEBR (Join-Path $release 'pebr_step3_run.py') --project-root $work --mode compare
Invoke-PEBR (Join-Path $release 'pebr_export_lab_inputs.py') --project-root $work
Invoke-PEBR (Join-Path $release 'pebr_revised_rule.py') --project-root $work --mode candidate
```

### 4.3 Run fresh selection, protected-word checks and fold sensitivity

```powershell
Invoke-PEBR (Join-Path $release 'pebr_validate_revised_selection.py') --project-root $work --mode preflight
Invoke-PEBR (Join-Path $release 'pebr_validate_revised_selection.py') --project-root $work --mode run
Invoke-PEBR (Join-Path $release 'pebr_step6_protected_words.py') --project-root $work --mode preflight
Invoke-PEBR (Join-Path $release 'pebr_step6_protected_words.py') --project-root $work --mode run
Invoke-PEBR (Join-Path $release 'pebr_step7_audit.py') --project-root $work
Invoke-PEBR (Join-Path $release 'pebr_step8_robustness.py') --project-root $work --mode preflight
Invoke-PEBR (Join-Path $release 'pebr_step8_robustness.py') --project-root $work --mode run
```

### 4.4 Run the Step 9 analyses and export

```powershell
Invoke-PEBR (Join-Path $release 'pebr_step9_complete.py') --project-root $work --phase preflight
Invoke-PEBR (Join-Path $release 'pebr_step9_complete.py') --project-root $work --phase all
```

The accepted Step 9 result is
`outputs/revision_rule_lab/completion9/s9_f9c03fbcc688`. Look in its
`publication_assets/` directory. For phase-level execution the runner accepts
`benchmarks`, `parameters`, `sizes`, `runtime`, `retrieval` and `assets`.
The runtime table describes the original machine. New timings are new
measurements.

## 5. Run the final extensions

The recorded full runs used CPU. Use the producing environment and the
staged working project from Section 4. The English and Chinese neural
baseline requirements are in their runner directories. Their preflights
verify the corpus and baseline sources. The REALM-inspired and ProdLDA
runners also verify the saved inputs they use.

```powershell
$py = $python310
$env:OMP_NUM_THREADS = '1'

Invoke-PEBR (Join-Path $release 'pebr_published_baseline_benchmark\run_benchmark.py') --project-root $work --mode preflight --device cpu
Invoke-PEBR (Join-Path $release 'pebr_published_baseline_benchmark\run_benchmark.py') --project-root $work --mode full --device cpu

Invoke-PEBR (Join-Path $release 'pebr_chinese_published_baselines\run_benchmark.py') --project-root $work --mode preflight --device cpu
Invoke-PEBR (Join-Path $release 'pebr_chinese_published_baselines\run_benchmark.py') --project-root $work --mode full --device cpu

Invoke-PEBR (Join-Path $release 'pebr_marginal_likelihood_benchmark\run_benchmark.py') --project-root $work --mode preflight
Invoke-PEBR (Join-Path $release 'pebr_marginal_likelihood_benchmark\run_benchmark.py') --project-root $work --mode full

Invoke-PEBR (Join-Path $release 'pebr_ntm_step1\run_pebr_ntm.py') --project-root $work --mode preflight --device cpu
Invoke-PEBR (Join-Path $release 'pebr_ntm_step1\run_pebr_ntm.py') --project-root $work --mode full --device cpu
```

The runners print the generated output directories under
`outputs/pebr_extensions/`. The following table identifies the archived
full runs and the corresponding aggregate references.

| Extension | Accepted run | Fits | Reference |
| --- | --- | ---: | --- |
| English EnCOT, FASTopic and NeuroMax | `full_a78f98dc4b66` | 15 | [English summary](reference_results/english_baselines/summary.csv) |
| Chinese EnCOT, FASTopic and NeuroMax | `full_5d8ff968911c` | 15 | [Chinese summary](reference_results/chinese_baselines/summary.csv) |
| REALM-inspired comparison | `full_20261106T034651308273Z` | 20 | [REALM-inspired summary](reference_results/realm_inspired/summary.csv) |
| ProdLDA | `full_7bd73b016645` | 30 | [ProdLDA summary](reference_results/prodlda/summary.csv) |

## Map runs to manuscript results

| Manuscript item | Producing stage |
| --- | --- |
| Main Table I and Supplement S1 | Frozen Step 2 preparation and original corpus configuration |
| Main Table II and Supplement S2–S3 benchmark tables | Step 9 benchmark grid and neural budget check, plus both published-baseline runs at K=12 |
| Main Table III and the S3 Chinese policy table | Fixed-cap PEBR and its controls, using the K=12 mask-3711 replay. The primary one-donor rows come from the REALM-inspired comparison |
| Supplement S3.1 | ProdLDA with original, matched-sharing and PEBR targets across five seeds on both corpora |
| Main Tables IV–V and Supplement S4–S5 | Fresh-selection summaries and Step 8 reference/fold checks |
| Main Table VI, Fig. 1 and Supplement S7.1 | Step 9 parameter and nested-size runs |
| Main Table VII and Supplement S6–S7.3 | Protected-word audit, transport evidence ledgers, original-post topic readout and recurrence |
| Supplement S8 | Step 9 runtime phase and recorded hardware |

The manuscript policy is `preserve_forward_fixed_cap`. The historical
audit label `proposed` identifies an earlier rule. Selected core aggregate
references are in [`reference_results/core/`](reference_results/core).

## Independent reproduction status

A fresh raw-data-to-paper run requires an authorized access route to the
Chinese source files and a verified way to obtain or rebuild the pinned
historical inputs. The saved-input replays above document the checks
completed so far. The data access and redistribution terms are tracked
separately from the code and its dependencies.
