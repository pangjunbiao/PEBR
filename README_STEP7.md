# PEBR Step 7 — verify baselines and export transportation evidence

Extract beside your existing `main.py`. Keep `pebr_step7_audit.py`, the
`pebr_review7` directory and `PEBR_STEP7_MANIFEST.json` together. These names
do not replace the Step 5 or Step 6 runners.

Run in PowerShell from the project root:

```powershell
& "C:\Users\Sabir\AppData\Local\Programs\Python\Python310\python.exe" .\pebr_step7_audit.py --project-root .
```

The program resolves the inspected, completed Step 6 v1.0.1 workspace,
then its frozen Step 5/3/2 lineage. It reads existing saved models,
checkpoints and baseline input caches. There is no neural/NMF/BTM training, encoder call,
download, package installation, parameter search or replacement of old
experiments. Numerical dependencies must match the audited project.
PyTorch and POT are used from your existing environment.
GloCOM's clustering input is reconstructed with its original fixed KMeans
settings; this does not change its trained topic model.

Outputs are under `outputs/revision_rule_lab/review7/s7_<short identity>/`.
The exact path is printed. Send `PEBR_step7_inspection.zip` for review,
including when the run reports a problem. Findings distinguish a demonstrated
failure, a missing check and an informative warning. No automatic method or
baseline repair is attempted.

## What is checked

- Existing BTM, kNNTM and GloCOM fits at K=8 and 12: both corpora, five
  seeds each, 60 models. Historical K=16 is inventoried without relabelling;
  missing all-method K=10 runs are explicitly flagged.
- Model receipts, count/vocabulary/document alignment, exact pinned source,
  word-vector alignment and coverage, finite training history, checkpoint
  identity and epoch coverage.
- Independent evaluation-only neural decoder and membership replay from CPU
  checkpoint tensors. kNNTM's signed topic scores and GloCOM's native
  topic-axis softmax are preserved.
- kNNTM's cached OT input matrices against checkpoint tensors and four fixed
  document-pair calculations. GloCOM's original-count global context is
  reconstructed using the pinned clustering settings.
- Original-reference NPMI, C_v, TD and five-seed matched-topic stability.

A replay can establish that the saved implementation is consistent. It does
not by itself establish convergence, suitable tuning or broad neural-model
inferiority. Loss histories and settings still require scientific review.

## Transportation evidence

1. **Protected-word cases:** K=12, first fresh outer bank 3712. Retain the four
   existing anchors (source rows 271, 357, 264, 14). Show the actual saved
   donors, transferred mass and probabilities for words genuinely present in
   the receiving post. Also export the first source-order predictive contrast
   meeting the disclosed criterion, if any, and the first negative-gain case.
   Outcome-selected illustrations are labelled explicitly. All selected cases
   retain the competing controls.
2. **Final-topic interpretation:** K=12, first fresh selection bank 3711,
   final seed 0 for display; seeds 0–4 for recurrence. The same four anchors
   select their dominant topics using original-post contributions. NMF,
   similarity sharing, the strong same-score prefix control and PEBR remain
   visible. No descriptor words are edited. Native LaTeX and two PDF/PNG
   recurrence figures are exported if Matplotlib is available.

These panels have different action banks. Do not imply the protected audit's
donors directly generated the final-topic table. Redactions and excerpt
omissions are marked. Review excerpts before publication. A donor never
establishes an unreported cause, and word prediction does not establish
passenger-service improvements.

## Is this the final implementation step?

That depends on the findings. Reference-initialization/family-fold sensitivity
has not been replaced by repeated masking. Old corpus-size, cap/lambda and
case assets must be regenerated for the exact revised rule if retained.
Full K=10 neural/BTM benchmarks need genuine K=10 fits if claimed. The next
action is chosen from these checks, not from a desired journal rating.
