# Step 8 runner validation

These checks validate the implementation; they are not favorable real-data
sensitivity results. Actual reference/fold experiments must run on the user's
complete project.

- Eighteen unit checks cover frozen settings, separate sensitivity axes,
  family integrity, masked/full donor exclusions, exact reconstruction and
  cache-tampering detection, complete/missing/duplicate reporting grids,
  path containment and explicit no-borrowing versus unavailable-donor fallback.
- On genuine exported primary/Chinese numerical inputs, the four-policy
  selector matched the frozen Step 5 implementation for 16 receiver cases,
  corresponding to 64 decisions. Selected donors, available donors,
  offered/realized mass and utility agreed. This checks numerical equivalence,
  not new performance; these checks used the existing K12/mask2711 inputs.
- The complete synthetic pipeline passed: exact two-corpus default selector
  replay, 120 fold-excluded references, 420 new final NMF fits, 750 case-fit
  records, 150 summaries, 360 paired metric contrasts and inspection ZIP export.
  The pre-existing accepted fixture inputs were hash-checked after the run.
- A restart reused all 420 completed final fits, fitted zero new final models,
  and retained all 750 case-fit records.
- Every runner module parses for Python 3.10. Typical new model-output paths
  are approximately 175 characters at the supplied Windows project root.
  Existing cache paths are read in place; the runner makes no encoder calls.

The genuine complete Step 7 results have been inspected separately; see
STEP7_REVIEW.md and STEP7_INDEPENDENT_CHECKS.json. The synthetic Step 8
rankings are not distributed as manuscript evidence.
