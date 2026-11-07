# Validation performed before delivery

The real Step 8 archive was inspected separately from the Step 9 code tests.
Its independent numerical checks are in STEP8_INDEPENDENT_CHECKS.json and its
interpretation is in STEP8_REVIEW.md.

Step 9 code validation:

- All Python modules parse with Python 3.10 syntax; no unintended control
  characters appear in source or LaTeX strings.
- Thirteen focused unit checks pass, covering fixed defaults, budget and job
  accounting, one-at-a-time sweeps, nested sample preparation, sample-level DF2
  vocabulary/zero-row handling, identity reuse and LaTeX formatting.
- The complete runner executed on an isolated synthetic fixture: 240 benchmark
  records, 60 parameter records, 1,000 size-analysis fits, fresh pipeline replay,
  and category retrieval from 150 current fitted representations. All five
  phase coverage checks passed and publication assets were exported.
- The native kNNTM/GloCOM trainers actually executed in the CPU fixture. Two
  additional K10/seed0 checks verified that 200 epochs plus resumed training to
  400 reproduces uninterrupted 400-epoch topic scores and original-post
  retrieval features exactly (maximum discrepancy zero).
- A second invocation reused every completed result phase without fitting
  again. Hashes of all accepted synthetic predecessor inputs stayed unchanged.
- On exported REAL counts, all 40 planned family-aware sample manifests were
  checked for DF2 vocabulary, retained nesting and nonempty folds. Ten
  diagnostic predictive references for the first 100-post sample in each
  corpus converged at the unchanged tolerance1e-4 / maximum2,000 iterations
  while excluding the receiving fold.
- Latest table/figure export was tested separately. Main fragments compile in
  a two-column LaTeX layout and supplementary fragments compile in an A4
  layout. Scientific plots and representative rendered table pages were
  visually checked. This checks fragment layout, not a compiled final IEEE
  manuscript with its existing text and bibliography.

The synthetic fixture uses 30 artificial posts per collection, deterministic
artificial encoder/word vectors and CPU neural training. Its size targets are
20/24/27/30; an unusually ill-conditioned 20-post/K12 test fit uses a test-only
iteration ceiling10,000. These fixture choices are not in the delivered study
protocol: real targets are100/200/400/600 and the original 2,000-iteration NMF
ceiling is unchanged. Historical fixture checkpoints are artificial test
inputs; their numerical scores and losses are not scientific results.

The real CUDA encoders, actual K10 baselines, complete real parameter/size
grids and actual hardware timings must run on the user's existing project.
No favorable Step 9 performance is asserted before that run and inspection.

If a scientific identity, convergence check or artifact verification fails,
the runner records a block and exports its inspection ZIP. It does not change
the rule, reseed until a better score appears, or mark missing phases complete.
