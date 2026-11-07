# Validation of the Step 7 runner

This is runner validation, not a new real-data performance result.

- Sixteen unit checks passed, including independent NumPy decoder and
  original-post membership equations compared with the pinned native
  PyTorch kNNTM and GloCOM classes. Signed kNNTM logits, GloCOM's topic-axis
  normalization, NMF scaling invariance, zero memberships, excerpt fidelity,
  redaction, source-order case selection and relocated paths were checked.
- A complete synthetic integration run passed for 60 existing-model cases:
  both collection roles, K=8/12, three baseline methods, five seeds each.
  Native checkpoint tensors, fixed-pair exact OT checks, original-reference
  coherence/diversity and five-seed stability were exercised.
- Transportation integration replayed protected word probabilities and
  donor transfer, exported all 80 anchor/method/final-seed topic selections,
  generated the LaTeX table and both PDF/PNG recurrence figures. The fixture
  figures were visually inspected for legibility.
- The complete engine also passed its accepted-source gates, final frozen-input
  recheck and inspection ZIP generation on that synthetic project.
- Every runner module parses for Python 3.10. The new output directory uses a
  short identity to avoid the previous Windows path-length problem.

The real frozen checkpoints and original source records remain on the user's
computer. Their full Step 7 replay must run there. Passing this package's tests
is not evidence of favorable real baseline results, adequate tuning or a
transportation-service benefit. The runner does not alter a model to obtain
such an outcome.
