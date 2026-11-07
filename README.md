# PEBR

Predictive Evidence Borrowing for Topic Modeling of Short Transport Posts.
The manuscript experiments use the `preserve_forward_fixed_cap` policy.

- [DATASETS.md](DATASETS.md) describes the input files, their expected
  locations and checksums, and how to obtain the available sources.
- [REPRODUCE.md](REPRODUCE.md) provides ordered PowerShell commands, the
  prerequisites for each stage, and a map from runs to paper results.
- [Reference results](reference_results) contains selected aggregate outputs
  for comparison with new runs.
- [Release provenance](release_provenance.json) records the accepted run IDs
  and pinned source fingerprints.

The verified Windows checks covered fresh English input preparation, the
full REALM-inspired comparison from saved inputs, and regeneration of the
Step 9 publication outputs. Independent end-to-end reproduction still needs
an authorized access route to the Chinese corpus and a way to obtain or
rebuild the pinned historical inputs.
