# Predictive Evidence Borrowing for Topic Modeling of Short Transport Posts

This candidate contains the archived experiment code and selected aggregate
reference results for the manuscript. The primary manuscript policy is
`preserve_forward_fixed_cap`. See [REPRODUCE.md](REPRODUCE.md) for the run
mapping, inputs, commands and current validation status.

The four extension runners are pinned to the hashes in the accepted run
manifests. The Chinese baseline runner comes from its original benchmark ZIP.
The ProdLDA runner comes from the archived source that produced the saved run.
`release_provenance.json` records run IDs and source digests.

The Windows checks covered fresh English input preparation, the full
REALM-inspired comparison using archived inputs, and regeneration of all 106
Step 9 publication assets. See [REPRODUCE.md](REPRODUCE.md) for the scope and
results of each check.

This package is ready for local Git staging. An independent end-to-end run
still needs an approved route to the Chinese corpus and a way to obtain or
rebuild the pinned predecessor artifacts. The historical runner documentation
includes examples from the original Windows machine.
