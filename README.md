# PEBR

Predictive Evidence Borrowing for Topic Modeling of Short Transport Posts.
The paper uses the `preserve_forward_fixed_cap` policy.

## Try the code with synthetic data

The Windows commands below use Git and Python 3.10. The tests exercise corpus
masking, evidence selection, and bounded transfer using synthetic inputs.

```powershell
git clone --branch quick-replay-v1 --single-branch https://github.com/pangjunbiao/PEBR.git
Set-Location PEBR
py -3.10 -m venv .venv
$python = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2
& $python -B verify_release.py
$env:PYTHONPATH = 'src'
& $python -B -m unittest tests.test_fresh tests.integration.test_borrowing tests.unit.test_transfer
```

The source check prints `RELEASE SOURCE PASS`, and the tests finish with `OK`.
These tests check the implementation using toy data. For manuscript results,
use the archived inputs and instructions below.

## Reproduce the manuscript outputs

- [Data and access](data/README.md) explains the files needed for replay and training.
- [Quick Replay release](https://github.com/pangjunbiao/PEBR/releases/tag/quick-replay-v1)
  provides scripts and Windows instructions for commit `072aeff`.
  Exact replay requires the approved companion input archive.
- [Reference results](reference_results/) contains selected aggregate outputs.
  [Release provenance](release_provenance.json) records the accepted run IDs and
  source fingerprints.
- [Historical experiments](REPRODUCE.md) maps the original runs to the paper
  and documents their dependencies. A full rebuild requires additional
  historical inputs and an appropriate CUDA environment.

On a clean Windows checkout with the archived inputs, Quick Replay
regenerated all 106 publication assets. Of these, 101 matched byte for byte
and five figure PDFs had different file hashes. The REALM-inspired comparison
reran 20 fits and reproduced its archived summary byte for byte. The complete
pipeline from raw data to all manuscript results still needs an independent
end-to-end run.
