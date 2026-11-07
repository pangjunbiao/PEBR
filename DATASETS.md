# Datasets

All paths are relative to the project root. Keep the original files in their
supplied order and encoding. The raw files and prepared corpus directories
are kept outside Git. The checksums identify the exact inputs expected by
the archived runners.

## Indian transport grievances

The transport-grievance corpus uses the `grievance.csv` file from
[Indian Transport SNAM](https://github.com/sohampoddar26/indian-transport-snam).
The English preparation command in [REPRODUCE.md](REPRODUCE.md) obtains and
checks the file automatically. Its local path is
`data/raw/indian_transport/grievance.csv`.

The loader reads UTF-8 with optional BOM. It uses the `text` column for
preparation and preserves the source-row order. Service-category annotations
are kept separately for evaluation. The frozen manuscript corpus contains
**619 posts and 1,296 vocabulary terms**.

## Pooled Chinese transport

The manuscript corpus pools three CSVs in the order shown below. Place
authorized copies under `data/raw/chinese_transport/`.

| File | Raw records |
| --- | ---: |
| `beijing.csv` | 812 |
| `shanghai.csv` | 588 |
| `xiamen.csv` | 673 |

The loader reads the `正文` text column and accepts UTF-8 with optional BOM
or GB18030. The configuration fixes the stopword list, tokenizer, encoder
and pooling order. After preparation, the frozen manuscript corpus contains
**2,053 posts and 5,002 vocabulary terms**.
See [the Chinese study configuration](configs/chinese/full_study.yaml).

**Access status.** The source citation, data custodian and authorized
reader-access route are pending author confirmation. The code archive
contains the expected file names and fingerprints. Readers need an approved
copy of the files to reproduce the Chinese results. Verify the same access
conditions for any derived inputs proposed for distribution.

## Check the source files

| Local path | Expected SHA-256 |
| --- | --- |
| `data/raw/indian_transport/grievance.csv` | `d0da254163384f6e1276f6fcb1cd3ca8cf4bd515a7e8817d9766b2f2bd7c9eeb` |
| `data/raw/dabiri_traffic/1_TrainingSet_3Class.csv` | `6afba982acaff9bb7bee28c10c0df7af5337819b49164043e14f2a318375a566` |
| `data/raw/chinese_transport/beijing.csv` | `5d7465bbad8da7d9ab31aaa917cfb4b7ea1e54dcaa259253ad1d1a2a8098d5e0` |
| `data/raw/chinese_transport/shanghai.csv` | `331035d8541251adf435c9135e249454ac15a85bb4eed96f29db3d09816d6079` |
| `data/raw/chinese_transport/xiamen.csv` | `2ef92be4bad856c9595b5bd57b13bfa9ebd2d2efaa673b6b17c3e628c9776ccc` |

From a checkout of this repository, use the following PowerShell block to
check the source files in a project containing the data. Paste the project
folder path when prompted.

```powershell
& {
    $dataProject = (Resolve-Path -LiteralPath (Read-Host 'Project folder containing data/raw') -ErrorAction Stop).Path
    $groups = (Get-Content -LiteralPath .\tools\input_manifest.json -Raw -ErrorAction Stop | ConvertFrom-Json).groups
    $files = @($groups.raw_english_inputs.PSObject.Properties) + @($groups.raw_chinese_inputs.PSObject.Properties)
    foreach ($file in $files) {
        $path = Join-Path $dataProject $file.Name
        $digest = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash
        if ($digest -ne $file.Value) { throw "Source checksum differs for $path" }
        Write-Host "Verified $($file.Name)"
    }
}
```

The project also needs the pinned encoders, baseline code and archived
intermediate artifacts described in [REPRODUCE.md](REPRODUCE.md).
