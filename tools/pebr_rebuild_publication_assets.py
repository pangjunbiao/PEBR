"""Rebuild the accepted Step 9 publication outputs from saved numerical inputs."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Checksum differs for {path}")
    return path


def archived_stage(project, path):
    parts = PureWindowsPath(path).parts
    return project.joinpath(*parts[parts.index("outputs"):])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    project, release, out = (value.resolve() for value in (args.project_root, args.release_root, args.out))
    if out.exists() or out.is_relative_to(project) or out.is_relative_to(release):
        parser.error("Choose a new output folder outside both input folders.")

    old = project / "outputs/revision_rule_lab/completion9/s9_f9c03fbcc688"
    pins = {
        "step9_report.json": "83be004852c2b6bba5d7b0b56916c2ad8290d59a766c8e04bfbd538b24658bc6",
        "protocol.json": "90ccb1255f4010fdf283851b95dcbb629c727b2a7fa4b3989facde074449c5eb",
        "asset_manifest.json": "3be860b85ebe8d2e910c14b16bc9132dbd022f3728f11966dd84ee0824577d83",
    }
    report, protocol, expected = [read(verified(old / name, sha)) for name, sha in pins.items()]
    manifest = verified(release / "PEBR_STEP9_MANIFEST.json",
                        "83661448151177ccfd9633fecf90698519f8f841ef26ce89d1eb884a66ba7eb8")
    for name, sha in read(manifest).items():
        verified(release / name, sha)

    for name, pin in {"numpy": "1.26.4", "scipy": "1.13.1", "matplotlib": "3.9.2"}.items():
        if version(name) != pin:
            raise ValueError(f"Use the archived environment with {name}=={pin}.")
    stages = {name: archived_stage(project, report[f"{name}_workspace"])
              for name in ("step5", "step6", "step7", "step8")}
    for record in read(old / "benchmark_results/seed_records.json"):
        if record["policy"] in ("knntm", "glocom") and record["epoch_budget"] == 400:
            path = Path(record["path"]).resolve()
            if not path.is_relative_to(project) or not (path / "training_losses.csv").is_file():
                raise FileNotFoundError(f"Archived training loss history is unavailable at {path}")

    sys.path[:0] = [str(release), str(release / "src")]
    from pebr_complete9.assets import run

    out.mkdir(parents=True)
    metadata = expected["FIGURE_CAPTIONS.md"]
    for group in metadata["phase_sources"].values():
        for name, sha in group.items():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(verified(old / name, sha), target)
    # Match the order of the frozen Step 2 collection descriptors.
    shapes = {"chinese_pooled": (2053, 5002), "transport_grievance": (619, 1296)}
    if set(shapes) != set(protocol["corpora"]):
        raise ValueError("Archived corpus coverage differs.")
    corpora = {name: {"collection": SimpleNamespace(counts=SimpleNamespace(shape=shape))}
               for name, shape in shapes.items()}
    run(SimpleNamespace(work=out, protocol=protocol, corpora=corpora,
                        lineage=metadata["existing_stage_identities"], **stages))

    generated = out / "publication_assets"
    paths = {p.relative_to(generated).as_posix() for p in generated.rglob("*") if p.is_file()}
    different = {name for name in paths & expected.keys()
                 if digest(generated / name) != expected[name]["sha256"]}
    figures = {name for name in different if name.startswith("figures/") and
               Path(name).suffix in (".png", ".pdf")}
    comparison = {
        "scope": "Step 9 publication exports from accepted saved inputs",
        "expected_assets": len(expected),
        "generated_assets": len(paths),
        "byte_identical": len(paths & expected.keys()) - len(different),
        "figures_with_different_bytes": sorted(figures),
        "other_differences": sorted(different - figures),
        "missing": sorted(expected.keys() - paths),
        "unexpected": sorted(paths - expected.keys()),
    }
    receipt = out / "publication_replay_report.json"
    receipt.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(paths)} assets. {comparison['byte_identical']} match byte for byte.")
    print(f"Report: {receipt}")
    return int(any(comparison[key] for key in ("other_differences", "missing", "unexpected")))


if __name__ == "__main__":
    raise SystemExit(main())
