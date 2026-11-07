"""Verify the release candidate's checked-in file contents."""

from hashlib import sha256
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "RELEASE_SHA256.json").read_text(encoding="utf-8"))
    issues = []
    for name, expected in manifest.items():
        file = root / name
        if not file.is_file():
            issues.append(f"Missing: {name}")
        elif sha256(file.read_bytes()).hexdigest() != expected:
            issues.append(f"Changed: {name}")
    if issues:
        print("\n".join(issues))
        return 1
    print(f"RELEASE SOURCE PASS: {len(manifest)} files match their SHA-256 hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
