"""Check archived PEBR inputs and reference outputs without running experiments."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def check_group(root, files):
    result = {'verified': 0, 'present_without_hash': 0, 'issues': []}
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            status = 'outside_project'
        elif not path.is_file():
            status = 'missing'
        elif expected is None:
            result['present_without_hash'] += 1
            continue
        else:
            try:
                actual = digest(path)
            except OSError:
                status = 'unreadable'
            else:
                if actual == expected:
                    result['verified'] += 1
                    continue
                status = 'checksum_mismatch'
        result['issues'].append({'path': name, 'status': status})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.out.resolve()
    if not (root / 'main.py').is_file() or not (root / 'outputs').is_dir():
        parser.error('Choose the original PyCharm project containing main.py and outputs.')
    if out.is_relative_to(root) or out.exists():
        parser.error('Choose a new report path outside the project.')
    manifest = Path(__file__).with_name('input_manifest.json')
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    groups = {}
    for name, files in spec['groups'].items():
        check = check_group(root, files)
        groups[name] = check
        print(f'{name} | {check["verified"]} hash-verified | '
              f'{check["present_without_hash"]} present | {len(check["issues"])} issues', flush=True)
    report = {'schema': 1, 'scope': spec['scope'], 'groups': groups}
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(f'Report: {out}', flush=True)
    return int(any(group['issues'] for group in groups.values()))


if __name__ == '__main__':
    raise SystemExit(main())
