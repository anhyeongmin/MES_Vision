"""Restore hash-pinned runtime assets from a private GitHub Release. Standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and digest(path) == entry['sha256']


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes destination: ' + name)
    return path


def restore(root=ROOT, *, verify_only=False, asset_dir=None):
    root = Path(root).resolve()
    spec = json.loads((root / 'runtime-assets.json').read_text(encoding='utf-8'))
    if spec['schema_version'] != 1:
        raise ValueError('Unknown asset manifest format')
    cache = root / '.cache/runtime-assets' / spec['tag']
    for entry in spec['files']:
        target = contained(root, entry['path'])
        if valid(target, entry):
            print('OK:', entry['path'], flush=True)
            continue
        if verify_only or entry['source'] == 'git':
            raise RuntimeError('Missing or modified asset: ' + entry['path'])
        if target.exists():
            raise RuntimeError('Existing file differs; preserve/move it before restoring: ' + str(target))
        parts = []
        for part in entry['parts']:
            if Path(part['name']).name != part['name']:
                raise ValueError('Invalid release filename')
            local = contained(Path(asset_dir).resolve() if asset_dir else cache, part['name'])
            if not valid(local, part):
                if asset_dir:
                    raise RuntimeError('Missing or corrupt local release part: ' + str(local))
                cache.mkdir(parents=True, exist_ok=True)
                print('DOWNLOAD:', part['name'], flush=True)
                subprocess.run(['gh', 'release', 'download', spec['tag'], '--repo', spec['repository'],
                                '--pattern', part['name'], '--dir', str(cache), '--clobber'], check=True)
                if not valid(local, part):
                    raise RuntimeError('Downloaded part checksum mismatch: ' + part['name'])
            parts.append(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial')
        with temporary.open('wb') as output:
            for part in parts:
                with part.open('rb') as stream:
                    shutil.copyfileobj(stream, output, length=4*1024*1024)
        if not valid(temporary, entry):
            raise RuntimeError('Assembled asset checksum mismatch: ' + entry['path'])
        temporary.replace(target)
        print('RESTORED:', entry['path'], flush=True)
    print('All runtime assets verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--asset-dir', type=Path, help='Optional offline release parts directory')
    args = parser.parse_args()
    try:
        restore(verify_only=args.verify_only, asset_dir=args.asset_dir)
    except Exception as exc:
        print('RESTORE FAILED:', exc, file=sys.stderr)
        raise SystemExit(1)
