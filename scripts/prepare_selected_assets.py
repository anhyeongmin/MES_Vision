"""Download pinned model/wheel assets and verify them without installing or executing.

Usage: python scripts/prepare_selected_assets.py [--verify-only]
Model revisions and expected content hashes are in models/selection-lock.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def target_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Asset path outside project')
    return path

def validate(path, asset):
    if path.stat().st_size != asset['size']:
        raise ValueError('File size mismatch: ' + asset['id'])
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {asset['size']}\0".encode('ascii'))
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(chunk)
            git.update(chunk)
    if asset.get('sha256'):
        if sha.hexdigest() != asset['sha256']:
            raise ValueError('SHA256 mismatch: ' + asset['id'])
    elif asset.get('git_blob_sha1'):
        if git.hexdigest() != asset['git_blob_sha1']:
            raise ValueError('Git blob mismatch: ' + asset['id'])
    else:
        raise ValueError('Expected digest absent: ' + asset['id'])
    return sha.hexdigest()

def download(asset, verify_only):
    path = target_path(asset['path'])
    if path.exists():
        digest = validate(path, asset)
    elif verify_only:
        raise FileNotFoundError(asset['path'])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.partial')
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > asset['size']:
                    raise ValueError('Partial file exceeds expected size')
                if offset != asset['size']:
                    headers = {'User-Agent': 'MES-Vision-pinned-assets/1.0'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    request = urllib.request.Request(asset['url'], headers=headers)
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if offset and response.status == 206:
                            if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                                raise ValueError('Invalid range response')
                            mode = 'ab'
                        elif response.status == 200:
                            mode, offset = 'wb', 0
                        else:
                            raise ValueError('Unexpected HTTP download status')
                        last_print = time.monotonic()
                        with partial.open(mode) as output:
                            while chunk := response.read(4 * 1024 * 1024):
                                offset += len(chunk)
                                if offset > asset['size']:
                                    raise ValueError('Response exceeds expected size')
                                output.write(chunk)
                                if time.monotonic() - last_print > 15:
                                    print(f"Downloading {asset['id']}: {offset / 1e9:.2f}/{asset['size'] / 1e9:.2f} GB", flush=True)
                                    last_print = time.monotonic()
                digest = validate(partial, asset)
                partial.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}: {asset['id']}", flush=True)
    print('Verified ' + asset['id'], flush=True)
    return dict(id=asset['id'], path=asset['path'], size=asset['size'], sha256=digest, status='verified')

def wheel_notices(assets):
    records = []
    for asset in assets:
        if asset['category'] != 'wheel':
            continue
        # Called only after the complete pinned asset set passes verification.
        with zipfile.ZipFile(target_path(asset['path'])) as wheel:
            for member in wheel.namelist():
                lower = member.lower()
                if member.endswith('/') or not ('.dist-info/licenses/' in lower or lower.endswith('.dist-info/metadata')):
                    continue
                relative = 'licenses/selection/wheel-notices/' + member
                dest = target_path(relative)
                data = wheel.read(member)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                records.append(dict(wheel=asset['path'], member=member, path=relative, sha256=hashlib.sha256(data).hexdigest()))
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    lock_path = ROOT / 'models' / 'selection-lock.json'
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    assets = sorted(lock['assets'], key=lambda asset: asset['size'])
    paths = [target_path(a['path']) for a in assets]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate asset target')
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, a, args.verify_only): a for a in assets}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(dict(id=futures[future]['id'], error=str(exc)))
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), selection_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(), status='verified' if not errors else 'incomplete', execution_tested=False, assets=sorted(results, key=lambda item: item['id']), errors=errors)
    if not errors:
        report['wheel_notice_files'] = wheel_notices(assets)
    report_path = ROOT / 'artifacts' / 'selected-assets-check.json'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], verified_count=len(results), errors=errors), ensure_ascii=False), flush=True)
    return 1 if errors else 0

if __name__ == '__main__':
    raise SystemExit(main())
