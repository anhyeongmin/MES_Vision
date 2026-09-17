"""Record installed versions and preserve distributed license notices (not a legal audit)."""
from pathlib import Path
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'requirements-lock.txt').write_text(freeze, encoding='utf-8')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    (ROOT / 'artifacts' / 'pip-check.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    check.check_returncode()
    packages = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name = dist.metadata['Name']
        notices = []
        for entry in dist.files or []:
            text = str(entry).replace('\\', '/')
            if '.dist-info/' not in text:
                continue
            tail = text.split('.dist-info/', 1)[1]
            if not ('license' in tail.lower() or Path(tail).name.upper().startswith(('NOTICE', 'COPYING', 'AUTHORS'))):
                continue
            source = Path(dist.locate_file(entry))
            if not source.is_file():
                continue
            target = ROOT / 'licenses' / 'dependencies' / f'{name}-{dist.version}' / tail
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            notices.append(target.relative_to(ROOT).as_posix())
        packages.append({
            'name': name, 'version': dist.version,
            'license_expression': dist.metadata.get('License-Expression'),
            'license_metadata': dist.metadata.get('License'),
            'project_urls': dist.metadata.get_all('Project-URL') or [],
            'saved_notice_files': notices,
        })
    report = {'scope': 'Installed package metadata and bundled dist-info notices only. Not a complete third-party or binary redistribution audit.', 'packages': packages}
    (ROOT / 'licenses' / 'dependency-inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Recorded {len(packages)} packages; pip check passed.')

if __name__ == '__main__':
    main()
