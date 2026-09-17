"""Offline registration report, optionally with isolated station software tests."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODULES = ['test_station', 'test_station_devices', 'test_station_ui', 'test_station_sort',
           'test_station_history', 'test_station_audit']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sources():
    files = [p for folder in ('src', 'tests', 'scripts') for p in (ROOT / folder).rglob('*.py')
             if '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def test_worker(output):
    sys.path.insert(0, str(ROOT / 'tests'))
    class Result(unittest.TextTestResult):
        def startTest(self, test):
            self.identities.append(test.id())
            super().startTest(test)
        identities = []
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not result.expectedFailures
    write(output, dict(status='passed' if passed else 'incomplete', tests_run=result.testsRun,
        skipped=len(result.skipped), seconds=round(time.monotonic() - started, 3), test_ids=result.identities,
        failures=[(t.id(), v) for t, v in result.failures], errors=[(t.id(), v) for t, v in result.errors],
        skips=[(t.id(), reason) for t, reason in result.skipped],
        expected_failures=[(t.id(), reason) for t, reason in result.expectedFailures],
        unexpected_successes=[t.id() for t in result.unexpectedSuccesses]))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=ROOT / 'artifacts/operation')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/station-check')
    parser.add_argument('--run-tests', action='store_true', help='Run injected station tests; no physical devices')
    parser.add_argument('--test-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.test_worker:
        return test_worker(args.test_worker)
    # Every run has its own directory; never overwrites earlier results.
    from uuid import uuid4
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    before = sources()
    from mes_vision.station.audit import audit, markdown
    report = audit(args.runtime)
    report['software_tests'] = {'status': 'not_run'}
    report['python'] = sys.version
    write(output / 'report.json', report)
    if args.run_tests:
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests'))))
        result_file = output / 'software-tests.json'
        with (output / 'software-tests.log').open('w', encoding='utf-8') as log:
            try:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--test-worker', str(result_file)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else {'status': 'failed'}
                if completed.returncode:
                    result['status'] = 'failed'
                result['exit_code'] = completed.returncode
            except subprocess.TimeoutExpired:
                result = {'status': 'failed', 'error': 'Software tests exceeded 600 seconds'}
        write(result_file, result)
        report['software_tests'] = result
    after = sources()
    write(output / 'source-hashes.json', before)
    report['source_unchanged_during_check'] = before == after
    report['software_tests'] = {k: v for k, v in report['software_tests'].items()
                                if k not in ('test_ids', 'failures', 'errors', 'skips', 'expected_failures', 'unexpected_successes')}
    write(output / 'report.json', report)
    (output / 'report.md').write_text(markdown(report), encoding='utf-8')
    print(str(output / 'report.md'), flush=True)
    if not report['source_unchanged_during_check'] or report['software_tests']['status'] not in {'passed', 'not_run'}:
        return 1
    return 0 if report['registration_status'] == 'checks_passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
