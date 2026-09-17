"""Run hardware-independent input integration tests and retain a JSON report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    destination = ROOT / "artifacts/input-check"
    destination.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_inputs.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "passed_tests": result.passed,
        "failures": [{"test": str(t), "detail": error} for t, error in result.failures + result.errors],
        "hardware_used": False, "product_accuracy_tested": False,
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
