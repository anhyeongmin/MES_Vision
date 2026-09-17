"""CPU regression, four export paths, and Qt labeler checks. No product training."""
from pathlib import Path
import os
import subprocess
import sys
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.data_management.fixtures import make_fixture
from mes_vision.data_management.export import export_collection
from mes_vision.training.data import read_json, write_json


def main():
    output = ROOT / "artifacts/data-management-check" / ("run-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {"passed": False, "path": str(output), "tests_run": result.testsRun,
              "errors": len(result.errors), "failures": len(result.failures), "synthetic": True,
              "product_accuracy_tested": False, "gpu_training_performed": False}
    try:
        if not result.wasSuccessful():
            raise ValueError("회귀 시험 실패: tests.log 확인")
        collection = make_fixture(output / "fixture")
        reports = []
        for role in ("object_detector", "known_defect_detector", "normal_bank", "challenge"):
            exported = export_collection(collection, output / role, role, defect_codes=["NG03", "NG06"])
            reports.append({"role": role, "status": exported["status"], "images": len(exported["items"])})
        report["exports"] = reports
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        with (output / "ui.log").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, str(ROOT / "scripts/verify_labeler.py"), "--output", str(output / "ui")],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if process.returncode:
            raise ValueError("Qt 라벨링 시험 실패: ui.log 확인")
        report["ui"] = read_json(output / "ui/report.json")
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    write_json(output / "report.json", report)
    write_json(output.parent / "report.json", report)
    print(f"passed={report['passed']}; {result.testsRun} regression tests; {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
