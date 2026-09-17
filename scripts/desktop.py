"""Local PySide6 entry point and isolated, cancellable inspection child."""
import argparse
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "artifacts/operation")
    parser.add_argument("--inspect-job", type=Path)
    parser.add_argument("--settings", action="store_true", help="Open application preferences after startup")
    args = parser.parse_args()
    if args.inspect_job:
        from mes_vision.training.data import read_json, write_json
        from mes_vision.vlm.worker import watch_parent
        watch_parent(os.getppid())
        request = read_json(args.inspect_job)
        try:
            from mes_vision.desktop.service import run_inspection
            run_inspection(request, ROOT)
            return 0
        except Exception as exc:
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output.parent / (output.name + "-error.json"), {"error": f"{type(exc).__name__}: {exc}"})
            return 1
    from mes_vision.operation.preferences import Preferences
    from mes_vision.i18n import set_language, install_qt_translator, install_display_font, tr
    preferences=Preferences(args.runtime); set_language(preferences.value["language"])
    from filelock import FileLock, Timeout
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv[:1]); app.setApplicationName("MES Vision")
    install_qt_translator(app)
    install_display_font(app)
    args.runtime.mkdir(parents=True, exist_ok=True)
    owner = FileLock(str(args.runtime / "desktop.lock"), timeout=0)
    try: owner.acquire()
    except Timeout:
        QMessageBox.information(None, "MES Vision", tr("같은 저장 폴더의 검사 프로그램이 이미 실행 중입니다."))
        return 2
    try:
        from mes_vision.station.window import StationWindow as DesktopWindow
        window = DesktopWindow(ROOT, args.runtime); window.show()
        if args.settings:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0,window.open_preferences)
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, tr("MES Vision 시작 실패"), str(exc)); return 1
    finally: owner.release()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
