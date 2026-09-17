"""Standalone manual camera inspection. Shares desktop lock; never connects Dobot."""
import multiprocessing
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from filelock import FileLock, Timeout
    from PySide6.QtWidgets import QApplication, QMessageBox
    from mes_vision.i18n import set_language, install_display_font, install_qt_translator
    from mes_vision.operation.catalog import OperationStore
    from mes_vision.station.manual_dialog import ManualInspectionDialog
    runtime=ROOT/'artifacts/operation'; runtime.mkdir(parents=True,exist_ok=True)
    app=QApplication(sys.argv[:1]); set_language('ko'); install_display_font(app); install_qt_translator(app)
    lock=FileLock(str(runtime/'desktop.lock'),timeout=0)
    try: lock.acquire()
    except Timeout:
        QMessageBox.information(None,'수동 검사','기존 프로그램을 닫거나, 기존 프로그램의 품목 설정 → 수동 도착 검사를 사용하세요.')
        return 2
    try:
        settings=OperationStore(runtime).equipment()['camera']
        dialog=ManualInspectionDialog(ROOT,runtime,settings)
        dialog.exec()
        return 0
    except Exception as exc:
        QMessageBox.critical(None,'수동 검사 시작 실패',str(exc)); return 1
    finally: lock.release()


if __name__=='__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
