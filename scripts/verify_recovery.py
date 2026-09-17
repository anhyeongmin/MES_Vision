"""Render actual recovery UI in a separate temporary runtime, without devices."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/recovery-check"); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from PySide6.QtWidgets import QApplication,QInputDialog,QFileDialog
    from PySide6.QtGui import QFont,QFontDatabase
    from mes_vision.operation.window import DesktopWindow
    from mes_vision.operation.faults import FaultJournal
    from mes_vision.training.data import write_json
    app=QApplication([]); font=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/malgun.ttf"
    if font.exists():
        families=QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families: app.setFont(QFont(families[0],10))
    with tempfile.TemporaryDirectory() as directory:
        w=DesktopWindow(ROOT,directory,start_worker=False); w.timer.stop(); w.analysis.timer.stop(); w.resize(1440,900); w.show(); app.processEvents()
        w.tick(); app.processEvents(); w.grab().save(str(args.output/"ready.png"))
        w.trip_fault("CAMERA_FAILURE","D405 영상 응답이 멈췄습니다. 연결 해제 후 케이블과 설치 상태를 확인하세요.")
        w.trip_fault("STORAGE_FAILURE","검사 원본 저장을 완료하지 못했습니다. 저장 공간과 권한을 확인하세요.")
        w.tick(); app.processEvents(); w.grab().save(str(args.output/"fault.png"))
        persisted=len(FaultJournal(directory).active)
        diagnostic=args.output/"diagnostics.json"
        # Keep every prior diagnostic run rather than replacing an operator export.
        from uuid import uuid4
        if diagnostic.exists(): diagnostic=args.output/("diagnostics-"+uuid4().hex+".json")
        with patch.object(QFileDialog,"getSaveFileName",return_value=(str(diagnostic),"")): w.export_diagnostics()
        with patch.object(QInputDialog,"getText",return_value=("software verification",True)),patch.object(QInputDialog,"getMultiLineText",return_value=("injected faults only; no device motion",True)):
            w.acknowledge_faults()
        w.tick(); app.processEvents(); w.grab().save(str(args.output/"recovered.png"))
        report={"screens":3,"faults_persisted":persisted,"acknowledged":not w.faults.active,"no_automatic_inspection_restart":not w.running,
            "diagnostic_file":diagnostic.name,"isolated_injected_faults":True,"physical_hardware_tested":False}
        write_json(args.output/"report.json",report); w.closing=True; w.close(); app.processEvents()
    print(report)


if __name__=="__main__": main()
