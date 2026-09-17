"""Capture actual Qt history/storage screens using isolated injected outputs only."""
from pathlib import Path
import argparse
import os
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/storage-check"); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont,QFontDatabase
    from mes_vision.operation.window import DesktopWindow
    from mes_vision.operation.storage_dialog import StorageDialog
    from mes_vision.training.data import write_json
    import test_operation
    app=QApplication([])
    font=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/malgun.ttf"
    if font.exists():
        families=QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families: app.setFont(QFont(families[0],10))
    f=test_operation.OperationTests(); f.setUp(); f.models.defect=True
    f.models.detections=(test_operation.detection(),test_operation.detection(x=210))
    f.engine.commit(f.inspect()); f.engine.close(); f.store.end_session(f.session)
    w=DesktopWindow(ROOT,f.root,start_worker=False); w.timer.stop(); w.analysis.timer.stop(); w.resize(1440,900); w.show(); w.nav.setCurrentRow(3)
    w.show_record(w.history_rows[0]["object_id"]); app.processEvents(); w.grab().save(str(args.output/"history.png"))
    d=StorageDialog(w.store,ROOT,w); d.show(); d.show_usage(d.service.usage()); d.show_preview(d.service.preview_archive()); app.processEvents(); d.grab().save(str(args.output/"storage.png"))
    with tempfile.TemporaryDirectory() as folder:
        backup=d.service.backup(Path(folder)/"backup"); restored=d.service.restore(backup["path"],Path(folder)/"restored")
        d.restore_done(restored); app.processEvents(); d.grab().save(str(args.output/"restore.png"))
        report={"ui_screens":3,"isolated_injected_outputs":True,"history_rows":len(w.history_rows),"backup_files":backup["files"],"restored_captures":restored["captures"],"physical_hardware_tested":False}
        write_json(args.output/"report.json",report)
    d.close(); d.deleteLater(); w.closing=True; w.close(); app.processEvents(); f.tearDown(); print(report)


if __name__=="__main__": main()
