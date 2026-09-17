"""Exercise the same start-button and child process used by the desktop UI."""
from pathlib import Path
import os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from mes_vision.station.photo_dialog import PhotoInspectionDialog
from mes_vision.training.data import write_json,require


def main():
    app=QApplication.instance() or QApplication([])
    directory=ROOT/'artifacts/model-ui-integration'
    window=PhotoInspectionDialog(ROOT,directory/'ui-worker-runtime'); window.show()
    failures=[]; window.guard=lambda fn: fn()  # Surface exceptions instead of opening a blocking error dialog.
    window.choose_samples(); began=time.monotonic(); window.start(); ticks=0; max_gap=0; last=began
    deadline=began+180
    while window.process is not None and time.monotonic()<deadline:
        app.processEvents(); ticks+=1; now=time.monotonic(); max_gap=max(max_gap,now-last); last=now; time.sleep(.01)
    if window.process is not None:
        window.stop()
        while window.process is not None: app.processEvents(); time.sleep(.01)
        raise TimeoutError('UI worker did not finish')
    require(window.report is not None and len(window.report['records'])==12,'UI worker did not publish its results')
    require(window.kind.currentData()=='synthetic','Synthetic input origin was lost in the UI')
    for index,row in enumerate(window.report['records']):
        window.files.selectRow(index); app.processEvents()
        if row['role']=='detail': require('보류' in window.reasons.toPlainText(),'Unvalidated criteria were approved')
    write_json(directory/'ui-worker-verification.json',{'status':'PASSED','photos':12,
        'output':str(window.output),'qt_events_processed':ticks,'observed_max_poll_gap_seconds':max_gap,
        'elapsed_seconds':time.monotonic()-began,'vlm_enqueued':False,'robot_connected':False})
    window.close(); app.processEvents(); print('UI start -> child inference -> verified results -> selection passed',flush=True)


if __name__=='__main__': main()
