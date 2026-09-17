"""Native Qt production screen checks, with separate temporary verification runtime."""
from pathlib import Path
import argparse
import os
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/operator-check")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from mes_vision.operation.window import DesktopWindow
    from mes_vision.operation.camera import D405Camera
    from mes_vision.robot.magician import Magician
    from mes_vision.training.data import write_json
    app=QApplication([]); app.setFont(QFont("Malgun Gothic",10))
    with tempfile.TemporaryDirectory() as folder:
        window=DesktopWindow(ROOT,folder,start_worker=False); window.resize(1440,900); window.show(); app.processEvents()
        window.tick(); window.timer.stop(); window.analysis.timer.stop()
        for i in range(5):
            window.nav.setCurrentRow(i); app.processEvents(); window.grab().save(str(args.output/f"page-{i+1}.png"))
        report={"screens":5,"production_no_device_start_blocked":not window.start.isEnabled(),"vlm_default_off":not window.queue.enabled(),
            "camera_devices":D405Camera.devices(),"serial_ports":Magician.ports(),"physical_commissioning_tested":False}
        write_json(args.output/"report.json",report)
        window.closing=True; window.close(); app.processEvents()
    print(report)

if __name__=="__main__": main()
