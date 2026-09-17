import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import subprocess
import sys
import time
import unittest
from pathlib import Path
from PySide6.QtWidgets import QApplication,QWidget,QDialog
from mes_vision.operation.single_instance import DesktopActivation,focus_existing,server_name

APP=QApplication.instance() or QApplication([])

class ActivationTests(unittest.TestCase):
    def test_relaunch_restores_existing_window_without_creating_another(self):
        with tempfile.TemporaryDirectory() as root:
            window=QWidget(); activation=DesktopActivation(root,window)
            child=subprocess.Popen([sys.executable,'-B','-c',
                'from PySide6.QtCore import QCoreApplication; from mes_vision.operation.single_instance import focus_existing; '
                'import sys; app=QCoreApplication([]); sys.exit(0 if focus_existing(sys.argv[1]) else 2)',root])
            end=time.monotonic()+5
            while time.monotonic()<end and (child.poll() is None or not window.isVisible()):
                APP.processEvents(); time.sleep(.005)
            self.assertEqual(child.wait(2),0); self.assertTrue(window.isVisible())
            activation.server.close(); window.close(); window.deleteLater(); APP.processEvents()

    def test_other_runtime_cannot_focus_this_instance(self):
        with tempfile.TemporaryDirectory() as root:
            window=QWidget(); activation=DesktopActivation(root,window)
            self.assertFalse(focus_existing(Path(root)/'other',30)); self.assertFalse(window.isVisible())
            self.assertEqual(server_name(root),server_name(Path(root)/'.'))
            activation.server.close(); window.deleteLater(); APP.processEvents()

if __name__=='__main__': unittest.main()
