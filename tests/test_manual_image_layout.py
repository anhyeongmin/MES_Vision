import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PySide6.QtWidgets import QApplication
from mes_vision.station.manual_dialog import ManualInspectionDialog

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]

class ManualImageLayoutTests(unittest.TestCase):
    def test_both_images_fit_without_child_clipping_after_resize(self):
        with tempfile.TemporaryDirectory() as temp:
            dialog=ManualInspectionDialog(ROOT,Path(temp),{'driver':'uvc'},camera=Mock())
            dialog.timer.stop()
            try:
                for panel in (dialog.preview,dialog.detail):
                    panel.canvas.set_rgb(np.zeros((720,720,3),dtype=np.uint8))
                dialog.show()
                for width,height in ((1240,900),(1475,1050),(1100,800)):
                    dialog.resize(width,height);APP.processEvents()
                    for panel in (dialog.preview,dialog.detail):
                        canvas=panel.canvas
                        self.assertGreaterEqual(canvas.height(),190)
                        self.assertTrue(panel.frame.rect().contains(canvas.geometry()))
                        scale,x,y=canvas.transform()
                        self.assertGreaterEqual(x,0);self.assertGreaterEqual(y,0)
                        self.assertLessEqual(x+720*scale,canvas.width()+1)
                        self.assertLessEqual(y+720*scale,canvas.height()+1)
            finally:
                dialog.hide();dialog.deleteLater();APP.processEvents()
