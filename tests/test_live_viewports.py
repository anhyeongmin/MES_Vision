"""Three-viewport live screen: pane wiring and square geometry.

Injected frames only; this is not a physical acceptance test. It covers the parts
of the redesigned live screen that can be checked without a camera or a robot:
that the live pane is actually fed, that all three panes stay 1:1, and that the
view area claims only the width its square panes need.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
import unittest
from dataclasses import replace
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QSizePolicy

QWIDGETSIZE_MAX = (1 << 24)-1
from mes_vision.station.window import StationWindow
from mes_vision.operation.shell import ViewArea, panel
from mes_vision.operation.image_view import ImagePanel
from mes_vision.i18n import tr, language
from mes_vision.qt_i18n import apply_language
from test_station_devices import live

APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]


def settle(predicate, seconds=5):
    deadline = time.monotonic()+seconds
    while not predicate() and time.monotonic() < deadline:
        APP.processEvents(); time.sleep(.005)
    if not predicate(): raise AssertionError('Software work timeout')


class Camera:
    """Minimal stand-in for CameraProcess: hands back one frame."""
    def __init__(self, frame): self.frame = frame; self.stopping = False
    def get_latest(self): return self.frame, time.monotonic()
    def isRunning(self): return True
    def poll(self): return []
    def close(self): pass


class LiveViewportTests(unittest.TestCase):
    def setUp(self):
        self.old = language()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.w = StationWindow(ROOT, self.root, start_worker=False)
        self.w.timer.stop(); self.w.analysis.timer.stop()
        # Geometry assertions need a real layout pass, which only happens once the
        # window is shown. The offscreen platform plugin supports this.
        self.w.show(); APP.processEvents()
        settle(lambda: not self.w.tasks and not self.w.analysis.refresh_read.busy)

    def tearDown(self):
        self.w.scan = None; self.w.capture_port = None; self.w.camera = None
        self.w.engine = None; self.w.robot = None; self.w.worker = None; self.w.running = False
        settle(lambda: not self.w.tasks and not self.w.analysis.refresh_read.busy and not self.w.vlm_read.busy)
        self.w.close(); self.w.close(); self.w.deleteLater(); APP.processEvents()
        apply_language(APP, self.old); self.temp.cleanup()

    def test_live_pane_receives_camera_frames(self):
        """tick() must feed self.live_view, not only the popup preview."""
        self.assertTrue(self.w.live_view.canvas.pixmap.isNull())
        frame = live(1, time.monotonic())
        self.w.camera = Camera(frame)
        self.w.tick()
        shown = self.w.live_view.canvas.pixmap
        self.assertFalse(shown.isNull(), 'live pane was never fed')
        height, width = frame.rgb.shape[:2]
        self.assertEqual((shown.width(), shown.height()), (width, height),
                         'live pane was handed something other than the camera frame')

    def test_judgement_column_has_no_maximum_width(self):
        """A maximum here makes QBoxLayout centre the column, leaving dead space
        on both sides. This is exactly what produced the empty band on the right."""
        holder = self.w.judgement_column
        self.assertEqual(holder.maximumWidth(), QWIDGETSIZE_MAX,
                         'judgement column has a maximum width; it will be centred')
        self.assertGreater(holder.minimumWidth(), 0,
                           'judgement column needs a minimum so it stays readable')

    def test_live_pane_is_not_labelled_rectified_while_showing_raw_frames(self):
        """The title may only claim rectification once something rectifies."""
        title = self.w.live_view.title_label.text()
        self.assertNotIn(tr('왜곡보정'), title)


class ViewAreaGeometryTests(unittest.TestCase):
    """Layout rules checked directly on the widgets.

    Not through StationWindow: responsive.fit_available_screen() clamps the window
    to the screen, and the offscreen screen is small enough that no spare width
    exists, which would make every assertion here pass for the wrong reason.
    """

    def build(self, width=1600, height=820):
        page = QWidget(); body = QHBoxLayout(page)
        # No margins: this harness isolates the layout rule, not the page padding.
        body.setContentsMargins(0, 0, 0, 0); body.setSpacing(10)
        card, card_layout = panel()
        views = ViewArea(ImagePanel(), ImagePanel(compact=True), ImagePanel(compact=True))
        card_layout.addWidget(views, 1)
        # stretch 1 on purpose: the layout offers the card more width than its square
        # panes can ever use. ViewArea must refuse it. This is the shape that produced
        # the empty band beside the live and detail panes.
        body.addWidget(card, 1)
        judgement = QWidget(); judgement.setMinimumWidth(430)
        judgement.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        body.addWidget(judgement, 1)
        page.resize(width, height); page.show()
        for _ in range(3): APP.processEvents()
        self.addCleanup(page.deleteLater); self.addCleanup(page.close)
        return page, card, views, judgement

    def test_every_pane_is_square(self):
        page, card, views, judgement = self.build()
        for pane in (views.primary, views.secondary, views.tertiary):
            holder = pane.holder
            box = holder.child.geometry()
            self.assertEqual(box.width(), box.height(), 'pane is not 1:1: '+repr(box))
            self.assertEqual(box.width(), min(holder.width(), holder.height()),
                             'pane does not fill the short side of its holder')

    def test_stacked_panes_are_half_the_tall_pane(self):
        page, card, views, judgement = self.build()
        big = views.primary.width()+views.CHROME
        small = views.column.width()+views.CHROME
        self.assertAlmostEqual(big, small*2+views.GAP, delta=2)

    def test_card_claims_only_the_width_its_panes_need(self):
        """The regression that showed as an empty band beside the panes.

        ViewArea caps its own width and Qt carries that maximum up through the card's
        layout, so the card must not keep width its panes cannot fill.
        """
        page, card, views, judgement = self.build()
        content = views.primary.width()+views.column.width()+views.GAP
        slack = card.width()-content
        self.assertLessEqual(slack, 40,
                             'card is %dpx wider than its panes; dead space returns' % slack)

    def test_judgement_column_reaches_the_page_edge(self):
        """Whatever the square panes cannot use belongs to the judgement column."""
        page, card, views, judgement = self.build()
        self.assertGreaterEqual(judgement.geometry().right(), page.width()-2,
                                'judgement column stops short; width is going nowhere')
        self.assertLessEqual(judgement.geometry().left()-card.geometry().right(), 12,
                             'gap between card and judgement column is wider than the layout spacing')


if __name__ == '__main__':
    unittest.main()
