import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from mes_vision.i18n import tr
from mes_vision.qt_i18n import apply_language
import test_station_ui as station_ui
APP=station_ui.APP


class HandoffHooksTests(unittest.TestCase):
    def test_failed_other_object_capture_cannot_leave_previous_reason_or_vlm(self):
        ui=station_ui.WindowTests('runTest'); ui.setUp(); self.addCleanup(ui.tearDown)
        ui.install(); overview=station_ui.InjectedModels()
        overview.detections=(station_ui.detection(x=40),station_ui.detection(x=320))
        ui.w.engine.service.overview_detector=overview; ui.drive()
        first,second=ui.w.display_data['targets']; ui.w.select_track(first['id'])
        ui.w.vlm_details.setPlainText('previous object explanation')
        second['detail']['sha256']='0'*64
        for _ in range(2):
            with self.assertRaises(ValueError): ui.w.select_track(second['id'])
            self.assertEqual(ui.w.live_details.toPlainText(),'')
            self.assertEqual(ui.w.vlm_details.toPlainText(),'')
            self.assertEqual(ui.w.verdict.objectName(),'verdictIdle')

    def test_actual_station_selection_banner_and_context_keep_baseline(self):
        ui=station_ui.WindowTests('runTest'); ui.setUp(); self.addCleanup(ui.tearDown)
        ui.install(); ui.w.engine.service.detail_detector.defect=True
        data=ui.drive(); target=data['targets'][0]
        self.assertEqual(target['decision'],'NG')
        self.assertEqual(ui.w.verdict.objectName(),'verdictNg')
        self.assertIn(target['id'].split(':')[-1],ui.w.verdict.main.text())
        self.assertEqual(ui.w.context.values['품목'].text(),ui.w.product['name'])
        self.assertEqual(ui.w.context.values['모델'].objectName(),'ctxValueOk')
        # Explanations are advisory; refreshing an empty/OFF VLM result preserves NG.
        ui.w.refresh_selected_vlm()
        self.assertEqual(ui.w.verdict.objectName(),'verdictNg')
        apply_language(APP,'en')
        self.assertEqual(ui.w.verdict.mark.text(),str(tr('불량')))
        ui.w.select_track(None)
        self.assertEqual(ui.w.verdict.objectName(),'verdictIdle')
        ui.w.camera_info=None; ui.w.engine_ready=False; ui.w.tick()
        self.assertEqual(ui.w.context.values['카메라'].objectName(),'ctxValueWarn')
        self.assertEqual(ui.w.context.values['모델'].objectName(),'ctxValueWarn')
