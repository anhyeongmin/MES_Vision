"""End-to-end USB camera protocol with injected images/models/motion acknowledgements."""
from dataclasses import replace
from unittest.mock import patch
import unittest
from mes_vision.inputs import SourceKind
from mes_vision.inputs.camera_identity import camera_uri
import test_station_ui as station_ui


class USBStationFlow(unittest.TestCase):
    def test_usb_overview_detail_saved_results_with_vlm_off(self):
        ui=station_ui.WindowTests('runTest'); ui.setUp(); self.addCleanup(ui.tearDown)
        original=station_ui.live
        def usb(sequence,stamp,**kwargs):
            return replace(original(sequence,stamp,**kwargs),source_kind=SourceKind.UVC,source_uri=camera_uri('uvc','TEST'),
                           media_time_seconds=None,host_read_completed_monotonic=stamp)
        with patch.object(station_ui,'live',side_effect=usb):
            ui.install(); ui.w.equipment['camera']['driver']='uvc'
            # This injected GUI loop publishes every 34 ms. Its old 10 ms age
            # budget was shorter than one publication interval and failed
            # depending on when rendering happened. Strict stale rejection is
            # exercised with deterministic timestamps in test_uvc_camera.
            settings=station_ui.StationSettings(ui.root); value=dict(settings.value)
            value['max_frame_age_seconds']=.2; settings.save(value)
            ui.w.engine.service.detail_detector.defect=True
            # The legacy NG fixture only names a defect. Supply a crop box to
            # exercise the new display without inventing one in production.
            from mes_vision.inspection import Box
            inspector=next(i for i in ui.w.engine.service.inspectors if i.check_id=='known_defects')
            original_inspect=inspector.inspect
            def with_box(crop):
                check=original_inspect(crop)
                return replace(check,findings=tuple(replace(f,crop_box=Box(10,10,20,20)) for f in check.findings))
            inspector.inspect=with_box
            data=ui.drive(); target=data['targets'][0]
            self.assertEqual(target['decision'],'NG'); self.assertIn('NG03',ui.w.live_details.toPlainText())
            self.assertEqual(data['profile']['camera_driver'],'uvc')
            self.assertEqual(data['overview']['frame_metadata']['source_kind'],'uvc')
            self.assertIn('host_read_completion',data['overview']['timing']['timing_basis'])
            self.assertNotEqual(data['overview']['frame_id'],target['detail']['frame_id'])
            self.assertFalse(ui.w.queue.list())
            ui.w.select_track(target['id']); self.assertFalse(ui.w.detail_image.canvas.pixmap.isNull())
            findings=[row for row in ui.w.detail_image.canvas.tracks if row.get('finding')]
            self.assertTrue(findings); self.assertTrue(all(row['track_id']==target['id'] for row in findings))
            self.assertEqual(findings[0]['box'],[190,40,200,50])
            self.assertIn('NG03',ui.w.live_details.toPlainText())
