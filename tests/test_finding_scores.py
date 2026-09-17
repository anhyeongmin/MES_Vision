import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from PySide6.QtWidgets import QApplication,QTextEdit
from mes_vision.inspection.finding_scores import score_band,visible_findings
from mes_vision.operation.finding_display import object_overlays,finding_summary,finding_details_html
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.region_backend import region_plan,QUESTIONS
from mes_vision.training.data import read_json,write_json,sha256
from mes_vision.anomaly.features import fingerprint

APP=QApplication.instance() or QApplication([])

class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        fixture=make_vlm_fixture(Path(self.temp.name)/'fixture');self.job=fixture['queue'].get(fixture['job_id'])
        self.path=Path(self.job['snapshot_path']);self.inspection=read_json(self.path/'inspection.json')
        self.obj=self.inspection['objects'][1];self.original=deepcopy(self.obj['checks'][0]['findings'][0])
    def findings(self,items):
        self.obj['checks'][0]['findings']=[dict(deepcopy(self.original),defect_code=code,score=score) for code,score in items]
    def persist(self):
        write_json(self.path/'inspection.json',self.inspection)
        m=read_json(self.path/'snapshot.json');m['files']['inspection.json']=sha256(self.path/'inspection.json');write_json(self.path/'snapshot.json',m)
        self.job['snapshot_digest']=fingerprint(m)
    def test_boundary_scores_and_missing_score(self):
        for score,expected in [(0.199999,'HIDDEN'),(.2,'SUSPECT'),(.499999,'SUSPECT'),(.5,'DETECTED'),(1,'DETECTED'),(None,'SUSPECT')]:
            self.assertEqual(score_band(dict(defect_code='NG05',score=score)),expected)
        for value in [float('nan'),float('inf'),True,-.1,1.01]:
            with self.assertRaises(ValueError):score_band(dict(defect_code='NG05',score=value))
    def test_user_example_same_filter_in_summary_overlay_and_vlm(self):
        self.findings([('NG05',.948),('NG06',.133)]);before=deepcopy(self.obj)
        self.assertEqual(finding_summary([self.obj]),'검출: NG05')
        overlays=object_overlays(self.obj);self.assertEqual(len(overlays),2)
        self.assertIn('[검출]',overlays[1]['label']);self.assertNotIn('NG06',overlays[1]['label'])
        self.persist();plan=region_plan(self.job,set(QUESTIONS))
        self.assertEqual([r['code'] for r in plan['regions']],['NG05'])
        self.assertEqual(plan['hidden_below_display_min'],1)
        self.assertEqual(self.obj,before)
        self.assertEqual(len(read_json(self.path/'inspection.json')['objects'][1]['checks'][0]['findings']),2)
    def test_multiple_defects_retained_and_suspect_yellow(self):
        self.findings([('NG05',.948),('NG03',.6),('NG06',.33)])
        text=finding_summary([self.obj]);self.assertIn('검출: NG03, NG05',text);self.assertIn('의심: NG06',text)
        overlay=object_overlays(self.obj)[-1];self.assertEqual(overlay['display_color'],'#d6a52b')
        self.persist();plan=region_plan(self.job,set(QUESTIONS));self.assertEqual(len(plan['regions']),3)
        self.assertEqual(plan['regions'][-1]['score_band'],'SUSPECT')
    def test_all_hidden_is_not_reinterpreted_as_ok(self):
        self.findings([('NG06',.133)]);self.persist()
        self.assertFalse(region_plan(self.job,set(QUESTIONS))['regions'])
        self.assertFalse(visible_findings(self.obj))
        self.assertEqual(len(object_overlays(self.obj)),1)
        self.assertIn('정상 확정은 아닙니다',finding_details_html([], [self.obj]))
    def test_hidden_bad_coordinates_still_rejected(self):
        self.findings([('NG05',.133)]);self.obj['checks'][0]['findings'][0]['original_box']['x1']+=1;self.persist()
        with self.assertRaisesRegex(ValueError,'coordinate mismatch'):region_plan(self.job,set(QUESTIONS))
    def test_qt_details_show_bands_without_low_score_and_escape_text(self):
        self.findings([('NG05',.948),('NG03',.33),('NG06',.133)])
        view=QTextEdit();view.setHtml(finding_details_html(['<not markup>'],[self.obj]))
        text=view.toPlainText();self.assertIn('[검출] NG05',text);self.assertIn('[의심] NG03',text)
        self.assertNotIn('NG06',text);self.assertIn('기준 미만 1건',text);self.assertIn('<not markup>',text)
        self.assertIn('#d6a52b',view.toHtml());view.deleteLater()

    def test_manual_table_and_selected_details_use_same_bands(self):
        from mes_vision.station.manual_dialog import ManualInspectionDialog
        from mes_vision.operation.catalog import default_equipment
        self.findings([('NG05',.948),('NG03',.33),('NG06',.133)])
        dialog=ManualInspectionDialog(Path(__file__).resolve().parents[1],Path(self.temp.name)/'manual-runtime',default_equipment()['camera'])
        dialog.objects=[dict(deepcopy(self.obj),manual_id='1')]
        dialog.details={'1':dict(associated=True,result={'objects':[self.obj]},snapshot=str(self.path),snapshot_digest=self.job['snapshot_digest'])}
        dialog.refresh_objects();dialog.select_object('1')
        summary=dialog.table.item(0,2).text();self.assertIn('검출: NG05',summary);self.assertIn('의심: NG03',summary);self.assertNotIn('NG06',summary)
        details=dialog.reasons.toPlainText();self.assertIn('[검출] NG05',details);self.assertIn('[의심] NG03',details);self.assertNotIn('NG06',details)
        self.assertEqual(len(dialog.detail.canvas.tracks),3)
        self.assertEqual(len(self.obj['checks'][0]['findings']),3)
        dialog.close();APP.processEvents();dialog.deleteLater()
