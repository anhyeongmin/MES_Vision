"""History/storage verification uses temporary data only; no physical devices."""
from pathlib import Path
from copy import deepcopy
import json,time,tempfile,unittest
from unittest.mock import patch
import test_station_ui as ui
import test_operation_storage as legacy
from test_operation import InjectedModels,detection
from mes_vision.operation.storage import StorageService,Search,inventory
from mes_vision.operation.maintenance import Maintenance
from mes_vision.operation.catalog import OperationStore
from mes_vision.station.sequence import ScanJournal
from mes_vision.station.history import sync,assets
from mes_vision.station.worker import read_capture
from mes_vision.training.data import read_json


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.ui=ui.WindowTests(); self.ui.setUp(); self.ui.install(); self.w=self.ui.w
        self.external=tempfile.TemporaryDirectory(); self.out=Path(self.external.name)
        self.service=Maintenance(self.w.store)
    def tearDown(self):
        self.ui.tearDown(); self.external.cleanup()
    def drive(self,*,ng=False,overlap=False):
        self.w.engine.service.detail_detector.defect=ng
        if overlap:
            model=InjectedModels(); model.detections=(detection(x=100),detection(x=120)); self.w.engine.service.overview_detector=model
        data=self.ui.drive(); ui.wait(lambda:not self.w.tasks and not self.w.history_busy)
        sync(self.w.store); return data
    def old(self,data):
        stamp=time.time()-1000*86400
        with self.w.scan_journal.connect() as db: db.execute('UPDATE cycle_events SET created=? WHERE cycle_id=?',(stamp,data['id']))
        with self.w.store.connect() as db: db.execute('DELETE FROM station_index_cycles')
        sync(self.w.store)
        self.service.save_policy(dict(self.service.policy(),archive_directory=str(self.out/'archive')))
    def test_search_filters_review_and_ui_use_same_station_record(self):
        data=self.drive(ng=True); target=data['targets'][0]
        result=self.service.search(Search(defect='NG03',decision='NG',product='test-part',text=data['id']))
        self.assertEqual(result['summary']['inspections'],1); row=result['rows'][0]; self.assertEqual(row['object_id'],target['id'])
        self.w.show_record(row['object_id']); self.assertIn('NG03',self.w.history_details.toPlainText())
        before=self.w.scan_journal.read(data['id']); self.w.store.review(row['object_id'],'operator','checked')
        self.assertEqual(self.w.scan_journal.read(data['id']),before)
        self.assertEqual(len(self.service.search(Search(review='reviewed'))['rows']),1)
        self.assertEqual(self.service.search(Search(until=1))['rows'],[])
        self.w.open_history_cycle(); self.assertEqual(self.w.selected_id,target['id'])
    def test_missing_closeup_review_stays_searchable_reviewable_and_exportable(self):
        data=self.drive(overlap=True); rows=self.service.search(Search(decision='REVIEW'))['rows']; self.assertEqual(len(rows),2)
        self.w.show_record(rows[0]['object_id']); self.assertIn('겹쳐',self.w.history_details.toPlainText())
        self.w.store.review(rows[0]['object_id'],'operator','no close-up')
        report=self.out/'report'; self.service.export(report,Search())
        result=read_json(report/'report.json'); self.assertEqual(result['counts']['REVIEW'],2)
        self.assertTrue(all(not r['has_detail_inspection'] for r in result['records']))
        self.assertEqual(len(list((report/'station-evidence'/data['id']).iterdir())),1)
    def test_export_copies_overview_detail_evidence_and_preserves_basic_reason(self):
        data=self.drive(ng=True); report=self.out/'report'; self.service.export(report,Search(decision='NG'))
        result=read_json(report/'report.json'); record=result['records'][0]
        self.assertEqual(record['inspection']['final_decision'],'NG'); self.assertEqual(len(record['evidence']),3)
        self.assertTrue((report/record['station']['overview']['path']).is_file())
        self.assertEqual(record['station']['target']['id'],data['targets'][0]['id'])
        actual=inventory(report); manifest=read_json(report/'export-manifest.json'); actual.pop('export-manifest.json')
        self.assertEqual(actual,manifest['files'])
    def test_hold_and_unreviewed_review_protect_whole_cycle(self):
        data=self.drive(overlap=True); self.old(data); before=self.service.preview_archive()
        self.assertFalse(before['candidates']); self.assertTrue(before['protected'])
        for row in self.service.search()['rows']: self.w.store.review(row['object_id'],'operator','confirmed')
        self.service.hold('station:'+data['id'],'keep all images'); self.assertFalse(self.service.preview_archive()['candidates'])
        self.service.hold('station:'+data['id'],None); self.assertEqual(len(self.service.preview_archive()['candidates']),1)
    def test_archive_moves_complete_cycle_keeps_search_and_backup_portable(self):
        data=self.drive(); self.old(data); originals=assets(data)
        row=self.service.search()['rows'][0]; self.w.store.review(row['object_id'],'operator','archived check')
        preview=self.service.preview_archive(); self.assertEqual(len(preview['candidates']),1)
        result=self.service.archive(preview); self.assertEqual(result['archived'],1)
        self.assertTrue(all(not p.exists() for p in originals))
        moved=self.w.scan_journal.read(data['id']); read_capture(moved['overview']); self.assertFalse(moved['coordinate_valid'])
        found=self.service.search(Search(archived='archived'))['rows']; self.assertEqual(len(found),1)
        self.w.show_record(found[0]['object_id']); self.assertFalse(self.w.saved_canvas.pixmap.isNull())
        self.service.backup(self.out/'backup'); restored=self.service.restore(self.out/'backup',self.out/'restored')
        store=OperationStore(restored['runtime']); record=StorageService(store).search()['rows'][0]
        self.assertIn('archived check',record['review']); opened=store.open_record(record['object_id'])
        self.assertTrue(Path(opened[0]['path']).is_relative_to(self.out/'restored'))
        self.assertFalse(ScanJournal(restored['runtime']).read(data['id'])['coordinate_valid'])
    def test_archive_recovers_after_link_before_deletion(self):
        data=self.drive(); self.old(data); preview=self.service.preview_archive(); original=self.service._intent
        def interrupt(identity,state,payload):
            original(identity,state,payload)
            if state=='LINKED': raise OSError('injected interruption')
        with patch.object(self.service,'_intent',side_effect=interrupt):
            with self.assertRaises(OSError): self.service.archive(preview)
        self.assertTrue(all(p.exists() for p in assets(data)))
        result=self.service.repair_archive(); self.assertEqual(result[0]['state'],'COMPLETED')
        self.assertTrue(all(not p.exists() for p in assets(data)))
    def test_changed_source_or_preview_never_deletes_originals(self):
        data=self.drive(); self.old(data); preview=self.service.preview_archive(); source=assets(data)[0]
        (source/'unregistered.txt').write_text('preserve',encoding='utf-8')
        with self.assertRaises(ValueError): self.service.archive(preview)
        self.assertTrue(source.exists()); self.assertTrue((source/'unregistered.txt').exists())
    def test_failed_cycle_and_unfinished_motion_are_protected(self):
        data=self.drive(); self.old(data); current=self.w.scan_journal.read(data['id']); old=current['revision']; current['revision']+=1
        current.update(state='FAILED',coordinate_valid=False); current['targets'][0]['status']='SORT_UNCONFIRMED'
        self.w.scan_journal.save(current,old,'UNIT_TEST_FAILURE')
        self.assertFalse(self.service.preview_archive()['candidates'])
    def test_usage_includes_station_images_and_invalid_original_blocks_export(self):
        data=self.drive(); self.assertGreater(self.service.usage()['active_bytes'],0)
        Path(data['overview']['path']).write_bytes(b'corrupt')
        with self.assertRaises(Exception): self.service.export(self.out/'corrupt',Search())
    def test_legacy_and_station_records_share_summary_filters_and_pages(self):
        fixture=legacy.StorageTests(); fixture.setUp()
        try:
            fixture.capture(); self.drive(ng=True)
            with fixture.store.connect() as source,self.w.store.connect() as dest:
                for table in ('sessions','captures','inspections','inspection_index'):
                    rows=source.execute('SELECT * FROM '+table).fetchall()
                    for row in rows: dest.execute('INSERT INTO '+table+' VALUES('+','.join('?' for _ in row)+')',tuple(row))
            first=self.service.search(page_size=1); second=self.service.search(first and Search(**first['filters']),page=1,page_size=1)
            self.assertEqual(first['summary']['inspections'],2); self.assertEqual(first['summary']['ok'],1); self.assertEqual(first['summary']['ng'],1)
            self.assertEqual({first['rows'][0]['record_kind'],second['rows'][0]['record_kind']},{'legacy','station'})
            self.w.show_record(self.service.search(Search(decision='OK'))['rows'][0]['object_id'])
            self.assertFalse(self.w.saved_canvas.pixmap.isNull())
        finally: fixture.tearDown()
    def test_history_vlm_request_uses_detail_id_and_links_selected_cycle(self):
        data=self.drive(); identity=data['targets'][0]['id']; before=self.w.scan_journal.read(data['id'])
        self.w.queue.set_enabled(True); self.w.show_record(identity); self.w.request_analysis()
        rows=self.service.search(Search(vlm='PENDING'))['rows']; self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['object_id'],identity)
        with self.w.scan_journal.connect() as db:
            job=db.execute('SELECT job_id FROM station_vlm_jobs WHERE cycle_id=? AND target_id=?',(data['id'],identity)).fetchone()[0]
        self.assertEqual(self.w.queue.get(job)['object_id'],data['targets'][0]['result']['object_id'])
        self.assertEqual(self.w.scan_journal.read(data['id']),before)
    def test_rebuild_reports_corruption_without_losing_known_record(self):
        data=self.drive(ng=True); path=Path(data['targets'][0]['result']['snapshot_path'])/'frame.png'; path.write_bytes(b'corrupt')
        result=self.service.rebuild_index(); self.assertEqual(len(result['errors']),1)
        self.assertEqual(self.service.search()['index_missing'],1)
        self.assertEqual(self.service.search()['index_missing'],1)
        self.assertEqual(self.service.search()['summary']['inspections'],1)
