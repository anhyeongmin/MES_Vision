"""Offline audit must not migrate a runtime or turn file checks into motion authority."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import sqlite3
import unittest

from mes_vision.station.audit import audit, markdown, registration_snapshot
from mes_vision.operation.catalog import default_equipment, new_product


def database(root, products=None, equipment=None):
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / 'operation.sqlite3') as db:
        db.executescript('CREATE TABLE products(id TEXT,version INTEGER,data TEXT); '
                        'CREATE TABLE equipment(version INTEGER,data TEXT);')
        db.execute('INSERT INTO equipment VALUES(?,?)', (1, json.dumps(equipment or default_equipment())))
        for p in products or []:
            db.execute('INSERT INTO products VALUES(?,?,?)', (p['id'], p['version'], json.dumps(p)))
    db.close()


class AuditTests(unittest.TestCase):
    def test_missing_runtime_is_reported_without_creating_it(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'not-created'
            report = audit(root)
            self.assertFalse(root.exists())
            self.assertEqual(report['registration_status'], 'needs_attention')
            self.assertFalse(report['motion_authorized'])
            self.assertTrue(all(c['status'] == 'not_assessed' for c in report['field_checks']))

    def test_reads_latest_versions_without_migration_or_data_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = new_product('previous'); p['version'] = 1
            latest = dict(p, version=2, name='current')
            database(root, [p, latest])
            before = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in root.iterdir()}
            _, products = registration_snapshot(root)
            self.assertEqual([p['name'] for p in products], ['current'])
            report = audit(root)
            after = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in root.iterdir()}
            self.assertEqual(before, after)
            self.assertIn('detail:' + p['id'], [c['id'] for c in report['checks']])
            self.assertEqual(report['physical_acceptance'], 'not_assessed')

    def test_corrupt_database_and_settings_are_reported_independently(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'operation.sqlite3').write_bytes(b'not a sqlite database')
            (root / 'station-settings.json').write_text('{}', encoding='utf-8')
            report = audit(root)
            bad = {c['id']: c for c in report['checks'] if c['status'] == 'needs_attention'}
            self.assertIn('registration', bad)
            self.assertIn('capture_setup', bad)

    def test_changed_overview_model_is_rejected_even_when_detail_check_passes(self):
        from mes_vision.station.recipe import recipe_path
        with TemporaryDirectory() as temporary:
            root = Path(temporary); p = new_product(); e = default_equipment(); e['camera'].update(width=640,height=480)
            database(root, [p], e)
            model = root / 'model.pth'; model.write_bytes(b'changed')
            path = recipe_path(root, p); path.parent.mkdir()
            path.write_text(json.dumps(dict(schema_version=1, product_id=p['id'], product_version=p['version'],
                equipment_version=e['version'], validation_reference='temporary test only', image_size=[640,480],
                capture_domains={'overview_objects': 'overview'}, detail_workspace={'roi': [[0,0],[640,0],[640,480],[0,480]],
                'excluded': [], 'validation_reference': 'temporary test only'},
                overview_asset={'weights': str(model), 'sha256': '0' * 64, 'class_names': ['test']})), encoding='utf-8')
            with patch('mes_vision.operation.catalog.readiness', return_value=[]):
                report = audit(root)
            item = next(c for c in report['checks'] if c['id'] == 'recipe:' + p['id'])
            self.assertEqual(item['status'], 'needs_attention')
            self.assertIn('Overview model changed', item['detail'])

    def test_wal_snapshot_includes_committed_data(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); database(root)
            db = sqlite3.connect(root / 'operation.sqlite3')
            try:
                db.execute('PRAGMA journal_mode=WAL')
                p = new_product('committed in WAL')
                db.execute('INSERT INTO products VALUES(?,?,?)', (p['id'], 1, json.dumps(p))); db.commit()
                self.assertEqual(registration_snapshot(root)[1][0]['name'], p['name'])
            finally:
                db.close()

    def test_markdown_never_calls_unrun_tests_passed(self):
        with TemporaryDirectory() as temporary:
            report = audit(Path(temporary) / 'absent')
        report['software_tests'] = {'status': 'not_run'}
        rendered = markdown(report)
        self.assertIn('실행하지 않았습니다', rendered)
        self.assertNotIn('software-tests.log', rendered)
        report['source_unchanged_during_check'] = False
        self.assertIn('소스가 변경됐습니다', markdown(report))


if __name__ == '__main__':
    unittest.main()
