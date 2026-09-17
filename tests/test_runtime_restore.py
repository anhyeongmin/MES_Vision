import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('restore_runtime',Path(__file__).resolve().parents[1]/'scripts/restore_runtime.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)


def entry(data):
    return {'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}


class RuntimeRestoreTests(unittest.TestCase):
    def fixture(self,root):
        release=root/'parts';release.mkdir()
        pieces=[b'first-part',b'second-part']
        parts=[]
        for i,data in enumerate(pieces):
            name=f'part{i}';(release/name).write_bytes(data);parts.append({'name':name,**entry(data)})
        full=b''.join(pieces)
        manifest={'schema_version':1,'repository':'unused','tag':'test','files':[{'path':'models/weights.bin','source':'release','parts':parts,**entry(full)}]}
        (root/'runtime-assets.json').write_text(json.dumps(manifest))
        return release,full,manifest

    def test_reassembles_and_verifies_with_no_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parts,data,_=self.fixture(root)
            restore.restore(root,asset_dir=parts)
            self.assertEqual((root/'models/weights.bin').read_bytes(),data)
            restore.restore(root,verify_only=True)

    def test_corrupt_download_never_installs_and_existing_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parts,_,_=self.fixture(root)
            (parts/'part1').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'corrupt'):restore.restore(root,asset_dir=parts)
            target=root/'models/weights.bin';self.assertFalse(target.exists())
            target.parent.mkdir(exist_ok=True);target.write_bytes(b'custom model')
            with self.assertRaisesRegex(RuntimeError,'Existing file differs'):restore.restore(root,asset_dir=parts)
            self.assertEqual(target.read_bytes(),b'custom model')

    def test_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parts,_,manifest=self.fixture(root)
            manifest['files'][0]['path']='../outside.bin'
            (root/'runtime-assets.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'escapes'):restore.restore(root,asset_dir=parts)
