import tempfile, unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from PIL import Image
from mes_vision.station.photo_inspection import ResidentPhotos,run_photos,open_results
from mes_vision.training.data import write_json,sha256
from test_photo_inspection import Registry,ROOT


class ResidentTests(unittest.TestCase):
    def test_models_loaded_once_across_roles_and_cycles_then_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bundle=dict(schema_version=1,purpose='saved_photo_inspection',product_id='ASH',production_ready=False,
                                      models=dict(overview={},objects={},defects={}))
            path=root/'bundle.json';write_json(path,bundle)
            photo=root/'input.png';Image.new('RGB',(160,100),'white').save(photo)
            registry=Registry()
            with patch('mes_vision.station.photo_inspection.load_bundle',return_value=bundle) as loader:
                with ResidentPhotos(path,ROOT,root/'runtime',registry) as resident:
                    for i,role in enumerate(('overview','detail','detail','overview')):
                        request=dict(bundle=str(path),runtime=str(root/'runtime'),output=str(root/str(i)),image_kind='real',
                                     images=[dict(role=role,path=str(photo),sha256=sha256(photo))])
                        report=run_photos(request,ROOT,resident=resident)
                        open_results(root/str(i)/'results.json')
                        self.assertFalse(report['robot_commands_enabled'])
                    self.assertEqual(registry.loaded,['objects','objects','defects'])
                    self.assertEqual(registry.closed,[]); self.assertEqual(loader.call_count,1)
                    altered=deepcopy(bundle);altered['product_id']='changed';write_json(path,altered)
                    request['output']=str(root/'bad')
                    with self.assertRaisesRegex(ValueError,'bundle changed'): run_photos(request,ROOT,resident=resident)
                    self.assertFalse((root/'bad').exists())
                self.assertCountEqual(registry.loaded,registry.closed)

    def test_partial_load_failure_releases_handles_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bundle=dict(models=dict(overview={},objects={},defects={}))
            path=root/'bundle.json';write_json(path,bundle)
            registry=Registry(); original=registry.create
            with patch('mes_vision.station.photo_inspection.load_bundle',return_value=bundle):
                with patch.object(registry,'create',side_effect=[original({},'objects'),RuntimeError('load failed')]):
                    with self.assertRaisesRegex(RuntimeError,'load failed'):
                        with ResidentPhotos(path,ROOT,root/'runtime',registry): pass
                self.assertEqual(registry.closed,['objects'])
                from filelock import FileLock
                with FileLock(str(root/'runtime/vlm/gpu-coordination/resident.lock'),timeout=0): pass
