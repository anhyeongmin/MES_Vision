import json
from pathlib import Path
import tempfile
import unittest
from mes_vision.operation.catalog import default_equipment
from mes_vision.operation.acquisition import configure_equipment


class FreshCloneAcquisitionTests(unittest.TestCase):
    def test_unregistered_camera_can_open_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'configs/camera/acquisition.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'schema_version':1,'processing':{'kind':'undistorted_center_square_v1','sensor_size':[1280,720],'calibration_file':'lens.json','calibration_sha256':'abc','camera_serial':'KNOWN'}}))
            original=default_equipment();configured=configure_equipment(root,original)
            self.assertNotIn('inspection_acquisition',configured['camera'])
            self.assertEqual(configured['workspace']['validation_reference'],'')
            self.assertEqual(original,default_equipment())
            original['camera']['serial']='WRONG'
            with self.assertRaisesRegex(ValueError,'another camera'):
                configure_equipment(root,original)
            original['camera']['serial']='KNOWN'
            configured=configure_equipment(root,original)
            self.assertEqual(configured['camera']['inspection_acquisition']['camera_serial'],'KNOWN')
            self.assertEqual(configured['workspace']['roi'],[[0,0],[720,0],[720,720],[0,720]])
