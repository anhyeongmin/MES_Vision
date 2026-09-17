from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from mes_vision.data_management.collection import Collection
from mes_vision.data_management.video import import_video
from mes_vision.data_management.export import export_collection
from mes_vision.training.data import sha256


class VideoCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.collection = Collection.create(self.root / 'data', 'ASH', kind='synthetic')
        self.video = self.root / 'input.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(30):
            writer.write(np.full((48, 64, 3), index * 7, dtype=np.uint8))
        writer.release()

    def tearDown(self):
        self.temp.cleanup()

    def test_sampling_provenance_and_no_automatic_labels(self):
        result = import_video(self.collection, self.video, 'one', interval_seconds=1, max_frames=10)
        self.assertEqual(result['frames'], 3)
        self.assertFalse(result['limit_reached'])
        records = Collection(self.collection.root).data['records']
        self.assertEqual([r['video_source']['frame_index'] for r in records], [0, 10, 20])
        for record in records:
            self.assertEqual(sha256(self.collection.root / record['video_source']['original']), sha256(self.video))
            self.assertEqual(record['split'], 'unassigned')
            self.assertFalse(record['reviewed'])
            self.assertEqual(record['objects'], [])
            self.assertEqual(sha256(self.collection.image_path(record)), record['image_sha256'])
        with self.assertRaises(ValueError):
            export_collection(self.collection, self.root / 'export', 'object_detector')

    def test_limit_and_duplicate_video_refused(self):
        result = import_video(self.collection, self.video, 'train', max_frames=2)
        self.assertTrue(result['limit_reached'])
        self.collection.assign_session('train', 'train')
        with self.assertRaises(ValueError):
            import_video(self.collection, self.video, 'test')
        self.assertEqual(len(self.collection.data['records']), 2)
        self.assertTrue(all(r['split'] == 'train' for r in self.collection.data['records']))

    def test_invalid_video_and_parameters_leave_no_records(self):
        broken = self.root / 'broken.mp4'
        broken.write_bytes(b'not a video')
        for path, kwargs in [(broken, {}), (self.video, {'interval_seconds': float('nan')}),
                             (self.video, {'max_frames': 0})]:
            with self.assertRaises(ValueError):
                import_video(self.collection, path, 'one', **kwargs)
        self.assertEqual(self.collection.data['records'], [])
        self.assertEqual(list((self.collection.root / 'images').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
