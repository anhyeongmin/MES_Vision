from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from mes_vision.data_management.collection import Collection, new_object, validate_record
from mes_vision.data_management.export import export_collection
from mes_vision.data_management.fixtures import make_fixture
from mes_vision.training.data import read_json, sha256, validate_dataset


class DataManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.collection = make_fixture(self.root / "fixture")
        self.first = self.collection.data["records"][0]["id"]

    def tearDown(self):
        self.temp.cleanup()

    def export(self, role, name=None):
        return export_collection(self.collection, self.root / (name or role), role, defect_codes=["NG03", "NG06"])

    def test_original_and_normalized_owned(self):
        record = self.collection.record(self.first)
        self.assertEqual(sha256(self.collection.root / record["original"]), record["original_sha256"])
        self.assertEqual(sha256(self.collection.image_path(record)), record["image_sha256"])
        self.assertEqual(record["split"], "train")
        self.assertEqual(record["reviewer"], "SYNTHETIC_FIXTURE")

    def test_edit_invalidates_review_and_preserves_history(self):
        record = self.collection.record(self.first)
        record["objects"][0]["note"] = "수정"
        self.collection.save_record(record)
        self.assertFalse(self.collection.record(self.first)["reviewed"])
        self.assertIsNone(self.collection.record(self.first)["reviewer"])
        self.assertTrue(list((self.collection.root / "history").glob("revision-*.json")))

    def test_export_boxes_and_original_crop_pixels(self):
        objects = self.export("object_detector")
        defects = self.export("known_defect_detector")
        self.assertEqual(len(objects["items"]), 3)
        self.assertEqual(len(defects["items"]), 6)
        data = read_json(self.root / "known_defect_detector/train/_annotations.coco.json")
        self.assertEqual(data["annotations"][0]["bbox"], [15, 15, 30, 55])
        self.assertEqual(len(data["annotations"]), 2)
        self.assertEqual(len(data["images"]), 2)
        normal_id = data["images"][0]["id"]
        self.assertFalse(any(a["image_id"] == normal_id for a in data["annotations"]))
        item = defects["items"][0]
        source = self.collection.record(item["capture_id"])
        with Image.open(self.collection.image_path(source)) as image, Image.open(self.root / "known_defect_detector" / item["file"]) as crop:
            np.testing.assert_array_equal(np.array(image.crop(item["crop_bounds_original"])), np.array(crop))
        self.assertEqual(validate_dataset(self.root / "known_defect_detector")["fingerprint"], defects["dataset_fingerprint"])

    def test_normal_bank_train_only_and_challenge_isolated(self):
        bank = self.export("normal_bank")
        challenge = self.export("challenge")
        self.assertEqual(len(bank["items"]), 1)
        self.assertTrue(all(i["split"] == "train" and i["object"]["condition"] == "NORMAL" for i in bank["items"]))
        self.assertEqual(len(challenge["items"]), 2)
        self.assertTrue(all(i["split"] == "challenge" for i in challenge["items"]))
        self.assertFalse((self.root / "normal_bank/dataset.json").exists())

    def test_same_specimen_split_rejected_without_commit(self):
        before = deepcopy(self.collection.data)
        record = self.collection.record(self.collection.data["records"][1]["id"])
        record["objects"][0]["specimen_id"] = "train-normal"
        with self.assertRaisesRegex(ValueError, "specimen"):
            self.collection.save_record(record)
        self.assertEqual(self.collection.data, before)
        self.assertEqual(Collection(self.collection.root).data, before)

    def test_session_assignment_atomic_and_import_inherits(self):
        record = self.collection.record(self.first)
        path = self.root / "new.png"
        Image.new("RGB", (320, 200), (9, 7, 2)).save(path)
        identity = self.collection.import_image(path, " session-train ")
        self.assertEqual(self.collection.record(identity)["split"], "train")
        self.collection.assign_session("session-train", "valid")
        self.assertEqual(self.collection.record(self.first)["split"], "valid")
        self.assertEqual(self.collection.record(identity)["split"], "valid")

    def test_duplicate_pixels_different_encoding_rejected(self):
        record = self.collection.record(self.first)
        path = self.root / "duplicate.bmp"
        with Image.open(self.collection.image_path(record)) as image:
            image.save(path)
        identity = self.collection.import_image(path, "another-session")
        with self.assertRaisesRegex(ValueError, "pixels"):
            self.collection.assign_session("another-session", "test")
        self.assertEqual(self.collection.record(identity)["split"], "unassigned")

    def test_stale_writer_rejected(self):
        stale = Collection(self.collection.root)
        record = self.collection.record(self.first)
        record["objects"][0]["note"] = "first writer"
        self.collection.save_record(record)
        with self.assertRaisesRegex(ValueError, "다른 창"):
            stale.save_record(stale.record(self.first))
        self.assertEqual(Collection(self.collection.root).record(self.first)["objects"][0]["note"], "first writer")

    def test_unreviewed_export_blocked(self):
        self.collection.save_record(self.collection.record(self.first))
        with self.assertRaisesRegex(ValueError, "검토되지"):
            self.export("normal_bank")

    def test_unknown_in_training_blocked(self):
        self.collection.assign_session("session-challenge", "train")
        with self.assertRaisesRegex(ValueError, "challenge"):
            self.export("object_detector")

    def test_missing_classes_fail_without_publishing(self):
        with self.assertRaises(ValueError):
            export_collection(self.collection, self.root / "bad-export", "known_defect_detector", defect_codes=["NG01", "NG03"])
        self.assertFalse((self.root / "bad-export").exists())
        failure = list(self.root.glob(".bad-export-building-*/export.json"))
        self.assertEqual(len(failure), 1)
        self.assertEqual(read_json(failure[0])["status"], "FAILED")

    def test_global_defect_crop_excluded_not_normalized(self):
        # Keep a second boxed NG specimen so coverage remains valid.
        path = self.root / "global.png"
        Image.new("RGB", (320, 200), (81, 64, 48)).save(path)
        identity = self.collection.import_image(path, "session-train")
        record = self.collection.record(identity)
        obj = new_object("train-global", [20, 25, 120, 155])
        obj.update(condition="KNOWN_NG", defects=[{"id": "global-defect", "code": "NG01", "bbox": None, "note": "돌기 전체 누락"}])
        record["objects"] = [obj]
        self.collection.save_record(record)
        self.collection.review(identity, "reviewer")
        result = self.export("known_defect_detector")
        self.assertFalse(any(item["capture_id"] == identity for item in result["items"]))
        self.assertTrue(any(item.get("object_id") == obj["id"] for item in result["excluded"]))

    def test_modified_image_rejected(self):
        record = self.collection.record(self.first)
        Image.new("RGB", (320, 200), (0, 0, 0)).save(self.collection.image_path(record))
        with self.assertRaisesRegex(ValueError, "변경"):
            self.export("object_detector")

    def test_no_overwrite_or_nested_export(self):
        self.export("normal_bank")
        with self.assertRaisesRegex(ValueError, "덮어"):
            self.export("normal_bank")
        with self.assertRaisesRegex(ValueError, "밖에"):
            export_collection(self.collection, self.collection.root / "output", "normal_bank")

    def test_empty_scene_requires_confirmation(self):
        record = self.collection.record(self.first)
        record["objects"] = []
        self.collection.save_record(record)
        with self.assertRaisesRegex(ValueError, "빈 작업대"):
            self.collection.review(self.first, "reviewer")
        record["empty_confirmed"] = True
        self.collection.save_record(record)
        self.collection.review(self.first, "reviewer")

    def test_incomplete_or_invalid_labels_rejected(self):
        original = self.collection.record(self.first)
        cases = [lambda r: r["objects"][0].update(bbox=[-1, 0, 100, 100]),
                 lambda r: r["objects"][1]["defects"][0].update(bbox=[0, 0, 10, 10]),
                 lambda r: r["objects"][1].update(condition="NORMAL"),
                 lambda r: r["objects"][0].update(condition="KNOWN_NG"),
                 lambda r: r["objects"][0].update(condition="UNKNOWN_NG"),
                 lambda r: r.update(empty_confirmed=True),
                 lambda r: r.update(excluded=True, exclude_reason="")]
        for edit in cases:
            record = deepcopy(original)
            edit(record)
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.collection.save_record(record)

    def test_excluded_unreviewed_can_be_skipped(self):
        record = self.collection.record(self.collection.data["records"][3]["id"])
        record.update(excluded=True, exclude_reason="가림")
        self.collection.save_record(record)
        report = self.export("object_detector")
        self.assertTrue(any(i["reason"] == "가림" for i in report["excluded"]))

    def test_overlapping_crop_rejected(self):
        record = self.collection.record(self.first)
        record["objects"][0]["bbox"] = [20, 25, 180, 155]
        self.collection.save_record(record)
        self.collection.review(self.first, "reviewer")
        with self.assertRaisesRegex(ValueError, "다른 물체"):
            self.export("normal_bank")

    def test_path_escape_rejected(self):
        record = self.collection.record(self.first)
        record["image"] = "../outside.png"
        with self.assertRaises(ValueError):
            self.collection.image_path(record)

    def test_exif_orientation_normalized_once(self):
        image = Image.new("RGB", (30, 20), (200, 100, 50))
        exif = Image.Exif()
        exif[274] = 6
        path = self.root / "oriented.jpg"
        image.save(path, exif=exif)
        identity = self.collection.import_image(path, "orientation")
        record = self.collection.record(identity)
        self.assertEqual((record["width"], record["height"]), (20, 30))
        with Image.open(self.collection.image_path(record)) as normalized:
            self.assertIsNone(normalized.getexif().get(274))
        self.assertTrue(record["transformations"])


if __name__ == "__main__":
    unittest.main()
