from dataclasses import replace
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.training.config import JobConfig
from mes_vision.training.data import validate_dataset, read_json, write_json
from mes_vision.training.fixtures import make_fixture
from mes_vision.training.jobs import verified_artifact


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mes-training-")
        self.addCleanup(self.tmp.cleanup)
        self.root = make_fixture(Path(self.tmp.name) / "data", "object_detector")

    def change(self, split, modify):
        path = self.root / split / "_annotations.coco.json"
        data = read_json(path)
        modify(data)
        write_json(path, data)

    def test_valid_objects_include_negative_images(self):
        report = validate_dataset(self.root)
        self.assertEqual(report["splits"]["train"]["negative_images"], 1)
        self.assertEqual(report["categories"][0]["label_index"], 0)

    def test_valid_defects_support_multiple_boxes_and_normal_crops(self):
        report = validate_dataset(make_fixture(Path(self.tmp.name) / "defects", "known_defect_detector"))
        self.assertEqual(report["splits"]["train"]["annotations"], 2)
        self.assertEqual(report["splits"]["train"]["negative_images"], 1)
        self.assertEqual([x["name"] for x in report["categories"]], ["NG03", "NG06"])

    def test_pixel_duplicate_between_splits_rejected(self):
        shutil.copyfile(self.root / "train/synthetic-0.png", self.root / "valid/synthetic-0.png")
        with self.assertRaisesRegex(ValueError, "same image pixels"):
            validate_dataset(self.root)

    def test_physical_specimen_leakage_rejected(self):
        self.change("valid", lambda d: d["images"][0].update(specimen_ids=["synthetic-train-0"]))
        with self.assertRaisesRegex(ValueError, "physical specimen"):
            validate_dataset(self.root)

    def test_capture_session_leakage_rejected(self):
        self.change("test", lambda d: d["images"][0].update(capture_session_id="synthetic-session-train"))
        with self.assertRaisesRegex(ValueError, "capture session"):
            validate_dataset(self.root)

    def test_annotation_referencing_missing_image_rejected(self):
        self.change("train", lambda d: d["annotations"][0].update(image_id=99))
        with self.assertRaisesRegex(ValueError, "references missing"):
            validate_dataset(self.root)

    def test_bbox_outside_original_dimensions_rejected(self):
        self.change("train", lambda d: d["annotations"][0].update(bbox=[490, 100, 200, 200]))
        with self.assertRaisesRegex(ValueError, "bbox outside"):
            validate_dataset(self.root)

    def test_wrong_area_rejected(self):
        self.change("train", lambda d: d["annotations"][0].update(area=1))
        with self.assertRaisesRegex(ValueError, "area"):
            validate_dataset(self.root)

    def test_category_mapping_mismatch_rejected(self):
        self.change("valid", lambda d: d["categories"][0].update(name="different"))
        with self.assertRaisesRegex(ValueError, "category mapping"):
            validate_dataset(self.root)

    def test_unobserved_class_rejected(self):
        self.change("train", lambda d: d["categories"].append({"id": 2, "name": "missing"}))
        with self.assertRaisesRegex(ValueError, "every registered class"):
            validate_dataset(self.root)

    def test_path_escape_rejected(self):
        self.change("train", lambda d: d["images"][0].update(file_name="../outside.png"))
        with self.assertRaisesRegex(ValueError, "unsafe file_name"):
            validate_dataset(self.root)

    def test_unknown_defect_and_ok_classes_rejected(self):
        meta = read_json(self.root / "dataset.json")
        meta["role"] = "known_defect_detector"
        write_json(self.root / "dataset.json", meta)
        with self.assertRaisesRegex(ValueError, "NG01..NG06"):
            validate_dataset(self.root)

    def test_changed_annotation_changes_dataset_fingerprint(self):
        before = validate_dataset(self.root)["fingerprint"]
        self.change("train", lambda d: d["annotations"][0].update(bbox=[101, 130, 200, 210]))
        self.assertNotEqual(before, validate_dataset(self.root)["fingerprint"])

    def test_duplicate_ids_rejected(self):
        self.change("train", lambda d: d["images"][1].update(id=1))
        with self.assertRaisesRegex(ValueError, "duplicate/invalid image"):
            validate_dataset(self.root)

    def test_rgb_dimensions_must_match_labels(self):
        self.change("train", lambda d: d["images"][0].update(width=100))
        with self.assertRaisesRegex(ValueError, "image size mismatch"):
            validate_dataset(self.root)

    def test_config_rejects_unknown_keys_and_invalid_values(self):
        config = JobConfig("object_detector", str(self.root), "weights", "a"*64)
        with self.assertRaises(ValueError):
            replace(config, batch_size=0)
        with self.assertRaises(ValueError):
            replace(config, lr=float("nan"))
        self.assertEqual(config.compatibility(), replace(config, epochs=200).compatibility())
        self.assertNotEqual(config.compatibility(), replace(config, seed=12).compatibility())

    def test_bad_checkpoint_hash_rejected(self):
        path = Path(self.tmp.name) / "bad.ckpt"
        path.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verified_artifact(Path(self.tmp.name), {"path": path.name, "sha256": "a"*64})

    def test_duplicate_json_keys_rejected(self):
        path = Path(self.tmp.name) / "invalid.json"
        path.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            read_json(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
