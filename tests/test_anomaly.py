from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from mes_vision.anomaly.bank import Bank, BankConfig, build_bank, coreset, checked_features, read_normal_export
from mes_vision.anomaly.features import DinoFeatures, fingerprint
from mes_vision.anomaly.scoring import AnomalyEngine, AnomalyInspector, Criteria, connected_regions, nearest_neighbors, save_score
from mes_vision.data_management.fixtures import make_fixture
from mes_vision.data_management.export import export_collection
from mes_vision.inputs import ImageSource
from mes_vision.inspection import Box, Crop, CheckStatus, Detection, InspectionPipeline, Mode
from mes_vision.inspection.adapters import MockDetector
from mes_vision.training.data import read_json, write_json


class TinyFeatures:
    """Deterministic unit fixture; intentionally not an AI-model accuracy test."""
    device = "cpu"
    signature = {"implementation": "synthetic-test-colors-v1", "size": 2}
    def extract(self, rgb):
        a = np.asarray(Image.fromarray(rgb).resize((2, 2))).astype(np.float32) + 1
        return a / np.linalg.norm(a, axis=-1, keepdims=True)


class AnomalyTests(unittest.TestCase):
    def test_resident_engine_preserves_status_regions_and_reference(self):
        from unittest.mock import patch
        import torch
        resident=AnomalyEngine(self.path,self.extractor,product_id='fixture-part',criteria=self.criteria(),
                               allow_synthetic=True,distance_execution='resident')
        legacy=self.engine(self.criteria())
        images=(self.rgb,np.full_like(self.rgb,32),np.full_like(self.rgb,224))
        expected=legacy.score_many(images)
        device_calls=[]
        def extract_device(values):
            device_calls.append(len(values))
            return tuple(torch.from_numpy(self.extractor.extract(rgb)) for rgb in values)
        self.extractor.extract_many_device=extract_device
        with patch('mes_vision.anomaly.scoring.nearest_neighbors',side_effect=AssertionError('legacy transfer path used')):
            actual=resident.score_many(images)
        self.assertEqual(device_calls,[3])
        for a,b in zip(expected,actual):
            for key in ('status','raw_score','regions','peak_nearest_reference','criteria','bank_digest','feature_fingerprint'):
                self.assertEqual(a[0][key],b[0][key])
            np.testing.assert_array_equal(a[1],b[1]); np.testing.assert_array_equal(a[2],b[2])
        neighbors=resident._neighbors; resident.prepare(); self.assertIs(neighbors,resident._neighbors)
        resident.close(); self.assertEqual(neighbors.resident_bytes,0)
        for method in (lambda:resident.score(self.rgb),lambda:resident.score_many(()),resident.prepare):
            with self.assertRaisesRegex(ValueError,'closed'): method()

    def test_resident_empty_and_failed_prepare_do_not_publish_state(self):
        from unittest.mock import patch
        resident=AnomalyEngine(self.path,self.extractor,product_id='fixture-part',allow_synthetic=True,distance_execution='resident')
        self.assertEqual(resident.score_many(()),()); self.assertIsNone(resident._neighbors)
        with patch('mes_vision.anomaly.resident.ResidentNeighbors',side_effect=RuntimeError('allocation failed')):
            with self.assertRaisesRegex(RuntimeError,'allocation failed'): resident.prepare()
        self.assertIsNone(resident._neighbors)
        resident.prepare(); self.assertIsNotNone(resident._neighbors)

    def test_batch_scores_preserve_each_objects_evidence(self):
        images=(self.rgb,np.full_like(self.rgb,32),np.full_like(self.rgb,224))
        engine=self.engine(self.criteria())
        groups=[]
        def extract_many(values):
            groups.append(len(values)); return tuple(self.extractor.extract(rgb) for rgb in values)
        self.extractor.extract_many=extract_many
        expected=[engine.score(rgb) for rgb in images]; actual=engine.score_many(images)
        self.assertEqual(groups,[3])
        for a,b in zip(expected,actual,strict=True):
            self.assertEqual(a[0]['status'],b[0]['status']); self.assertEqual(a[0]['raw_score'],b[0]['raw_score'])
            np.testing.assert_array_equal(a[1],b[1]); np.testing.assert_array_equal(a[2],b[2])

    def test_batch_feature_count_mismatch_is_rejected(self):
        self.extractor.extract_many=lambda values: ()
        with self.assertRaisesRegex(ValueError,'count mismatch'): self.engine().score_many((self.rgb,self.rgb))

    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.collection = make_fixture(self.root / "fixture")
        self.export = self.root / "normal"
        export_collection(self.collection, self.export, "normal_bank")
        self.extractor = TinyFeatures()
        self.path = self.root / "bank"
        self.meta = build_bank(self.export, self.path, self.extractor, config=BankConfig(16, 4, 3), allow_synthetic=True)
        self.bank = Bank(self.path, self.extractor.signature, product_id="fixture-part", allow_synthetic=True)
        item = read_json(self.export / "export.json")["items"][0]
        with Image.open(self.export / item["file"]) as image:
            self.rgb = np.array(image)

    def tearDown(self):
        self.temp.cleanup()

    def engine(self, criteria=None):
        return AnomalyEngine(self.path, self.extractor, product_id="fixture-part", criteria=criteria, allow_synthetic=True)

    def criteria(self, **changes):
        values = dict(version="synthetic-only-criteria", bank_digest=self.bank.digest, product_id="fixture-part",
                      pass_max=.01, fail_min=.2, pixel_threshold=.1, validated=True,
                      validation_reference="synthetic test fixture, not product validation", kind="synthetic")
        values.update(changes)
        return Criteria(**values)

    def test_normal_roundtrip_no_threshold_never_pass(self):
        details, grid, _ = self.engine().score(self.rgb)
        self.assertEqual(details["status"], "UNCERTAIN")
        self.assertIsNone(details["criteria"])
        self.assertLess(details["raw_score"], 1e-6)
        self.assertEqual(grid.shape, (2, 2))
        self.assertEqual(details["regions"], [])

    def test_synthetic_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            Bank(self.path, self.extractor.signature, product_id="fixture-part")
        with self.assertRaisesRegex(ValueError, "synthetic"):
            read_normal_export(self.export)

    def test_bank_product_and_preprocessing_binding(self):
        with self.assertRaisesRegex(ValueError, "product"):
            Bank(self.path, self.extractor.signature, product_id="other", allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, "version"):
            Bank(self.path, dict(self.extractor.signature, size=3), product_id="fixture-part", allow_synthetic=True)

    def test_bank_file_tamper_and_nan_rejected(self):
        a = np.load(self.path / "features.npy")
        a[0, 0] = np.nan
        np.save(self.path / "features.npy", a)
        with self.assertRaisesRegex(ValueError, "integrity"):
            Bank(self.path, self.extractor.signature, product_id="fixture-part", allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            checked_features(a)

    def test_source_review_split_condition_and_hash_checked(self):
        original = read_json(self.export / "export.json")
        for edit in (lambda r: r["items"][0].update(split="valid"),
                     lambda r: r["items"][0]["object"].update(condition="UNKNOWN_NG"),
                     lambda r: r["items"][0].update(image_sha256="0"*64),
                     lambda r: r["items"].append(deepcopy(r["items"][0]))):
            report = deepcopy(original)
            edit(report)
            write_json(self.export / "export.json", report)
            with self.assertRaises(ValueError):
                read_normal_export(self.export, allow_synthetic=True)
        write_json(self.export / "export.json", original)

    def test_source_snapshot_review_required(self):
        snapshot = read_json(self.export / "collection-snapshot.json")
        snapshot["records"][0]["reviewed"] = False
        write_json(self.export / "collection-snapshot.json", snapshot)
        with self.assertRaisesRegex(ValueError, "reviewed"):
            read_normal_export(self.export, allow_synthetic=True)

    def test_out_of_export_path_rejected(self):
        report = read_json(self.export / "export.json")
        report["items"][0]["file"] = "../outside.png"
        write_json(self.export / "export.json", report)
        with self.assertRaisesRegex(ValueError, "path"):
            read_normal_export(self.export, allow_synthetic=True)

    def test_chunked_distances_match_brute_force_and_ties(self):
        rng = np.random.default_rng(4)
        q, m = rng.random((9, 3)).astype(np.float32), rng.random((11, 3)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        m[5] = m[0]
        expected = np.linalg.norm(q[:, None]-m[None, :], axis=2)
        values, indices = nearest_neighbors(q, m, query_chunk=2, bank_chunk=3)
        np.testing.assert_allclose(values, expected.min(axis=1), atol=2e-7)
        np.testing.assert_array_equal(indices, expected.argmin(axis=1))
        same, first = nearest_neighbors(m[:1], m, bank_chunk=1)
        self.assertEqual(float(same[0]), 0)
        self.assertEqual(int(first[0]), 0)

    def test_coreset_deterministic_unique_and_bounded(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(30, 5)).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        config = BankConfig(30, 6, 3)
        selected = coreset(a, config)
        np.testing.assert_array_equal(selected, coreset(a, config))
        self.assertEqual(len(set(selected)), 6)
        self.assertTrue(np.all((0 <= selected) & (selected < 30)))

    def test_reservoir_and_bank_caps_reported(self):
        meta = build_bank(self.export, self.root / "small-bank", self.extractor,
                          config=BankConfig(3, 2, 2), allow_synthetic=True)
        self.assertEqual((meta["total_patches"], meta["candidate_patches"], meta["bank_patches"]), (4, 3, 2))

    def test_threshold_scope_and_invalid_ranges_rejected(self):
        for changes in ({"pass_max": .3}, {"pixel_threshold": 0}, {"fail_min": float("nan")},
                        {"pass_max": True}, {"validation_reference": None}):
            with self.assertRaises(ValueError): self.criteria(**changes)
        with self.assertRaisesRegex(ValueError, "another bank"):
            self.engine(self.criteria(bank_digest="0"*64))
        with self.assertRaises(ValueError): self.engine(self.criteria(kind="real"))

    def test_validated_pass_fail_and_unvalidated_uncertain(self):
        self.assertEqual(self.engine(self.criteria()).score(self.rgb)[0]["status"], "PASS")
        changed = np.zeros_like(self.rgb)
        changed[..., 0] = 255
        details, _, _ = self.engine(self.criteria()).score(changed)
        self.assertEqual(details["status"], "FAIL")
        self.assertTrue(details["regions"])
        details, _, _ = self.engine(self.criteria(validated=False, validation_reference=None)).score(changed)
        self.assertEqual(details["status"], "UNCERTAIN")

    def test_review_band_and_boundary_equality(self):
        changed = np.zeros_like(self.rgb)
        changed[..., 0] = 255
        value = self.engine().score(changed)[0]["raw_score"]
        self.assertEqual(self.engine(self.criteria(pass_max=value, fail_min=value+.1, pixel_threshold=value+.01)).score(changed)[0]["status"], "PASS")
        self.assertEqual(self.engine(self.criteria(pass_max=value-.1, fail_min=value, pixel_threshold=value)).score(changed)[0]["status"], "FAIL")
        self.assertEqual(self.engine(self.criteria(pass_max=value-.1, fail_min=value+.1, pixel_threshold=value)).score(changed)[0]["status"], "UNCERTAIN")

    def test_non_square_region_mapping_and_disconnected_areas(self):
        grid = np.array([[.8, 0, 0], [0, 0, .9]], np.float32)
        regions = connected_regions(grid, 300, 80, .5)
        self.assertEqual([r["box"] for r in regions], [[0., 0., 100., 40.], [200., 40., 300., 80.]])
        self.assertEqual(connected_regions(grid, 300, 80, None), [])

    def test_adapter_crop_provenance_and_final_policy_unchanged(self):
        image_path = self.root / "frame.png"
        changed = np.zeros((200, 300, 3), np.uint8)
        changed[..., 0] = 255
        Image.fromarray(changed).save(image_path)
        with ImageSource(image_path) as source:
            frame = source.read().frame
        inspector = AnomalyInspector(self.engine(self.criteria()), evidence_dir=self.root / "evidence")
        detector = MockDetector((Detection(Box(20, 30, 120, 160), .9, 0, "fixture-part"), Detection(Box(180, 30, 280, 160), .9, 0, "fixture-part")))
        result = InspectionPipeline(detector, (inspector,), mode=Mode.SIMULATION, product_id="fixture-part").run(frame)
        self.assertEqual(len(result.objects), 2)
        for obj in result.objects:
            check = next(c for c in obj.checks if c.check_id == "anomaly")
            self.assertEqual(check.status, CheckStatus.FAIL)
            self.assertEqual(check.findings[0].defect_code, "NG_UNKNOWN")
            self.assertEqual(check.findings[0].original_box.x1, obj.crop["bounds_original_xyxy"]["x1"])
            self.assertEqual(check.details["object_id"], obj.object_id)
            self.assertTrue(Path(check.details["evidence_directory"]).is_dir())
        self.assertIsNone(result.final_decision)
        self.assertFalse(result.robot_commands_enabled)

    def test_uncertain_findings_do_not_claim_ng_unknown(self):
        rgb = np.zeros_like(self.rgb); rgb[..., 0] = 255
        inspector = AnomalyInspector(self.engine(self.criteria(validated=False)))
        crop = Crop("frame", "object", Box(0, 0, rgb.shape[1], rgb.shape[0]), Box(0, 0, rgb.shape[1], rgb.shape[0]), rgb)
        check = inspector.inspect(crop)
        self.assertEqual(check.status, CheckStatus.UNCERTAIN)
        self.assertTrue(check.findings)
        self.assertTrue(all(f.defect_code is None for f in check.findings))

    def test_wrong_pipeline_product_rejected(self):
        inspector = AnomalyInspector(self.engine())
        with self.assertRaisesRegex(ValueError, "product"):
            InspectionPipeline(MockDetector(()), (inspector,), mode=Mode.SIMULATION, product_id="other")

    def test_failure_does_not_reuse_previous_score(self):
        inspector = AnomalyInspector(self.engine(self.criteria()))
        crop = Crop("frame", "object", Box(0, 0, 100, 130), Box(0, 0, 100, 130), self.rgb)
        self.assertEqual(inspector.inspect(crop).status, CheckStatus.PASS)
        self.extractor.extract = lambda rgb: np.full((2, 2, 3), np.nan, np.float32)
        with self.assertRaises(ValueError): inspector.inspect(crop)

    def test_artifact_grid_scale_and_original_image_preserved(self):
        details, grid, nearest = self.engine().score(self.rgb)
        report = save_score(self.root / "score", self.rgb, details, grid, nearest)
        np.testing.assert_array_equal(np.load(self.root / "score/patch-distances.npy"), grid)
        with Image.open(self.root / "score/input.png") as image:
            np.testing.assert_array_equal(np.array(image), self.rgb)
        self.assertEqual(report["visual_scale"]["max"], 2)
        with self.assertRaises(ValueError): save_score(self.root / "score", self.rgb, details, grid, nearest)

    def test_empty_features_and_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError): checked_features(np.empty((0, 3), np.float32))
        with self.assertRaises(ValueError): BankConfig(2, 3)
        with self.assertRaises(ValueError): DinoFeatures(self.root, image_size=225)

    def test_bank_not_overwritten(self):
        with self.assertRaises(ValueError):
            build_bank(self.export, self.path, self.extractor, allow_synthetic=True)

    def test_invalid_bank_dimensions_and_counts_rejected(self):
        original = read_json(self.path / "bank.json")
        for change in ({"grid_shape": [0, 2, 3]}, {"grid_shape": [2, 2, 400]}, {"bank_patches": 10000}):
            write_json(self.path / "bank.json", dict(original, **change))
            with self.assertRaises(ValueError):
                Bank(self.path, self.extractor.signature, product_id="fixture-part", allow_synthetic=True)


if __name__ == "__main__":
    unittest.main()
