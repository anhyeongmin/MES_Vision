from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from mes_vision.data_management.collection import validate_groups, validate_record, utc
from mes_vision.data_management.export import clean_crop
from mes_vision.training.data import read_json, require, sha256, write_json
from .features import fingerprint


@dataclass(frozen=True)
class BankConfig:
    max_candidates: int = 16384
    max_bank: int = 2048
    projection_dim: int = 32
    seed: int = 42

    def __post_init__(self):
        require(type(self.max_candidates) is int and 1 <= self.max_candidates <= 65536, "max_candidates must be 1..65536")
        require(type(self.max_bank) is int and 1 <= self.max_bank <= min(self.max_candidates, 8192), "max_bank must be <= max_candidates and <=8192")
        require(type(self.projection_dim) is int and 1 <= self.projection_dim <= 384, "projection_dim must be 1..384")
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "invalid seed")


def checked_features(features):
    features = np.asarray(features)
    require(features.dtype == np.float32 and features.ndim in {2, 3} and min(features.shape) > 0,
            "nonempty float32 features required")
    require(np.isfinite(features).all(), "nonfinite features")
    require(np.allclose(np.linalg.norm(features, axis=-1), 1, atol=2e-4), "features must be L2 normalized")
    return features


def safe_image(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "invalid reference image path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root / "images"), "reference image escaped images directory")
    return path


def read_normal_export(root, *, allow_synthetic=False):
    root = Path(root).resolve()
    report = read_json(root / "export.json")
    snapshot = read_json(root / "collection-snapshot.json")
    require(report["status"] == "COMPLETE" and report["role"] == "normal_bank", "reviewed normal_bank export required")
    require(report["kind"] in {"real", "synthetic"}, "invalid reference kind")
    require(report["kind"] == "real" or allow_synthetic, "synthetic references require explicit synthetic mode")
    for key, other in (("collection_id", "id"), ("collection_revision", "revision"), ("product_id", "product_id"), ("kind", "kind")):
        require(report[key] == snapshot[other], "export/snapshot identity mismatch")
    require(isinstance(report["product_id"], str) and report["product_id"].strip(), "product_id required")
    require(report["items"], "normal reference export is empty")
    validate_groups(snapshot["records"])
    records = {r["id"]: r for r in snapshot["records"]}
    require(len(records) == len(snapshot["records"]), "duplicate capture identities")
    seen = set()
    for item in report["items"]:
        record = records[item["capture_id"]]
        validate_record(record, complete=True)
        require(record["reviewed"] and not record["excluded"] and record["reviewer"] and record["split"] == "train", "reference must be reviewed, included train data")
        obj = next(o for o in record["objects"] if o["id"] == item["object"]["id"])
        require(obj == item["object"] and obj["condition"] == "NORMAL" and not obj["defects"], "only reviewed NORMAL objects may enter bank")
        require(item["split"] == "train" and item["capture_session_id"] == record["capture_session_id"], "reference split/session mismatch")
        require(list(clean_crop(record, obj)) == item["crop_bounds_original"], "reference crop bounds mismatch")
        path = safe_image(root, item["file"])
        require(item["file"] not in seen, "duplicate reference file")
        seen.add(item["file"])
        require(sha256(path) == item["image_sha256"], "reference image hash mismatch")
        with Image.open(path) as image:
            bounds = item["crop_bounds_original"]
            require(image.mode == "RGB" and image.size == (bounds[2]-bounds[0], bounds[3]-bounds[1]) and image.getexif().get(274, 1) == 1,
                    "reference image geometry/mode changed")
    return root, report, snapshot


def coreset(features, config):
    """Farthest-first traversal in a seeded random projection; retain full vectors."""
    checked_features(features)
    count = min(config.max_bank, len(features))
    if count == len(features): return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    dimension = min(config.projection_dim, features.shape[1])
    projection = rng.standard_normal((features.shape[1], dimension)).astype(np.float32) / np.float32(np.sqrt(dimension))
    points = features @ projection
    chosen = np.empty(count, dtype=np.int64)
    nearest = np.full(len(features), np.inf, dtype=np.float32)
    next_index = int(rng.integers(len(features)))
    for index in range(count):
        chosen[index] = next_index
        distance = np.square(points-points[next_index]).sum(axis=1)
        np.minimum(nearest, distance, out=nearest)
        nearest[chosen[:index+1]] = -1
        next_index = int(nearest.argmax())
    return chosen


def build_bank(normal_export, output, extractor, *, config=BankConfig(), allow_synthetic=False):
    root, report, snapshot = read_normal_export(normal_export, allow_synthetic=allow_synthetic)
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(root), "use a new bank directory outside the export")
    rng = np.random.default_rng(config.seed)
    pool = keys = origins = None
    grid_shape = None
    total = 0
    for image_index, item in enumerate(report["items"]):
        path = safe_image(root, item["file"])
        require(sha256(path) == item["image_sha256"], "reference changed before feature extraction")
        with Image.open(path) as image:
            grid = checked_features(extractor.extract(np.asarray(image).copy()))
        require(grid.ndim == 3, "extractor must return spatial features")
        if grid_shape is None: grid_shape = list(grid.shape)
        require(list(grid.shape) == grid_shape, "feature grid changed during registration")
        features = grid.reshape(-1, grid.shape[-1])
        references = np.column_stack((np.full(len(features), image_index), np.arange(len(features)))).astype(np.int64)
        priorities = rng.random(len(features))
        pool = features.copy() if pool is None else np.concatenate((pool, features))
        origins = references if origins is None else np.concatenate((origins, references))
        keys = priorities if keys is None else np.concatenate((keys, priorities))
        if len(pool) > config.max_candidates:
            keep = np.argsort(keys, kind="stable")[:config.max_candidates]
            pool, origins, keys = pool[keep], origins[keep], keys[keep]
        total += len(features)
    indices = coreset(pool, config)
    memory, provenance = pool[indices].copy(), origins[indices].copy()
    # Recheck both manifests; a concurrent replacement cannot silently change provenance.
    source_hashes = {name: sha256(root / name) for name in ("export.json", "collection-snapshot.json")}
    require(read_json(root / "export.json") == report and read_json(root / "collection-snapshot.json") == snapshot,
            "normal export changed during registration")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ("." + output.name + "-building-" + uuid4().hex[:8])
    staging.mkdir()
    np.save(staging / "features.npy", memory, allow_pickle=False)
    np.save(staging / "origins.npy", provenance, allow_pickle=False)
    write_json(staging / "source-export.json", report)
    write_json(staging / "source-collection.json", snapshot)
    meta = {"schema_version": 1, "bank_id": uuid4().hex, "created_at_utc": utc(), "status": "COMPLETE",
            "product_id": report["product_id"], "kind": report["kind"], "production_ready": False,
            "feature_signature": extractor.signature, "feature_fingerprint": fingerprint(extractor.signature),
            "grid_shape": grid_shape, "config": asdict(config), "source_hashes": source_hashes,
            "source_images": len(report["items"]), "total_patches": total, "candidate_patches": len(pool),
            "bank_patches": len(memory), "selection": "seeded uniform-priority reservoir then projected farthest-first",
            "metric": "euclidean_on_L2_normalized_vectors", "image_score": "max_patch_distance",
            "thresholds": None, "files": {name: sha256(staging / name) for name in
                ("features.npy", "origins.npy", "source-export.json", "source-collection.json")}}
    write_json(staging / "bank.json", meta)
    Bank(staging, extractor.signature, product_id=report["product_id"], allow_synthetic=allow_synthetic)
    require(not output.exists(), "bank output created concurrently")
    staging.rename(output)
    return meta


class Bank:
    def __init__(self, directory, signature, *, product_id, allow_synthetic=False):
        self.root = Path(directory).resolve()
        self.meta = read_json(self.root / "bank.json")
        meta = self.meta
        require(meta["schema_version"] == 1 and meta["status"] == "COMPLETE", "incomplete or unsupported bank")
        require(meta["product_id"] == product_id and product_id, "bank product mismatch")
        require(meta["kind"] in {"real", "synthetic"} and (meta["kind"] == "real" or allow_synthetic), "synthetic bank requires explicit synthetic mode")
        require(meta["feature_signature"] == signature and meta["feature_fingerprint"] == fingerprint(signature), "feature/model/preprocessing version mismatch")
        require(meta["metric"] == "euclidean_on_L2_normalized_vectors" and meta["image_score"] == "max_patch_distance", "unsupported scoring method")
        config = BankConfig(**meta["config"])
        shape = meta["grid_shape"]
        require(isinstance(shape, list) and len(shape) == 3 and all(type(x) is int and x > 0 for x in shape)
                and max(shape[:2]) <= 64 and shape[2] <= 384, "invalid bank grid shape")
        require(type(meta["bank_patches"]) is int and 1 <= meta["bank_patches"] <= config.max_bank,
                "bank patch count exceeds configured limit")
        require(type(meta["source_images"]) is int and meta["source_images"] > 0, "invalid bank source count")
        for name in ("features.npy", "origins.npy", "source-export.json", "source-collection.json"):
            require(sha256(self.root / name) == meta["files"][name], f"bank integrity mismatch: {name}")
        self.features = checked_features(np.load(self.root / "features.npy", allow_pickle=False))
        self.origins = np.load(self.root / "origins.npy", allow_pickle=False)
        require(self.features.ndim == 2 and self.features.shape == (meta["bank_patches"], meta["grid_shape"][2]), "bank feature shape mismatch")
        require(self.origins.dtype == np.int64 and self.origins.shape == (len(self.features), 2), "bank provenance shape mismatch")
        require((self.origins >= 0).all() and (self.origins[:, 0] < meta["source_images"]).all()
                and (self.origins[:, 1] < meta["grid_shape"][0]*meta["grid_shape"][1]).all(), "bank provenance index mismatch")
        self.features.setflags(write=False)
        self.origins.setflags(write=False)
        self.digest = fingerprint(meta)
