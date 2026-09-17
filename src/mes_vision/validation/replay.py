"""Resident full-scene replay using reviewed held-out collection images."""
from copy import deepcopy
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
import csv
import json
import sqlite3
import time

from mes_vision.data_management.collection import Collection, validate_groups, validate_record
from mes_vision.decision import FrameEvidence, apply_policy
from mes_vision.inputs import ImageSource, InputStatus
from mes_vision.inspection import CheckStatus, InspectionPipeline, Mode
from mes_vision.inspection.contracts import DetectionBatch
from mes_vision.operation.engine import Models, FrozenDetector
from mes_vision.operation.quality import quality, inside, in_workspace
from mes_vision.training.data import require, sha256, write_json
from mes_vision.vlm.snapshots import save_snapshot
from .metrics import evaluate


def session_configuration(runtime, session_id):
    """Read a frozen session without initializing or migrating the operating database."""
    database = Path(runtime).resolve() / "operation.sqlite3"
    with closing(sqlite3.connect(database.as_uri()+"?mode=ro", uri=True)) as db:
        row = db.execute("SELECT product,equipment FROM sessions WHERE id=?", (session_id,)).fetchone()
    require(row is not None, "Unknown operation session")
    return json.loads(row[0]), json.loads(row[1])


def select_collection(directory, splits=("test", "challenge")):
    require(splits and len(set(splits)) == len(splits) and set(splits) <= {"test", "challenge"}, "Only held-out test/challenge splits may be evaluated")
    collection = Collection(directory)
    validate_groups(collection.data["records"])
    records = []
    for record in collection.data["records"]:
        if record["excluded"] or record["split"] not in splits: continue
        require(record["reviewed"], "Selected image has not been reviewed")
        validate_record(record, complete=True)
        require(record["split"] == "challenge" or not any(o["condition"] in {"UNKNOWN_NG", "UNCERTAIN"} for o in record["objects"]),
            "Unknown/uncertain truth must remain in the challenge split")
        require(sha256(collection.image_path(record)) == record["image_sha256"], "Collection image hash mismatch")
        records.append(deepcopy(record))
    require(records, "No reviewed held-out images")
    require(len({r["id"] for r in records}) == len(records), "Duplicate capture ID")
    return collection, records


def truth_records(records):
    return [{"capture_id": r["id"], "objects": [{"id": o["id"], "bbox": o["bbox"], "condition": o["condition"],
        "codes": sorted({d["code"] for d in o["defects"]})} for o in r["objects"]]} for r in records]


class ReplayRunner:
    def __init__(self, root, product, equipment, *, models=None):
        self.product, self.equipment = deepcopy(product), deepcopy(equipment)
        self.models = models or Models(root, self.product)

    def inspect(self, frame):
        require(not frame.is_live, "Replay accepts file images only")
        p, e = self.product, self.equipment
        require((frame.width, frame.height) == (e["camera"]["width"], e["camera"]["height"]), "Image resolution differs from equipment configuration")
        batch = self.models.detector.detect(frame)
        detections = tuple(d for d in batch.detections if inside(((d.box.x1+d.box.x2)/2, (d.box.y1+d.box.y2)/2), e["workspace"]["roi"]))
        batch = DetectionBatch(batch.frame_id, batch.image_size, detections, batch.model,
            candidate_threshold=batch.candidate_threshold, excluded_reserved_slots=batch.excluded_reserved_slots)
        good, measurements = quality(frame.rgb, [d.box for d in detections], p["quality"])
        workspace_good = (bool(detections) and bool(e["workspace"]["validation_reference"])
            and p["workspace_id"] == e["id"] and all(in_workspace(d.box, e["workspace"]) for d in detections))
        # The live engine does not inspect or publish bad-quality/out-of-workspace crops.
        # Replay retains their detections as REVIEW with unexecuted checks for the denominator.
        inspectors = tuple(self.models.inspectors) if good and workspace_good else ()
        raw = InspectionPipeline(FrozenDetector(batch), inspectors, mode=Mode.MODEL_FILE,
            product_id=p["id"], expected_count=p["expected_count"] if p["count_mode"] == "fixed" else None).run(frame)
        q = p["quality"]
        evidence = FrameEvidence(raw.run_id, frame.frame_id, p["id"], "real", q["version"],
            CheckStatus.PASS if good else CheckStatus.UNCERTAIN, CheckStatus.PASS if workspace_good else CheckStatus.UNCERTAIN,
            bool(q["validation_reference"]), q["validation_reference"])
        raw.config.update(replay=True, product_version=p["version"], equipment_version=e["version"],
            quality_measurements=measurements, replay_quality_gate_passed=bool(good and workspace_good))
        return apply_policy(raw, self.models.policy, evidence)

    def run(self, collection_directory, output, *, splits=("test", "challenge"), threshold=.5, allow_synthetic=False):
        from .metrics import match_boxes
        match_boxes([], [], threshold)
        collection, records = select_collection(collection_directory, splits)
        require(collection.data["product_id"] == self.product["id"], "Collection and product do not match")
        require(collection.data["kind"] == "real" or allow_synthetic, "Synthetic collection requires explicit test opt-in")
        # Evaluate all selected labels, including those outside the configured ROI, so exclusion cannot hide misses.
        truth = truth_records(records)
        output = Path(output).resolve()
        require(not output.is_relative_to(collection.root), "Replay output must be outside the source collection")
        output.mkdir(parents=True, exist_ok=False)
        meta = {"schema_version": 1, "status": "RUNNING", "kind": collection.data["kind"], "splits": list(splits),
            "collection_id": collection.data["id"], "collection_revision": collection.data["revision"],
            "product_accuracy_validated": False, "field_acceptance": "PENDING", "robot_commands_enabled": False,
            "scope": "Independent saved full scenes; no camera timing, stabilization, tracking, VLM or robot execution"}
        write_json(output / "product.json", self.product); write_json(output / "equipment.json", self.equipment)
        write_json(output / "collection-snapshot.json", collection.data); write_json(output / "truth.json", truth)
        predictions = []
        try:
            began = time.perf_counter(); self.models.load()
            meta["load_and_warmup_seconds"] = time.perf_counter()-began
            write_json(output / "policy.json", asdict(self.models.policy))
            for index, record in enumerate(records):
                began = time.perf_counter()
                row = {"capture_id": record["id"], "image_sha256": record["image_sha256"], "objects": []}
                try:
                    path = collection.image_path(record)
                    require(sha256(path) == record["image_sha256"], "Image changed during replay")
                    with ImageSource(path) as source:
                        event = source.read(); require(event.status == InputStatus.FRAME, event.message)
                        frame = event.frame
                    result = self.inspect(frame)
                    inference_ms = (time.perf_counter()-began)*1000
                    require(not result.robot_commands_enabled, "Replay must not enable robot commands")
                    save_snapshot(output / "snapshots" / f"{index:06d}", frame, result, kind=collection.data["kind"])
                    require(result.execution_status != "ERROR", "; ".join(result.issues))
                    row.update(status="COMPLETED", inference_and_decode_ms=inference_ms,
                        objects=[{"id": o.object_id, "bbox": list(asdict(o.effective_box).values()),
                            "decision": o.final_decision, "codes": o.decision_details.get("defect_codes", []) if o.final_decision == "NG" else []} for o in result.objects])
                except Exception as exc:
                    row.update(status="ERROR", objects=[], error=f"{type(exc).__name__}: {exc}")
                row["elapsed_ms"] = (time.perf_counter()-began)*1000
                predictions.append(row)
                write_json(output / "predictions.json", predictions)
            report = evaluate(truth, predictions, threshold=threshold)
            meta["status"] = "COMPLETED_WITH_ERRORS" if report["failed_frames"] else "COMPLETED"
            report["provenance"] = meta
            write_json(output / "report.json", report)
            with (output / "objects.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                fields = ["capture_id", "truth_id", "prediction_id", "condition", "outcome", "iou", "expected_codes", "predicted_codes"]
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(report["objects"])
            return report
        except Exception as exc:
            meta.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try: self.models.close()
            finally: write_json(output / "run.json", meta)
