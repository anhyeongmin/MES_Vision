from dataclasses import asdict
from pathlib import Path

from mes_vision.data_management.fixtures import make_fixture
from mes_vision.data_management.export import export_collection
from mes_vision.inputs import ImageSource
from mes_vision.inspection import Box, Detection, CheckResult, CheckStatus, Finding, InspectionPipeline, Mode
from mes_vision.inspection.adapters import MockDetector, MockInspector
from .snapshots import save_snapshot
from .backend import GenerationConfig
from .queue import AnalysisQueue


def make_vlm_fixture(root, *, enabled=False, generation=None):
    root = Path(root).resolve()
    collection = make_fixture(root / "source")
    export_collection(collection, root / "normal-reference", "normal_bank")
    record = collection.data["records"][0]
    with ImageSource(collection.image_path(record)) as source:
        frame = source.read().frame
    def known(crop, model):
        bad = crop.object_id.endswith("0002")
        return CheckResult("known_defects", crop.frame_id, crop.object_id, CheckStatus.FAIL if bad else CheckStatus.PASS, model,
            findings=(Finding("균열", "NG03", .9, Box(15, 15, 45, 70)),) if bad else (), criteria_version="SYNTHETIC-ONLY-v1")
    pipeline = InspectionPipeline(MockDetector(tuple(Detection(Box(*o["bbox"]), .99, 0, "fixture-part") for o in record["objects"])),
                                  (MockInspector("known_defects", known),), mode=Mode.SIMULATION, product_id="fixture-part")
    result = pipeline.run(frame)
    snapshot = root / "snapshot"
    save_snapshot(snapshot, frame, result, kind="synthetic", normal_export=root / "normal-reference",
                  criteria={"version": "SYNTHETIC-DEMO", "description": "Compare visible marks with the normal reference. Do not infer manufacturing quality."})
    queue = AnalysisQueue(root / "queue")
    queue.set_enabled(enabled)
    config = generation or asdict(GenerationConfig())
    identity = queue.enqueue(snapshot, result.objects[1].object_id, config)
    return {"queue": queue, "snapshot": snapshot, "job_id": identity, "result": result, "frame": frame}
