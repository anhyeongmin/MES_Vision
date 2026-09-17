"""Generate a clearly labelled simulation: three objects, multiple defects, one error."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw
from mes_vision.inputs import ImageSource
from mes_vision.inspection import Box, CheckResult, CheckStatus, Detection, Finding, InspectionPipeline, Mode
from mes_vision.inspection.adapters import MockDetector, MockInspector
from mes_vision.inspection.geometry import extract_crop


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (1000, 540), "#eef2f7")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "SIMULATION | 3 OBJECTS | NO FINAL OK/NG DECISION", fill="#14253b", font_size=24)
    draw.text((28, 56), "Fixed mock detections and checks. Not AI defect accuracy.", fill="#52677e", font_size=18)
    boxes = (Box(60, 150, 260, 380), Box(390, 150, 590, 380), Box(720, 150, 920, 380))
    for box in boxes:
        draw.rounded_rectangle((box.x1, box.y1, box.x2-1, box.y2-1), radius=16, fill="#346cb3")
        draw.ellipse((box.x1+70, box.y1+70, box.x1+130, box.y1+130), fill="#eef2f7")
    draw.line([(415, 300), (448, 315), (435, 350)], fill="#172030", width=4)
    draw.rectangle((530, 320, 559, 339), fill="#9ab8db")
    draw.rectangle((800, 325, 835, 350), fill="#506583")
    input_path = args.output / "synthetic-input.png"
    image.save(input_path)
    with ImageSource(input_path) as source:
        frame = source.read().frame
    def known(crop, model):
        if crop.object_id.endswith("0003"):
            raise RuntimeError("SIMULATED_KNOWN_MODEL_FAILURE")
        findings = ()
        status = CheckStatus.PASS
        if crop.object_id.endswith("0002"):
            findings = (Finding("crack", "NG03", .94, Box(25, 150, 60, 202)),
                        Finding("surface defect", "NG06", .89, Box(140, 170, 170, 190)))
            status = CheckStatus.FAIL
        return CheckResult("known_defects", crop.frame_id, crop.object_id, status, model, findings,
                           messages=("SIMULATED_CHECK",), criteria_version="demo-only-v1")
    def anomaly(crop, model):
        suspicious = crop.object_id.endswith("0003")
        return CheckResult("anomaly", crop.frame_id, crop.object_id,
                           CheckStatus.UNCERTAIN if suspicious else CheckStatus.PASS, model,
                           findings=(Finding("unclassified appearance", crop_box=Box(80, 175, 115, 200)),) if suspicious else (),
                           messages=("SIMULATED_SCORE_NOT_PROBABILITY",), raw_score=8.1 if suspicious else .2,
                           criteria_version="demo-only-v1")
    pipe = InspectionPipeline(MockDetector(tuple(Detection(box, .99, 0, "demo_part") for box in boxes)),
                              (MockInspector("known_defects", known), MockInspector("anomaly", anomaly)),
                              mode=Mode.SIMULATION, expected_count=3)
    report = pipe.run(frame)
    (args.output / "result.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    labels = ["OBJ 1 | mock PASS", "OBJ 2 | NG03 + NG06", "OBJ 3 | ERROR + UNCERTAIN"]
    colors = ["#237792", "#c14530", "#a67500"]
    for i, item in enumerate(report.objects):
        box = item.effective_box
        draw.rectangle((box.x1, box.y1, box.x2-1, box.y2-1), outline=colors[i], width=4)
        draw.text((box.x1-12, 408), labels[i], fill=colors[i], font_size=18)
        crop = extract_crop(frame, item.object_id, item.detection.box)
        Image.fromarray(crop.rgb).save(args.output / f"object-{i+1}.png")
        for check in item.checks:
            for finding in check.findings:
                if finding.original_box:
                    evidence = finding.original_box
                    draw.rectangle((evidence.x1, evidence.y1, evidence.x2-1, evidence.y2-1), outline="#ffe28d", width=2)
    draw.text((28, 477), "All 3 objects received anomaly checks. Missing geometry/VLM remain NOT_RUN.", fill="#344b62", font_size=18)
    draw.text((28, 507), "Coordinates: original RGB pixels. No robot commands or grasp coordinates.", fill="#344b62", font_size=16)
    overlay.save(args.output / "overlay.png")
    print(json.dumps({"mode": report.mode, "objects": len(report.objects), "status": report.execution_status,
                      "final_decision": report.final_decision, "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
