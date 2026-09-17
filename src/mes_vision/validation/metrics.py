"""Full-scene evaluation. Missing/error frames cannot disappear from denominators."""
import math
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from mes_vision.inspection import Box
from mes_vision.inspection.contracts import DEFECT_CODES
from mes_vision.training.data import require

CONDITIONS = ("NORMAL", "KNOWN_NG", "UNKNOWN_NG", "UNCERTAIN")
OUTCOMES = ("OK", "NG", "REVIEW", "MISSED", "ERROR")


def distribution(values):
    values = list(values)
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values), "Invalid timing sample")
    if not values:
        return {"n": 0, **{k: None for k in ("mean", "median", "p95", "p99", "max")}}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "max": max(values)}


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def match_boxes(truth, predictions, threshold=.5):
    require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 < threshold <= 1, "Invalid IoU threshold")
    for box in [*truth, *predictions]: Box(*box)
    if not truth or not predictions: return []
    overlaps = np.array([[iou(a, b) for b in predictions] for a in truth])
    # One additional valid match outweighs the sum of all IoU tie breakers.
    rewards = np.where(overlaps >= threshold, min(len(truth), len(predictions)) + 1 + overlaps, 0)
    rows, cols = linear_sum_assignment(rewards, maximize=True)
    return [(int(r), int(c), float(overlaps[r, c])) for r, c in zip(rows, cols) if overlaps[r, c] >= threshold]


def _index(frames):
    result = {}
    for frame in frames:
        key = frame["capture_id"]
        require(isinstance(key, str) and key.strip() and key not in result, "Missing/duplicate capture ID")
        ids = set()
        for obj in frame["objects"]:
            require(isinstance(obj["id"], str) and obj["id"] and obj["id"] not in ids, "Missing/duplicate object ID")
            ids.add(obj["id"]); Box(*obj["bbox"])
            codes = obj.get("codes", [])
            require(isinstance(codes, list) and len(set(codes)) == len(codes) and all(c in DEFECT_CODES for c in codes), "Invalid defect codes")
        result[key] = frame
    return result


def evaluate(truth_frames, prediction_frames, *, threshold=.5):
    truth, predicted = _index(truth_frames), _index(prediction_frames)
    require(truth and truth.keys() == predicted.keys(), "Evaluation requires exactly one outcome for every selected frame")
    match_boxes([], [], threshold)  # Validate even for entirely empty scenes.
    confusion = {c: dict.fromkeys(OUTCOMES, 0) for c in CONDITIONS}
    rows, false_positives, matched, missed, error_objects, predictions, failed_frames = [], 0, 0, 0, 0, 0, 0
    code_total, code_found = Counter(), Counter()
    timings = []
    for key, expected in truth.items():
        actual = predicted[key]
        require(actual["status"] in ("COMPLETED", "ERROR"), "Missing frame execution status")
        require(actual["status"] != "ERROR" or actual.get("error"), "Failed frame requires an error reason")
        for obj in expected["objects"]:
            require(obj["condition"] in CONDITIONS, "Unreviewed/invalid ground truth")
            require(obj["condition"] != "NORMAL" or not obj.get("codes"), "Normal truth cannot have defect codes")
            require(obj["condition"] != "KNOWN_NG" or obj.get("codes"), "Known NG truth needs codes")
        for obj in actual["objects"]:
            require(obj["decision"] in OUTCOMES[:3], "Unknown prediction decision; do not silently convert to OK")
            require(obj["decision"] == "NG" or not obj.get("codes"), "Only final NG codes belong in this report")
        require(actual["status"] != "ERROR" or not actual["objects"], "Failed frames must not expose partial predictions")
        timings.append(actual["elapsed_ms"])
        pairs = match_boxes([o["bbox"] for o in expected["objects"]], [o["bbox"] for o in actual["objects"]], threshold)
        assignments = {r: (c, overlap) for r, c, overlap in pairs}
        matched += len(pairs); predictions += len(actual["objects"])
        failed_frames += actual["status"] == "ERROR"
        for i, obj in enumerate(expected["objects"]):
            pair = assignments.get(i)
            candidate = actual["objects"][pair[0]] if pair else None
            outcome = candidate["decision"] if candidate else "ERROR" if actual["status"] == "ERROR" else "MISSED"
            confusion[obj["condition"]][outcome] += 1
            missed += outcome == "MISSED"; error_objects += outcome == "ERROR"
            codes = obj.get("codes", [])
            for code in codes:
                code_total[code] += 1
                code_found[code] += bool(candidate and candidate["decision"] == "NG" and code in candidate.get("codes", []))
            rows.append({"capture_id": key, "truth_id": obj["id"], "prediction_id": candidate["id"] if candidate else None,
                         "condition": obj["condition"], "outcome": outcome, "iou": pair[1] if pair else None,
                         "expected_codes": codes, "predicted_codes": candidate.get("codes", []) if candidate else []})
        for i, obj in enumerate(actual["objects"]):
            if i not in {c for _, c, _ in pairs}:
                false_positives += 1
                rows.append({"capture_id": key, "truth_id": None, "prediction_id": obj["id"], "condition": "NO_OBJECT",
                             "outcome": obj["decision"], "iou": None, "expected_codes": [], "predicted_codes": obj.get("codes", [])})
    total = sum(sum(r.values()) for r in confusion.values())
    normal = confusion["NORMAL"]; ng = {k: confusion["KNOWN_NG"][k]+confusion["UNKNOWN_NG"][k] for k in OUTCOMES}
    return {"schema_version": 1, "iou_threshold": threshold, "frames": len(truth), "failed_frames": failed_frames,
            "truth_objects": total, "predicted_objects": predictions, "matched": matched, "missed": missed,
            "unassessed_due_to_error": error_objects, "false_positives": false_positives,
            "detection_recall_all_selected": rate(matched, total), "detection_precision": rate(matched, predictions),
            "ng_passed_as_ok": rate(ng["OK"], sum(ng.values())), "normal_rejected_as_ng": rate(normal["NG"], sum(normal.values())),
            "truth_review_rate": rate(sum(r["REVIEW"] for r in confusion.values()), total),
            "unknown_ng_rejected": rate(confusion["UNKNOWN_NG"]["NG"], sum(confusion["UNKNOWN_NG"].values())),
            "confusion": confusion, "defect_code_recall": {c: rate(code_found[c], n) for c, n in sorted(code_total.items())},
            "frame_elapsed_ms": distribution(timings), "objects": rows,
            "acceptance": "NOT_ASSESSED", "notes": ["UNCERTAIN truth is reported separately, never treated as normal.",
                "NG escape, MISSED and ERROR are separate outcomes; low NG-to-OK alone is not success.",
                "Object occurrences in images are the unit; repeated images of one specimen are not independent samples.",
                "No defect localization mAP, tracking accuracy or robot success is implied."]}
