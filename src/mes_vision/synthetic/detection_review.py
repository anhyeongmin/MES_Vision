"""Diagnostic one-to-one detection counts; never an inspection PASS policy."""


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_image(row, threshold, overlap=0.5):
    """Score-ordered, class-aware greedy matching at the declared IoU."""
    targets = row["targets"]
    predictions = sorted((p for p in row["predictions"] if p["score"] > threshold), key=lambda p: p["score"], reverse=True)
    matched, pairs, false_positives = set(), [], []
    for p in predictions:
        choices = [(iou(p["box"], t["box"]), i) for i, t in enumerate(targets)
                   if i not in matched and t["label"] == p["label"]]
        best, index = max(choices, default=(0, -1))
        if best >= overlap and index >= 0:
            matched.add(index)
            pairs.append({"label": p["label"], "iou": best, "score": p["score"]})
        else:
            false_positives.append(p)
    return {"tp": pairs, "fp": false_positives, "fn": [t for i, t in enumerate(targets) if i not in matched]}


def summarize(rows, threshold, labels):
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    errors, negative_images, false_alarm_images, missing_detection_images = [], 0, 0, 0
    for row in rows:
        result = match_image(row, threshold)
        for key in ("tp", "fp", "fn"):
            for item in result[key]:
                counts[item["label"]][key] += 1
        selected = [p for p in row["predictions"] if p["score"] > threshold]
        if not row["targets"]:
            negative_images += 1
            false_alarm_images += bool(selected)
        elif not selected:
            missing_detection_images += 1
        if result["fp"] or result["fn"]:
            errors.append({"file_name": row["file_name"], **result})

    def rates(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return {**c, "precision": tp/(tp+fp) if tp+fp else None,
                "recall": tp/(tp+fn) if tp+fn else None,
                "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}

    total = rates({key: sum(c[key] for c in counts.values()) for key in ("tp", "fp", "fn")})
    return {"threshold": threshold, "iou": 0.5, "images": len(rows), "total": total,
            "classes": {label: rates(c) for label, c in counts.items()},
            "negative_images": negative_images, "false_alarm_images": false_alarm_images,
            "positive_images_with_no_detection": missing_detection_images, "errors": errors}


def select_threshold(validation, labels):
    candidates = [summarize(validation, i/100, labels) for i in range(5, 96, 5)]
    # Equal F1: prefer recall, then precision, then the lower threshold.
    best = max(candidates, key=lambda r: (r["total"]["f1"], r["total"]["recall"] or 0,
                                          r["total"]["precision"] or 0, -r["threshold"]))
    return best["threshold"], [{k: v for k, v in r.items() if k != "errors"} for r in candidates]
