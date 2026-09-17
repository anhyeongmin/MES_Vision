"""Export matched synthetic comparison figures from saved predictions; no inference."""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ash_reinforcement import BATCH, LABELS, assert_matched
from mes_vision.synthetic.detection_review import match_image
from mes_vision.training.data import read_json, write_json, require


def plot_curve(review):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (BATCH / "detail-defects/engine/metrics.csv").open(encoding="utf-8") as stream:
        history = [(int(float(r["epoch"]))+1, float(r["val/mAP_50_95"]))
                   for r in csv.DictReader(stream) if r.get("val/mAP_50_95")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(*zip(*history), marker="o", color="#087f8c")
    best = max(history, key=lambda r: r[1])
    axes[0].scatter(*best, color="#e67e22", s=90, zorder=3, label=f"Selected epoch {best[0]}")
    axes[0].set(xlabel="Completed epoch", ylabel="Validation mAP 50:95", title="Validation only; test excluded from selection")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for offset, name, color in ((-.18, "baseline", "#778899"), (.18, "reinforced", "#087f8c")):
        report = read_json(review / name / "predicted_crop_chain-fixed-threshold-counts.json")
        bars = axes[1].bar([i+offset for i in range(6)], [report["classes"][k]["tp"] for k in LABELS],
                          width=.35, label=name, color=color)
        axes[1].bar_label(bars, fontsize=9)
    axes[1].set(xticks=list(range(6)), xticklabels=LABELS, ylim=(0, 20), ylabel="Matched defects / 18 per class",
                title="Same new test; predicted crop; fixed threshold 0.85")
    axes[1].legend(loc="lower right")
    fig.suptitle("SYNTHETIC ONLY - same 7 CAD shapes, new rendering conditions", fontsize=12)
    fig.savefig(review / "learning-and-comparison.png", dpi=150)
    plt.close(fig)


def paired_review(review, *, selected_thresholds=False):
    from PIL import Image, ImageDraw, ImageFont
    a = read_json(review / "baseline/predicted_crop_chain-test-predictions.json")
    b = read_json(review / "reinforced/predicted_crop_chain-test-predictions.json")
    assert_matched(a, b)
    thresholds = [.85, .85]
    if selected_thresholds:
        selections = read_json(review / "thresholds.json")["models"]
        thresholds = [selections[name]["predicted_crop_chain"] for name in ("baseline", "reinforced")]
    def error(row, threshold):
        result = match_image(row, threshold)
        return len(result["fp"]) + len(result["fn"])
    pairs = sorted(zip(a, b), key=lambda pair: (-error(pair[1], thresholds[1]), -error(pair[0], thresholds[0]), pair[0]["file_name"]))
    selected, seen = [], set()
    for old, new in pairs:
        label = old["targets"][0]["label"] if old["targets"] else "OK"
        if label not in seen:
            selected.append((old, new))
            seen.add(label)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    sheet = Image.new("RGB", (1000, 100 + 225*len(selected)), (22, 25, 30))
    draw = ImageDraw.Draw(sheet)
    title = f"Validation thresholds {thresholds[0]:.2f} / {thresholds[1]:.2f}" if selected_thresholds else "Same threshold 0.85"
    draw.text((12, 10), f"SYNTHETIC TEST | {title} | Green: GT | Red: prediction", fill="white", font=font)
    draw.text((12, 35), "Predicted-object crop chain; zoomed display. One example/class; reinforced errors first.", fill="white", font=small)
    draw.text((12, 66), "BASELINE", fill="white", font=font)
    draw.text((510, 66), "REINFORCED", fill="white", font=font)
    full = {r["file_name"]: r for r in read_json(review / "full-frames/test/_annotations.coco.json")["images"]}
    for index, pair in enumerate(selected):
        info = full[pair[0]["file_name"]]
        x1, y1, x2, y2 = info["object_box"]
        window = (max(0, x1-35), max(0, y1-35), min(info["width"], x2+35), min(info["height"], y2+35))
        for column, row in enumerate(pair):
            threshold = thresholds[column]
            with Image.open(review / "full-frames/test" / row["file_name"]) as source:
                im = source.convert("RGB")
            overlay = ImageDraw.Draw(im)
            for target in row["targets"]:
                overlay.rectangle(target["box"], outline=(50, 255, 90), width=3)
            for p in row["predictions"]:
                if p["score"] > threshold:
                    overlay.rectangle(p["box"], outline=(255, 70, 70), width=2)
                    overlay.text((p["box"][0], max(0, p["box"][1]-18)), f"{p['label']} {p['score']:.2f}", fill=(255, 70, 70), font=small)
            im = im.crop(window)
            im.thumbnail((480, 187))
            x, y = column*500+10, index*225+100
            sheet.paste(im, (x+(480-im.width)//2, y))
            result = match_image(row, threshold)
            label = row["file_name"].replace("reinforce-test-", "").replace(".png", "")
            draw.text((x, y+190), f"{label} | TP {len(result['tp'])} FP {len(result['fp'])} FN {len(result['fn'])}", fill="white", font=small)
    stem = "paired-chain-selected" if selected_thresholds else "paired-chain-review"
    sheet.save(review / (stem + ".png"))
    write_json(review / (stem + "-selection.json"), {"thresholds": thresholds,
        "selection": "one example/class; reinforced errors first then baseline errors; not a random sample",
        "images": [a["file_name"] for a, _ in selected]})


if __name__ == "__main__":
    review = BATCH / "review"
    require(read_json(review / "summary.json")["status"] == "COMPLETED", "comparison incomplete")
    plot_curve(review)
    paired_review(review)
    paired_review(review, selected_thresholds=True)
    print("Saved learning curve and matched image review")
