"""Fetch official RF-DETR Small weights and retain their license/provenance."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.9.4"
URL = "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"
EXPECTED_MD5 = "fb37061c1af7bace359c91b723a8d5c1"
WEIGHTS = ROOT / "models" / "rf-detr-small.pth"

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    return partial

def main():
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        print("Downloading official RF-DETR Small checkpoint...", flush=True)
        partial = fetch(URL, WEIGHTS)
        if digest(partial, "md5") != EXPECTED_MD5:
            raise RuntimeError("Official checkpoint checksum mismatch; refusing to load it.")
        partial.replace(WEIGHTS)
    if digest(WEIGHTS, "md5") != EXPECTED_MD5:
        raise RuntimeError("Existing checkpoint does not match the pinned official asset.")
    base = f"https://raw.githubusercontent.com/roboflow/rf-detr/{VERSION}/"
    sources = {
        "RF-DETR-LICENSE.txt": base + "LICENSE",
        "RF-DETR-README.md": base + "README.md",
        "RF-DETR-model_weights.py.txt": base + "src/rfdetr/assets/model_weights.py",
    }
    records = []
    for filename, url in sources.items():
        target = ROOT / "licenses" / filename
        if not target.exists():
            fetch(url, target).replace(target)
        records.append({"file": str(target.relative_to(ROOT)), "source": url, "sha256": digest(target, "sha256")})
    manifest = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "RF-DETR Small (Detection)", "rfdetr_version": VERSION,
        "weights_file": str(WEIGHTS.relative_to(ROOT)), "source_url": URL,
        "bytes": WEIGHTS.stat().st_size, "md5": digest(WEIGHTS, "md5"),
        "sha256": digest(WEIGHTS, "sha256"), "license": "Apache-2.0",
        "license_scope": "Official Small detection checkpoint and rfdetr code; not Plus components or the entire dependency stack.",
        "pretraining": "Official COCO checkpoint; NOT trained for the project's OK/NG classes.",
        "sources": records,
    }
    (ROOT / "models" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == "__main__":
    main()
