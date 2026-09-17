from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mes_vision.training.data import read_json, require, sha256


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_rgb(rgb):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            and min(rgb.shape[:2]) > 0, "8-bit original RGB image required")


class DinoFeatures:
    """Frozen FP32 last-layer spatial tokens; no remote code or default center crop."""
    def __init__(self, project_root, *, device="cuda", image_size=448, max_batch_size=None):
        require(type(image_size) is int and 112 <= image_size <= 896 and image_size % 14 == 0,
                "feature image_size must be a multiple of 14 in [112, 896]")
        require(device in {"cuda", "cpu"}, "device must be cuda or cpu")
        if max_batch_size is None: max_batch_size=4 if device=='cuda' else 1
        require(type(max_batch_size) is int and max_batch_size in (1,2,4),'Invalid DINO batch limit')
        self.max_batch_size=max_batch_size
        self.root = Path(project_root).resolve()
        self.directory = self.root / "models/dinov2-small"
        self.device, self.image_size, self.model = device, image_size, None
        self.signature = {"implementation": "mes-dino-normal-reference-v1", "assets": self.verify_assets(),
            "transformers": importlib.metadata.version("transformers"), "torch": importlib.metadata.version("torch"),
            "pillow": importlib.metadata.version("pillow"), "dtype": "float32",
            "preprocessing": {"size": image_size, "resize": "full_rgb_square_PIL_BICUBIC", "center_crop": False,
                "padding": False, "mean": [.485, .456, .406], "std": [.229, .224, .225],
                "patch_size": 14, "features": "last_hidden_state_without_CLS_L2_normalized", "alignment": "none"}}
        if max_batch_size>1: self.signature['batch_limit']=max_batch_size

    def verify_assets(self):
        lock = read_json(self.root / "models/selection-lock.json")
        assets = {}
        for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
            asset = next(a for a in lock["assets"] if a["id"] == "dinov2/" + name)
            path = self.directory / name
            require(path.stat().st_size == asset["size"], f"DINO asset size mismatch: {name}")
            digest = sha256(path)
            if asset.get("sha256"):
                require(digest == asset["sha256"], f"DINO SHA256 mismatch: {name}")
            else:
                blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
                with path.open("rb") as stream:
                    for part in iter(lambda: stream.read(1024 * 1024), b""):
                        blob.update(part)
                require(blob.hexdigest() == asset.get("git_blob_sha1"), f"DINO Git blob mismatch: {name}")
            assets[name] = digest
        return assets

    def load(self):
        if self.model is not None: return
        import torch
        from transformers import AutoModel
        require(self.verify_assets() == self.signature["assets"], "DINO assets changed before load")
        require(self.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable; choose cpu explicitly")
        model = AutoModel.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True).float().to(self.device).eval()
        require(model.config.model_type == "dinov2" and model.config.patch_size == 14 and model.config.hidden_size == 384,
                "Only the pinned plain DINOv2 Small patch layout is supported")
        self.model = model

    def extract(self, rgb):
        return self.extract_many((rgb,))[0]

    def extract_many(self, images):
        return tuple(value.cpu().numpy().copy() for value in self.extract_many_device(images))

    def extract_many_device(self, images):
        """Same FP32 features, retained on the model device for immediate scoring."""
        images=tuple(images)
        require(len(images)<=4,'At most four current images per DINO request')
        for rgb in images: check_rgb(rgb)
        if not images: return ()
        self.load()
        outputs=[]
        for start in range(0,len(images),self.max_batch_size):
            outputs.extend(self._extract_batch_device(images[start:start+self.max_batch_size]))
        return tuple(outputs)

    def _extract_batch_device(self, images):
        import torch
        from torch.nn.functional import normalize
        resized=np.stack([np.asarray(Image.fromarray(rgb).resize((self.image_size,self.image_size),Image.Resampling.BICUBIC)) for rgb in images])
        pixels = torch.from_numpy(resized).permute(0,3,1,2).to(self.device, dtype=torch.float32) / 255
        mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            tokens = self.model(pixel_values=(pixels-mean)/std).last_hidden_state[:, 1:]
            side = self.image_size // 14
            require(tokens.shape == (len(images), side*side, 384) and torch.isfinite(tokens).all().item(), "Invalid DINO spatial tokens")
            require(torch.all(torch.linalg.vector_norm(tokens, dim=-1) > 1e-8).item(), "Degenerate DINO feature")
            values=normalize(tokens, dim=-1).reshape(len(images),side,side,384)
            return tuple(values[i] for i in range(len(images)))

    def close(self):
        self.model = None
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
