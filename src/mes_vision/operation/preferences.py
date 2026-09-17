"""Versioned application preferences, independent of product/equipment and operating authority."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from filelock import FileLock
from mes_vision.i18n import LANGUAGES

DEFAULTS = {"schema_version": 1, "language": "ko", "text_scale": 100, "start_page": 0,
    "remember_window": True, "window_geometry": "", "theme": "light"}


def validate(value):
    # Additive migration: existing v1 files preserve language, scale and geometry.
    if isinstance(value, dict) and set(value) == set(DEFAULTS) - {'theme'}:
        value = dict(value, theme='light')
    if not isinstance(value, dict) or set(value) != set(DEFAULTS): raise ValueError("Invalid preference fields")
    if value['theme'] not in ('light', 'dark'): raise ValueError('Unsupported theme')
    if type(value["schema_version"]) is not int or value["schema_version"] != 1: raise ValueError("Unsupported preference version")
    if not isinstance(value["language"], str) or value["language"] not in LANGUAGES: raise ValueError("Unsupported display language")
    if type(value["text_scale"]) is not int or value["text_scale"] not in {100, 115, 130}: raise ValueError("Invalid text size")
    if type(value["start_page"]) is not int or not 0 <= value["start_page"] < 5: raise ValueError("Invalid start page")
    if type(value["remember_window"]) is not bool: raise ValueError("Invalid window preference")
    geometry = value["window_geometry"]
    if not isinstance(geometry, str) or len(geometry) > 8192: raise ValueError("Invalid window geometry")
    try: bytes.fromhex(geometry)
    except ValueError as exc: raise ValueError("Invalid window geometry") from exc
    return deepcopy(value)


class Preferences:
    def __init__(self, runtime):
        self.path = Path(runtime) / "preferences.json"
        self.value = deepcopy(DEFAULTS); self.warning = None
        try: raw = self.path.read_bytes() if self.path.exists() else None
        except OSError as exc:
            self.warning = str(exc); self.digest = None; return
        self.digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
        if raw is not None:
            try: self.value = validate(json.loads(raw))
            except (ValueError, TypeError) as exc: self.warning = str(exc)

    def save(self, value):
        value = validate(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path)+".lock", timeout=3):
            raw = self.path.read_bytes() if self.path.exists() else None
            digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
            if digest != self.digest: raise ValueError("Preferences changed in another window. Reopen Settings.")
            if self.warning and raw is not None:
                backup = self.path.with_name("preferences-invalid-"+uuid4().hex+".json")
                backup.write_bytes(raw)
            encoded = (json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
            temporary = self.path.with_name(".preferences-"+uuid4().hex+".tmp")
            try:
                temporary.write_bytes(encoded); temporary.replace(self.path)
            finally:
                if temporary.exists(): temporary.unlink()
            self.value = value; self.digest = hashlib.sha256(encoded).hexdigest(); self.warning = None

    def save_geometry(self, geometry):
        # Merge with newly saved preferences so closing never undoes a language change.
        current = Preferences(self.path.parent)
        if current.warning or not current.value["remember_window"]: return
        value = deepcopy(current.value); value["window_geometry"] = geometry
        current.save(value); self.value = current.value; self.digest = current.digest
