"""Planar pixel-to-robot XY calibration with explicit lens and holdout validation."""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
from pathlib import Path

import cv2
import numpy as np

from mes_vision.anomaly.features import fingerprint
from mes_vision.robot.contracts import number
from mes_vision.training.data import require, read_json


def text(value): return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class GeometryContext:
    camera_id: str
    mount_revision: str
    acquisition_revision: str
    robot_base_id: str
    tool_frame_id: str
    image_size: tuple[int, int]
    transformations: tuple[str, ...] = ()

    def __post_init__(self):
        require(all(text(v) for v in (self.camera_id, self.mount_revision, self.acquisition_revision, self.robot_base_id, self.tool_frame_id)), "camera/mount/acquisition/base/tool identity required")
        require(isinstance(self.image_size, tuple) and len(self.image_size) == 2
                and all(type(v) is int and 0 < v <= 20000 for v in self.image_size), "invalid image resolution")
        require(isinstance(self.transformations, tuple) and all(text(v) for v in self.transformations), "image transformation signature required")


@dataclass(frozen=True)
class Lens:
    mode: str  # none_verified or opencv_brown5; not an arbitrary RealSense distortion enum
    reference: str
    camera_matrix: tuple | None = None
    coefficients: tuple | None = None

    def __post_init__(self):
        require(self.mode in {"none_verified", "opencv_brown5"} and text(self.reference), "lens model and measurement reference required")
        if self.mode == "none_verified":
            require(self.camera_matrix is None and self.coefficients is None, "no distortion mode cannot contain coefficients")
        else:
            matrix = np.asarray(self.camera_matrix, dtype=float)
            coeffs = np.asarray(self.coefficients, dtype=float)
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and coeffs.shape == (5,) and np.isfinite(coeffs).all(), "invalid Brown5 intrinsics")
            require(all(number(v) for row in self.camera_matrix for v in row) and all(number(v) for v in self.coefficients), "intrinsics must be numeric, not boolean/string")
            require(matrix[0, 0] > 0 and matrix[1, 1] > 0 and np.array_equal(matrix[2], [0, 0, 1])
                    and matrix[0, 1] == matrix[1, 0] == 0, "unsupported camera matrix")


@dataclass(frozen=True)
class Pair:
    point_id: str
    role: str
    measurement_session: str
    pixel: tuple[float, float]
    robot_xy_mm: tuple[float, float]

    def __post_init__(self):
        require(text(self.point_id) and text(self.measurement_session) and self.role in {"fit", "check"}, "point ID, role and measurement session required")
        require(all(isinstance(v, tuple) and len(v) == 2 and all(number(x) for x in v) for v in (self.pixel, self.robot_xy_mm)), "point coordinates must be finite pairs")


@dataclass(frozen=True)
class Limits:
    fit_max_mm: float
    check_max_mm: float
    check_rmse_mm: float
    min_check_coverage: float

    def __post_init__(self):
        require(all(number(v) for v in asdict(self).values()) and self.fit_max_mm > 0 and self.check_max_mm > 0
                and 0 < self.check_rmse_mm <= self.check_max_mm and 0 < self.min_check_coverage <= 1, "explicit valid error/coverage limits required")


@dataclass(frozen=True)
class Specification:
    version: str
    product_id: str
    kind: str
    context: GeometryContext
    lens: Lens
    plane_z_mm: float
    plane_tolerance_mm: float
    application_polygon: tuple[tuple[float, float], ...]
    limits: Limits
    pairs: tuple[Pair, ...]
    schema_version: int = 1

    def __post_init__(self):
        require(text(self.version) and text(self.product_id) and self.kind in {"real", "synthetic"}, "explicit calibration version/product/kind required")
        require(type(self.schema_version) is int and self.schema_version == 1, "unsupported calibration specification")
        require(isinstance(self.context, GeometryContext) and isinstance(self.lens, Lens) and isinstance(self.limits, Limits), "incomplete calibration settings")
        require(number(self.plane_z_mm) and number(self.plane_tolerance_mm) and self.plane_tolerance_mm >= 0, "explicit plane height and tolerance required")
        require(isinstance(self.pairs, tuple) and all(isinstance(p, Pair) for p in self.pairs), "invalid point list")
        require(len({p.point_id for p in self.pairs}) == len(self.pairs), "duplicate point IDs")
        require(len({p.pixel for p in self.pairs}) == len(self.pairs) and len({p.robot_xy_mm for p in self.pairs}) == len(self.pairs), "duplicate/reused fit or check coordinates")
        train = [p for p in self.pairs if p.role == "fit"]; check = [p for p in self.pairs if p.role == "check"]
        require(len(train) >= 6 and len(check) >= 4 and len(self.pairs) <= 1000, "need at least 6 fit points and 4 independent check points; max 1000")
        require(not {p.measurement_session for p in train} & {p.measurement_session for p in check}, "check measurements must be from independent sessions")
        polygon = np.asarray(self.application_polygon, dtype=float)
        require(all(len(p) == 2 and all(number(v) for v in p) for p in self.application_polygon), "invalid polygon coordinates")
        require(polygon.ndim == 2 and polygon.shape[1] == 2 and 3 <= len(polygon) <= 32 and np.isfinite(polygon).all(), "invalid application polygon")
        contour = polygon.astype(np.float32)
        require(cv2.isContourConvex(contour) and abs(cv2.contourArea(contour)) > 1e-6, "application polygon must be ordered, convex, and nonzero")
        w, h = self.context.image_size
        require(all(0 <= x < w and 0 <= y < h for x, y in [*self.application_polygon, *(p.pixel for p in self.pairs)]), "pixel outside original image")
        require(all(inside(p.pixel, polygon) for p in check), "holdout points must lie in the application region")


def spec_from_dict(data):
    d = deepcopy(data)
    c = d["context"]
    d["context"] = GeometryContext(**dict(c, image_size=tuple(c["image_size"]), transformations=tuple(c.get("transformations", ()))))
    lens = d["lens"]
    d["lens"] = Lens(**dict(lens, camera_matrix=tuple(tuple(r) for r in lens["camera_matrix"]) if lens.get("camera_matrix") is not None else None,
                           coefficients=tuple(lens["coefficients"]) if lens.get("coefficients") is not None else None))
    d["limits"] = Limits(**d["limits"])
    d["application_polygon"] = tuple(tuple(p) for p in d["application_polygon"])
    d["pairs"] = tuple(Pair(**dict(p, pixel=tuple(p["pixel"]), robot_xy_mm=tuple(p["robot_xy_mm"]))) for p in d["pairs"])
    return Specification(**d)


def inside(point, polygon):
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), (float(point[0]), float(point[1])), False) >= 0


def hull(points): return cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2).astype(float)


def undistort(points, lens):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    require(np.isfinite(p).all(), "nonfinite pixels")
    if lens.mode == "none_verified": return p.copy()
    matrix = np.asarray(lens.camera_matrix, dtype=np.float64)
    coeffs = np.asarray(lens.coefficients, dtype=np.float64)
    mapped = cv2.undistortPoints(p.reshape(-1, 1, 2), matrix, coeffs, P=matrix,
                                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)).reshape(-1, 2)
    require(np.isfinite(mapped).all(), "lens inversion failed")
    normalized = np.column_stack(((mapped[:, 0]-matrix[0, 2])/matrix[0, 0], (mapped[:, 1]-matrix[1, 2])/matrix[1, 1], np.ones(len(p))))
    back = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), matrix, coeffs)[0].reshape(-1, 2)
    require(np.max(np.linalg.norm(back-p, axis=1)) <= 1e-4, "lens inverse did not converge")
    return mapped


def project(matrix, points):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ np.asarray(matrix, dtype=float).T
    require(np.isfinite(q).all() and np.all(np.abs(q[:, 2]) > 1e-10), "projective horizon/invalid transform")
    return q[:, :2]/q[:, 2:3]


def normalized_dlt(pixels, xy):
    def normalize(p):
        center = p.mean(axis=0)
        scale = np.sqrt(2)/np.sqrt(np.mean(np.sum((p-center)**2, axis=1)))
        require(np.isfinite(scale) and scale > 0, "degenerate point extent")
        t = np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1]])
        return project(t, p), t
    p, tp = normalize(pixels); q, tq = normalize(xy)
    rows = []
    for (x, y), (u, v) in zip(p, q):
        rows.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    _, singular, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
    condition = float(singular[0]/max(singular[-2], 1e-300))
    require(condition < 1e8, "degenerate or ill-conditioned correspondence layout")
    normalized = vt[-1].reshape(3, 3)
    require(abs(np.linalg.det(normalized)) > 1e-12, "singular homography")
    matrix = np.linalg.inv(tq) @ normalized @ tp
    require(abs(matrix[2, 2]) > 1e-12, "unstable homography normalization")
    return matrix/matrix[2, 2], condition


def fit(spec):
    spec = spec_from_dict(asdict(spec))
    train = [p for p in spec.pairs if p.role == "fit"]
    raw = np.array([p.pixel for p in train], dtype=float)
    ideal = undistort(raw, spec.lens)
    xy = np.array([p.robot_xy_mm for p in train], dtype=float)
    raw_hull, ideal_hull = hull(raw), hull(ideal)
    require(all(inside(p, raw_hull) for p in spec.application_polygon), "application region extends beyond measured fit points")
    matrix, condition = normalized_dlt(ideal, xy)
    denominators = np.column_stack((ideal_hull, np.ones(len(ideal_hull)))) @ matrix[2]
    require(np.all(denominators > 1e-8) or np.all(denominators < -1e-8), "projective horizon crosses calibration region")
    predicted = project(matrix, undistort([p.pixel for p in spec.pairs], spec.lens))
    residuals = [{"point_id": p.point_id, "role": p.role, "predicted_xy_mm": guess.tolist(),
                  "error_mm": float(np.linalg.norm(guess-np.asarray(p.robot_xy_mm)))} for p, guess in zip(spec.pairs, predicted)]
    errors = {role: np.array([r["error_mm"] for r in residuals if r["role"] == role]) for role in ("fit", "check")}
    check_hull = hull([p.pixel for p in spec.pairs if p.role == "check"])
    coverage = abs(cv2.contourArea(check_hull.astype(np.float32)))/abs(cv2.contourArea(np.array(spec.application_polygon, dtype=np.float32)))
    metrics = {"fit_max_mm": float(errors["fit"].max()), "check_max_mm": float(errors["check"].max()),
               "check_rmse_mm": float(np.sqrt(np.mean(errors["check"]**2))), "check_coverage": coverage, "normalized_dlt_condition": condition}
    failures = [name.upper() + "_EXCEEDED" for name in ("fit_max_mm", "check_max_mm", "check_rmse_mm") if metrics[name] > getattr(spec.limits, name)]
    if coverage < spec.limits.min_check_coverage: failures.append("CHECK_COVERAGE_INSUFFICIENT")
    data = {"schema_version": 1, "algorithm": "normalized-dlt-all-points-v1", "runtime": {"opencv": cv2.__version__, "numpy": np.__version__}, "specification": asdict(spec),
            "matrix_ideal_pixel_to_robot_xy": matrix.tolist(), "fit_hull_raw_pixels": raw_hull.tolist(),
            "fit_hull_ideal_pixels": ideal_hull.tolist(), "metrics": metrics, "residuals": residuals,
            "failures": failures, "acceptance": None, "status": "CHECKS_FAILED" if failures else "AWAITING_ACCEPTANCE"}
    return Calibration(data)


class Calibration:
    def __init__(self, data):
        self._data = deepcopy(data)

    @property
    def data(self): return deepcopy(self._data)

    @property
    def digest(self): return fingerprint(self._data)

    @property
    def identity(self): return self._data["specification"]["version"] + ":" + self.digest

    @property
    def ready(self): return self._data["status"] == "ACCEPTED"

    def accept(self, reference):
        require(self._data["status"] == "AWAITING_ACCEPTANCE" and text(reference), "passed independent checks and explicit acceptance record required")
        d = self.data
        d["status"] = "ACCEPTED"; d["acceptance"] = {"reference": reference}
        return Calibration(d)

    def map_xy(self, pixel, context, *, plane_z_mm, kind):
        require(self.ready, "CALIBRATION_NOT_ACCEPTED")
        spec = spec_from_dict(self._data["specification"])
        require(context == spec.context and kind == spec.kind, "CALIBRATION_CONTEXT_OR_KIND_CHANGED")
        require(number(plane_z_mm) and abs(plane_z_mm-spec.plane_z_mm) <= spec.plane_tolerance_mm, "INSPECTION_PLANE_CHANGED_OR_UNKNOWN")
        require(isinstance(pixel, (tuple, list)) and len(pixel) == 2 and all(number(v) for v in pixel), "finite original pixel required")
        require(inside(pixel, spec.application_polygon), "OUTSIDE_CALIBRATED_APPLICATION_REGION")
        ideal = undistort([pixel], spec.lens)[0]
        require(inside(ideal, self._data["fit_hull_ideal_pixels"]), "OUTSIDE_MEASURED_IDEAL_PIXEL_REGION")
        mapped = project(self._data["matrix_ideal_pixel_to_robot_xy"], [ideal])[0]
        return tuple(float(v) for v in mapped)

    def save(self, path):
        import json
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"data": self._data, "sha256": self.digest}, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    envelope = read_json(path)
    require(set(envelope) == {"data", "sha256"} and fingerprint(envelope["data"]) == envelope["sha256"], "calibration integrity mismatch")
    data = envelope["data"]
    rebuilt = fit(spec_from_dict(data["specification"]))
    if data["acceptance"] is not None:
        require(set(data["acceptance"]) == {"reference"}, "invalid acceptance metadata")
        rebuilt = rebuilt.accept(data["acceptance"]["reference"])
    # Numerical replay tolerates platform rounding only; hashes still bind the exact saved artifact.
    expected = rebuilt.data
    require(set(data) == set(expected) and data["schema_version"] == expected["schema_version"] and data["algorithm"] == expected["algorithm"]
            and data["runtime"] == expected["runtime"]
            and data["status"] == expected["status"] and data["failures"] == expected["failures"], "calibration replay metadata mismatch")
    for key in ("matrix_ideal_pixel_to_robot_xy", "fit_hull_raw_pixels", "fit_hull_ideal_pixels"):
        require(np.allclose(data[key], expected[key], rtol=1e-10, atol=1e-10), "saved transform differs from measurements")
    require(fingerprint(data["specification"]) == fingerprint(expected["specification"]), "invalid specification serialization")
    for key in expected["metrics"]: require(math.isclose(data["metrics"][key], expected["metrics"][key], rel_tol=1e-8, abs_tol=1e-9), "saved metrics changed")
    require(len(data["residuals"]) == len(expected["residuals"]), "residual inventory mismatch")
    for old, new in zip(data["residuals"], expected["residuals"]):
        require(set(old) == set(new) and old["point_id"] == new["point_id"] and old["role"] == new["role"]
                and np.allclose(old["predicted_xy_mm"], new["predicted_xy_mm"], rtol=1e-10, atol=1e-9)
                and math.isclose(old["error_mm"], new["error_mm"], rel_tol=1e-8, abs_tol=1e-9), "saved residuals changed")
    return Calibration(data)
