"""Synthetic camera geometry only. No measured D405 or Dobot values."""
import numpy as np
from .core import GeometryContext, Lens, Pair, Limits, Specification, project


def make_spec():
    matrix = np.array([[.5, .04, -50], [-.02, -.6, 70], [.0003, -.0001, 1.]])
    fit_points = ((5, 5), (240, 5), (475, 5), (475, 175), (240, 175), (5, 175), (5, 90), (475, 90))
    checks = ((18, 18), (462, 18), (462, 162), (18, 162), (230, 87))
    pairs = tuple(Pair(f"{role}-{i}", role, f"SYNTHETIC-{role}-session", pixel, tuple(float(v) for v in project(matrix, [pixel])[0]))
                  for role, points in (("fit", fit_points), ("check", checks)) for i, pixel in enumerate(points))
    # Convert numpy scalars to strict Python JSON-compatible numbers.
    pairs = tuple(Pair(p.point_id, p.role, p.measurement_session, p.pixel, tuple(float(v) for v in p.robot_xy_mm)) for p in pairs)
    return Specification("SYNTHETIC-plane-v1", "fixture-part", "synthetic",
        GeometryContext("SYNTHETIC-camera", "mount-v1", "rgb-v1", "robot-base-v1", "tool-v1", (480, 180)),
        Lens("none_verified", "SYNTHETIC-undistorted-projection"), 20., 0.,
        ((15., 15.), (465., 15.), (465., 165.), (15., 165.)), Limits(.1, .1, .05, .7), pairs)
