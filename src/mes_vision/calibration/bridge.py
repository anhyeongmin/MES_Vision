from mes_vision.inspection import Mode
from mes_vision.robot import MappedTarget
from mes_vision.training.data import require
from .core import Calibration, spec_from_dict


def map_target(calibration, inspection, object_id, pixel, context, *, plane_z_mm, r_deg, grasp_policy_version):
    """Use an explicitly selected grasp pixel, never assume a detection center is safe."""
    require(isinstance(calibration, Calibration), "calibration required")
    spec = spec_from_dict(calibration.data["specification"])
    require(inspection.config.get("product_id") == spec.product_id, "CALIBRATION_PRODUCT_MISMATCH")
    frame = inspection.frame
    require((frame["width"], frame["height"]) == context.image_size and tuple(frame["transformations"]) == context.transformations
            and frame["coordinate_space"] == "input_rgb_pixels", "FRAME_IMAGE_GEOMETRY_CHANGED")
    objects = [o for o in inspection.objects if o.object_id == object_id]
    require(len(objects) == 1 and objects[0].decision_details.get("identity_valid") is True, "CURRENT_DECIDED_OBJECT_REQUIRED")
    obj = objects[0]
    require(obj.object_id.startswith(inspection.run_id + ":OBJ"), "OBJECT_RUN_MISMATCH")
    kind = "synthetic" if inspection.mode == Mode.SIMULATION else "real"
    x, y = calibration.map_xy(pixel, context, plane_z_mm=plane_z_mm, kind=kind)
    require(obj.effective_box.x1 <= pixel[0] < obj.effective_box.x2 and obj.effective_box.y1 <= pixel[1] < obj.effective_box.y2, "GRASP_PIXEL_OUTSIDE_OBJECT")
    return MappedTarget(inspection.run_id, frame["frame_id"], object_id, calibration.identity, grasp_policy_version,
                        kind, tuple(pixel), x, y, r_deg, True, calibration.data["acceptance"]["reference"])
