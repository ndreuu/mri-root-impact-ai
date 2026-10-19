"""Independent stdlib checks of source-bound RSNA keypoints and v9 crops.

This checks recorded geometry and provenance, without opening images or claiming
that native numbering or cross-series registration is anatomically correct.
"""

import math

from .contracts import manifest_digest
from .rsna_geometry import (box_corners_lps, crop_box, level_segments,
                            lps_to_pixel, pixel_to_lps, segment_plane_intersection)
from .rsna_results import LEVELS, SIDES, validate_native


SEQUENCES = ("sagittal_t1", "sagittal_t2", "axial_t2")


def _require(condition, message):
    if not condition:
        raise ValueError("RSNA spatial: " + message)


def _required(value, fields, name):
    _require(isinstance(value, dict) and set(fields.split()) <= set(value),
             name + " is missing required fields")


def _number(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, count, name):
    _require(isinstance(value, list) and len(value) == count and all(_number(v) for v in value),
             name + " requires finite numeric coordinates")
    return value


def _close(actual, expected, name, tolerance=1e-5):
    _vector(actual, len(expected), name)
    _require(all(abs(a - b) <= tolerance for a, b in zip(actual, expected)), name + " disagrees with source geometry")


def _ref(reference):
    _required(reference, "series_id image_id frame_index", "image_ref")
    _require(set(reference) == {"series_id", "image_id", "frame_index"}
             and isinstance(reference["series_id"], str) and isinstance(reference["image_id"], str)
             and type(reference["frame_index"]) is int and reference["frame_index"] >= 0,
             "invalid image_ref")
    return reference["series_id"], reference["image_id"], reference["frame_index"]


def _box(value, frame, name):
    _require(isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value),
             name + " requires four integer half-open bounds")
    left, top, right, bottom = value
    _require(0 <= left < right <= frame["columns"] and 0 <= top < bottom <= frame["rows"],
             name + " is empty or outside the source matrix")
    return value


def validate_rsna_spatial(study, native, spatial):
    """Validate native/spatial source bindings; raise ValueError on disagreement."""
    validate_native(study, native)
    _required(spatial, "kind schema_version study_id study_digest coordinate_system frames keypoints rois inputs registration_status anatomical_numbering_status", "root")
    _require(spatial["kind"] == "rsna_source_geometry" and spatial["schema_version"] == "1.0",
             "unsupported kind or version")
    _require(spatial["study_id"] == study["study_id"] and spatial["study_digest"] == manifest_digest(study),
             "study identity or digest mismatch")
    _require(spatial["coordinate_system"] == "DICOM_LPS_mm", "coordinate system must be DICOM LPS millimeters")
    _require(spatial["registration_status"] == "unreviewed" and spatial["anatomical_numbering_status"] == "unreviewed",
             "this adapter does not independently confirm registration or numbering")
    for name in ("frames", "keypoints", "rois", "inputs"):
        _require(isinstance(spatial[name], list), name + " must be an array")
    expected = {}
    for series in study["series"]:
        if series["sequence"] not in SEQUENCES:
            continue
        for image in series["images"]:
            _require(image["frame_count"] == 1 and image["sha256"] is not None,
                     "source-bound RSNA geometry requires pinned single-frame inputs")
            expected[(series["series_id"], image["image_id"], 0)] = (series["sequence"], image["sha256"])
    frames = {}
    for frame in spatial["frames"]:
        _required(frame, "image_ref sequence sha256 position orientation spacing rows columns", "frame")
        key = _ref(frame["image_ref"])
        _require(key in expected and key not in frames, "frame is absent from manifest or repeated")
        _require((frame["sequence"], frame["sha256"]) == expected[key], "frame sequence or SHA disagrees with manifest")
        _vector(frame["position"], 3, "position")
        orientation = _vector(frame["orientation"], 6, "orientation")
        spacing = _vector(frame["spacing"], 2, "spacing")
        _require(min(spacing) > 0, "spacing must be positive")
        _require(type(frame["rows"]) is int and type(frame["columns"]) is int
                 and min(frame["rows"], frame["columns"]) > 0, "invalid source matrix")
        pixel_to_lps(frame, 0, 0)  # Also enforces orthonormal IOP axes.
        if frame["sequence"].startswith("sagittal"):
            _require(orientation[1] > .99 and orientation[5] < -.99, "unsupported sagittal side/polarity")
        else:
            _require(orientation[0] > .98 and orientation[4] > .8, "unsupported axial side/polarity")
        frames[key] = frame
    _require(set(frames) == set(expected), "source frame coverage differs from the required manifest series")
    declared_uids, missing_uid = set(), False
    for frame in frames.values():
        uid = frame.get("frame_of_reference_uid")
        _require(uid is None or isinstance(uid, str), "invalid FrameOfReferenceUID metadata")
        if isinstance(uid, str) and uid.strip():
            declared_uids.add(uid.strip())
        else:
            missing_uid = True
    conflict = len(declared_uids) > 1
    _require(type(native.get("different_frame_of_reference_uids")) is bool
             and native["different_frame_of_reference_uids"] == conflict,
             "native FrameOfReferenceUID conflict flag disagrees with source frames")
    policy = native.get("registration_policy")
    _require(isinstance(policy, str) and policy in {
        "require_same_declared_FrameOfReferenceUID", "explicit_research_identity_LPS_unverified"},
        "registration policy must be explicit")
    _require(spatial.get("registration_policy") == policy, "registration policies disagree")
    if (conflict or missing_uid) and policy == "require_same_declared_FrameOfReferenceUID":
        _require(all(item["status"] != "predicted" for item in native["predictions"]),
                 "strict registration policy forbids prediction with missing or conflicting FrameOfReferenceUIDs")

    points = {}
    for point in spatial["keypoints"]:
        _required(point, "sequence label image_ref column row norm_xy lps_mm heatmap_peak heatmap_shape percentile_mask", "keypoint")
        sequence = point["sequence"]
        _require(isinstance(sequence, str) and sequence in SEQUENCES, "invalid keypoint sequence")
        level = point.get("native_level") if sequence == "axial_t2" else point["label"]
        _require(isinstance(level, str) and level in LEVELS, "invalid keypoint native level")
        _require(sequence != "axial_t2" or point["label"] == "center", "axial keypoint must be center")
        _require((sequence, level) not in points, "duplicate keypoint for one sequence/level")
        key = _ref(point["image_ref"])
        _require(key in frames and frames[key]["sequence"] == sequence, "keypoint source sequence disagrees")
        frame = frames[key]
        x, y = point["column"], point["row"]
        _require(_number(x) and _number(y) and 0 <= x < frame["columns"] and 0 <= y < frame["rows"],
                 "keypoint pixel is outside the source matrix")
        _close(point["norm_xy"], [x / frame["columns"], y / frame["rows"]], "normalized keypoint")
        _close(point["lps_mm"], pixel_to_lps(frame, x, y), "keypoint LPS")
        _require(_number(point["heatmap_peak"]) and 0 <= point["heatmap_peak"] <= 1,
                 "invalid sigmoid heatmap peak")
        shape = point["heatmap_shape"]
        _require(isinstance(shape, list) and len(shape) == 2
                 and all(type(v) is int and v > 0 for v in shape), "invalid heatmap shape")
        _require(point["percentile_mask"] == 99.8, "unexpected keypoint extraction percentile")
        points[(sequence, level)] = point
    locations = {item["level"]: item for item in native["localizations"]}
    for level, item in locations.items():
        _require((item["status"] == "located") == (("sagittal_t2", level) in points),
                 "native localization disagrees with SagT2 keypoint coverage")
        refs = {_ref(r) for r in item["image_refs"]}
        _require(refs <= set(frames), "localization uses an undeclared source frame")
        for (sequence, point_level), point in points.items():
            if point_level == level:
                _require(_ref(point["image_ref"]) in refs, "localization omits a used same-level keypoint frame")

    predicted_levels = {item["level"] for item in native["predictions"] if item["status"] == "predicted"}
    segments = None
    if predicted_levels:
        _require(all(("sagittal_t2", level) in points for level in LEVELS),
                 "axial level assignment requires all five SagT2 points")
        segments = level_segments({level: points[("sagittal_t2", level)]["lps_mm"] for level in LEVELS})
        if "level_segments_lps_mm" in spatial:
            recorded = spatial["level_segments_lps_mm"]
            _require(isinstance(recorded, dict) and set(recorded) == set(LEVELS), "invalid recorded native level segments")
            for level in LEVELS:
                _require(isinstance(recorded[level], list) and len(recorded[level]) == 2, "invalid recorded level segment")
                for actual, wanted in zip(recorded[level], segments[level]):
                    _close(actual, wanted, "recorded SagT2 level segment")

    def check_level_plane(frame, level):
        matches = [candidate for candidate, (upper, lower) in segments.items()
                   if segment_plane_intersection(frame, upper, lower, tolerance=1e-6)]
        _require(matches == [level], "axial source plane does not uniquely intersect its native SagT2 level segment")
        upper, lower = segments[level]
        da, db = lps_to_pixel(frame, upper)[2], lps_to_pixel(frame, lower)[2]
        fraction = da / (da - db) if abs(da - db) > 1e-6 else .5
        intersection = [upper[i] + fraction * (lower[i] - upper[i]) for i in range(3)]
        x, y, _ = lps_to_pixel(frame, intersection)
        _require(_number(x) and _number(y) and 0 <= x < frame["columns"] and 0 <= y < frame["rows"],
                 "native level segment intersects outside axial source field of view")

    for level in predicted_levels:
        _require(("axial_t2", level) in points, "successful classifier requires its same-level axial keypoint")
        point_frame = frames[_ref(points[("axial_t2", level)]["image_ref"])]
        check_level_plane(point_frame, level)

    rois = {}
    for roi in spatial["rois"]:
        _required(roi, "roi_id level side sequence image_ref base_crop_box source_crop_box source_corners_lps_mm horizontal_flip resize_shape", "ROI")
        rid, level, side, sequence = roi["roi_id"], roi["level"], roi["side"], roi["sequence"]
        _require(isinstance(rid, str) and bool(rid.strip()) and rid not in rois, "ROI IDs must be unique nonempty strings")
        _require(isinstance(level, str) and level in LEVELS and isinstance(side, str) and side in SIDES
                 and isinstance(sequence, str) and sequence in SEQUENCES, "invalid ROI level/side/sequence")
        key = _ref(roi["image_ref"])
        _require(key in frames and frames[key]["sequence"] == sequence, "ROI source sequence disagrees")
        frame = frames[key]
        base = _box(roi["base_crop_box"], frame, "base crop")
        box = _box(roi["source_crop_box"], frame, "side crop")
        _require((sequence, level) in points, "ROI has no same-level automatic keypoint")
        point = points[(sequence, level)]
        sagittal = sequence != "axial_t2"
        if sagittal:
            center = point["norm_xy"]
        else:
            x, y, _ = lps_to_pixel(frame, point["lps_mm"])
            center = [x / frame["columns"], y / frame["rows"]]
        wanted = crop_box(frame, center, [35, 50] if sagittal else [45, 75], [-5, 0] if sagittal else [5, 0])
        _require(base == wanted, "base crop disagrees with fixed v9 physical keypoint crop")
        side_box = list(base)
        if not sagittal:
            width = base[2] - base[0]
            if side == "right":
                side_box[2] = base[0] + int(width * .6)
            else:
                side_box[0] += int(width * .4)
        _require(box == side_box, "side crop disagrees with v9 patient-side mapping")
        _require(type(roi["horizontal_flip"]) is bool
                 and roi["horizontal_flip"] == (not sagittal and side == "right"), "incorrect side flip")
        _require(roi["resize_shape"] == [224, 224], "unexpected classifier resize shape")
        corners = roi["source_corners_lps_mm"]
        _require(isinstance(corners, list) and len(corners) == 4, "ROI requires four physical footprint corners")
        for actual, wanted_corner in zip(corners, box_corners_lps(frame, box)):
            _close(actual, wanted_corner, "ROI half-pixel physical corner")
        rois[rid] = roi

    predictions = {(item["level"], item["side"]): item for item in native["predictions"]}
    inputs = {}
    for item in spatial["inputs"]:
        _required(item, "level side roi_ids", "classifier input")
        level, side = item["level"], item["side"]
        _require(isinstance(level, str) and level in LEVELS and isinstance(side, str) and side in SIDES,
                 "invalid classifier input level/side")
        key = (level, side)
        _require(key not in inputs and predictions[key]["status"] == "predicted",
                 "duplicate classifier input or inputs recorded for not_run")
        ids = item["roi_ids"]
        _require(isinstance(ids, dict) and set(ids) == set(SEQUENCES), "classifier requires all three sequences")
        refs = {_ref(r) for r in locations[level]["image_refs"]}
        axial_orientations = set()
        for sequence in SEQUENCES:
            names = ids[sequence]
            _require(isinstance(names, list) and len(names) == (10 if sequence == "axial_t2" else 15),
                     "classifier input slice counts must be 15/15/10")
            for rid in names:
                _require(isinstance(rid, str) and rid in rois, "classifier input refers to an absent ROI")
                roi = rois[rid]
                _require((roi["level"], roi["side"], roi["sequence"]) == (level, side, sequence),
                         "classifier input uses a different level, side or sequence")
                ref = _ref(roi["image_ref"])
                refs.add(ref)
                if sequence == "axial_t2":
                    frame = frames[ref]
                    axial_orientations.add(tuple(round(v, 6) for v in frame["orientation"]))
                    check_level_plane(frame, level)
                    point_frame = frames[_ref(points[("axial_t2", level)]["image_ref"])]
                    _close(frame["orientation"], point_frame["orientation"],
                           "axial keypoint/input orientation group", tolerance=1e-6)
        _require(len(axial_orientations) == 1, "classifier axial input combines different orientation groups")
        _require({_ref(r) for r in predictions[key]["image_refs"]} == refs,
                 "native classifier refs must equal used ROI and localization refs")
        inputs[key] = item
    _require(set(inputs) == {key for key, item in predictions.items() if item["status"] == "predicted"},
             "successful native predictions and classifier inputs disagree")
