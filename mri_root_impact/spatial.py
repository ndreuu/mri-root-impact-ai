"""Versioned source-frame geometry and conservative SpineNet label observations.

This module uses only the standard library. It validates metadata; the sagittal
loader is responsible for verifying these geometry values against DICOM bytes.
"""

import copy
import hashlib
import json
import math
import re

from .contracts import ContractError, manifest_digest, validate_bundle, validate_study


VERSION = "1.0"
VERTEBRAE = ("L1", "L2", "L3", "L4", "L5", "S1")
DISCS = tuple("{}-{}".format(a, b) for a, b in zip(VERTEBRAE, VERTEBRAE[1:]))
REVIEW_REASON = "Native model numbering has not received independent anatomical review."
DISC_REASON = "Disc source-frame ROI has not been derived or reviewed; adjacent vertebra labels are insufficient."


def _require(condition, location, message):
    if not condition:
        raise ContractError("{}: {}".format(location, message))


def _object(value, fields, location):
    _require(isinstance(value, dict) and set(value) == set(fields.split()), location,
             "expected exactly fields: " + fields)


def _array(value, length, location):
    _require(isinstance(value, list) and len(value) == length, location,
             "expected array of length {}".format(length))


def _number(value, location, positive=False):
    _require(type(value) in (int, float), location, "expected finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite and (not positive or value > 0), location,
             "expected positive finite number" if positive else "expected finite number")


def _vector(value, length, location):
    _array(value, length, location)
    for number in value:
        _number(number, location)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _geometry(study, geometry):
    _object(geometry, "series_id volume_shape pixel_spacing_mm slice_center_spacing_mm coordinate_system slices", "geometry")
    _require(geometry["coordinate_system"] == "DICOM_LPS_mm", "geometry.coordinate_system", "expected DICOM_LPS_mm")
    _array(geometry["volume_shape"], 3, "geometry.volume_shape")
    for value in geometry["volume_shape"]:
        _require(type(value) is int and value > 0, "geometry.volume_shape", "expected positive integer dimensions")
    _number(geometry["pixel_spacing_mm"], "geometry.pixel_spacing_mm", positive=True)
    _number(geometry["slice_center_spacing_mm"], "geometry.slice_center_spacing_mm", positive=True)
    source = [s for s in study["series"] if s["series_id"] == geometry["series_id"]]
    _require(len(source) == 1, "geometry.series_id", "series is not in study")
    _require(source[0]["sequence"] in ("sagittal_t1", "sagittal_t2"), "geometry.series_id", "expected sagittal sequence")
    images = {image["image_id"]: image for image in source[0]["images"]}
    _require(all(image["frame_count"] == 1 for image in images.values()), "geometry", "only single-frame images are supported")
    _array(geometry["slices"], geometry["volume_shape"][2], "geometry.slices")
    _require(len(geometry["slices"]) == len(images), "geometry.slices", "must cover every selected series image exactly once")
    seen, orientation, normal, previous, direction = set(), None, None, None, None
    for index, item in enumerate(geometry["slices"]):
        _object(item, "slice_index image_ref image_position_patient_mm image_orientation_patient pixel_spacing_mm slice_thickness_mm", "geometry.slice")
        _require(type(item["slice_index"]) is int and item["slice_index"] == index, "geometry.slice_index", "must match zero-based array position")
        ref = item["image_ref"]
        _object(ref, "series_id image_id frame_index", "geometry.image_ref")
        _require(ref["series_id"] == geometry["series_id"] and isinstance(ref["image_id"], str)
                 and ref["image_id"] in images, "geometry.image_ref", "does not resolve in selected series")
        _require(type(ref["frame_index"]) is int and ref["frame_index"] == 0, "geometry.image_ref.frame_index", "expected single-frame index zero")
        _require(ref["image_id"] not in seen, "geometry.image_ref", "duplicate source image")
        seen.add(ref["image_id"])
        _vector(item["image_position_patient_mm"], 3, "geometry.image_position_patient_mm")
        _vector(item["image_orientation_patient"], 6, "geometry.image_orientation_patient")
        _vector(item["pixel_spacing_mm"], 2, "geometry.slice.pixel_spacing_mm")
        _require(all(_close(v, geometry["pixel_spacing_mm"]) and v > 0 for v in item["pixel_spacing_mm"]),
                 "geometry.slice.pixel_spacing_mm", "must match isotropic input scalar")
        _number(item["slice_thickness_mm"], "geometry.slice_thickness_mm", positive=True)
        row, column = item["image_orientation_patient"][:3], item["image_orientation_patient"][3:]
        _require(abs(_dot(row, row) - 1) <= 1e-3 and abs(_dot(column, column) - 1) <= 1e-3
                 and abs(_dot(row, column)) <= 1e-3,
                 "geometry.image_orientation_patient", "direction cosines must be orthonormal")
        if orientation is None:
            orientation = item["image_orientation_patient"]
            normal = [row[1] * column[2] - row[2] * column[1],
                      row[2] * column[0] - row[0] * column[2],
                      row[0] * column[1] - row[1] * column[0]]
        _require(all(_close(a, b) for a, b in zip(orientation, item["image_orientation_patient"])),
                 "geometry.image_orientation_patient", "orientation changes across input volume")
        position = item["image_position_patient_mm"]
        if previous is not None:
            difference = [a - b for a, b in zip(position, previous)]
            step = _dot(difference, normal)
            _require(abs(abs(step) - geometry["slice_center_spacing_mm"]) <= 1e-3, "geometry.slices", "irregular center spacing")
            current_direction = 1 if step > 0 else -1
            _require(direction is None or current_direction == direction, "geometry.slices", "slice order reverses")
            direction = current_direction
            _require(all(abs(d - step * n) <= 1e-3 for d, n in zip(difference, normal)),
                     "geometry.slices", "in-plane source origins drift")
        previous = position


def _polygon(value, location):
    _array(value, 4, location)
    for point in value:
        _vector(point, 2, location)


def _provenance(value):
    _object(value, "source_commit weights_sha256", "model_provenance")
    for field, length in (("source_commit", 40), ("weights_sha256", 64)):
        _require(isinstance(value[field], str) and re.fullmatch("[a-f0-9]{%d}" % length, value[field]) is not None,
                 "model_provenance." + field, "expected lowercase hex digest")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assessment(identifier, level, zone, status, reason, label=None, refs=None):
    return {"assessment_id": identifier,
            "target": {"level": level, "side": "not_applicable", "zone": zone},
            "status": status, "reason": reason,
            "observation": None if label is None else {
                "label": label, "scale_id": "spinenet_predicted_vertebral_label", "measurements": []},
            "image_refs": [] if refs is None else copy.deepcopy(refs), "upstream_refs": []}


def build_localization(study, geometry, native_detections, model_provenance):
    """Preserve unreviewed native numbering; return strict C001 and spatial v1.

    ``slice_nos`` index the supplied H×W×S input. ``polys`` and
    ``average_polygon`` use (column x, row y) coordinates in that input.
    The average is retained as a native summary, never a source-frame ROI.
    """
    validate_study(study)
    _geometry(study, geometry)
    _provenance(model_provenance)
    _require(isinstance(native_detections, list), "native_detections", "expected array")
    normalized = []
    for native in native_detections:
        _object(native, "predicted_label slice_nos polys average_polygon", "native_detection")
        _require(isinstance(native["predicted_label"], str) and bool(native["predicted_label"].strip()),
                 "native_detection.predicted_label", "expected nonempty raw string")
        _require(isinstance(native["slice_nos"], list) and bool(native["slice_nos"]), "native_detection.slice_nos", "expected nonempty array")
        _array(native["polys"], len(native["slice_nos"]), "native_detection.polys")
        _polygon(native["average_polygon"], "native_detection.average_polygon")
        seen = set()
        for index, polygon in zip(native["slice_nos"], native["polys"]):
            _require(type(index) is int and 0 <= index < geometry["volume_shape"][2], "native_detection.slice_nos", "index outside volume")
            _require(index not in seen, "native_detection.slice_nos", "duplicate slice for detection")
            seen.add(index)
            _polygon(polygon, "native_detection.polys")
        normalized.append(copy.deepcopy(native))

    result_id = "spinenet-" + _digest({"geometry": geometry, "native_detections": normalized,
                                        "model_provenance": model_provenance, "study_digest": manifest_digest(study)})[:24]
    detections, assessments, label_ids = [], [], {level: [] for level in VERTEBRAE}
    for index, native in enumerate(normalized):
        detection_id = "detection-{:04d}".format(index + 1)
        assessment_id = "localization-{:04d}".format(index + 1)
        label = native["predicted_label"]
        if label in label_ids:
            label_ids[label].append(detection_id)
        polygons, refs = [], []
        for slice_index, polygon in zip(native["slice_nos"], native["polys"]):
            frame = geometry["slices"][slice_index]
            ref = copy.deepcopy(frame["image_ref"])
            refs.append(ref)
            ipp, iop, spacing = (frame["image_position_patient_mm"], frame["image_orientation_patient"], frame["pixel_spacing_mm"])
            lps = [[ipp[axis] + x * spacing[1] * iop[axis] + y * spacing[0] * iop[axis + 3]
                    for axis in range(3)] for x, y in polygon]
            for point in lps:
                _vector(point, 3, "spatial.polygon_lps_mm")
            height, width = geometry["volume_shape"][:2]
            polygons.append({"slice_index": slice_index, "image_ref": ref, "polygon_xy": polygon,
                             "polygon_lps_mm": lps,
                             "in_bounds": all(0 <= x <= width - 1 and 0 <= y <= height - 1 for x, y in polygon)})
        detections.append({"detection_id": detection_id, "assessment_id": assessment_id,
                           "native_label": label, "native_average_polygon_xy": native["average_polygon"], "polygons": polygons})
        reason = REVIEW_REASON if label in VERTEBRAE else "Unsupported native label retained without renumbering. " + REVIEW_REASON
        assessments.append(_assessment(assessment_id, label if label in VERTEBRAE else "unknown",
                                       "vertebral_body", "indeterminate", reason, label, refs))

    vertebra_coverage = []
    for level in VERTEBRAE:
        ids = label_ids[level]
        reason = REVIEW_REASON
        if not ids:
            reason = "Expected vertebral label was not returned; absence is not a negative finding."
            assessments.append(_assessment("missing-" + level, level, "vertebral_body", "indeterminate", reason))
        elif len(ids) > 1:
            reason = "Duplicate native label makes anatomical numbering ambiguous. " + REVIEW_REASON
            for detection in detections:
                if detection["native_label"] == level:
                    next(a for a in assessments if a["assessment_id"] == detection["assessment_id"])["reason"] = reason
        vertebra_coverage.append({"level": level, "detection_ids": ids, "status": "indeterminate", "reason": reason})

    disc_coverage = []
    for level, upper, lower in zip(DISCS, VERTEBRAE, VERTEBRAE[1:]):
        assessments.append(_assessment("disc-" + level, level, "disc", "not_run", DISC_REASON))
        disc_coverage.append({"level": level, "status": "not_run", "reason": DISC_REASON,
                              "adjacent_detection_ids": {upper: label_ids[upper], lower: label_ids[lower]}})
    result = {"schema_version": VERSION, "kind": "module_result", "result_id": result_id,
              "study_id": study["study_id"], "synthetic": study["synthetic"], "study_digest": manifest_digest(study),
              "component": "localization", "producer": {
                  "name": "MRI Root Impact SpineNet spatial adapter", "version": VERSION, "method": "model",
                  "model": {"name": "SpineNet detect_vb", "version": model_provenance["source_commit"],
                            "weights_sha256": model_provenance["weights_sha256"]}}, "assessments": assessments}
    artifact = {"schema_version": VERSION, "kind": "localization_spatial", "study_id": study["study_id"],
                "study_digest": manifest_digest(study), "synthetic": study["synthetic"], "result_id": result_id,
                "model_provenance": copy.deepcopy(model_provenance), "geometry": copy.deepcopy(geometry),
                "detections": detections,
                "coverage": {"vertebrae": vertebra_coverage, "disc_levels": disc_coverage,
                             "unsupported_detection_ids": [d["detection_id"] for d in detections if d["native_label"] not in VERTEBRAE],
                             "anatomical_review_status": "not_reviewed", "full_expected_numbering": False},
                "limitations": list(study["limitations"]) + [
                    REVIEW_REASON, "Polygons describe vertebral bodies; no voxel masks or disc source-frame ROIs are provided.",
                    "Native average polygons are summaries, not polygons on any specific source frame.",
                    "Coordinates belong only to the selected sagittal series; no cross-series registration is asserted.",
                    "Spatial validation checks metadata consistency; DICOM geometry must be verified by the source loader."]}
    validate_bundle(study, [result])
    return result, artifact


def validate_spatial(study, result, artifact):
    """Reject inconsistent source references, derived coordinates, numbering or C001.

    Reconstructing the canonical outputs also ensures missing and duplicate labels
    cannot be promoted to assessed status by editing one part of a saved artifact.
    This does not independently establish anatomical correctness or read DICOM.
    """
    validate_bundle(study, [result])
    _object(artifact, "schema_version kind study_id study_digest synthetic result_id model_provenance geometry detections coverage limitations", "spatial")
    _require(isinstance(artifact["detections"], list), "spatial.detections", "expected array")
    native = []
    for detection in artifact["detections"]:
        _object(detection, "detection_id assessment_id native_label native_average_polygon_xy polygons", "spatial.detection")
        _require(isinstance(detection["polygons"], list), "spatial.detection.polygons", "expected array")
        for polygon in detection["polygons"]:
            _object(polygon, "slice_index image_ref polygon_xy polygon_lps_mm in_bounds", "spatial.polygon")
        native.append({"predicted_label": detection["native_label"], "slice_nos": [p["slice_index"] for p in detection["polygons"]],
                       "polys": [p["polygon_xy"] for p in detection["polygons"]], "average_polygon": detection["native_average_polygon_xy"]})
    expected_result, expected_artifact = build_localization(study, artifact["geometry"], native, artifact["model_provenance"])
    # Canonical JSON equality distinguishes false from 0 and rejects NaN/Infinity.
    try:
        _require(_digest(result) == _digest(expected_result), "result", "does not match native spatial observations")
        _require(_digest(artifact) == _digest(expected_artifact), "spatial", "derived fields or source references are inconsistent")
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError("spatial: not finite canonical JSON") from error
