"""Validate native RSNA predictions and retain uncertainty in C001 results.

This boundary does not read images, load models, confirm anatomical numbering,
or infer root impact. Native probabilities describe the model's three MRI
classes; they are not probabilities of a clinical diagnosis.
"""

import copy
import math
import re

from .contracts import manifest_digest, validate_bundle


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
SIDES = ("left", "right")
CLASSES = ("normal_mild", "moderate", "severe")
COMPONENTS = ("foraminal_stenosis", "subarticular_stenosis")
NUMBERING_REASON = (
    "Native RSNA numbering is unreviewed; anatomical disc numbering and "
    "registration have not been independently confirmed."
)


def _require(condition, message):
    if not condition:
        raise ValueError("RSNA native: " + message)


def _fields(value, expected, name):
    _require(isinstance(value, dict) and set(value) == set(expected.split()),
             name + " must contain exactly " + expected)


def _reason(value, missing, name):
    if missing:
        _require(isinstance(value, str) and bool(value.strip()),
                 name + " requires an explicit nonempty reason")
    else:
        _require(value is None, name + " success requires a null reason")


def _native_structure(study, native):
    validate_bundle(study, [])
    required = set("kind schema_version study_id study_digest assets class_order localizations predictions limitations".split())
    _require(isinstance(native, dict) and required <= set(native),
             "missing required root fields")
    _require(native["kind"] == "rsna_native_predictions" and native["schema_version"] == "1.0",
             "unsupported kind or schema version")
    _require(native["study_id"] == study["study_id"] and native["study_digest"] == manifest_digest(study),
             "study identity or digest mismatch")
    _require(native["class_order"] == list(CLASSES), "class order must be normal_mild, moderate, severe")
    _require(isinstance(native["limitations"], list) and all(
        isinstance(item, str) and bool(item.strip()) for item in native["limitations"]),
        "limitations must be an array of nonempty strings")
    assets = native["assets"]
    _require(isinstance(assets, dict), "assets must be an object")
    _require(isinstance(assets.get("source_commit"), str) and bool(
        re.fullmatch(r"[0-9a-f]{40}", assets["source_commit"])), "assets requires a source commit")
    weights = assets.get("weights")
    _require(isinstance(weights, dict) and {"classifier", "sagittal_keypoints"} <= set(weights),
             "assets requires classifier and sagittal_keypoints weights")
    for role, entry in weights.items():
        _require(isinstance(role, str) and bool(role.strip()) and isinstance(entry, dict)
                 and isinstance(entry.get("sha256"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])),
                 "each checkpoint role requires a lowercase SHA-256")

    locations = native["localizations"]
    _require(isinstance(locations, list) and len(locations) == len(LEVELS),
             "localizations must cover exactly five disc levels")
    by_level = {}
    for item in locations:
        _fields(item, "level status reason image_refs", "localization")
        level = item["level"]
        _require(isinstance(level, str) and level in LEVELS and level not in by_level,
                 "localization levels must be unique and cover L1-L2 through L5-S1")
        _require(isinstance(item["status"], str) and item["status"] in {"located", "not_located"},
                 "invalid localization status")
        missing = item["status"] == "not_located"
        _reason(item["reason"], missing, "localization")
        _require(isinstance(item["image_refs"], list) and (missing or bool(item["image_refs"])),
                 "located disc requires source image references")
        by_level[level] = item

    predictions = native["predictions"]
    _require(isinstance(predictions, list) and len(predictions) == len(LEVELS) * len(SIDES),
             "predictions must cover exactly ten level/side pairs")
    seen = set()
    for item in predictions:
        _fields(item, "level side status reason probabilities image_refs", "prediction")
        level, side = item["level"], item["side"]
        _require(isinstance(level, str) and level in LEVELS and isinstance(side, str) and side in SIDES,
                 "invalid prediction level or side")
        _require((level, side) not in seen, "duplicate prediction level/side")
        seen.add((level, side))
        _require(isinstance(item["status"], str) and item["status"] in {"predicted", "not_run"},
                 "invalid prediction status")
        missing = item["status"] == "not_run"
        _reason(item["reason"], missing, "prediction")
        _require(isinstance(item["image_refs"], list) and (missing or bool(item["image_refs"])),
                 "predicted values require source image references")
        if missing:
            _require(item["probabilities"] is None, "not_run prediction must not contain probabilities")
            continue
        _require(by_level[level]["status"] == "located", "predicted values require native localization of the same level")
        probabilities = item["probabilities"]
        _fields(probabilities, "foraminal_stenosis subarticular_stenosis", "probabilities")
        for component in COMPONENTS:
            values = probabilities[component]
            _require(isinstance(values, list) and len(values) == len(CLASSES),
                     "each component requires three class probabilities")
            for value in values:
                _require(type(value) in {int, float}, "class probabilities must be finite numbers")
                try:
                    valid = math.isfinite(value) and 0 <= value <= 1
                except OverflowError:
                    valid = False
                _require(valid, "class probabilities must be finite and between zero and one")
            _require(abs(sum(values) - 1.0) <= 1e-5, "class probabilities must sum to one")


def _producer(native, role):
    return {"name": "rsna-moyasii-adapter", "version": "1.0", "method": "model",
            "model": {"name": "rsna2024-moyasii-" + role,
                      "version": native["assets"]["source_commit"],
                      "weights_sha256": native["assets"]["weights"][role]["sha256"]}}


def _envelope(study, native, result_id, component, role):
    return {"schema_version": "1.0", "kind": "module_result", "result_id": result_id,
            "study_id": study["study_id"], "synthetic": study["synthetic"],
            "study_digest": manifest_digest(study), "component": component,
            "producer": _producer(native, role), "assessments": []}


def _build(study, native):
    localization = _envelope(study, native, "rsna-localization-001", "localization", "sagittal_keypoints")
    locations = {item["level"]: item for item in native["localizations"]}
    for level in LEVELS:
        item = locations[level]
        located = item["status"] == "located"
        localization["assessments"].append({
            "assessment_id": "disc-" + level,
            "target": {"level": level, "side": "not_applicable", "zone": "disc"},
            "status": "indeterminate",
            "reason": NUMBERING_REASON if located else item["reason"] + " " + NUMBERING_REASON,
            "observation": {"label": level, "scale_id": "rsna2024:native-disc-level:v1", "measurements": []} if located else None,
            "image_refs": copy.deepcopy(item["image_refs"]), "upstream_refs": []})
    results = [localization]
    predictions = {(item["level"], item["side"]): item for item in native["predictions"]}
    for component, short, zone in (("foraminal_stenosis", "foraminal", "foraminal"),
                                   ("subarticular_stenosis", "subarticular", "subarticular")):
        result = _envelope(study, native, "rsna-" + short + "-001", component, "classifier")
        for level in LEVELS:
            for side in SIDES:
                item = predictions[(level, side)]
                predicted = item["status"] == "predicted"
                observation = None
                if predicted:
                    probabilities = item["probabilities"][component]
                    label = CLASSES[max(range(len(CLASSES)), key=lambda index: probabilities[index])]
                    observation = {"label": label, "scale_id": "rsna2024:stenosis-severity:v1",
                                   "measurements": [{"name": "p_" + name, "value": value, "unit": "probability"}
                                                    for name, value in zip(CLASSES, probabilities)]}
                result["assessments"].append({
                    "assessment_id": short + "-" + level + "-" + side,
                    "target": {"level": level, "side": side, "zone": zone},
                    "status": "indeterminate" if predicted else "not_run",
                    "reason": NUMBERING_REASON if predicted else item["reason"],
                    "observation": observation, "image_refs": copy.deepcopy(item["image_refs"]),
                    "upstream_refs": [{"result_id": localization["result_id"], "assessment_id": "disc-" + level}]})
        results.append(result)
    return results


def validate_native(study, native):
    """Reject malformed or incomplete native results without reading image files."""
    _native_structure(study, native)
    validate_bundle(study, _build(study, native))


def build_results(study, native):
    """Build five localizations and twenty side/condition assessments in C001."""
    _native_structure(study, native)
    results = _build(study, native)
    validate_bundle(study, results)
    return results
