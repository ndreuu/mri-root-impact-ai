"""Validate versioned JSON boundaries without reading images or inferring findings."""

import hashlib
import json
import math
from pathlib import Path
import re


VERSION = "1.0"
VERTEBRAE = {"L1", "L2", "L3", "L4", "L5", "S1"}
DISC_LEVELS = {"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"}
SIDES = {"left", "right", "midline", "bilateral", "not_applicable", "unknown"}
ZONES = {"vertebral_body", "disc", "central", "subarticular", "foraminal", "extraforaminal", "unknown"}
COMPONENTS = {"localization", "disc_analysis", "foraminal_stenosis", "subarticular_stenosis"}
STATUSES = {"assessed", "indeterminate", "not_run", "error"}
SEQUENCES = {"sagittal_t1", "sagittal_t2", "axial_t2", "other", "unknown"}


class ContractError(ValueError):
    """A JSON payload violates the versioned project contract."""


def load_document(path):
    """Read strict JSON without silently replacing repeated object keys."""
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ContractError("JSON: duplicate object key: " + key)
            value[key] = item
        return value

    def reject_constant(value):
        raise ContractError("JSON: non-finite constant: " + value)

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def _require(condition, location, message):
    if not condition:
        raise ContractError("{}: {}".format(location, message))


def _object(value, fields, location):
    _require(isinstance(value, dict), location, "expected object")
    _require(set(value) == set(fields.split()), location, "expected fields: " + fields)


def _text(value, location):
    _require(isinstance(value, str) and bool(value.strip()), location, "expected nonempty string")


def _identifier(value, location):
    _text(value, location)
    _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)), location, "invalid identifier")


def _enum(value, choices, location):
    _require(isinstance(value, str) and value in choices, location, "expected one of " + ", ".join(sorted(choices)))


def _list(value, location, nonempty=False):
    _require(isinstance(value, list), location, "expected array")
    _require(not nonempty or len(value) > 0, location, "array must not be empty")


def _integer(value, minimum, location):
    _require(type(value) is int and value >= minimum, location, "expected integer >= {}".format(minimum))


def _sha256(value, location):
    _require(isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{64}", value)), location, "expected lowercase SHA-256")


def _unique(value, seen, location):
    _require(value not in seen, location, "duplicate identifier or reference")
    seen.add(value)


def manifest_digest(study):
    """SHA-256 of UTF-8 JSON, sorted object keys, compact separators, array order retained."""
    try:
        encoded = json.dumps(study, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError("study: not canonical JSON: " + str(error)) from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_study(study):
    _object(study, "schema_version kind study_id synthetic series limitations", "study")
    _require(study["schema_version"] == VERSION, "study.schema_version", "unsupported version")
    _require(study["kind"] == "study_manifest", "study.kind", "expected study_manifest")
    _identifier(study["study_id"], "study.study_id")
    _require(type(study["synthetic"]) is bool, "study.synthetic", "expected boolean")
    _list(study["limitations"], "study.limitations")
    for limitation in study["limitations"]:
        _text(limitation, "study.limitations[]")
    _list(study["series"], "study.series")
    _require(bool(study["series"]) or bool(study["limitations"]), "study", "empty study requires a limitation explaining missing data")
    series_ids, image_ids = set(), set()
    for series in study["series"]:
        _object(series, "series_id sequence images", "series")
        _identifier(series["series_id"], "series.series_id")
        _unique(series["series_id"], series_ids, "series.series_id")
        _enum(series["sequence"], SEQUENCES, "series.sequence")
        _list(series["images"], "series.images", nonempty=True)
        for image in series["images"]:
            _object(image, "image_id path frame_count sha256", "image")
            _identifier(image["image_id"], "image.image_id")
            _unique(image["image_id"], image_ids, "image.image_id")
            _text(image["path"], "image.path")
            _require(Path(image["path"]).is_absolute(), "image.path", "expected absolute local path")
            _integer(image["frame_count"], 1, "image.frame_count")
            if image["sha256"] is not None:
                _sha256(image["sha256"], "image.sha256")


def _validate_target(target, component):
    _object(target, "level side zone", "target")
    _enum(target["level"], VERTEBRAE | DISC_LEVELS | {"unknown"}, "target.level")
    _enum(target["side"], SIDES, "target.side")
    _enum(target["zone"], ZONES, "target.zone")
    if component == "localization":
        _enum(target["zone"], {"vertebral_body", "disc", "unknown"}, "target.zone")
        _enum(target["side"], {"not_applicable", "unknown"}, "target.side")
        if target["level"] != "unknown" and target["zone"] != "unknown":
            levels = VERTEBRAE if target["zone"] == "vertebral_body" else DISC_LEVELS
            _enum(target["level"], levels, "target.level")
    else:
        _enum(target["level"], DISC_LEVELS | {"unknown"}, "target.level")
        _require(target["zone"] != "vertebral_body", "target.zone", "not a disc-level zone")
        if component in {"foraminal_stenosis", "subarticular_stenosis"}:
            zone = "foraminal" if component == "foraminal_stenosis" else "subarticular"
            _enum(target["zone"], {zone, "unknown"}, "target.zone")
            _enum(target["side"], {"left", "right", "unknown"}, "target.side")


def _validate_observation(observation):
    _object(observation, "label scale_id measurements", "observation")
    _text(observation["label"], "observation.label")
    _text(observation["scale_id"], "observation.scale_id")
    _list(observation["measurements"], "observation.measurements")
    names = set()
    for measurement in observation["measurements"]:
        _object(measurement, "name value unit", "measurement")
        _identifier(measurement["name"], "measurement.name")
        _unique(measurement["name"], names, "measurement.name")
        value = measurement["value"]
        _require(type(value) in {int, float}, "measurement.value", "expected finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        _require(finite, "measurement.value", "expected finite number")
        _text(measurement["unit"], "measurement.unit")


def _validate_producer(producer, synthetic, items):
    _object(producer, "name version method model", "producer")
    _text(producer["name"], "producer.name")
    _text(producer["version"], "producer.version")
    _enum(producer["method"], {"synthetic", "manual", "model"}, "producer.method")
    if producer["method"] == "synthetic":
        _require(synthetic, "producer.method", "synthetic producer requires synthetic study")
    if producer["method"] == "model":
        if producer["model"] is None:
            _require(all(item["status"] in {"not_run", "error"} for item in items), "producer.model", "missing model permits only not_run/error assessments")
            return
        _object(producer["model"], "name version weights_sha256", "producer.model")
        _text(producer["model"]["name"], "producer.model.name")
        _text(producer["model"]["version"], "producer.model.version")
        _sha256(producer["model"]["weights_sha256"], "producer.model.weights_sha256")
    else:
        _require(producer["model"] is None, "producer.model", "non-model producer requires null")


def validate_bundle(study, results):
    """Check result envelopes, input identity, source images, and supplied upstream results."""
    validate_study(study)
    _list(results, "results")
    images = {
        (series["series_id"], image["image_id"]): image
        for series in study["series"] for image in series["images"]
    }
    digest = manifest_digest(study)
    by_id, assessments = {}, {}
    for result in results:
        _object(result, "schema_version kind result_id study_id synthetic study_digest component producer assessments", "result")
        _require(result["schema_version"] == VERSION, "result.schema_version", "unsupported version")
        _require(result["kind"] == "module_result", "result.kind", "expected module_result")
        _identifier(result["result_id"], "result.result_id")
        _require(result["result_id"] not in by_id, "result.result_id", "duplicate result identifier")
        by_id[result["result_id"]] = result
        _require(result["study_id"] == study["study_id"], "result.study_id", "does not match study")
        _require(type(result["synthetic"]) is bool and result["synthetic"] == study["synthetic"], "result.synthetic", "does not match study")
        _sha256(result["study_digest"], "result.study_digest")
        _require(result["study_digest"] == digest, "result.study_digest", "manifest has changed or belongs to another result")
        _enum(result["component"], COMPONENTS, "result.component")
        _list(result["assessments"], "result.assessments", nonempty=True)
        local_ids = set()
        for item in result["assessments"]:
            _object(item, "assessment_id target status reason observation image_refs upstream_refs", "assessment")
            _identifier(item["assessment_id"], "assessment.assessment_id")
            _unique(item["assessment_id"], local_ids, "assessment.assessment_id")
            assessments[(result["result_id"], item["assessment_id"])] = item
            _validate_target(item["target"], result["component"])
            _enum(item["status"], STATUSES, "assessment.status")
            if item["status"] == "assessed":
                _require(item["reason"] is None, "assessment.reason", "assessed requires null")
                _require("unknown" not in item["target"].values(), "assessment.target", "unknown target requires indeterminate status")
                _require(item["observation"] is not None, "assessment.observation", "assessed requires explicit native value")
            else:
                _text(item["reason"], "assessment.reason")
                if item["status"] in {"not_run", "error"}:
                    _require(item["observation"] is None, "assessment.observation", "not_run/error requires null")
            if item["observation"] is not None:
                _validate_observation(item["observation"])
            _list(item["image_refs"], "assessment.image_refs", nonempty=item["observation"] is not None)
            refs = set()
            for reference in item["image_refs"]:
                _object(reference, "series_id image_id frame_index", "image_ref")
                _identifier(reference["series_id"], "image_ref.series_id")
                _identifier(reference["image_id"], "image_ref.image_id")
                _integer(reference["frame_index"], 0, "image_ref.frame_index")
                key = (reference["series_id"], reference["image_id"])
                _require(key in images, "image_ref", "series/image is not in the study")
                _require(reference["frame_index"] < images[key]["frame_count"], "image_ref.frame_index", "outside declared frame_count")
                _unique(key + (reference["frame_index"],), refs, "image_ref")
            _list(item["upstream_refs"], "assessment.upstream_refs")
            upstream = set()
            for reference in item["upstream_refs"]:
                _object(reference, "result_id assessment_id", "upstream_ref")
                _identifier(reference["result_id"], "upstream_ref.result_id")
                _identifier(reference["assessment_id"], "upstream_ref.assessment_id")
                _require(reference["result_id"] != result["result_id"], "upstream_ref", "must reference an earlier, separate result")
                _unique((reference["result_id"], reference["assessment_id"]), upstream, "upstream_ref")
        _validate_producer(result["producer"], study["synthetic"], result["assessments"])

    dependencies = {result_id: set() for result_id in by_id}
    for result_id, result in by_id.items():
        for item in result["assessments"]:
            for reference in item["upstream_refs"]:
                key = (reference["result_id"], reference["assessment_id"])
                _require(key in assessments, "upstream_ref", "referenced assessment was not supplied")
                dependencies[result_id].add(reference["result_id"])
                if item["status"] == "assessed":
                    _require(assessments[key]["status"] == "assessed", "upstream_ref", "assessed output depends on unresolved upstream assessment")
                    source = assessments[key]
                    if (by_id[reference["result_id"]]["component"] == "localization"
                            and source["target"]["level"] in DISC_LEVELS
                            and item["target"]["level"] in DISC_LEVELS):
                        _require(source["target"]["level"] == item["target"]["level"], "upstream_ref", "disc level contradicts its localization source")

    visited, active = set(), set()

    def visit(result_id):
        _require(result_id not in active, "upstream_refs", "cyclic result dependency")
        if result_id in visited:
            return
        active.add(result_id)
        for dependency in dependencies[result_id]:
            visit(dependency)
        active.remove(result_id)
        visited.add(result_id)

    for result_id in by_id:
        visit(result_id)
