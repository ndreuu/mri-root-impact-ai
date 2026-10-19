"""Expand the F001 vectors into fictitious, fully bound C001 documents.

No MRI file, trained model, clinical reviewer, or registration is represented by
these fixtures. They exercise the fusion boundary using actual contract shapes.
"""

import copy
import hashlib
import json
from pathlib import Path

from mri_root_impact.contracts import manifest_digest


ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "harness" / "fixtures" / "fusion-decision-cases.json"
GRADES = ("normal_mild", "moderate", "severe")
SERIES = (("sag-t2", "sagittal_t2"), ("sag-t1", "sagittal_t1"), ("ax-t2", "axial_t2"))


def canonical_bytes(document):
    return json.dumps(document, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def document_sha256(document):
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def load_vectors():
    with VECTORS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def case_by_id(case_id):
    return next(case for case in load_vectors()["cases"] if case["case_id"] == case_id)


def _result(study, result_id, component, assessments, method="synthetic"):
    return {
        "schema_version": "1.0", "kind": "module_result",
        "result_id": result_id, "study_id": study["study_id"],
        "study_digest": manifest_digest(study), "synthetic": True,
        "component": component,
        "producer": {
            "name": "fictitious-fusion-vector", "version": "1.0", "method": method,
            "model": ({"name": "fictitious-rsna-model", "version": "synthetic-only",
                       "weights_sha256": "a" * 64} if method == "model" else None),
        },
        "assessments": assessments,
    }


def _refs():
    return [{"series_id": name, "image_id": name + "-image", "frame_index": 0}
            for name, _ in SERIES]


def rebind_review(review, artifacts):
    """Refresh a synthetic context after a deliberate fixture mutation."""
    if review is None:
        return None
    review["input_artifacts_sha256"] = {item["artifact_id"]: item["sha256"] for item in artifacts}
    return document_sha256(review)


def build_case(case):
    """Return study, results, artifact bindings, optional review, and review SHA.

    Repeated evidence keys deliberately repeat result identity. The engine must
    collapse equal documents before C001 validation and reject changed content.
    """
    from mri_fusion.contracts import artifact_binding

    vector = copy.deepcopy(case)
    defaults = load_vectors()["defaults"]
    inputs = vector["inputs"]
    study = {
        "schema_version": "1.0", "kind": "study_manifest",
        "study_id": "synthetic-" + vector["case_id"].lower(), "synthetic": True,
        "series": [{"series_id": name, "sequence": sequence, "images": [{
            "image_id": name + "-image",
            "path": "/synthetic/fusion-vectors/" + name + ".dcm",
            "frame_count": 1, "sha256": None,
        }]} for name, sequence in SERIES],
        "limitations": ["Fictitious metadata only; image files and review evidence do not exist."],
    }
    levels = sorted({branch["level"] for branch in inputs["branches"] if branch["level"] != "unknown"})
    localization_id = "synthetic-localization"
    results = []
    if levels:
        items = [{
            "assessment_id": "loc-" + level.lower(),
            "target": {"level": level, "side": "not_applicable", "zone": "disc"},
            "status": "assessed", "reason": None,
            "observation": {"label": level, "scale_id": "fixture:disc-level:v1", "measurements": []},
            "image_refs": [_refs()[0]], "upstream_refs": [],
        } for level in levels]
        results.append(_result(study, localization_id, "localization", items))

    review_rows = []
    for index, branch in enumerate(inputs["branches"]):
        evidence_key = branch.get("evidence_key", "item-{:02d}".format(index))
        result_id = "synthetic-stenosis-" + evidence_key.lower()
        assessment_id = "assessment-" + evidence_key.lower()
        target = {name: branch[name] for name in ("level", "side", "zone")}
        status = branch.get("status", defaults["status"])
        observation = None
        if status not in ("not_run", "error"):
            probabilities = branch.get("probabilities", defaults["probabilities"])
            measurements = [] if probabilities is None else [
                {"name": "p_" + grade, "value": probability, "unit": "probability"}
                for grade, probability in zip(GRADES, probabilities)
            ]
            observation = {"label": branch["label"],
                           "scale_id": branch.get("scale_id", defaults["scale_id"]),
                           "measurements": measurements}
        reason = None if status == "assessed" else "Fictitious vector source status: " + status
        if branch.get("native_level") is not None:
            reason += "; unsupported native_level=" + branch["native_level"]
        item = {
            "assessment_id": assessment_id, "target": target, "status": status,
            "reason": reason, "observation": observation, "image_refs": _refs(),
            "upstream_refs": ([{"result_id": localization_id,
                                "assessment_id": "loc-" + branch["level"].lower()}]
                              if branch["level"] != "unknown" else []),
        }
        component = "foraminal_stenosis" if branch["zone"] == "foraminal" else "subarticular_stenosis"
        method = inputs.get("producer_method", defaults["producer_method"])
        results.append(_result(study, result_id, component, [item], method=method))
        states = dict(defaults["reviews"])
        states.update(inputs.get("reviews", {}))
        row = {
            "source_ref": {"result_id": result_id, "assessment_id": assessment_id},
            "target": copy.deepcopy(target), "reviewer_id": "synthetic-reviewer",
            "states": states, "covered_series_ids": sorted(ref["series_id"] for ref in item["image_refs"]),
            "evidence_refs": ["synthetic-review-proof"],
        }
        if row not in review_rows:
            review_rows.append(row)

    if "disc_evidence" in inputs:
        disc = inputs["disc_evidence"]
        status = disc.get("status", "indeterminate")
        item = {
            "assessment_id": "disc-context",
            "target": {"level": disc.get("level", "L4-L5"),
                       "side": disc.get("side", "unknown"), "zone": disc.get("zone", "disc")},
            "status": status,
            "reason": None if status == "assessed" else disc.get("reason", "Fictitious unresolved disc context"),
            "observation": (None if status in ("not_run", "error") else {
                "label": disc.get("label", disc.get("presence", "unknown")),
                "scale_id": disc.get("scale_id", "fixture:unaccepted-disc-scale:v1"), "measurements": [],
            }),
            "image_refs": _refs(), "upstream_refs": [],
        }
        results.append(_result(study, "synthetic-disc-context", "disc_analysis", [item]))

    artifacts = []
    used_artifact_ids = set()
    for result in results:
        artifact_id = "artifact-" + result["result_id"]
        if artifact_id in used_artifact_ids:
            continue
        used_artifact_ids.add(artifact_id)
        artifacts.append(artifact_binding(artifact_id, result, document_sha256(result)))
    proof = {
        "schema_version": "1.0", "kind": "fusion_review_evidence",
        "evidence_id": "synthetic-review-proof", "study_id": study["study_id"],
        "study_digest": manifest_digest(study), "synthetic": True,
        "description": "Fictitious technical review proof; no actual review, images or registration.",
    }
    artifacts.append(artifact_binding("synthetic-review-proof", proof, document_sha256(proof)))
    for flag, kind in (("localizer_comparison", "localizer_comparison"),
                       ("allow_unverified_registration", "rsna_spatial_evidence"),
                       ("same_FrameOfReferenceUID", "rsna_spatial_evidence")):
        if flag not in inputs:
            continue
        context_evidence = {
            "schema_version": "1.0", "kind": kind,
            "study_id": study["study_id"], "study_digest": manifest_digest(study),
            "synthetic": True, flag: inputs[flag],
            "anatomical_numbering_confirmed": False, "cross_series_registration_confirmed": False,
            "description": "Fictitious native/model agreement cannot substitute for independent review.",
        }
        artifact_id = "synthetic-context-" + flag.lower()
        artifacts.append(artifact_binding(artifact_id, context_evidence, document_sha256(context_evidence)))
    review = {
        "schema_version": "1.0", "kind": "fusion_review_context",
        "context_id": "synthetic-fusion-context", "study_id": study["study_id"],
        "study_digest": manifest_digest(study), "synthetic": True,
        "input_artifacts_sha256": {}, "reviews": review_rows,
    }
    review_sha = rebind_review(review, artifacts)
    if inputs.get("input_sha256_matches") is False:
        review["input_artifacts_sha256"][artifacts[0]["artifact_id"]] = "0" * 64
        review_sha = document_sha256(review)
    if "review_context" in inputs and inputs["review_context"] is None:
        review, review_sha = None, None
    return study, results, artifacts, review, review_sha
