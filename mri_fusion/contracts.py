"""Strict, dependency-free boundaries for the separate MRI fusion contract."""

import hashlib
import json
from pathlib import Path
import re

from mri_root_impact.contracts import (
    ContractError,
    _validate_observation,
    _validate_producer,
    _validate_target,
    manifest_digest,
    validate_bundle,
    validate_study,
)


POLICY_SHA256 = "c3e574a12603a0a438eb1db50b9ce21b47b53fcae9b5d2900bbbfd3cc8744fed"
POLICY_PATH = Path(__file__).resolve().parents[1] / "harness" / "fusion-policy.json"
REVIEW_STATES = {
    "anatomical_numbering": {"confirmed", "unreviewed", "unknown", "unsupported", "failed", "contradictory"},
    "side_assignment": {"confirmed", "unreviewed", "unknown", "failed", "contradictory"},
    "source_geometry": {"verified", "unreviewed", "unknown", "failed", "contradictory"},
    "cross_series_registration": {"verified", "not_required_with_evidence", "unreviewed", "unknown", "failed", "contradictory"},
    "diagnostic_adequacy": {"sufficient", "insufficient", "unreviewed", "unknown", "failed", "contradictory"},
}


class FusionContractError(ValueError):
    """Fusion input or output violates its versioned boundary."""


def _require(condition, location, message):
    if not condition:
        raise FusionContractError("{}: {}".format(location, message))


def _object(value, fields, location):
    _require(isinstance(value, dict), location, "expected object")
    _require(set(value) == set(fields.split()), location, "expected fields: " + fields)


def _text(value, location):
    _require(isinstance(value, str) and bool(value.strip()), location, "expected nonempty string")


def _identifier(value, location):
    _text(value, location)
    _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)), location, "invalid identifier")


def _sha(value, location):
    _require(isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{64}", value)), location, "expected lowercase SHA-256")


def _list(value, location, nonempty=False):
    _require(isinstance(value, list), location, "expected array")
    _require(not nonempty or bool(value), location, "array must not be empty")


def _unique_texts(values, location, identifiers=False, nonempty=False):
    _list(values, location, nonempty)
    seen = set()
    for value in values:
        (_identifier if identifiers else _text)(value, location + "[]")
        _require(value not in seen, location, "duplicate reference")
        seen.add(value)
    return seen


def _digest(document, location):
    try:
        return manifest_digest(document)
    except ContractError as error:
        raise FusionContractError(location + ": " + str(error)) from error


def load_policy(path=None):
    """Accept only the exact, immutable F001 policy 0.1.0 bytes."""
    policy_path = POLICY_PATH if path is None else Path(path)
    try:
        raw = policy_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        _require(digest == POLICY_SHA256, "policy", "bytes do not match pinned F001 policy 0.1.0")
        # Parse the bytes just hashed; a second filesystem read could change
        # between validation and parsing. Exact pinning already guarantees that
        # these bytes have the reviewed policy's strict JSON structure.
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ContractError, ValueError) as error:
        if isinstance(error, FusionContractError):
            raise
        raise FusionContractError("policy: " + str(error)) from error
    return document, digest


def artifact_binding(artifact_id, document, file_sha256):
    """Bind loaded JSON content and the independently verified file bytes."""
    _identifier(artifact_id, "artifact_id")
    _require(isinstance(document, dict), "artifact.document", "expected object")
    _text(document.get("kind"), "artifact.document.kind")
    _sha(file_sha256, "artifact.sha256")
    result_id = document.get("result_id") if document["kind"] == "module_result" else None
    if document["kind"] == "module_result":
        _identifier(result_id, "artifact.result_id")
    return {
        "artifact_id": artifact_id,
        "kind": document["kind"],
        "result_id": result_id,
        "sha256": file_sha256,
        "document_digest": _digest(document, "artifact.document"),
    }


def _validate_bindings(results, artifacts):
    _list(artifacts, "artifacts")
    by_id, by_result = {}, {}
    for artifact in artifacts:
        _object(artifact, "artifact_id kind result_id sha256 document_digest", "artifact")
        _identifier(artifact["artifact_id"], "artifact.artifact_id")
        _text(artifact["kind"], "artifact.kind")
        _sha(artifact["sha256"], "artifact.sha256")
        _sha(artifact["document_digest"], "artifact.document_digest")
        _require(artifact["artifact_id"] not in by_id, "artifact.artifact_id", "duplicate identifier")
        by_id[artifact["artifact_id"]] = artifact
        if artifact["kind"] == "module_result":
            _identifier(artifact["result_id"], "artifact.result_id")
            _require(artifact["result_id"] not in by_result, "artifact.result_id", "result has multiple bindings")
            by_result[artifact["result_id"]] = artifact
        else:
            _require(artifact["result_id"] is None, "artifact.result_id", "non-module artifact requires null")
    _require(set(by_result) == {result["result_id"] for result in results}, "artifacts", "module bindings must match every supplied result exactly")
    for result in results:
        _require(by_result[result["result_id"]]["document_digest"] == _digest(result, "result"), "artifact.document_digest", "module document has changed")
    return by_id


def validate_inputs(study, results, artifacts, review_context=None, review_sha256=None):
    """Validate bound C001 inputs; return reviews without promoting any status.

    Missing states and partial series coverage are valid incomplete evidence. The
    engine applies their gates. This boundary cannot inspect evidence documents
    or verify actual file bytes; the runner must verify those before binding.
    """
    try:
        validate_bundle(study, results)
    except (ContractError, TypeError, KeyError) as error:
        raise FusionContractError("C001: " + str(error)) from error
    bindings = _validate_bindings(results, artifacts)
    if review_context is None:
        _require(review_sha256 is None, "review_sha256", "requires a review context")
        return {}
    _sha(review_sha256, "review_sha256")
    _object(review_context, "schema_version kind context_id study_id study_digest synthetic input_artifacts_sha256 reviews", "review_context")
    _require(review_context["schema_version"] == "1.0", "review_context.schema_version", "unsupported version")
    _require(review_context["kind"] == "fusion_review_context", "review_context.kind", "expected fusion_review_context")
    _identifier(review_context["context_id"], "review_context.context_id")
    _require(review_context["study_id"] == study["study_id"], "review_context.study_id", "does not match study")
    _sha(review_context["study_digest"], "review_context.study_digest")
    _require(review_context["study_digest"] == _digest(study, "study"), "review_context.study_digest", "does not match study")
    _require(type(review_context["synthetic"]) is bool and review_context["synthetic"] == study["synthetic"], "review_context.synthetic", "does not match study")
    expected_hashes = {key: value["sha256"] for key, value in bindings.items()}
    _require(isinstance(review_context["input_artifacts_sha256"], dict) and review_context["input_artifacts_sha256"] == expected_hashes, "review_context.input_artifacts_sha256", "must bind every input artifact exactly")
    assessments = {
        (result["result_id"], assessment["assessment_id"]): assessment
        for result in results for assessment in result["assessments"]
    }
    series_ids = {series["series_id"] for series in study["series"]}
    _list(review_context["reviews"], "review_context.reviews")
    reviews = {}
    for review in review_context["reviews"]:
        _object(review, "source_ref target reviewer_id states covered_series_ids evidence_refs", "review")
        _object(review["source_ref"], "result_id assessment_id", "review.source_ref")
        for key in ("result_id", "assessment_id"):
            _identifier(review["source_ref"][key], "review.source_ref." + key)
        reference = (review["source_ref"]["result_id"], review["source_ref"]["assessment_id"])
        _require(reference in assessments, "review.source_ref", "assessment was not supplied")
        _require(reference not in reviews, "review.source_ref", "duplicate review entry")
        _require(review["target"] == assessments[reference]["target"], "review.target", "does not match original assessment target")
        _text(review["reviewer_id"], "review.reviewer_id")
        _require(isinstance(review["states"], dict), "review.states", "expected object")
        _require(set(review["states"]).issubset(REVIEW_STATES), "review.states", "unknown gate")
        for gate, state in review["states"].items():
            _require(isinstance(state, str) and state in REVIEW_STATES[gate], "review.states." + gate, "unsupported state for this gate")
        coverage = _unique_texts(review["covered_series_ids"], "review.covered_series_ids", identifiers=True)
        _require(coverage.issubset(series_ids), "review.covered_series_ids", "unknown study series")
        evidence_refs = _unique_texts(review["evidence_refs"], "review.evidence_refs", identifiers=True, nonempty=True)
        for evidence in evidence_refs:
            _require(evidence in bindings, "review.evidence_refs", "artifact was not supplied")
            _require(bindings[evidence]["kind"] == "fusion_review_evidence", "review.evidence_refs", "proof requires fusion_review_evidence, not model agreement or run metadata")
        reviews[reference] = review
    return reviews


def validate_output(study, report):
    """Validate output structure, anatomy, gates, coverage and reference sets.

    The runner additionally recomputes the report from the bound original files;
    an internally consistent edited snapshot is not proof of source fidelity.
    """
    try:
        validate_study(study)
    except ContractError as error:
        raise FusionContractError("study: " + str(error)) from error
    _object(report, "schema_version kind fusion_id study_id study_digest synthetic policy_id policy_version policy_sha256 input_artifacts review_context_sha256 branches unresolved_native_evidence raw_disc_evidence root_summaries root_candidates coverage herniation_context etiology inference_performed fusion_performed global_root_clearance clinical_diagnosis", "fusion")
    _require(report["schema_version"] == "1.0" and report["kind"] == "mri_root_impact_fusion", "fusion", "unsupported output schema")
    _identifier(report["fusion_id"], "fusion.fusion_id")
    _require(report["study_id"] == study["study_id"], "fusion.study_id", "does not match study")
    _require(report["study_digest"] == _digest(study, "study"), "fusion.study_digest", "does not match study")
    _require(type(report["synthetic"]) is bool and report["synthetic"] == study["synthetic"], "fusion.synthetic", "does not match study")
    policy, policy_sha = load_policy()
    _require(report["policy_id"] == policy["policy_id"] and report["policy_version"] == policy["policy_version"] and report["policy_sha256"] == policy_sha, "fusion.policy", "does not match pinned policy")
    if report["review_context_sha256"] is not None:
        _sha(report["review_context_sha256"], "fusion.review_context_sha256")
    for key, expected in (("herniation_context", "not_evaluated"), ("etiology", "unassigned")):
        _require(report[key] == expected, "fusion." + key, "unsupported interpretation")
    for key, expected in (("inference_performed", False), ("fusion_performed", True), ("global_root_clearance", False)):
        _require(report[key] is expected, "fusion." + key, "incorrect execution or conclusion flag")
    _require(report["clinical_diagnosis"] is None, "fusion.clinical_diagnosis", "clinical diagnosis is outside Stage I")
    bindings = _output_bindings(report["input_artifacts"])
    levels = [entry["level"] for entry in policy["anatomy"]]
    anatomy = {entry["level"]: entry for entry in policy["anatomy"]}
    targets = [(level, side, zone) for level in levels for side in ("left", "right") for zone in ("foraminal", "subarticular")]
    _list(report["branches"], "fusion.branches")
    _require(len(report["branches"]) == 20, "fusion.branches", "expected twenty canonical branches")
    decisions = set(policy["grade_decisions_after_gates"].values())
    gate_names = {"source_assessed", "reviews_passed", "known_scale_and_label", "native_probabilities_usable", "evidence_consistent"}
    branch_map, mapped_groups, seen_evidence = {}, {}, set()
    for branch, (level, side, zone) in zip(report["branches"], targets):
        _object(branch, "branch_id target component rule_id decision reason_codes mapped_root candidate_root evidence gate_states herniation_context etiology", "branch")
        branch_id = zone + "-" + level + "-" + side
        _require(branch["branch_id"] == branch_id, "branch.branch_id", "canonical branch order or identifier mismatch")
        _require(branch["target"] == {"level": level, "side": side, "zone": zone}, "branch.target", "does not match canonical branch")
        component = zone + "_stenosis"
        _require(branch["component"] == component, "branch.component", "does not match zone")
        _require(branch["rule_id"] == policy["rules"][component]["rule_id"], "branch.rule_id", "does not match component")
        _require(isinstance(branch["decision"], str) and branch["decision"] in decisions, "branch.decision", "unsupported decision")
        _unique_texts(branch["reason_codes"], "branch.reason_codes")
        _object(branch["gate_states"], " ".join(sorted(gate_names)), "branch.gate_states")
        _require(all(type(value) is bool for value in branch["gate_states"].values()), "branch.gate_states", "every gate must be boolean")
        expected_root = dict(anatomy[level][zone], side=side)
        eligible = all(branch["gate_states"].values())
        _require(branch["mapped_root"] == (expected_root if eligible else None), "branch.mapped_root", "mapping requires every gate and explicit anatomy")
        if branch["decision"] == "potential_impact":
            _require(eligible and branch["candidate_root"] == expected_root, "branch.candidate_root", "positive candidate requires eligible anatomical mapping")
        else:
            _require(branch["candidate_root"] is None, "branch.candidate_root", "candidate is only allowed for potential_impact")
        if branch["decision"] != "indeterminate":
            _require(eligible, "branch.decision", "positive or limited negative requires every gate")
        _require(branch["herniation_context"] == "not_evaluated" and branch["etiology"] == "unassigned", "branch", "unsupported herniation or causal interpretation")
        _list(branch["evidence"], "branch.evidence", nonempty=eligible)
        for evidence in branch["evidence"]:
            _output_evidence(study, evidence, bindings, report["review_context_sha256"], seen_evidence)
            _require(evidence["component"] == component and evidence["target"] == branch["target"], "branch.evidence", "source must match branch target and component")
        if eligible:
            _eligible_output_evidence(branch, policy)
        branch_map[branch_id] = branch
        if eligible:
            mapped_groups.setdefault((expected_root["root"], side), []).append(branch_id)
    _list(report["unresolved_native_evidence"], "fusion.unresolved_native_evidence")
    for evidence in report["unresolved_native_evidence"]:
        _output_evidence(study, evidence, bindings, report["review_context_sha256"], seen_evidence, unresolved=True)
        _require(evidence["component"] in {"foraminal_stenosis", "subarticular_stenosis"} and "unknown" in evidence["target"].values(), "unresolved_native_evidence", "requires unknown stenosis target")
        _require(evidence["reason_codes"] == ["unknown_target"], "unresolved_native_evidence.reason_codes", "expected unknown_target")
    _list(report["raw_disc_evidence"], "fusion.raw_disc_evidence")
    for evidence in report["raw_disc_evidence"]:
        _output_evidence(study, evidence, bindings, report["review_context_sha256"], seen_evidence)
        _require(evidence["component"] == "disc_analysis", "raw_disc_evidence", "expected uninterpreted disc_analysis evidence")
    _output_aggregation(report, branch_map, mapped_groups, anatomy)


def _output_bindings(artifacts):
    _list(artifacts, "fusion.input_artifacts")
    bindings, result_ids = {}, set()
    for binding in artifacts:
        _object(binding, "artifact_id kind result_id sha256 document_digest", "artifact")
        _identifier(binding["artifact_id"], "artifact.artifact_id")
        _text(binding["kind"], "artifact.kind")
        _sha(binding["sha256"], "artifact.sha256")
        _sha(binding["document_digest"], "artifact.document_digest")
        _require(binding["artifact_id"] not in bindings, "artifact.artifact_id", "duplicate artifact")
        bindings[binding["artifact_id"]] = binding
        if binding["kind"] == "module_result":
            _identifier(binding["result_id"], "artifact.result_id")
            _require(binding["result_id"] not in result_ids, "artifact.result_id", "result has multiple bindings")
            result_ids.add(binding["result_id"])
        else:
            _require(binding["result_id"] is None, "artifact.result_id", "non-module artifact requires null")
    return bindings


def _output_evidence(study, evidence, bindings, review_sha, seen, unresolved=False):
    fields = "source_ref artifact_id artifact_sha256 component target status reason observation producer image_refs upstream_refs review_ref review_states review_series_coverage"
    _object(evidence, fields + (" reason_codes" if unresolved else ""), "evidence")
    _object(evidence["source_ref"], "result_id assessment_id", "evidence.source_ref")
    for key in ("result_id", "assessment_id"):
        _identifier(evidence["source_ref"][key], "evidence.source_ref." + key)
    reference = (evidence["source_ref"]["result_id"], evidence["source_ref"]["assessment_id"])
    _require(reference not in seen, "evidence.source_ref", "duplicate source assessment")
    seen.add(reference)
    _identifier(evidence["artifact_id"], "evidence.artifact_id")
    binding = bindings.get(evidence["artifact_id"])
    _require(binding is not None and binding["kind"] == "module_result" and binding["result_id"] == reference[0], "evidence.artifact_id", "must bind the original module result")
    _require(evidence["artifact_sha256"] == binding["sha256"], "evidence.artifact_sha256", "does not match artifact binding")
    _require(isinstance(evidence["component"], str) and evidence["component"] in {"foraminal_stenosis", "subarticular_stenosis", "disc_analysis"}, "evidence.component", "unsupported component")
    _require(isinstance(evidence["status"], str) and evidence["status"] in {"assessed", "indeterminate", "not_run", "error"}, "evidence.status", "unsupported source status")
    if evidence["status"] == "assessed":
        _require(evidence["reason"] is None and evidence["observation"] is not None, "evidence", "assessed requires observation and null reason")
        _require(isinstance(evidence["target"], dict) and "unknown" not in evidence["target"].values(), "evidence.target", "assessed target cannot be unknown")
    else:
        _text(evidence["reason"], "evidence.reason")
        if evidence["status"] in {"not_run", "error"}:
            _require(evidence["observation"] is None, "evidence.observation", "not_run/error cannot contain observation")
    try:
        _validate_target(evidence["target"], evidence["component"])
        if evidence["observation"] is not None:
            _validate_observation(evidence["observation"])
        _validate_producer(evidence["producer"], study["synthetic"], [evidence])
    except ContractError as error:
        raise FusionContractError("evidence: " + str(error)) from error
    images = {(series["series_id"], image["image_id"]): image for series in study["series"] for image in series["images"]}
    _list(evidence["image_refs"], "evidence.image_refs", nonempty=evidence["observation"] is not None)
    image_seen = set()
    for image in evidence["image_refs"]:
        _object(image, "series_id image_id frame_index", "evidence.image_ref")
        _identifier(image["series_id"], "image_ref.series_id")
        _identifier(image["image_id"], "image_ref.image_id")
        key = (image["series_id"], image["image_id"])
        _require(key in images and type(image["frame_index"]) is int and 0 <= image["frame_index"] < images[key]["frame_count"], "evidence.image_ref", "unknown image or frame")
        full_key = key + (image["frame_index"],)
        _require(full_key not in image_seen, "evidence.image_refs", "duplicate image reference")
        image_seen.add(full_key)
    _list(evidence["upstream_refs"], "evidence.upstream_refs")
    upstream_seen = set()
    module_ids = {binding["result_id"] for binding in bindings.values() if binding["kind"] == "module_result"}
    for upstream in evidence["upstream_refs"]:
        _object(upstream, "result_id assessment_id", "evidence.upstream_ref")
        _identifier(upstream["result_id"], "upstream_ref.result_id")
        _identifier(upstream["assessment_id"], "upstream_ref.assessment_id")
        key = (upstream["result_id"], upstream["assessment_id"])
        _require(key not in upstream_seen and key[0] in module_ids and key[0] != reference[0], "evidence.upstream_ref", "duplicate, self or unbound upstream reference")
        upstream_seen.add(key)
    if evidence["review_ref"] is not None:
        _object(evidence["review_ref"], "context_id source_ref", "evidence.review_ref")
        _identifier(evidence["review_ref"]["context_id"], "evidence.review_ref.context_id")
        _require(review_sha is not None and evidence["review_ref"]["source_ref"] == evidence["source_ref"], "evidence.review_ref", "must match source and a bound review context")
    _require(isinstance(evidence["review_states"], dict) and set(evidence["review_states"]).issubset(REVIEW_STATES), "evidence.review_states", "unknown review gate")
    for gate, state in evidence["review_states"].items():
        _require(isinstance(state, str) and state in REVIEW_STATES[gate], "evidence.review_states", "unsupported gate state")
    _require(isinstance(evidence["review_series_coverage"], str) and evidence["review_series_coverage"] in {"complete", "partial", "missing"}, "evidence.review_series_coverage", "unsupported coverage")


def _eligible_output_evidence(branch, policy):
    labels = set()
    for evidence in branch["evidence"]:
        _require(evidence["status"] == "assessed", "branch.gate_states.source_assessed", "mapped evidence must remain assessed")
        observation = evidence["observation"]
        _require(observation["scale_id"] == policy["accepted_input"]["scale_id"] and observation["label"] in policy["accepted_input"]["labels"], "branch.gate_states.known_scale_and_label", "mapped evidence requires accepted native grade")
        _require(evidence["review_ref"] is not None and evidence["review_series_coverage"] == "complete", "branch.gate_states.reviews_passed", "mapped evidence requires a bound review covering inputs")
        for gate, accepted in policy["required_review_states"].items():
            accepted_states = accepted if isinstance(accepted, list) else [accepted]
            _require(evidence["review_states"].get(gate) in accepted_states, "branch.gate_states.reviews_passed", "mapped evidence requires every review gate")
        probabilities = {item["name"]: item for item in observation["measurements"] if item["name"].startswith("p_")}
        class_order = policy["accepted_input"]["labels"]
        if probabilities or evidence["producer"]["method"] == "model":
            _require(set(probabilities) == {"p_" + label for label in class_order}, "branch.gate_states.native_probabilities_usable", "mapped model evidence requires all native class probabilities")
            for measure in probabilities.values():
                _require(measure["unit"] == "probability" and 0 <= measure["value"] <= 1, "branch.gate_states.native_probabilities_usable", "invalid probability value or unit")
            values = [probabilities["p_" + label]["value"] for label in class_order]
            _require(abs(sum(values) - 1.0) <= policy["native_probabilities"]["sum_absolute_tolerance"], "branch.gate_states.native_probabilities_usable", "probabilities must sum to one")
            winners = [label for label, value in zip(class_order, values) if value == max(values)]
            _require(winners == [observation["label"]], "branch.gate_states.native_probabilities_usable", "requires one probability maximum matching label")
        labels.add(observation["label"])
    _require(len(labels) == 1, "branch.gate_states.evidence_consistent", "mapped evidence must agree on grade")
    _require(branch["decision"] == policy["grade_decisions_after_gates"][next(iter(labels))], "branch.decision", "does not match eligible native grade")


def _output_aggregation(report, branches, mapped_groups, anatomy):
    _object(report["coverage"], "expected_branches potential_impact no_candidate_by_evaluated_criterion indeterminate unresolved_native_evidence duplicate_result_documents", "fusion.coverage")
    for key, value in report["coverage"].items():
        _require(type(value) is int and value >= 0, "fusion.coverage." + key, "expected nonnegative integer")
    expected_counts = {decision: sum(branch["decision"] == decision for branch in branches.values()) for decision in ("potential_impact", "no_candidate_by_evaluated_criterion", "indeterminate")}
    expected_counts.update(expected_branches=20, unresolved_native_evidence=len(report["unresolved_native_evidence"]))
    for key, value in expected_counts.items():
        _require(report["coverage"][key] == value, "fusion.coverage." + key, "does not match branches")
    _list(report["root_summaries"], "fusion.root_summaries")
    summaries = {}
    for summary in report["root_summaries"]:
        _object(summary, "root side decision coverage branch_refs supporting_branch_refs expected_branch_ids unresolved_expected_branch_ids global_root_clearance", "root_summary")
        _require(isinstance(summary["root"], str) and summary["root"] in {"L1", "L2", "L3", "L4", "L5", "S1"} and isinstance(summary["side"], str) and summary["side"] in {"left", "right"}, "root_summary", "unsupported root or side")
        group = (summary["root"], summary["side"])
        _require(group not in summaries and group in mapped_groups, "root_summary", "duplicate or unverified root group")
        summaries[group] = summary
        expected = {branch_id for branch_id, branch in branches.items() if branch["target"]["side"] == group[1] and anatomy[branch["target"]["level"]][branch["target"]["zone"]]["root"] == group[0]}
        mapped = set(mapped_groups[group])
        supporting = {branch_id for branch_id in mapped if branches[branch_id]["decision"] == "potential_impact"}
        unresolved = {branch_id for branch_id in expected if branches[branch_id]["mapped_root"] is None or branches[branch_id]["decision"] == "indeterminate"}
        for key, target in (("branch_refs", mapped), ("supporting_branch_refs", supporting), ("expected_branch_ids", expected), ("unresolved_expected_branch_ids", unresolved)):
            refs = _unique_texts(summary[key], "root_summary." + key)
            _require(refs == target, "root_summary." + key, "does not match contributing branches")
        decision = "potential_impact" if supporting else ("indeterminate" if unresolved else "no_candidate_by_evaluated_criterion")
        _require(summary["decision"] == decision, "root_summary.decision", "does not match branch decisions")
        _require(summary["coverage"] == ("partial" if unresolved else "complete_within_policy"), "root_summary.coverage", "does not match unresolved expected branches")
        _require(summary["global_root_clearance"] is False, "root_summary.global_root_clearance", "global clearance is outside policy")
    _require(set(summaries) == set(mapped_groups), "fusion.root_summaries", "must retain every verified root group")
    _list(report["root_candidates"], "fusion.root_candidates")
    candidates = {}
    for candidate in report["root_candidates"]:
        _object(candidate, "root side branch_refs", "root_candidate")
        _require(isinstance(candidate["root"], str) and isinstance(candidate["side"], str), "root_candidate", "expected root and side strings")
        group = (candidate["root"], candidate["side"])
        _require(group not in candidates and group in summaries and summaries[group]["decision"] == "potential_impact", "root_candidate", "requires a unique positive root summary")
        refs = _unique_texts(candidate["branch_refs"], "root_candidate.branch_refs", nonempty=True)
        _require(refs == set(summaries[group]["supporting_branch_refs"]), "root_candidate.branch_refs", "must retain exact positive support")
        candidates[group] = candidate
    _require(set(candidates) == {key for key, summary in summaries.items() if summary["decision"] == "potential_impact"}, "fusion.root_candidates", "must match positive summaries")
