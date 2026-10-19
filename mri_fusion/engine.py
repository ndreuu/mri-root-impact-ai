"""Deterministic research fusion; no image/model imports and no source edits."""

import copy
import math

from mri_root_impact.contracts import manifest_digest
from .contracts import FusionContractError, load_policy, validate_inputs


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
SIDES = ("left", "right")
ZONES = ("foraminal", "subarticular")
CLASSES = ("normal_mild", "moderate", "severe")


def normalize_results(results):
    """Collapse identical repeated identities, never choose a changed version."""
    if not isinstance(results, list):
        raise FusionContractError("results must be an array")
    unique, repeated = {}, 0
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("result_id"), str):
            raise FusionContractError("result requires a result_id")
        key = result["result_id"]
        if key in unique:
            if manifest_digest(unique[key]) != manifest_digest(result):
                raise FusionContractError("Same result identity has different content")
            repeated += 1
        else:
            unique[key] = result
    return [unique[k] for k in sorted(unique)], repeated


def branch_id(level, side, zone):
    return "{}-{}-{}".format(zone, level, side)


def _probabilities(result, assessment, policy):
    observation = assessment["observation"]
    if observation is None:
        return False, []
    if observation["scale_id"] != policy["accepted_input"]["scale_id"]:
        return False, []
    values = {m["name"]: m for m in observation["measurements"] if m["name"].startswith("p_")}
    names = {"p_" + c for c in CLASSES}
    if not values:
        missing = result["producer"]["method"] == "model"
        return not missing, ["native_probabilities_missing"] if missing else []
    if set(values) != names:
        raise FusionContractError("Native probabilities require all three classes")
    probabilities = []
    for name in CLASSES:
        measure = values["p_" + name]
        value = measure["value"]
        try:
            valid = (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)
        except OverflowError:
            valid = False
        if not valid or measure["unit"] != "probability":
            raise FusionContractError("Invalid native probability value or unit")
        probabilities.append(value)
    if abs(sum(probabilities) - 1.) > policy["native_probabilities"]["sum_absolute_tolerance"]:
        raise FusionContractError("Native class probabilities do not sum to one")
    maximum = max(probabilities)
    winners = [c for c, p in zip(CLASSES, probabilities) if p == maximum]
    if len(winners) != 1:
        return False, ["tied_probability_maximum"]
    if observation["label"] in CLASSES and observation["label"] != winners[0]:
        raise FusionContractError("Native label differs from its unique probability maximum")
    return True, []


def _all_series(result_id, assessment_id, results):
    items = {(r["result_id"], a["assessment_id"]): a for r in results for a in r["assessments"]}
    pending, visited, series = [(result_id, assessment_id)], set(), set()
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        item = items[key]
        series.update(ref["series_id"] for ref in item["image_refs"])
        pending.extend((ref["result_id"], ref["assessment_id"]) for ref in item["upstream_refs"])
    return series


def _review_reasons(review, required_series, context):
    if review is None:
        return False, "missing", ["review_context_missing" if context is None else "review_target_missing",
                                  "numbering_unreviewed", "registration_unverified"]
    accepted = {"anatomical_numbering": {"confirmed"}, "side_assignment": {"confirmed"},
                "source_geometry": {"verified"}, "cross_series_registration": {"verified", "not_required_with_evidence"},
                "diagnostic_adequacy": {"sufficient"}}
    names = {"anatomical_numbering": "numbering", "side_assignment": "side_assignment",
             "source_geometry": "source_geometry", "cross_series_registration": "registration",
             "diagnostic_adequacy": "diagnostic_adequacy"}
    reasons = []
    for gate, states in accepted.items():
        state = review["states"].get(gate, "unreviewed")
        if state not in states:
            reasons.append(names[gate] + "_" + state)
    coverage = "complete" if required_series <= set(review["covered_series_ids"]) else "partial"
    if coverage != "complete":
        reasons.append("review_series_coverage_incomplete")
    return not reasons, coverage, reasons


def _snapshot(result, item, binding, review, coverage, context):
    return {"source_ref": {"result_id": result["result_id"], "assessment_id": item["assessment_id"]},
            "artifact_id": binding["artifact_id"], "artifact_sha256": binding["sha256"],
            "component": result["component"], "target": copy.deepcopy(item["target"]),
            "status": item["status"], "reason": item["reason"], "observation": copy.deepcopy(item["observation"]),
            "producer": copy.deepcopy(result["producer"]), "image_refs": copy.deepcopy(item["image_refs"]),
            "upstream_refs": copy.deepcopy(item["upstream_refs"]),
            "review_ref": None if review is None else {"context_id": context["context_id"], "source_ref": copy.deepcopy(review["source_ref"])},
            "review_states": {} if review is None else copy.deepcopy(review["states"]),
            "review_series_coverage": coverage}


def _root_summaries(branches, policy):
    table = {row["level"]: row for row in policy["anatomy"]}
    grouped = {}
    for row in branches:
        if row["mapped_root"] is not None:
            root = row["mapped_root"]
            grouped.setdefault((root["root"], root["side"]), []).append(row)
    summaries = []
    for (root, side), rows in sorted(grouped.items()):
        expected = [row["branch_id"] for row in branches if row["target"]["side"] == side
                    and table[row["target"]["level"]][row["target"]["zone"]]["root"] == root]
        unresolved = [row["branch_id"] for row in branches if row["branch_id"] in expected
                      and (row["mapped_root"] is None or row["decision"] == "indeterminate")]
        positive = [row["branch_id"] for row in rows if row["decision"] == "potential_impact"]
        decision = ("potential_impact" if positive else "indeterminate" if unresolved
                    else "no_candidate_by_evaluated_criterion")
        summaries.append({"root": root, "side": side, "decision": decision,
                          "coverage": "partial" if unresolved else "complete_within_policy",
                          "branch_refs": [r["branch_id"] for r in rows], "supporting_branch_refs": positive,
                          "expected_branch_ids": expected, "unresolved_expected_branch_ids": unresolved,
                          "global_root_clearance": False})
    return summaries


def fuse(study, results, artifacts, review_context=None, review_sha256=None, policy_path=None):
    """Apply pinned rules to bound C001 documents, preserving unresolved evidence.

    Artifact byte hashes are assertions supplied by the I/O boundary. This pure
    function checks their document binding; runner verifies actual file bytes.
    It validates typed review attestations, not the clinical truth of a review.
    """
    policy, policy_sha = load_policy(policy_path)
    results, repeated = normalize_results(results)
    reviews = validate_inputs(study, results, artifacts, review_context, review_sha256)
    by_result = {b["result_id"]: b for b in artifacts if b["kind"] == "module_result"}
    table = {row["level"]: row for row in policy["anatomy"]}
    grouped, unresolved, discs = {}, [], []
    for result in results:
        for item in result["assessments"]:
            key = result["result_id"], item["assessment_id"]
            review = reviews.get(key)
            required_series = _all_series(*key, results)
            review_ok, review_coverage, review_reasons = _review_reasons(review, required_series, review_context)
            source = _snapshot(result, item, by_result[result["result_id"]], review, review_coverage, review_context)
            if result["component"] == "disc_analysis":
                discs.append(source)
                continue
            if result["component"] not in policy["accepted_input"]["components"]:
                continue
            probabilities_ok, probability_reasons = _probabilities(result, item, policy)
            target = item["target"]
            if target["level"] not in LEVELS or target["side"] not in SIDES or target["zone"] not in ZONES:
                source["reason_codes"] = ["unknown_target"]
                unresolved.append(source)
                continue
            grouped.setdefault((target["level"], target["side"], target["zone"]), []).append(
                (result, item, source, review_ok, review_reasons, probabilities_ok, probability_reasons))
    branches = []
    for level in LEVELS:
        for side in SIDES:
            for zone in ZONES:
                component = zone + "_stenosis"
                sources = grouped.get((level, side, zone), [])
                row = {"branch_id": branch_id(level, side, zone), "target": {"level": level, "side": side, "zone": zone},
                       "component": component, "rule_id": policy["rules"][component]["rule_id"],
                       "decision": "indeterminate", "reason_codes": [], "mapped_root": None, "candidate_root": None,
                       "evidence": [copy.deepcopy(s[2]) for s in sources], "gate_states": {
                           "source_assessed": bool(sources) and all(s[1]["status"] == "assessed" for s in sources),
                           "reviews_passed": bool(sources) and all(s[3] for s in sources),
                           "known_scale_and_label": bool(sources) and all(s[1]["observation"] is not None
                               and s[1]["observation"]["scale_id"] == policy["accepted_input"]["scale_id"]
                               and s[1]["observation"]["label"] in CLASSES for s in sources),
                           "native_probabilities_usable": bool(sources) and all(s[5] for s in sources),
                           "evidence_consistent": bool(sources)},
                       "herniation_context": "not_evaluated", "etiology": "unassigned"}
                reasons = []
                if not sources:
                    reasons.extend(["source_missing", "review_target_missing"])
                labels = set()
                for _, item, _, _, review_reasons, _, probability_reasons in sources:
                    if item["status"] != "assessed":
                        reasons.append("source_" + item["status"])
                    observation = item["observation"]
                    if observation is not None:
                        labels.add((observation["scale_id"], observation["label"]))
                        if observation["scale_id"] != policy["accepted_input"]["scale_id"]:
                            reasons.append("unsupported_scale")
                        if observation["label"] not in CLASSES:
                            reasons.append("unsupported_label")
                    reasons.extend(review_reasons + probability_reasons)
                if len(labels) > 1:
                    row["gate_states"]["evidence_consistent"] = False
                    reasons.append("same_target_evidence_conflict")
                if all(row["gate_states"].values()):
                    root = {"root": table[level][zone]["root"], "side": side, "role": table[level][zone]["role"]}
                    row["mapped_root"] = root
                    grade = sources[0][1]["observation"]["label"]
                    row["decision"] = policy["grade_decisions_after_gates"][grade]
                    if row["decision"] == "potential_impact":
                        row["candidate_root"] = dict(root)
                        reasons.append("severe_stenosis_research_candidate")
                    elif grade == "moderate":
                        reasons.append("moderate_threshold_unestablished")
                    else:
                        reasons.append("no_candidate_by_evaluated_criterion")
                reasons.append("herniation_not_evaluated")
                row["reason_codes"] = sorted(set(reasons))
                branches.append(row)
    summaries = _root_summaries(branches, policy)
    report = {"schema_version": "1.0", "kind": "mri_root_impact_fusion", "fusion_id": "pending",
              "study_id": study["study_id"], "study_digest": manifest_digest(study), "synthetic": study["synthetic"],
              "policy_id": policy["policy_id"], "policy_version": policy["policy_version"], "policy_sha256": policy_sha,
              "input_artifacts": copy.deepcopy(sorted(artifacts, key=lambda b: b["artifact_id"])),
              "review_context_sha256": review_sha256, "branches": branches,
              "unresolved_native_evidence": unresolved, "raw_disc_evidence": discs,
              "root_summaries": summaries, "root_candidates": [{"root": r["root"], "side": r["side"],
                  "branch_refs": list(r["supporting_branch_refs"])} for r in summaries if r["decision"] == "potential_impact"],
              "coverage": {"expected_branches": 20, "potential_impact": sum(b["decision"] == "potential_impact" for b in branches),
                  "no_candidate_by_evaluated_criterion": sum(b["decision"] == "no_candidate_by_evaluated_criterion" for b in branches),
                  "indeterminate": sum(b["decision"] == "indeterminate" for b in branches),
                  "unresolved_native_evidence": len(unresolved), "duplicate_result_documents": repeated},
              "herniation_context": "not_evaluated", "etiology": "unassigned", "inference_performed": False,
              "fusion_performed": True, "global_root_clearance": False, "clinical_diagnosis": None}
    report["fusion_id"] = "fusion-" + manifest_digest({k: v for k, v in report.items() if k != "fusion_id"})[:24]
    return report
