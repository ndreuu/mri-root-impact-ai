"""Check hashes, source binding and review boundaries using fictitious data."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mri_fusion.contracts import (
    FusionContractError,
    POLICY_PATH,
    POLICY_SHA256,
    artifact_binding,
    load_policy,
    validate_inputs,
    validate_output,
)
from mri_root_impact.contracts import manifest_digest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "harness" / "fixtures" / "contracts"


class FusionContractTests(unittest.TestCase):
    def setUp(self):
        self.study = json.loads((FIXTURES / "study.json").read_text())
        self.results = [json.loads((FIXTURES / name).read_text()) for name in ("localization.json", "stenosis-assessed.json")]
        self.artifacts = [artifact_binding("module-" + str(index), result, str(index) * 64) for index, result in enumerate(self.results)]
        proof = {
            "kind": "fusion_review_evidence",
            "synthetic": True,
            "description": "Fictitious proof for tests only; no real anatomical review.",
        }
        self.artifacts.append(artifact_binding("synthetic-proof", proof, "a" * 64))
        assessment = self.results[1]["assessments"][0]
        self.review = {
            "schema_version": "1.0",
            "kind": "fusion_review_context",
            "context_id": "synthetic-review-001",
            "study_id": self.study["study_id"],
            "study_digest": manifest_digest(self.study),
            "synthetic": True,
            "input_artifacts_sha256": {artifact["artifact_id"]: artifact["sha256"] for artifact in self.artifacts},
            "reviews": [{
                "source_ref": {"result_id": self.results[1]["result_id"], "assessment_id": assessment["assessment_id"]},
                "target": copy.deepcopy(assessment["target"]),
                "reviewer_id": "fictitious-test-reviewer",
                "states": {
                    "anatomical_numbering": "confirmed",
                    "side_assignment": "confirmed",
                    "source_geometry": "verified",
                    "cross_series_registration": "not_required_with_evidence",
                    "diagnostic_adequacy": "sufficient",
                },
                "covered_series_ids": [series["series_id"] for series in self.study["series"]],
                "evidence_refs": ["synthetic-proof"],
            }],
        }

    def validate(self, review=True):
        return validate_inputs(self.study, self.results, self.artifacts, self.review if review else None, "b" * 64 if review else None)

    def test_pinned_policy_copy_accepts_only_exact_original_bytes(self):
        policy, digest = load_policy()
        self.assertEqual(digest, POLICY_SHA256)
        self.assertEqual(policy["policy_version"], "0.1.0")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_bytes(POLICY_PATH.read_bytes())
            self.assertEqual(load_policy(path), (policy, digest))
            for mutation in ("positive_labels", "unknown_flag", "mapping"):
                changed = copy.deepcopy(policy)
                if mutation == "positive_labels":
                    changed["threshold"]["positive_labels"].append("moderate")
                elif mutation == "mapping":
                    changed["anatomy"][-1]["subarticular"]["root"] = "L6"
                else:
                    changed["unknown_flag"] = True
                path.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                    load_policy(path)

    def test_policy_reformatting_and_missing_file_are_not_pinned_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            with self.assertRaises(FusionContractError):
                load_policy(path)
            path.write_bytes(POLICY_PATH.read_bytes() + b"\n")
            with self.assertRaises(FusionContractError):
                load_policy(path)

    def test_artifact_binding_preserves_raw_and_canonical_hash_roles(self):
        result = self.results[1]
        raw = json.dumps(result, indent=3).encode()
        binding = artifact_binding("test-result", result, hashlib.sha256(raw).hexdigest())
        self.assertEqual(set(binding), {"artifact_id", "kind", "result_id", "sha256", "document_digest"})
        self.assertEqual(binding["document_digest"], manifest_digest(result))
        self.assertNotEqual(binding["sha256"], binding["document_digest"])
        self.assertEqual(binding["result_id"], result["result_id"])
        self.assertIsNone(self.artifacts[-1]["result_id"])

    def test_bound_review_returns_reference_map_without_mutation(self):
        before = copy.deepcopy((self.study, self.results, self.artifacts, self.review))
        reviews = self.validate()
        reference = (self.results[1]["result_id"], self.results[1]["assessments"][0]["assessment_id"])
        self.assertEqual(reviews, {reference: self.review["reviews"][0]})
        self.assertEqual((self.study, self.results, self.artifacts, self.review), before)
        self.assertEqual(self.validate(review=False), {})

    def test_missing_review_states_and_partial_coverage_are_valid_incomplete_evidence(self):
        self.review["reviews"][0]["states"] = {}
        self.review["reviews"][0]["covered_series_ids"] = []
        self.assertEqual(len(self.validate()), 1)

    def test_review_does_not_promote_indeterminate_source(self):
        assessment = self.results[1]["assessments"][0]
        assessment.update(status="indeterminate", reason="Synthetic numbering unresolved")
        self.artifacts[1] = artifact_binding("module-1", self.results[1], "1" * 64)
        self.validate()
        self.assertEqual(assessment["status"], "indeterminate")
        self.assertEqual(assessment["reason"], "Synthetic numbering unresolved")

    def test_tampered_result_or_digest_and_missing_binding_are_rejected(self):
        for mutation in ("result", "digest", "missing", "extra_result_binding", "duplicate_artifact"):
            with self.subTest(mutation=mutation):
                artifacts, results = copy.deepcopy((self.artifacts, self.results))
                if mutation == "result":
                    results[1]["assessments"][0]["observation"]["label"] = "severe"
                elif mutation == "digest":
                    artifacts[1]["document_digest"] = "f" * 64
                elif mutation == "missing":
                    artifacts.pop(1)
                elif mutation == "extra_result_binding":
                    extra = copy.deepcopy(artifacts[1])
                    extra.update(artifact_id="extra", result_id="unknown-result")
                    artifacts.append(extra)
                else:
                    artifacts.append(copy.deepcopy(artifacts[-1]))
                with self.assertRaises(FusionContractError):
                    validate_inputs(self.study, results, artifacts)

    def test_module_id_cannot_be_disguised_as_nonmodule_binding(self):
        self.artifacts[1]["kind"] = "geometry_metadata"
        with self.assertRaises(FusionContractError):
            self.validate()
        self.artifacts[1]["result_id"] = None
        with self.assertRaises(FusionContractError):
            self.validate()

    def test_changed_study_or_invalid_c001_never_reaches_review(self):
        self.study["series"][0]["images"][0]["frame_count"] += 1
        with self.assertRaises(FusionContractError):
            self.validate()

    def test_context_must_match_study_identity_and_full_artifact_hash_map(self):
        for mutation in ("study_id", "study_digest", "synthetic", "hash_value", "hash_missing", "hash_extra"):
            review = copy.deepcopy(self.review)
            if mutation == "study_id":
                review["study_id"] = "other-study"
            elif mutation == "study_digest":
                review["study_digest"] = "f" * 64
            elif mutation == "synthetic":
                review["synthetic"] = False
            elif mutation == "hash_value":
                review["input_artifacts_sha256"]["module-1"] = "f" * 64
            elif mutation == "hash_missing":
                review["input_artifacts_sha256"].pop("synthetic-proof")
            else:
                review["input_artifacts_sha256"]["unknown"] = "f" * 64
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_inputs(self.study, self.results, self.artifacts, review, "b" * 64)

    def test_review_requires_its_file_hash_and_hash_cannot_exist_alone(self):
        for value in (None, "invalid", "B" * 64, 1):
            with self.subTest(value=value), self.assertRaises(FusionContractError):
                validate_inputs(self.study, self.results, self.artifacts, self.review, value)
        with self.assertRaises(FusionContractError):
            validate_inputs(self.study, self.results, self.artifacts, None, "b" * 64)

    def test_review_ref_target_and_duplicate_reviews_are_rejected(self):
        for mutation in ("ref", "target", "duplicate", "reviewer"):
            review = copy.deepcopy(self.review)
            if mutation == "ref":
                review["reviews"][0]["source_ref"]["assessment_id"] = "not-supplied"
            elif mutation == "target":
                review["reviews"][0]["target"]["side"] = "right"
            elif mutation == "reviewer":
                review["reviews"][0]["reviewer_id"] = " "
            else:
                review["reviews"].append(copy.deepcopy(review["reviews"][0]))
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_inputs(self.study, self.results, self.artifacts, review, "b" * 64)

    def test_gate_specific_states_reject_success_word_from_other_gate(self):
        for gate, state in (("source_geometry", "confirmed"), ("side_assignment", "verified"), ("diagnostic_adequacy", "verified"), ("unknown_gate", "verified")):
            review = copy.deepcopy(self.review)
            review["reviews"][0]["states"][gate] = state
            with self.subTest(gate=gate), self.assertRaises(FusionContractError):
                validate_inputs(self.study, self.results, self.artifacts, review, "b" * 64)

    def test_series_coverage_rejects_unknown_duplicate_and_wrong_type(self):
        for coverage in (["absent-series"], ["sag-t1", "sag-t1"], "sag-t1"):
            review = copy.deepcopy(self.review)
            review["reviews"][0]["covered_series_ids"] = coverage
            with self.subTest(coverage=coverage), self.assertRaises(FusionContractError):
                validate_inputs(self.study, self.results, self.artifacts, review, "b" * 64)

    def test_evidence_requires_known_explicit_proof_not_model_metadata(self):
        for evidence_refs in ([], ["absent-proof"], ["module-1"], ["synthetic-proof", "synthetic-proof"]):
            review = copy.deepcopy(self.review)
            review["reviews"][0]["evidence_refs"] = evidence_refs
            with self.subTest(evidence_refs=evidence_refs), self.assertRaises(FusionContractError):
                validate_inputs(self.study, self.results, self.artifacts, review, "b" * 64)

    def test_unknown_or_missing_context_review_and_binding_fields_are_rejected(self):
        for object_name in ("context", "review", "binding"):
            for mutation in ("missing", "extra"):
                review, artifacts = copy.deepcopy((self.review, self.artifacts))
                target = {"context": review, "review": review["reviews"][0], "binding": artifacts[0]}[object_name]
                if mutation == "missing":
                    target.pop(next(iter(target)))
                else:
                    target["bypass_review"] = True
                with self.subTest(object_name=object_name, mutation=mutation), self.assertRaises(FusionContractError):
                    validate_inputs(self.study, self.results, artifacts, review, "b" * 64)

    def output_report(self, positive=False):
        policy, digest = load_policy()
        branches = []
        for anatomy in policy["anatomy"]:
            for side in ("left", "right"):
                for zone in ("foraminal", "subarticular"):
                    component = zone + "_stenosis"
                    branches.append({
                        "branch_id": zone + "-" + anatomy["level"] + "-" + side,
                        "target": {"level": anatomy["level"], "side": side, "zone": zone},
                        "component": component,
                        "rule_id": policy["rules"][component]["rule_id"],
                        "decision": "indeterminate",
                        "reason_codes": ["missing_source"],
                        "mapped_root": None,
                        "candidate_root": None,
                        "evidence": [],
                        "gate_states": dict.fromkeys(("source_assessed", "reviews_passed", "known_scale_and_label", "native_probabilities_usable", "evidence_consistent"), False),
                        "herniation_context": "not_evaluated",
                        "etiology": "unassigned",
                    })
        report = {
            "schema_version": "1.0", "kind": "mri_root_impact_fusion", "fusion_id": "synthetic-output-001",
            "study_id": self.study["study_id"], "study_digest": manifest_digest(self.study), "synthetic": True,
            "policy_id": policy["policy_id"], "policy_version": policy["policy_version"], "policy_sha256": digest,
            "input_artifacts": copy.deepcopy(self.artifacts), "review_context_sha256": None,
            "branches": branches, "unresolved_native_evidence": [], "raw_disc_evidence": [],
            "root_summaries": [], "root_candidates": [],
            "coverage": {"expected_branches": 20, "potential_impact": 0, "no_candidate_by_evaluated_criterion": 0, "indeterminate": 20, "unresolved_native_evidence": 0, "duplicate_result_documents": 0},
            "herniation_context": "not_evaluated", "etiology": "unassigned", "inference_performed": False,
            "fusion_performed": True, "global_root_clearance": False, "clinical_diagnosis": None,
        }
        if positive:
            source = copy.deepcopy(self.results[1])
            source["assessments"][0]["observation"].update(label="severe", scale_id=policy["accepted_input"]["scale_id"])
            report["input_artifacts"][1] = artifact_binding("module-1", source, "1" * 64)
            assessment = source["assessments"][0]
            source_ref = {"result_id": source["result_id"], "assessment_id": assessment["assessment_id"]}
            evidence = {key: copy.deepcopy(assessment[key]) for key in ("target", "status", "reason", "observation", "image_refs", "upstream_refs")}
            evidence.update(source_ref=source_ref, artifact_id="module-1", artifact_sha256="1" * 64, component=source["component"], producer=copy.deepcopy(source["producer"]), review_ref={"context_id": self.review["context_id"], "source_ref": copy.deepcopy(source_ref)}, review_states=copy.deepcopy(self.review["reviews"][0]["states"]), review_series_coverage="complete")
            branch = next(item for item in branches if item["branch_id"] == "foraminal-L4-L5-left")
            root = {"root": "L4", "side": "left", "role": "exiting"}
            branch.update(decision="potential_impact", mapped_root=copy.deepcopy(root), candidate_root=copy.deepcopy(root), evidence=[evidence], gate_states=dict.fromkeys(branch["gate_states"], True), reason_codes=["severe_stenosis"])
            report["review_context_sha256"] = "b" * 64
            report["coverage"].update(potential_impact=1, indeterminate=19)
            report["root_summaries"] = [{
                "root": "L4", "side": "left", "decision": "potential_impact", "coverage": "partial",
                "branch_refs": [branch["branch_id"]], "supporting_branch_refs": [branch["branch_id"]],
                "expected_branch_ids": ["subarticular-L3-L4-left", branch["branch_id"]],
                "unresolved_expected_branch_ids": ["subarticular-L3-L4-left"], "global_root_clearance": False,
            }]
            report["root_candidates"] = [{"root": "L4", "side": "left", "branch_refs": [branch["branch_id"]]}]
        return report

    def test_output_empty_evidence_preserves_twenty_unknown_branches(self):
        report = self.output_report()
        before = copy.deepcopy(report)
        self.assertIsNone(validate_output(self.study, report))
        self.assertEqual(report, before)
        self.assertEqual(report["root_candidates"], [])

    def test_output_positive_requires_reviewed_explicit_root_and_preserves_partial_coverage(self):
        report = self.output_report(positive=True)
        self.assertIsNone(validate_output(self.study, report))
        self.assertEqual(report["root_summaries"][0]["coverage"], "partial")
        self.assertEqual(report["root_candidates"][0]["root"], "L4")

    def test_output_rejects_candidate_from_failed_gate_or_unresolved_source(self):
        for mutation in ("gate", "source", "review", "probable_grade"):
            report = self.output_report(positive=True)
            branch = next(item for item in report["branches"] if item["candidate_root"] is not None)
            if mutation == "gate":
                branch["gate_states"]["reviews_passed"] = False
            elif mutation == "source":
                branch["evidence"][0].update(status="indeterminate", reason="Synthetic unresolved numbering")
            elif mutation == "review":
                branch["evidence"][0]["review_states"]["cross_series_registration"] = "unreviewed"
            else:
                branch["evidence"][0]["observation"]["label"] = "moderate"
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_output(self.study, report)

    def test_output_rejects_wrong_anatomical_mapping_and_nonpositive_candidate(self):
        for mutation in ("root", "role", "side", "negative_candidate"):
            report = self.output_report(positive=True)
            branch = next(item for item in report["branches"] if item["candidate_root"] is not None)
            if mutation == "negative_candidate":
                branch["decision"] = "indeterminate"
            else:
                branch["mapped_root"][mutation] = {"root": "L6", "role": "traversing", "side": "right"}[mutation]
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_output(self.study, report)

    def test_output_rejects_incorrect_coverage_and_root_summary_support(self):
        for mutation in ("count", "summary_refs", "expected", "unresolved", "candidate_refs", "no_summary", "extra_candidate"):
            report = self.output_report(positive=True)
            if mutation == "count":
                report["coverage"]["indeterminate"] = 0
            elif mutation == "summary_refs":
                report["root_summaries"][0]["supporting_branch_refs"] = []
            elif mutation == "expected":
                report["root_summaries"][0]["expected_branch_ids"] = ["foraminal-L4-L5-left"]
            elif mutation == "unresolved":
                report["root_summaries"][0]["unresolved_expected_branch_ids"] = []
            elif mutation == "candidate_refs":
                report["root_candidates"][0]["branch_refs"] = ["foraminal-L4-L5-right"]
            elif mutation == "no_summary":
                report["root_summaries"] = []
            else:
                report["root_candidates"].append(copy.deepcopy(report["root_candidates"][0]))
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_output(self.study, report)

    def test_output_rejects_schema_extension_branch_loss_and_claimed_diagnosis(self):
        for mutation in ("extension", "missing_branch", "order", "diagnosis", "clearance", "inference", "policy"):
            report = self.output_report()
            if mutation == "extension":
                report["clinical_probability"] = 0.9
            elif mutation == "missing_branch":
                report["branches"].pop()
            elif mutation == "order":
                report["branches"][0], report["branches"][1] = report["branches"][1], report["branches"][0]
            elif mutation == "diagnosis":
                report["clinical_diagnosis"] = "radiculopathy"
            elif mutation == "clearance":
                report["global_root_clearance"] = True
            elif mutation == "inference":
                report["inference_performed"] = True
            else:
                report["policy_sha256"] = "f" * 64
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_output(self.study, report)

    def test_output_evidence_cannot_change_bound_artifact_or_duplicate_source(self):
        for mutation in ("hash", "binding", "duplicate", "image", "review_ref"):
            report = self.output_report(positive=True)
            branch = next(item for item in report["branches"] if item["candidate_root"] is not None)
            evidence = branch["evidence"][0]
            if mutation == "hash":
                evidence["artifact_sha256"] = "f" * 64
            elif mutation == "binding":
                evidence["artifact_id"] = "synthetic-proof"
            elif mutation == "duplicate":
                branch["evidence"].append(copy.deepcopy(evidence))
            elif mutation == "image":
                evidence["image_refs"][0]["frame_index"] = 999
            else:
                evidence["review_ref"]["source_ref"]["assessment_id"] = "other-assessment"
            with self.subTest(mutation=mutation), self.assertRaises(FusionContractError):
                validate_output(self.study, report)

    def test_output_mapped_model_cannot_claim_usable_missing_invalid_or_tied_probabilities(self):
        cases = ([], [0.9], [0.5, 0, 0.5], [0.8, 0.1, 0.1], [0.2, 0.3, 0.7])
        for values in cases:
            report = self.output_report(positive=True)
            branch = next(item for item in report["branches"] if item["candidate_root"] is not None)
            evidence = branch["evidence"][0]
            evidence["producer"].update(method="model", model={"name": "fictitious-model", "version": "test-only", "weights_sha256": "c" * 64})
            evidence["observation"]["measurements"] = [{"name": "p_" + label, "value": value, "unit": "probability"} for label, value in zip(("normal_mild", "moderate", "severe"), values)]
            with self.subTest(values=values), self.assertRaises(FusionContractError):
                validate_output(self.study, report)

    def test_duplicate_identity_does_not_confuse_json_boolean_with_number(self):
        from mri_fusion.engine import normalize_results
        first = copy.deepcopy(self.results[1])
        first["assessments"][0]["observation"]["measurements"] = [{"name": "extra_native_measurement", "value": 1, "unit": "fixture"}]
        changed = copy.deepcopy(first)
        changed["assessments"][0]["observation"]["measurements"][0]["value"] = True
        self.assertEqual(first, changed, "Python equality alone must not decide JSON identity")
        self.assertNotEqual(manifest_digest(first), manifest_digest(changed))
        with self.assertRaises(FusionContractError):
            normalize_results([first, changed])

    def test_duplicate_identity_collapses_reordered_object_keys_without_extra_support(self):
        from mri_fusion.engine import normalize_results
        first = copy.deepcopy(self.results[1])
        reordered = dict(reversed(list(first.items())))
        reordered["producer"] = dict(reversed(list(first["producer"].items())))
        before = copy.deepcopy([first, reordered])
        normalized, count = normalize_results([first, reordered])
        self.assertEqual(count, 1)
        self.assertEqual(normalized, [first])
        self.assertEqual([first, reordered], before)


if __name__ == "__main__":
    unittest.main()
