"""Execute all F001 decisions against genuine synthetic C001/review bundles."""

import copy
import unittest

from mri_root_impact.contracts import validate_bundle
from tests.fusion_fixtures import (
    build_case,
    case_by_id,
    document_sha256,
    load_vectors,
    rebind_review,
)


def _unique_results(results):
    """Only exact duplicate documents can be normalized for the C001 boundary."""
    unique = []
    for result in results:
        if result not in unique:
            unique.append(result)
    return unique


def _branch(report, source):
    target = {key: source[key] for key in ("level", "side", "zone")}
    return next(branch for branch in report["branches"] if branch["target"] == target)


class FusionVectorTests(unittest.TestCase):
    def fuse(self, fixture):
        from mri_fusion.engine import fuse
        study, results, artifacts, review, review_sha = fixture
        return fuse(study, results, artifacts, review_context=review, review_sha256=review_sha)

    def assert_invariants(self, report):
        self.assertEqual(len(report["branches"]), 20)
        targets = [tuple(branch["target"][key] for key in ("level", "side", "zone"))
                   for branch in report["branches"]]
        self.assertEqual(len(set(targets)), 20)
        self.assertEqual({level for level, _, _ in targets},
                         {"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"})
        self.assertEqual({side for _, side, _ in targets}, {"left", "right"})
        self.assertEqual({zone for _, _, zone in targets}, {"foraminal", "subarticular"})
        self.assertIs(report["inference_performed"], False)
        self.assertIs(report["global_root_clearance"], False)
        for branch in report["branches"]:
            self.assertEqual(branch["herniation_context"], "not_evaluated")
            self.assertEqual(branch["etiology"], "unassigned")
            self.assertIsInstance(branch["reason_codes"], list)
            self.assertIsInstance(branch["evidence"], list)
            if branch["decision"] == "potential_impact":
                self.assertIsNotNone(branch["candidate_root"])
                self.assertEqual(branch["candidate_root"], branch["mapped_root"])
            else:
                self.assertIsNone(branch["candidate_root"])

    def execute_vector(self, case):
        fixture = build_case(case)
        before = copy.deepcopy(fixture)
        study, results, _, _, _ = fixture
        if case["case_id"] != "F001-22":
            self.assertIsNone(validate_bundle(study, _unique_results(results)))
        expected = case["expected"]
        if expected.get("operation") == "invalid_input_error":
            from mri_fusion.contracts import FusionContractError
            with self.assertRaises(FusionContractError):
                self.fuse(fixture)
            self.assertEqual(fixture, before)
            return

        report = self.fuse(fixture)
        from mri_fusion.contracts import validate_output
        self.assertIsNone(validate_output(study, report))
        self.assertEqual(fixture, before, "Fusion must preserve every original C001 and review field")
        self.assert_invariants(report)
        branches = report["branches"]
        supplied = case["inputs"]["branches"]
        primary = _branch(report, supplied[0]) if "branch" in expected else None
        for key, value in expected.get("branch", {}).items():
            self.assertEqual(primary[key], value)
        if "reason" in expected:
            self.assertIn(expected["reason"], primary["reason_codes"])
        if "mapped_root" in expected:
            self.assertEqual(primary["mapped_root"], expected["mapped_root"])
        if "candidate_roots" in expected:
            actual = [branch["candidate_root"] for branch in branches if branch["candidate_root"] is not None]
            self.assertCountEqual(actual, expected["candidate_roots"])
        if "root_summary" in expected:
            wanted = expected["root_summary"]
            summary = next(row for row in report["root_summaries"]
                           if (row["root"], row["side"]) == (wanted["root"], wanted["side"]))
            for key, value in wanted.items():
                if key == "supporting_branches":
                    self.assertEqual(len(summary["supporting_branch_refs"]), value)
                else:
                    self.assertEqual(summary[key], value)
        comparisons = {
            "total_branch_count": len(branches),
            "indeterminate_branch_count": sum(branch["decision"] == "indeterminate" for branch in branches),
            "no_candidate_branch_count": sum(branch["decision"] == "no_candidate_by_evaluated_criterion" for branch in branches),
            "named_candidate_count": sum(branch["candidate_root"] is not None for branch in branches),
            "root_summary_count": len(report["root_summaries"]),
            "unresolved_native_evidence_count": len(report["unresolved_native_evidence"]),
        }
        for key, actual in comparisons.items():
            if key in expected:
                self.assertEqual(actual, expected[key], key)
        if "root_summaries" in expected:
            self.assertEqual(report["root_summaries"], expected["root_summaries"])
        if "unique_supporting_evidence_count" in expected:
            self.assertEqual(len(primary["evidence"]), expected["unique_supporting_evidence_count"])
        if expected.get("same_target_conflict") is False:
            for branch in branches:
                self.assertNotIn("same_target_evidence_conflict", branch["reason_codes"])
        if "winning_root" in expected:
            self.assertIsNone(report.get("winning_root"))

    def test_vectors_have_unique_ids_and_declared_fictitious_provenance(self):
        document = load_vectors()
        self.assertIs(document["synthetic"], True)
        self.assertEqual(len(document["cases"]), 34)
        self.assertEqual(len({case["case_id"] for case in document["cases"]}), 34)
        for case in document["cases"]:
            study, results, _, review, _ = build_case(case)
            self.assertIs(study["synthetic"], True)
            self.assertTrue(all(result["synthetic"] for result in results))
            if review is not None:
                self.assertIs(review["synthetic"], True)
                self.assertTrue(all(row["reviewer_id"] == "synthetic-reviewer" for row in review["reviews"]))

    def test_review_cannot_promote_indeterminate_native_severe(self):
        report = self.fuse(build_case(case_by_id("F001-13")))
        source = case_by_id("F001-13")["inputs"]["branches"][0]
        branch = _branch(report, source)
        self.assertEqual(branch["decision"], "indeterminate")
        self.assertIsNone(branch["mapped_root"])
        self.assertEqual(report["root_summaries"], [])

    def test_one_reviewed_normal_branch_does_not_clear_its_other_root_site(self):
        report = self.fuse(build_case(case_by_id("F001-10")))
        summary = next(row for row in report["root_summaries"] if (row["root"], row["side"]) == ("L5", "left"))
        self.assertEqual(summary["decision"], "indeterminate")
        self.assertEqual(summary["coverage"], "partial")
        self.assertEqual(len(summary["branch_refs"]), 1)
        self.assertEqual(len(summary["expected_branch_ids"]), 2)
        self.assertEqual(len(summary["unresolved_expected_branch_ids"]), 1)
        self.assertEqual(summary["supporting_branch_refs"], [])

    def test_concordant_different_sources_retain_both_without_voting(self):
        case = copy.deepcopy(case_by_id("F001-23"))
        case["inputs"]["branches"][1]["label"] = "severe"
        report = self.fuse(build_case(case))
        branch = _branch(report, case["inputs"]["branches"][0])
        self.assertEqual(branch["decision"], "potential_impact")
        self.assertEqual(len(branch["evidence"]), 2)
        summary = next(row for row in report["root_summaries"] if (row["root"], row["side"]) == ("L5", "left"))
        self.assertEqual(len(summary["supporting_branch_refs"]), 1)
        self.assertEqual(summary["coverage"], "partial")

    def test_partial_probability_set_is_invalid_instead_of_normalized(self):
        from mri_fusion.contracts import FusionContractError, artifact_binding
        fixture = list(build_case(case_by_id("F001-24")))
        study, results, artifacts, review, _ = fixture
        result = next(result for result in results if result["component"] == "subarticular_stenosis")
        result["assessments"][0]["observation"]["measurements"].pop()
        artifact_id = "artifact-" + result["result_id"]
        index = next(index for index, item in enumerate(artifacts) if item["artifact_id"] == artifact_id)
        artifacts[index] = artifact_binding(artifact_id, result, document_sha256(result))
        fixture[4] = rebind_review(review, artifacts)
        validate_bundle(study, results)
        with self.assertRaises(FusionContractError):
            self.fuse(fixture)

    def test_review_must_cover_every_model_input_series(self):
        fixture = list(build_case(case_by_id("F001-01")))
        review = fixture[3]
        review["reviews"][0]["covered_series_ids"].remove("sag-t1")
        fixture[4] = document_sha256(review)
        report = self.fuse(fixture)
        branch = _branch(report, case_by_id("F001-01")["inputs"]["branches"][0])
        self.assertEqual(branch["decision"], "indeterminate")
        self.assertIsNone(branch["mapped_root"])
        self.assertEqual(report["root_summaries"], [])

    def test_complete_normal_inputs_still_leave_herniation_unassessed(self):
        report = self.fuse(build_case(case_by_id("F001-28")))
        self.assertEqual(len(report["root_summaries"]), 12)
        for summary in report["root_summaries"]:
            self.assertEqual(summary["decision"], "no_candidate_by_evaluated_criterion")
            self.assertEqual(summary["coverage"], "complete_within_policy")
            self.assertEqual(summary["supporting_branch_refs"], [])

    def test_disc_context_is_retained_without_inventing_causal_root_rules(self):
        for case_id in ("F001-29", "F001-30", "F001-31", "F001-32"):
            with self.subTest(case_id=case_id):
                fixture = build_case(case_by_id(case_id))
                disc_result = next(result for result in fixture[1] if result["component"] == "disc_analysis")
                original = disc_result["assessments"][0]
                report = self.fuse(fixture)
                self.assertEqual(report["herniation_context"], "not_evaluated")
                self.assertEqual(report["etiology"], "unassigned")
                self.assertEqual(len(report["raw_disc_evidence"]), 1)
                evidence = report["raw_disc_evidence"][0]
                self.assertEqual(evidence["source_ref"], {"result_id": disc_result["result_id"],
                                                         "assessment_id": original["assessment_id"]})
                for field in ("target", "status", "observation"):
                    self.assertEqual(evidence[field], original[field])


def _vector_method(case):
    def test(self):
        self.execute_vector(case)
    test.__doc__ = case["description"]
    return test


for _case in load_vectors()["cases"]:
    setattr(FusionVectorTests, "test_" + _case["case_id"].lower().replace("-", "_"), _vector_method(_case))


if __name__ == "__main__":
    unittest.main()
