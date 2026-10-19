"""Synthetic native-result checks; no patient files or model imports."""

import copy
import unittest

from mri_root_impact.contracts import manifest_digest, validate_bundle
from mri_root_impact.rsna_results import CLASSES, LEVELS, SIDES, build_results, validate_native


def example():
    study = {"schema_version": "1.0", "kind": "study_manifest", "study_id": "synthetic-rsna-results",
             "synthetic": True, "limitations": [], "series": [
                 {"series_id": "sag-t2", "sequence": "sagittal_t2", "images": [
                     {"image_id": "sag-image", "path": "/synthetic/no-image.dcm", "frame_count": 2, "sha256": None}]}]}
    refs = [{"series_id": "sag-t2", "image_id": "sag-image", "frame_index": 0}]
    native = {"kind": "rsna_native_predictions", "schema_version": "1.0", "study_id": study["study_id"],
              "study_digest": manifest_digest(study), "class_order": list(CLASSES), "limitations": [],
              "assets": {"source_commit": "a" * 40, "weights": {
                  "classifier": {"sha256": "b" * 64}, "sagittal_keypoints": {"sha256": "c" * 64},
                  "axial_keypoints": {"sha256": "d" * 64}}},
              "localizations": [{"level": level, "status": "located", "reason": None, "image_refs": copy.deepcopy(refs)} for level in LEVELS],
              "predictions": [{"level": level, "side": side, "status": "predicted", "reason": None,
                               "probabilities": {"foraminal_stenosis": [0.1, 0.2, 0.7], "subarticular_stenosis": [1.0, 0.0, 0.0]},
                               "image_refs": copy.deepcopy(refs)} for level in LEVELS for side in SIDES]}
    return study, native


class RSNAResultsTests(unittest.TestCase):
    def setUp(self):
        self.study, self.native = example()

    def test_success_has_exact_coverage_native_scale_and_unresolved_numbering(self):
        before = copy.deepcopy((self.study, self.native))
        self.assertIsNone(validate_native(self.study, self.native))
        results = build_results(self.study, self.native)
        self.assertEqual([result["result_id"] for result in results],
                         ["rsna-localization-001", "rsna-foraminal-001", "rsna-subarticular-001"])
        self.assertEqual([len(result["assessments"]) for result in results], [5, 10, 10])
        self.assertEqual((self.study, self.native), before)
        self.assertIsNone(validate_bundle(self.study, results))
        for result in results:
            for item in result["assessments"]:
                self.assertEqual(item["status"], "indeterminate")
                self.assertIn("unreviewed", item["reason"])
        self.assertEqual(results[1]["assessments"][0]["observation"]["label"], "severe")
        observation = results[2]["assessments"][0]["observation"]
        self.assertEqual(observation["label"], "normal_mild")
        self.assertEqual(observation["scale_id"], "rsna2024:stenosis-severity:v1")
        self.assertEqual([item["unit"] for item in observation["measurements"]], ["probability"] * 3)
        self.assertNotIn("root_impacts", results[2])

    def test_provenance_and_same_level_upstreams_are_preserved(self):
        results = build_results(self.study, self.native)
        self.assertEqual(results[0]["producer"]["model"]["weights_sha256"], "c" * 64)
        self.assertEqual(results[1]["producer"]["model"]["weights_sha256"], "b" * 64)
        for result in results[1:]:
            for item in result["assessments"]:
                self.assertEqual(item["upstream_refs"], [{"result_id": "rsna-localization-001",
                                                        "assessment_id": "disc-" + item["target"]["level"]}])

    def test_absent_roi_stays_not_run_with_no_fabricated_grade(self):
        self.native["localizations"][0].update(status="not_located", reason="No valid automatic ROI", image_refs=[])
        for item in self.native["predictions"][:2]:
            item.update(status="not_run", reason="No valid automatic ROI", probabilities=None, image_refs=[])
        results = build_results(self.study, self.native)
        self.assertEqual(results[0]["assessments"][0]["status"], "indeterminate")
        self.assertIsNone(results[0]["assessments"][0]["observation"])
        for result in results[1:]:
            for item in result["assessments"][:2]:
                self.assertEqual(item["status"], "not_run")
                self.assertEqual(item["reason"], "No valid automatic ROI")
                self.assertIsNone(item["observation"])

    def test_probability_validation_rejects_nonfinite_boolean_and_invalid_simplex(self):
        invalid = [[float("nan"), 0, 1], [float("inf"), 0, 0], [True, 0, 0], ["1", 0, 0],
                   [-0.1, 0.1, 1], [1.1, 0, 0], [0.2, 0.2, 0.2], [0.5, 0.5],
                   (0.1, 0.2, 0.7), [10 ** 1000, 0, 0]]
        for values in invalid:
            with self.subTest(values=str(values)[:50]):
                native = copy.deepcopy(self.native)
                native["predictions"][0]["probabilities"]["foraminal_stenosis"] = values
                with self.assertRaises(ValueError):
                    validate_native(self.study, native)

    def test_simplex_tolerance_and_tie_rule_are_deterministic(self):
        self.native["predictions"][0]["probabilities"]["foraminal_stenosis"] = [0.5, 0.5, 0.000001]
        self.assertEqual(build_results(self.study, self.native)[1]["assessments"][0]["observation"]["label"], "normal_mild")

    def test_target_coverage_rejects_missing_duplicate_unknown_and_swapped_sides(self):
        mutations = [lambda n: n["localizations"].pop(),
                     lambda n: n["localizations"][0].update(level="L2-L3"),
                     lambda n: n["localizations"][0].update(level="L6-S1"),
                     lambda n: n["predictions"].pop(),
                     lambda n: n["predictions"][0].update(side="right"),
                     lambda n: n["predictions"][0].update(side="bilateral"),
                     lambda n: n["predictions"][0].update(level="unknown")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                native = copy.deepcopy(self.native)
                mutate(native)
                with self.assertRaises(ValueError):
                    build_results(self.study, native)

    def test_input_order_does_not_change_result_order(self):
        expected = build_results(self.study, self.native)
        self.native["localizations"].reverse()
        self.native["predictions"].reverse()
        self.assertEqual(build_results(self.study, self.native), expected)

    def test_missing_or_invalid_source_refs_fail_bundle_validation(self):
        mutations = [lambda r: r.clear(), lambda r: r.append(copy.deepcopy(r[0])),
                     lambda r: r[0].update(frame_index=2), lambda r: r[0].update(frame_index=True),
                     lambda r: r[0].update(series_id="absent"), lambda r: r[0].update(image_id="absent")]
        for container in ("localizations", "predictions"):
            for index, mutate in enumerate(mutations):
                with self.subTest(container=container, index=index):
                    native = copy.deepcopy(self.native)
                    mutate(native[container][0]["image_refs"])
                    with self.assertRaises(ValueError):
                        validate_native(self.study, native)

    def test_source_refs_are_deep_copied(self):
        results = build_results(self.study, self.native)
        results[0]["assessments"][0]["image_refs"][0]["image_id"] = "changed"
        self.assertEqual(self.native["localizations"][0]["image_refs"][0]["image_id"], "sag-image")

    def test_identity_class_order_and_weight_hashes_are_checked(self):
        mutations = [lambda n: n.update(study_id="other"), lambda n: n.update(study_digest="e" * 64),
                     lambda n: n.update(schema_version="2.0"), lambda n: n.update(kind="unknown"),
                     lambda n: n.update(class_order=["severe", "moderate", "normal_mild"]),
                     lambda n: n["assets"].update(source_commit="a" * 39),
                     lambda n: n["assets"]["weights"]["classifier"].update(sha256="B" * 64),
                     lambda n: n["assets"]["weights"]["sagittal_keypoints"].update(sha256="c" * 63),
                     lambda n: n["assets"]["weights"].pop("classifier")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                native = copy.deepcopy(self.native)
                mutate(native)
                with self.assertRaises(ValueError):
                    validate_native(self.study, native)

    def test_status_and_reason_cannot_hide_missing_inference(self):
        mutations = [lambda n: n["localizations"][0].update(status="assessed"),
                     lambda n: n["localizations"][0].update(status="not_located", reason=""),
                     lambda n: n["predictions"][0].update(status="assessed"),
                     lambda n: n["predictions"][0].update(status="not_run", reason=None),
                     lambda n: n["predictions"][0].update(status="not_run", reason="Failed"),
                     lambda n: n["predictions"][0].update(reason="success with unexplained fallback"),
                     lambda n: n["localizations"][0].update(status="not_located", reason="Missing ROI", image_refs=[])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                native = copy.deepcopy(self.native)
                mutate(native)
                with self.assertRaises(ValueError):
                    validate_native(self.study, native)

    def test_optional_root_qc_is_allowed_but_required_native_fields_are_not_optional(self):
        self.native["qc"] = {"numbering_review": "not_performed"}
        self.native["timings"] = {"classifier_seconds": 1.2}
        validate_native(self.study, self.native)
        for key in list(self.native):
            if key in {"qc", "timings"}:
                continue
            native = copy.deepcopy(self.native)
            del native[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native(self.study, native)


if __name__ == "__main__":
    unittest.main()
