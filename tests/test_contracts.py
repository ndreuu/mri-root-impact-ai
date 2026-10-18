"""Validate synthetic MRI module data without running inference or reading images."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mri_root_impact.contracts import (
    ContractError,
    load_document,
    manifest_digest,
    validate_bundle,
    validate_study,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "harness" / "fixtures" / "contracts"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.study = self.load_fixture("study.json")
        self.localization = self.load_fixture("localization.json")
        self.indeterminate = self.load_fixture("stenosis-indeterminate.json")
        self.not_run = self.load_fixture("stenosis-not-run.json")
        self.assessed = self.load_fixture("stenosis-assessed.json")
        self.results = [self.localization, self.indeterminate, self.not_run, self.assessed]
        self.temp = tempfile.TemporaryDirectory(prefix="mri-root-contract-test-")
        self.addCleanup(self.temp.cleanup)
        self.temp_dir = Path(self.temp.name)

    @staticmethod
    def load_fixture(name):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    def cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "mri_root_impact", *map(str, arguments)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )

    def write_json(self, name, payload):
        path = self.temp_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def assert_cli_error(self, completed):
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("Traceback", completed.stderr)
        error = json.loads(completed.stderr)
        self.assertIs(error["ok"], False)
        self.assertIsInstance(error["error"], str)
        self.assertTrue(error["error"].strip())

    def test_fixture_bundle_is_valid_and_validation_does_not_mutate_it(self):
        before = copy.deepcopy((self.study, self.results))
        self.assertIsNone(validate_bundle(self.study, self.results))
        self.assertEqual((self.study, self.results), before)

    def test_manifest_digest_is_canonical_and_bound_to_manifest_content(self):
        reordered = dict(reversed(list(self.study.items())))
        self.assertEqual(manifest_digest(reordered), manifest_digest(self.study))
        changed = copy.deepcopy(self.study)
        changed["series"][0]["images"][0]["frame_count"] += 1
        self.assertNotEqual(manifest_digest(changed), manifest_digest(self.study))

    def test_manifest_does_not_require_image_files_to_exist(self):
        study = copy.deepcopy(self.study)
        for series in study["series"]:
            for image in series["images"]:
                image["path"] = "/nonexistent-synthetic-mri-contract-test/" + image["image_id"]
        self.assertIsNone(validate_study(study))

    def test_missing_series_requires_explicit_limitation(self):
        study = copy.deepcopy(self.study)
        study["series"] = []
        study["limitations"] = ["Synthetic fixture: no MRI series supplied."]
        self.assertIsNone(validate_study(study))
        study["limitations"] = []
        with self.assertRaises(ContractError):
            validate_study(study)

    def test_manifest_rejects_invalid_image_metadata_and_duplicate_ids(self):
        mutations = [
            ("boolean frame count", lambda s: s["series"][0]["images"][0].update(frame_count=True)),
            ("zero frame count", lambda s: s["series"][0]["images"][0].update(frame_count=0)),
            ("relative image path", lambda s: s["series"][0]["images"][0].update(path="images/image.dcm")),
            ("invalid checksum", lambda s: s["series"][0]["images"][0].update(sha256="not-a-sha256")),
            ("duplicate series id", lambda s: s["series"].append(copy.deepcopy(s["series"][0]))),
            ("duplicate image id", lambda s: s["series"][0]["images"].append(copy.deepcopy(s["series"][0]["images"][0]))),
        ]
        for name, mutate in mutations:
            with self.subTest(case=name):
                study = copy.deepcopy(self.study)
                mutate(study)
                with self.assertRaises(ContractError):
                    validate_study(study)

    def test_schema_rejects_missing_and_unexpected_fields(self):
        for target in ("study", "result", "assessment", "target"):
            for mutation in ("missing", "extra"):
                with self.subTest(target=target, mutation=mutation):
                    study, results = copy.deepcopy((self.study, self.results))
                    objects = {
                        "study": study,
                        "result": results[-1],
                        "assessment": results[-1]["assessments"][0],
                        "target": results[-1]["assessments"][0]["target"],
                    }
                    obj = objects[target]
                    if mutation == "missing":
                        del obj[next(iter(obj))]
                    else:
                        obj["unexpected_contract_field"] = True
                    with self.assertRaises(ContractError):
                        validate_bundle(study, results)

    def test_indeterminate_result_stays_unknown_with_no_fabricated_negative(self):
        assessment = self.indeterminate["assessments"][0]
        self.assertEqual(assessment["status"], "indeterminate")
        self.assertTrue(assessment["reason"])
        assessment["observation"] = None
        before = copy.deepcopy(assessment)
        validate_bundle(self.study, self.results)
        self.assertEqual(assessment, before)
        self.assertIsNone(assessment["observation"])
        not_run = self.not_run["assessments"][0]
        self.assertEqual(not_run["status"], "not_run")
        self.assertTrue(not_run["reason"])
        self.assertIsNone(not_run["observation"])

    def test_raw_observation_is_preserved_for_assessed_and_indeterminate_results(self):
        assessment = self.assessed["assessments"][0]
        assessment["observation"]["label"] = "normal_mild"
        assessment["observation"]["scale_id"] = "synthetic_raw_classes_v1"
        observation = copy.deepcopy(assessment["observation"])
        validate_bundle(self.study, self.results)
        self.assertEqual(assessment["observation"], observation)
        assessment["status"] = "indeterminate"
        assessment["reason"] = "Raw score retained; localization remains uncertain."
        validate_bundle(self.study, self.results)
        self.assertEqual(assessment["observation"], observation)
        assessment["image_refs"] = []
        with self.assertRaises(ContractError):
            validate_bundle(self.study, self.results)

    def test_status_and_observation_must_not_conflict(self):
        mutations = [
            ("assessed without observation", lambda a: a.update(observation=None)),
            ("assessed without images", lambda a: a.update(image_refs=[])),
            ("assessed with error reason", lambda a: a.update(reason="Failed to run")),
            ("indeterminate without reason", lambda a: a.update(status="indeterminate", reason=None)),
            ("not run with observation", lambda a: a.update(status="not_run", reason="Model not configured")),
            ("error with observation", lambda a: a.update(status="error", reason="Processing failed")),
        ]
        for name, mutate in mutations:
            with self.subTest(case=name):
                results = copy.deepcopy(self.results)
                mutate(results[-1]["assessments"][0])
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)

    def test_assessed_targets_must_be_known_and_match_component(self):
        changes = [
            {"level": "unknown"},
            {"side": "unknown"},
            {"zone": "unknown"},
            {"side": "bilateral"},
            {"level": "L4"},
            {"zone": "vertebral_body"},
        ]
        for change in changes:
            with self.subTest(change=change):
                results = copy.deepcopy(self.results)
                results[-1]["assessments"][0]["target"].update(change)
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)
        results = copy.deepcopy(self.results)
        results[0]["assessments"][0]["target"]["side"] = "left"
        with self.assertRaises(ContractError):
            validate_bundle(self.study, results)

    def test_measurement_values_must_be_finite_numbers_and_not_booleans(self):
        self.assessed["assessments"][0]["observation"]["measurements"] = [
            {"name": "synthetic_score", "value": 0, "unit": "arbitrary"}
        ]
        validate_bundle(self.study, self.results)
        for value in (True, float("nan"), float("inf"), float("-inf"), "2.5"):
            with self.subTest(value=repr(value)):
                results = copy.deepcopy(self.results)
                results[-1]["assessments"][0]["observation"]["measurements"] = [
                    {"name": "synthetic_score", "value": value, "unit": "arbitrary"}
                ]
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)

    def test_image_references_must_resolve_and_respect_frame_bounds(self):
        source = self.assessed["assessments"][0]["image_refs"][0]
        series = next(s for s in self.study["series"] if s["series_id"] == source["series_id"])
        image = next(i for i in series["images"] if i["image_id"] == source["image_id"])
        changes = [
            {"series_id": "unknown-series"},
            {"image_id": "unknown-image"},
            {"frame_index": -1},
            {"frame_index": True},
            {"frame_index": image["frame_count"]},
        ]
        other_series = next((s for s in self.study["series"] if s["series_id"] != source["series_id"]), None)
        if other_series is not None:
            changes.append({"series_id": other_series["series_id"]})
        for change in changes:
            with self.subTest(change=change):
                results = copy.deepcopy(self.results)
                results[-1]["assessments"][0]["image_refs"][0].update(change)
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)

    def test_result_and_assessment_ids_must_be_unique(self):
        results = copy.deepcopy(self.results)
        results.append(copy.deepcopy(results[-1]))
        with self.assertRaises(ContractError):
            validate_bundle(self.study, results)
        results = copy.deepcopy(self.results)
        results[-1]["assessments"].append(copy.deepcopy(results[-1]["assessments"][0]))
        with self.assertRaises(ContractError):
            validate_bundle(self.study, results)

    def test_results_are_bound_to_study_id_provenance_and_manifest_digest(self):
        for field, value in (
            ("study_id", "another-synthetic-study"),
            ("synthetic", False),
            ("study_digest", "0" * 64),
        ):
            with self.subTest(field=field):
                results = copy.deepcopy(self.results)
                results[-1][field] = value
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)

    def test_upstream_references_must_exist_and_assessed_depends_on_assessed(self):
        upstream = self.localization["assessments"][0]
        valid_reference = {
            "result_id": self.localization["result_id"],
            "assessment_id": upstream["assessment_id"],
        }
        self.assessed["assessments"][0]["upstream_refs"] = [valid_reference]
        validate_bundle(self.study, self.results)
        for field in ("result_id", "assessment_id"):
            with self.subTest(field=field):
                results = copy.deepcopy(self.results)
                results[-1]["assessments"][0]["upstream_refs"][0][field] = "missing-upstream"
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)
        for status in ("indeterminate", "not_run", "error"):
            with self.subTest(upstream_status=status):
                results = copy.deepcopy(self.results)
                results[0]["assessments"][0].update(
                    status=status, reason="Synthetic upstream unavailable", observation=None
                )
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)

    def test_upstream_self_references_and_cycles_are_rejected(self):
        assessment = self.assessed["assessments"][0]
        assessment["upstream_refs"] = [{
            "result_id": self.assessed["result_id"],
            "assessment_id": assessment["assessment_id"],
        }]
        with self.assertRaises(ContractError):
            validate_bundle(self.study, self.results)
        assessment["upstream_refs"] = [{
            "result_id": self.localization["result_id"],
            "assessment_id": self.localization["assessments"][0]["assessment_id"],
        }]
        self.localization["assessments"][0]["upstream_refs"] = [{
            "result_id": self.assessed["result_id"],
            "assessment_id": assessment["assessment_id"],
        }]
        with self.assertRaises(ContractError):
            validate_bundle(self.study, self.results)

    def test_assessed_disc_level_must_match_its_localization_source(self):
        source = self.localization["assessments"][0]
        assessment = self.assessed["assessments"][0]
        assessment["upstream_refs"] = [{
            "result_id": self.localization["result_id"],
            "assessment_id": source["assessment_id"],
        }]
        self.assertEqual(assessment["target"]["level"], source["target"]["level"])
        validate_bundle(self.study, self.results)
        assessment["target"]["level"] = next(
            level for level in ("L3-L4", "L4-L5")
            if level != source["target"]["level"]
        )
        with self.assertRaises(ContractError):
            validate_bundle(self.study, self.results)

    def test_unconfigured_model_permits_not_run_or_error_but_no_observations(self):
        self.not_run["producer"].update(method="model", model=None)
        for status in ("not_run", "error"):
            with self.subTest(status=status):
                self.not_run["assessments"][0]["status"] = status
                self.assertIsNone(validate_bundle(self.study, self.results))
        for result_id in (self.assessed["result_id"], self.indeterminate["result_id"]):
            with self.subTest(observed_result=result_id):
                results = copy.deepcopy(self.results)
                result = next(item for item in results if item["result_id"] == result_id)
                self.assertIsNotNone(result["assessments"][0]["observation"])
                result["producer"].update(method="model", model=None)
                with self.assertRaises(ContractError):
                    validate_bundle(self.study, results)

    def test_strict_json_reader_and_cli_reject_duplicate_keys_and_nonfinite_constants(self):
        duplicate = '{"study_id":"shadowed-study",' + json.dumps(self.study)[1:]
        documents = {"duplicate-key": duplicate}
        for constant in ("NaN", "Infinity", "-Infinity"):
            documents[constant] = '{"synthetic_score":' + constant + '}'
        for name, contents in documents.items():
            with self.subTest(document=name):
                path = self.temp_dir / (name + ".json")
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ContractError):
                    load_document(path)
                self.assert_cli_error(self.cli("validate", "--study", path))

    def test_cli_validates_study_and_result_bundle_as_json(self):
        arguments = ["validate", "--study", FIXTURES / "study.json"]
        for name in (
            "localization.json", "stenosis-indeterminate.json",
            "stenosis-not-run.json", "stenosis-assessed.json",
        ):
            arguments.extend(["--result", FIXTURES / name])
        completed = self.cli(*arguments)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIs(json.loads(completed.stdout)["ok"], True)

    def test_cli_reports_missing_malformed_and_invalid_inputs_without_traceback(self):
        malformed = self.temp_dir / "malformed.json"
        malformed.write_text('{"study_id":', encoding="utf-8")
        invalid = self.write_json("invalid-study.json", {"study_id": "incomplete"})
        for path in (self.temp_dir / "missing.json", malformed, invalid):
            with self.subTest(path=path.name):
                self.assert_cli_error(self.cli("validate", "--study", path))
        invalid_result = copy.deepcopy(self.assessed)
        invalid_result["study_id"] = "wrong-study"
        path = self.write_json("invalid-result.json", invalid_result)
        self.assert_cli_error(self.cli(
            "validate", "--study", FIXTURES / "study.json", "--result", path
        ))


if __name__ == "__main__":
    unittest.main()
