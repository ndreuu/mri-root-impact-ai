"""Exercise fusion file boundaries without images, optional packages, or models."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from mri_fusion.contracts import FusionContractError, POLICY_PATH, artifact_binding
from mri_root_impact.contracts import manifest_digest
from tests.fusion_fixtures import build_case, canonical_bytes, case_by_id, document_sha256


ROOT = Path(__file__).resolve().parents[1]


class FusionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mri-fusion-runner-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.inputs = self.directory / "inputs"
        self.inputs.mkdir()
        self.invalid_attempts = 0
        fixture = build_case(case_by_id("F001-01"))
        self.study, self.results = copy.deepcopy(fixture[:2])
        self.study_path = self.write_json(self.inputs / "study.json", self.study)
        self.result_paths = tuple(self.write_json(self.inputs / (result["result_id"] + ".json"), result)
                                  for result in self.results)

    @staticmethod
    def write_json(path, document):
        path.write_bytes(canonical_bytes(document))
        return path

    def fuse(self, output=None, **kwargs):
        from mri_fusion.runner import run_fusion
        return run_fusion(self.study_path, output or self.directory / "output",
                          result_paths=kwargs.pop("result_paths", self.result_paths), **kwargs)

    @staticmethod
    def snapshot(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def assert_invalid(self, error_pattern=None, **kwargs):
        before = self.snapshot(self.inputs)
        self.invalid_attempts += 1
        kwargs.setdefault("output", self.directory / ("invalid-output-" + str(self.invalid_attempts)))
        assertion = (self.assertRaises(FusionContractError) if error_pattern is None else
                     self.assertRaisesRegex(FusionContractError, error_pattern))
        with assertion:
            self.fuse(**kwargs)
        self.assertEqual(self.snapshot(self.inputs), before)

    def test_raw_synthetic_inputs_save_twenty_unknown_branches_without_inference(self):
        before = self.snapshot(self.inputs)
        output = self.directory / "output"
        self.fuse(output=output)
        self.assertEqual({path.name for path in output.iterdir()}, {"fusion.json", "report.md", "report.html", "run.json"})
        report = json.loads((output / "fusion.json").read_text())
        self.assertEqual(report["kind"], "mri_root_impact_fusion")
        self.assertEqual(len(report["branches"]), 20)
        self.assertTrue(all(row["decision"] == "indeterminate" for row in report["branches"]))
        self.assertTrue(all(row["candidate_root"] is None for row in report["branches"]))
        self.assertEqual(report["root_summaries"], [])
        self.assertEqual(report["root_candidates"], [])
        self.assertIs(report["inference_performed"], False)
        self.assertIsNone(report["clinical_diagnosis"])
        self.assertIn("неопредел", (output / "report.md").read_text().lower())
        self.assertIn("неопредел", (output / "report.html").read_text().lower())
        self.assertEqual(self.snapshot(self.inputs), before)

    def test_valid_empty_result_set_stays_unknown(self):
        output = self.directory / "empty-output"
        self.fuse(output=output, result_paths=())
        report = json.loads((output / "fusion.json").read_text())
        self.assertEqual(len(report["branches"]), 20)
        self.assertEqual(report["coverage"]["indeterminate"], 20)
        self.assertEqual(report["root_candidates"], [])

    def test_existing_output_is_not_overwritten(self):
        output = self.directory / "output"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("preserve existing run", encoding="utf-8")
        before = self.snapshot(output)
        self.assert_invalid(output=output)
        self.assertEqual(self.snapshot(output), before)

    def test_input_directory_and_input_file_cannot_be_output(self):
        for output in (self.inputs, self.study_path, self.inputs / "nested-output"):
            with self.subTest(output=output.name):
                self.assert_invalid(output=output)
        self.assertFalse((self.inputs / "nested-output").exists())

    def test_output_inside_git_checkout_is_rejected_before_writes(self):
        output = ROOT / "fusion-test-output-must-not-be-created"
        self.assertFalse(output.exists())
        self.assert_invalid(output=output)
        self.assertFalse(output.exists())

    def test_output_symlink_to_source_directory_is_rejected(self):
        alias = self.directory / "output-alias"
        alias.symlink_to(self.inputs, target_is_directory=True)
        self.assert_invalid(output=alias)
        self.assertTrue(alias.is_symlink())

    def test_output_cannot_be_nested_in_pinned_image_source_directory(self):
        image_sources = self.directory / "image-sources"
        image_sources.mkdir()
        self.study["synthetic"] = False
        self.study["limitations"] = ["Fictitious contract-test bytes; real-study flag tests source protection only."]
        for series in self.study["series"]:
            for image in series["images"]:
                path = image_sources / (image["image_id"] + ".bin")
                payload = ("Fictitious source bytes: " + image["image_id"]).encode()
                path.write_bytes(payload)
                image["path"] = str(path)
                image["sha256"] = hashlib.sha256(payload).hexdigest()
        self.write_json(self.study_path, self.study)
        for path, result in zip(self.result_paths, self.results):
            result["synthetic"] = False
            result["study_digest"] = manifest_digest(self.study)
            result["producer"]["method"] = "manual"
            self.write_json(path, result)
        before = self.snapshot(image_sources)
        output = image_sources / "nested-output"
        self.assert_invalid(output=output)
        self.assertEqual(self.snapshot(image_sources), before)
        self.assertFalse(output.exists())

    def test_frozen_run_mode_requires_all_three_run_directories(self):
        names = ("spinenet_run", "rsna_run", "crosscheck_run")
        for count in (1, 2):
            for omitted in range(3):
                selected = [name for index, name in enumerate(names) if (index == omitted) == (count == 1)]
                kwargs = {name: self.inputs / name for name in selected}
                with self.subTest(arguments=selected):
                    self.assert_invalid(result_paths=(), **kwargs)

    def test_frozen_metadata_change_during_upstream_validation_is_not_rebound(self):
        spine = self.directory / "frozen-spine"
        rsna = self.directory / "frozen-rsna"
        crosscheck = self.directory / "frozen-crosscheck"
        for directory in (spine, rsna, crosscheck):
            directory.mkdir()
        common = {"execution.json", "run.json", "localization.json", "native.json", "spatial.json"}
        for directory, names in ((spine, common),
                                 (rsna, common | {"foraminal.json", "subarticular.json"}),
                                 (crosscheck, {"crosscheck.json", "run.json"})):
            for name in names:
                self.write_json(directory / name, {"fictitious_metadata": "before-validation"})
        inventory = {
            key: {"run_directory": str(directory), "files_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.iterdir()
            }} for key, directory in (("spinenet", spine), ("rsna", rsna))
        }
        changed_path = spine / "native.json"
        changed_bytes = canonical_bytes({"fictitious_metadata": "changed-during-validation"})
        before_inputs = self.snapshot(self.inputs)
        before_frozen = {str(path): path.read_bytes()
                         for directory in (spine, rsna, crosscheck) for path in directory.iterdir()}

        def checked_old_inventory(*_args):
            changed_path.write_bytes(changed_bytes)
            return {}, {}, inventory

        output = self.directory / "race-output"
        with mock.patch("scripts.compare_localizers._read_runs", side_effect=checked_old_inventory) as checked:
            with mock.patch("scripts.compare_localizers.compare_levels") as compare:
                with self.assertRaisesRegex(FusionContractError, "Input changed across frozen validation"):
                    self.fuse(output=output, result_paths=(), spinenet_run=spine,
                              rsna_run=rsna, crosscheck_run=crosscheck)
                checked.assert_called_once()
                compare.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(self.snapshot(self.inputs), before_inputs)
        self.assertEqual(changed_path.read_bytes(), changed_bytes)
        for path, payload in before_frozen.items():
            if Path(path) != changed_path:
                self.assertEqual(Path(path).read_bytes(), payload)

    def test_verify_only_requires_boolean(self):
        for value in (0, 1, "false", None, [], {}):
            with self.subTest(value=repr(value)):
                self.assert_invalid(verify_only=value)
        self.assertFalse((self.directory / "output").exists())

    def test_missing_source_file_is_an_error_without_successful_report(self):
        self.assert_invalid(result_paths=(self.inputs / "missing-result.json",))
        self.assertFalse((self.directory / "output" / "fusion.json").exists())

    def test_malformed_json_is_rejected_without_rewriting_inputs(self):
        malformed = self.inputs / "malformed.json"
        for contents in (b'{"kind":', b'{"x":NaN}', b'{"x":Infinity}'):
            malformed.write_bytes(contents)
            with self.subTest(contents=contents):
                self.assert_invalid(result_paths=(malformed,))
        self.assertFalse((self.directory / "output" / "fusion.json").exists())

    def test_repeated_json_keys_are_rejected(self):
        duplicate = self.inputs / "duplicate.json"
        raw = canonical_bytes(self.results[0])
        duplicate.write_bytes(b'{"result_id":"shadowed",' + raw[1:])
        self.assert_invalid(result_paths=(duplicate,))

    def test_modified_policy_is_not_an_alternate_threshold(self):
        policy = self.inputs / "policy.json"
        policy.write_bytes(POLICY_PATH.read_bytes() + b"\n")
        self.assert_invalid(policy_path=policy)

    def test_verify_only_recomputes_report_without_any_file_writes(self):
        output = self.directory / "output"
        self.fuse(output=output)
        before = self.snapshot(self.directory)
        self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_verify_only_fails_for_absent_run_without_creating_output(self):
        output = self.directory / "absent-output"
        self.assert_invalid(output=output, verify_only=True)
        self.assertFalse(output.exists())

    def test_changed_input_bytes_invalidate_frozen_result_even_if_json_is_equal(self):
        output = self.directory / "output"
        self.fuse(output=output)
        source = self.result_paths[-1]
        source.write_bytes(source.read_bytes() + b"\n")
        before = self.snapshot(self.directory)
        with self.assertRaises(FusionContractError):
            self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_tampered_fusion_json_is_rejected_without_repairing_it(self):
        output = self.directory / "output"
        self.fuse(output=output)
        path = output / "fusion.json"
        report = json.loads(path.read_text())
        report["branches"][0]["decision"] = "potential_impact"
        self.write_json(path, report)
        before = self.snapshot(self.directory)
        with self.assertRaises(FusionContractError):
            self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_tampered_human_report_is_rejected_without_repairing_it(self):
        output = self.directory / "output"
        self.fuse(output=output)
        path = output / "report.md"
        path.write_text(path.read_text() + "\nAltered report.\n")
        before = self.snapshot(self.directory)
        with self.assertRaises(FusionContractError):
            self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_tampered_html_report_is_rejected_without_repairing_it(self):
        output = self.directory / "output"
        self.fuse(output=output)
        path = output / "report.html"
        path.write_text(path.read_text() + "\n<p>Altered clinical explanation.</p>\n")
        before = self.snapshot(self.directory)
        with self.assertRaises(FusionContractError):
            self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_html_edit_with_replaced_checksum_still_fails_source_recomputation(self):
        output = self.directory / "output"
        self.fuse(output=output)
        path = output / "report.html"
        path.write_text(path.read_text().replace("Предварительное заключение", "Изменённое заключение"))
        metadata_path = output / "run.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["artifacts_sha256"]["report.html"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_json(metadata_path, metadata)
        before = self.snapshot(self.directory)
        with self.assertRaisesRegex(FusionContractError, "Physician HTML differs"):
            self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_changed_code_binding_invalidates_verify_only(self):
        from mri_fusion import runner
        output = self.directory / "output"
        self.fuse(output=output)
        changed = runner._code_hashes()
        self.assertTrue({"mri_root_impact/contracts.py", "mri_root_impact/control_study.py",
                         "scripts/compare_localizers.py", "mri_fusion/engine.py"}.issubset(changed))
        changed[next(iter(changed))] = "f" * 64
        before = self.snapshot(self.directory)
        with mock.patch.object(runner, "_code_hashes", return_value=changed):
            with self.assertRaises(FusionContractError):
                self.fuse(output=output, verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_nonobject_run_metadata_has_structured_cli_error_and_no_verify_writes(self):
        output = self.directory / "output"
        self.fuse(output=output)
        command = [sys.executable, str(ROOT / "scripts" / "fuse_mri.py"),
                   "--study", str(self.study_path), "--output", str(output), "--verify-only"]
        for path in self.result_paths:
            command.extend(("--result", str(path)))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for malformed in ([], None, 17, "completed"):
            with self.subTest(metadata=malformed):
                self.write_json(output / "run.json", malformed)
                before = self.snapshot(self.directory)
                with self.assertRaises(FusionContractError):
                    self.fuse(output=output, verify_only=True)
                completed = subprocess.run(command, cwd=ROOT, env=environment,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, check=False, timeout=30)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(completed.stdout, "")
                self.assertNotIn("Traceback", completed.stderr)
                error = json.loads(completed.stderr)
                self.assertIs(error["ok"], False)
                self.assertIsInstance(error["error"], str)
                self.assertTrue(error["error"].strip())
                self.assertEqual(self.snapshot(self.directory), before)

    def write_positive_review(self):
        _, _, _, review, _ = build_case(case_by_id("F001-01"))
        proof = {
            "schema_version": "1.0", "kind": "fusion_review_evidence",
            "evidence_id": "synthetic-review-proof", "study_id": self.study["study_id"],
            "study_digest": manifest_digest(self.study), "synthetic": True,
            "description": "Fictitious review proof; no actual images, reviewer or registration.",
        }
        proof_path = self.write_json(self.inputs / "review-proof.json", proof)
        evidence_id = "evidence-" + proof["evidence_id"]
        bindings = [artifact_binding("result-" + result["result_id"], result, document_sha256(result))
                    for result in self.results]
        bindings.append(artifact_binding(evidence_id, proof, document_sha256(proof)))
        review["input_artifacts_sha256"] = {item["artifact_id"]: item["sha256"] for item in bindings}
        for row in review["reviews"]:
            row["evidence_refs"] = [evidence_id]
        review_path = self.write_json(self.inputs / "review.json", review)
        return review_path, proof_path, proof

    def test_explicit_fictitious_review_can_enable_one_synthetic_candidate(self):
        review_path, proof_path, _ = self.write_positive_review()
        output = self.directory / "positive-output"
        self.fuse(output=output, review_path=review_path, evidence_paths=(proof_path,))
        report = json.loads((output / "fusion.json").read_text())
        positives = [branch for branch in report["branches"] if branch["candidate_root"] is not None]
        self.assertEqual(len(positives), 1)
        self.assertEqual(positives[0]["candidate_root"], {"root": "L5", "role": "traversing", "side": "left"})
        self.assertIs(report["synthetic"], True)
        self.assertIs(report["inference_performed"], False)
        before = self.snapshot(self.directory)
        self.fuse(output=output, review_path=review_path, evidence_paths=(proof_path,), verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_review_evidence_document_must_match_study_and_schema(self):
        review_path, proof_path, proof = self.write_positive_review()
        for field, value in (("kind", "unrelated_evidence"), ("schema_version", "2.0"),
                             ("study_id", "other-study"), ("study_digest", "f" * 64),
                             ("synthetic", False)):
            changed = copy.deepcopy(proof)
            changed[field] = value
            self.write_json(proof_path, changed)
            review = json.loads(review_path.read_text())
            review["input_artifacts_sha256"]["evidence-synthetic-review-proof"] = document_sha256(changed)
            self.write_json(review_path, review)
            with self.subTest(field=field):
                self.assert_invalid(error_pattern="Review evidence", review_path=review_path, evidence_paths=(proof_path,))

    def test_synthetic_review_proof_cannot_be_applied_to_real_study(self):
        _, proof_path, proof = self.write_positive_review()
        self.study["synthetic"] = False
        self.write_json(self.study_path, self.study)
        for path, result in zip(self.result_paths, self.results):
            result["synthetic"] = False
            result["study_digest"] = manifest_digest(self.study)
            result["producer"]["method"] = "manual"
            self.write_json(path, result)
        proof["study_digest"] = manifest_digest(self.study)
        self.write_json(proof_path, proof)
        self.assert_invalid(error_pattern="Review evidence", evidence_paths=(proof_path,))


if __name__ == "__main__":
    unittest.main()
