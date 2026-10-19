"""RSNA runner lifecycle checks without subprocess inference or model packages."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from mri_root_impact.contracts import load_document, manifest_digest
from mri_root_impact.rsna_runner import run_rsna


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "harness/fixtures/contracts/study.json"
SYNTHETIC_ASSETS = {
    "source_commit": "a" * 40,
    "source_files_sha256": {"src/predict.py": "b" * 64},
    "source_tree_sha256": "c" * 64,
    "weights": {"classifier": {"filename": "synthetic.pth", "size_bytes": 1, "sha256": "d" * 64}},
    "weights_sha256": "e" * 64,
}


class RSNARunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rsna-runner-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = load_document(FIXTURE)
        self.study_path = self.root / "study.json"
        self.study_path.write_text(json.dumps(self.study), encoding="utf-8")
        self.source, self.weights = self.root / "source", self.root / "weights"
        self.source.mkdir()
        self.weights.mkdir()
        self.output = self.root / "run"
        asset_patch = patch("mri_root_impact.rsna_runner.verify_rsna_assets", return_value=SYNTHETIC_ASSETS)
        self.verify_assets = asset_patch.start()
        self.addCleanup(asset_patch.stop)

    def run_model(self, output=None, python=None, **options):
        return run_rsna(self.study_path, self.source, self.weights,
                        sys.executable if python is None else python,
                        self.output if output is None else output, **options)

    def failed_report(self, output=None):
        report = load_document((self.output if output is None else output) / "run.json")
        self.assertEqual(report["status"], "error")
        self.assertIsNone(report["inference_performed"])
        self.assertIsNone(report["root_impacts"])
        self.assertIsNone(report.get("stenosis_predictions"))
        self.assertEqual(report["study_id"], self.study["study_id"])
        return report

    def test_worker_nonzero_exit_preserves_unknown_result_and_exit_code(self):
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", return_value=Mock(returncode=7)):
            with self.assertRaises(ValueError):
                self.run_model()
        report = self.failed_report()
        self.assertEqual(report["worker_exit_code"], 7)
        self.assertFalse(report["timed_out"])
        self.assertFalse((self.output / "foraminal.json").exists())
        self.assertFalse((self.output / "subarticular.json").exists())

    def test_zero_exit_without_required_artifacts_is_not_completed(self):
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", return_value=Mock(returncode=0)):
            with self.assertRaises(ValueError):
                self.run_model()
        report = self.failed_report()
        self.assertEqual(report["worker_exit_code"], 0)
        self.assertIn("validation_error", report)

    def test_worker_success_header_cannot_replace_the_seven_required_artifacts(self):
        def launch_without_artifacts(*args, **kwargs):
            (self.output / "execution.json").write_text(json.dumps({
                "kind": "rsna_mri_execution", "status": "completed",
                "study_id": self.study["study_id"], "study_digest": manifest_digest(self.study),
                "inference_performed": True, "source_bytes_unchanged": True,
                "root_impacts": None, "anatomical_review": "indeterminate",
                "registration_status": "unreviewed", "artifacts_sha256": {},
            }), encoding="utf-8")
            return Mock(returncode=0)

        with patch("mri_root_impact.rsna_runner.subprocess.Popen", side_effect=launch_without_artifacts):
            with self.assertRaises(ValueError):
                self.run_model()
        self.assertIn("validation_error", self.failed_report())

    def test_launch_failure_leaves_checked_failure_report(self):
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", side_effect=OSError("synthetic launch failure")):
            with self.assertRaises(ValueError):
                self.run_model()
        report = self.failed_report()
        self.assertIsNone(report["worker_exit_code"])
        self.assertEqual(report["launch_error_type"], "OSError")
        self.assertTrue((self.output / "worker.log").is_file())

    def test_timeout_terminates_only_mock_process_group_and_keeps_unknown(self):
        process = Mock(returncode=-signal.SIGTERM, pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("synthetic-worker", 1), None]
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", return_value=process):
            with patch("mri_root_impact.rsna_runner.os.killpg") as kill:
                with self.assertRaises(ValueError):
                    self.run_model(timeout=1)
        kill.assert_called_once_with(123456, signal.SIGTERM)
        self.assertTrue(self.failed_report()["timed_out"])

    def test_unresponsive_worker_escalates_to_mock_sigkill(self):
        process = Mock(returncode=-signal.SIGKILL, pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("synthetic-worker", 1),
                                    subprocess.TimeoutExpired("synthetic-worker", 5), None]
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", return_value=process):
            with patch("mri_root_impact.rsna_runner.os.killpg") as kill:
                with self.assertRaises(ValueError):
                    self.run_model(timeout=1)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(123456, signal.SIGTERM), (123456, signal.SIGKILL)])
        self.assertTrue(self.failed_report()["timed_out"])

    def test_existing_output_cannot_overwrite_previous_run(self):
        self.output.mkdir()
        sentinel = self.output / "run.json"
        sentinel.write_bytes(b"existing result must survive")
        with patch("mri_root_impact.rsna_runner.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                self.run_model()
        launch.assert_not_called()
        self.verify_assets.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"existing result must survive")

    def test_git_source_and_weights_output_destinations_are_protected(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        (checkout / ".git").mkdir()
        outputs = [checkout / "run", self.source / "run", self.weights / "run"]
        with patch("mri_root_impact.rsna_runner.subprocess.Popen") as launch:
            for output in outputs:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    self.run_model(output=output)
                self.assertFalse(output.exists())
        launch.assert_not_called()
        self.verify_assets.assert_not_called()

    def test_output_cannot_contain_a_declared_source_image(self):
        self.study["series"][0]["images"][0]["path"] = str(self.output / "source.dcm")
        self.study_path.write_text(json.dumps(self.study), encoding="utf-8")
        with patch("mri_root_impact.rsna_runner.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                self.run_model()
        launch.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_default_registration_and_modality_policies_stay_strict(self):
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", return_value=Mock(returncode=1)) as launch:
            with self.assertRaises(ValueError):
                self.run_model()
        command = self.failed_report()["command"]
        self.assertNotIn("--allow-unverified-registration", command)
        self.assertNotIn("--allow-missing-modality", command)
        self.assertEqual(command, launch.call_args.args[0])
        self.assertIs(launch.call_args.kwargs["start_new_session"], True)
        environment = launch.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONPATH"], str(ROOT))
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_explicit_research_opt_ins_are_visible_in_persisted_worker_command(self):
        with patch("mri_root_impact.rsna_runner.subprocess.Popen", return_value=Mock(returncode=1)) as launch:
            with self.assertRaises(ValueError):
                self.run_model(allow_unverified_registration=True, allow_missing_modality=True)
        command = self.failed_report()["command"]
        self.assertIn("--allow-unverified-registration", command)
        self.assertIn("--allow-missing-modality", command)
        self.assertEqual(command, launch.call_args.args[0])

    def test_invalid_parameters_fail_before_creating_output_or_launching(self):
        invalid = ({"threads": True}, {"threads": 0}, {"threads": 9},
                   {"timeout": True}, {"timeout": 0}, {"timeout": 7201}, {"timeout": 1.5},
                   {"allow_missing_modality": "yes"}, {"allow_unverified_registration": 1})
        with patch("mri_root_impact.rsna_runner.subprocess.Popen") as launch:
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.run_model(**options)
                self.assertFalse(self.output.exists())
        launch.assert_not_called()
        self.verify_assets.assert_not_called()

    def test_missing_or_nonexecutable_python_fails_before_output(self):
        nonexecutable = self.root / "not-python"
        nonexecutable.write_text("synthetic executable fixture", encoding="utf-8")
        nonexecutable.chmod(0o600)
        with patch("mri_root_impact.rsna_runner.subprocess.Popen") as launch:
            for python in (self.root / "missing-python", nonexecutable):
                with self.subTest(python=python), self.assertRaises(ValueError):
                    self.run_model(python=python)
                self.assertFalse(self.output.exists())
        launch.assert_not_called()
        self.verify_assets.assert_not_called()

    def test_unverified_assets_fail_before_output_or_worker_launch(self):
        self.verify_assets.side_effect = ValueError("Synthetic checkpoint hash mismatch")
        with patch("mri_root_impact.rsna_runner.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                self.run_model()
        launch.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
