import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from mri_root_impact.contracts import load_document, manifest_digest
from mri_root_impact.spinenet_assets import verify_assets
from mri_root_impact.spinenet_runner import external_run_directory, run_spinenet, validate_worker_outputs


FIXTURE = Path(__file__).resolve().parents[1] / "harness/fixtures/contracts/study.json"


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spinenet-run-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = load_document(FIXTURE)
        self.study_path = self.root / "study.json"
        self.study_path.write_text(json.dumps(self.study))
        self.output = self.root / "run"
        self.source = self.root / "source"
        self.source.mkdir()

    def run_model(self, **options):
        return run_spinenet(self.study_path, self.source, sys.executable, self.output, **options)

    def test_existing_output_and_git_or_upstream_destinations_are_rejected(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "new output"):
            self.run_model()
        with self.assertRaisesRegex(ValueError, "upstream"):
            external_run_directory(self.study_path, self.study, self.source / "output", self.source)
        parent = self.root / "checkout"
        parent.mkdir()
        (parent / ".git").mkdir()
        with self.assertRaisesRegex(ValueError, "outside Git"):
            external_run_directory(self.study_path, self.study, parent / "new", self.source)

    def test_invalid_parameters_fail_before_creating_output(self):
        for options in ({"threads": True}, {"threads": 0}, {"timeout": 0}, {"timeout": 3.5},
                        {"allow_missing_modality": "yes"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_model(**options)
            self.assertFalse(self.output.exists())

    def test_worker_failure_is_preserved_without_a_negative_finding(self):
        process = Mock(returncode=2)
        with patch("mri_root_impact.spinenet_runner.subprocess.Popen", return_value=process) as launch:
            with self.assertRaisesRegex(ValueError, "run failed"):
                self.run_model()
        report = load_document(self.output / "run.json")
        self.assertEqual(report["status"], "error")
        self.assertIsNone(report["root_impacts"])
        self.assertIsNone(report["inference_performed"])
        self.assertFalse((self.output / "localization.json").exists())
        self.assertEqual(launch.call_args.kwargs["env"]["PYTHONPATH"], str(Path(__file__).resolve().parents[1]))

    def test_timeout_terminates_worker_and_leaves_a_report(self):
        process = Mock(returncode=-15, pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), None]
        with patch("mri_root_impact.spinenet_runner.subprocess.Popen", return_value=process), \
                patch("mri_root_impact.spinenet_runner.os.killpg") as kill:
            with self.assertRaisesRegex(ValueError, "run failed"):
                self.run_model(timeout=1)
        self.assertTrue(load_document(self.output / "run.json")["timed_out"])
        kill.assert_called_once()

    def test_zero_exit_without_valid_artifacts_cannot_pass(self):
        with patch("mri_root_impact.spinenet_runner.subprocess.Popen", return_value=Mock(returncode=0)):
            with self.assertRaisesRegex(ValueError, "run failed"):
                self.run_model()
        report = load_document(self.output / "run.json")
        self.assertEqual(report["status"], "error")
        self.assertIn("validation_error_type", report)

    def test_executable_launch_failure_leaves_a_report(self):
        with patch("mri_root_impact.spinenet_runner.subprocess.Popen", side_effect=OSError("unavailable")):
            with self.assertRaisesRegex(ValueError, "run failed"):
                self.run_model()
        self.assertEqual(load_document(self.output / "run.json")["launch_error_type"], "OSError")

    def test_empty_hash_map_cannot_replace_the_required_artifacts(self):
        self.output.mkdir()
        (self.output / "execution.json").write_text(json.dumps({
            "kind": "localization_execution", "study_id": self.study["study_id"],
            "study_digest": manifest_digest(self.study), "status": "completed",
            "inference_performed": True, "artifacts_sha256": {},
        }))
        with self.assertRaisesRegex(ValueError, "four required"):
            validate_worker_outputs(self.study, self.output)

    def test_malformed_or_wrong_study_execution_is_not_success(self):
        self.output.mkdir()
        for payload in ([], None, {"kind": "localization_execution", "study_id": "another-study"}):
            with self.subTest(payload=payload):
                (self.output / "execution.json").write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "this manifest"):
                    validate_worker_outputs(self.study, self.output)

    def test_cli_malformed_spatial_is_structured_json_error(self):
        fixture_root = FIXTURE.parent
        spatial = self.root / "spatial.json"
        for payload in ([], None):
            spatial.write_text(json.dumps(payload))
            result = subprocess.run([sys.executable, "-m", "mri_root_impact", "validate", "--study", str(FIXTURE),
                                     "--result", str(fixture_root / "localization.json"), "--spatial", str(spatial)],
                                    cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(result.stderr)["ok"])
            self.assertNotIn("Traceback", result.stderr)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spinenet-assets-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "spinenet").mkdir()
        (self.root / "spinenet/__init__.py").write_text("# no imports or models\n")
        (self.root / "LICENCE.md").write_text("synthetic license fixture\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "spinenet", "LICENCE.md"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                        "commit", "-qm", "synthetic source"], check=True)
        revision = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        self.weight = self.root / "spinenet/weights/detector/checkpoint.pt"
        self.weight.parent.mkdir(parents=True)
        self.weight.write_bytes(b"not a model; hash only")
        self.inventory = {"commit": revision, "license_url": "https://example.test/license", "files": {
            "detector/checkpoint.pt": {"size_bytes": self.weight.stat().st_size,
                                       "sha256": hashlib.sha256(self.weight.read_bytes()).hexdigest()}}}

    def verify(self):
        with patch("mri_root_impact.spinenet_assets.load_document", return_value=self.inventory):
            return verify_assets(self.root)

    def test_verified_assets_are_bound_to_bytes_and_aggregate_digest(self):
        result = self.verify()
        self.assertEqual(len(result["weights_sha256"]), 64)
        self.assertIn("spinenet/__init__.py", result["source_files_sha256"])
        self.weight.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.verify()

    def test_extra_checkpoint_cannot_be_chosen_by_modification_time(self):
        self.weight.with_name("unexpected.pt").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            self.verify()

    def test_modified_or_extra_source_is_rejected_before_import(self):
        (self.root / "spinenet/extra.py").write_text("raise RuntimeError('must not import')\n")
        with self.assertRaisesRegex(ValueError, "additional"):
            self.verify()
        (self.root / "spinenet/extra.py").unlink()
        (self.root / "spinenet/__init__.py").write_text("raise RuntimeError('must not import')\n")
        with self.assertRaisesRegex(ValueError, "verification failed"):
            self.verify()

    def test_git_assume_unchanged_cannot_hide_modified_source(self):
        subprocess.run(["git", "-C", str(self.root), "update-index", "--assume-unchanged", "spinenet/__init__.py"], check=True)
        (self.root / "spinenet/__init__.py").write_text("# modified but hidden from git diff\n")
        with self.assertRaisesRegex(ValueError, "pinned Git object"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
