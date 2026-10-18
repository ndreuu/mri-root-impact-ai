"""Infrastructure contracts; these checks deliberately perform no MRI inference."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness.py"
FIXTURE = ROOT / "harness" / "fixtures" / "synthetic-study.json"
STAGES = {
    "localization": "not_configured",
    "disc_analysis": "not_configured",
    "stenosis": "not_configured",
    "fusion": "not_configured",
}
SYNTHETIC_MANIFEST = {
    "case_id": "synthetic-smoke-001",
    "synthetic": True,
    "input_kind": "manifest_only",
    "series": [],
}


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("mri_root_harness_under_test", SCRIPT)
        cls.harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.harness)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mri-root-harness-test-")
        self.addCleanup(self.temp.cleanup)
        self.temp_dir = Path(self.temp.name)

    def cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, arguments)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )

    def successful_json(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIsInstance(payload, dict)
        return payload

    def write_manifest(self, manifest):
        path = self.temp_dir / "study.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def load_registry(self, tasks, **overrides):
        path = self.temp_dir / "tasks.json"
        registry = {"schema_version": 1, "tasks": tasks}
        registry.update(overrides)
        path.write_text(json.dumps(registry), encoding="utf-8")
        return self.harness.load_tasks(path)

    @staticmethod
    def task(task_id, status="ready", depends_on=None, evidence=None):
        return {
            "id": task_id,
            "title": "Task " + task_id,
            "status": status,
            "depends_on": [] if depends_on is None else depends_on,
            "evidence": [] if evidence is None else evidence,
        }

    def assert_smoke_report(self, report, case_id):
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["kind"], "harness_smoke")
        self.assertIs(report["synthetic"], True)
        self.assertEqual(report["case_id"], case_id)
        self.assertIs(report["inference_performed"], False)
        self.assertIn("root_impacts", report)
        self.assertIsNone(report["root_impacts"])
        self.assertEqual(report["stages"], STAGES)
        self.harness.validate_smoke_report(report)

    def test_doctor_and_status_return_json(self):
        # Do not invoke `check` here: check runs this suite itself.
        for command in ("doctor", "status"):
            with self.subTest(command=command):
                self.successful_json(self.cli(command))

    def test_smoke_roundtrip_keeps_unknown_clinical_result_null(self):
        manifest = dict(SYNTHETIC_MANIFEST, case_id="synthetic-roundtrip-test")
        fixture = self.write_manifest(manifest)
        fixture_before = fixture.read_bytes()
        output = self.temp_dir / "artifacts"

        result = self.successful_json(
            self.cli("smoke", "--fixture", fixture, "--output", output)
        )

        report_path = Path(result["report"])
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        self.assertEqual(report_path.resolve(), (output / "report.json").resolve())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assert_smoke_report(report, manifest["case_id"])
        self.assertEqual(fixture.read_bytes(), fixture_before)

    def test_default_fixture_is_synthetic_manifest_and_stays_unchanged(self):
        fixture_before = FIXTURE.read_bytes()
        self.assertEqual(json.loads(fixture_before), SYNTHETIC_MANIFEST)
        output = self.temp_dir / "default-fixture-run"

        self.successful_json(self.cli("smoke", "--output", output))

        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assert_smoke_report(report, SYNTHETIC_MANIFEST["case_id"])
        self.assertEqual(FIXTURE.read_bytes(), fixture_before)

    def test_smoke_rejects_invalid_manifest_before_persisting(self):
        invalid_manifests = {
            "non_synthetic": dict(SYNTHETIC_MANIFEST, synthetic=False),
            "missing_case_id": {
                key: value for key, value in SYNTHETIC_MANIFEST.items() if key != "case_id"
            },
            "nonempty_series": dict(SYNTHETIC_MANIFEST, series=[{"series_id": "001"}]),
        }
        for name, manifest in invalid_manifests.items():
            with self.subTest(manifest=name):
                fixture = self.write_manifest(manifest)
                fixture_before = fixture.read_bytes()
                output = self.temp_dir / name

                result = self.cli("smoke", "--fixture", fixture, "--output", output)

                self.assertNotEqual(result.returncode, 0)
                self.assertIsInstance(json.loads(result.stderr), dict)
                self.assertFalse((output / "report.json").exists())
                self.assertEqual(fixture.read_bytes(), fixture_before)

    def test_smoke_rejects_malformed_json_before_persisting(self):
        fixture = self.temp_dir / "malformed.json"
        fixture.write_text('{"case_id":', encoding="utf-8")
        output = self.temp_dir / "malformed-run"

        result = self.cli("smoke", "--fixture", fixture, "--output", output)

        self.assertNotEqual(result.returncode, 0)
        self.assertIsInstance(json.loads(result.stderr), dict)
        self.assertFalse((output / "report.json").exists())

    def test_smoke_validator_rejects_fabricated_negative_or_inferred_results(self):
        report = {
            "schema_version": 1,
            "kind": "harness_smoke",
            "synthetic": True,
            "case_id": SYNTHETIC_MANIFEST["case_id"],
            "inference_performed": False,
            "root_impacts": None,
            "stages": STAGES.copy(),
        }
        self.harness.validate_smoke_report(report)
        for field, value in (("root_impacts", []), ("inference_performed", True)):
            with self.subTest(field=field):
                invalid = copy.deepcopy(report)
                invalid[field] = value
                with self.assertRaises(ValueError):
                    self.harness.validate_smoke_report(invalid)

    def test_task_registry_accepts_completed_prerequisite_with_evidence(self):
        tasks = [
            self.task("T001", "done", evidence=["tests/test_harness.py"]),
            self.task("T002", "ready", depends_on=["T001"]),
        ]
        loaded = self.load_registry(tasks)
        self.assertEqual(loaded, tasks)

    def test_task_registry_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.load_registry([self.task("T001"), self.task("T001")])

    def test_task_registry_rejects_missing_dependency(self):
        with self.assertRaises(ValueError):
            self.load_registry([self.task("T001", depends_on=["T999"])])

    def test_task_registry_rejects_dependency_cycle(self):
        with self.assertRaises(ValueError):
            self.load_registry(
                [
                    self.task("T001", "todo", depends_on=["T002"]),
                    self.task("T002", "todo", depends_on=["T001"]),
                ]
            )

    def test_task_registry_requires_finished_dependencies_for_active_or_done_tasks(self):
        for status in ("ready", "in_progress", "done"):
            with self.subTest(status=status):
                evidence = ["some-evidence.txt"] if status == "done" else []
                with self.assertRaises(ValueError):
                    self.load_registry(
                        [
                            self.task("T001", "ready"),
                            self.task("T002", status, ["T001"], evidence),
                        ]
                    )

    def test_task_registry_requires_evidence_for_done_task(self):
        with self.assertRaises(ValueError):
            self.load_registry([self.task("T001", "done")])

    def test_invalid_status_type_returns_structured_cli_error(self):
        invalid_task = self.task("T001", status=[])
        with self.assertRaises(ValueError):
            self.load_registry([invalid_task])

        registry_dir = self.temp_dir / "harness"
        registry_dir.mkdir()
        (registry_dir / "tasks.json").write_text(
            json.dumps({"schema_version": 1, "tasks": [invalid_task]}),
            encoding="utf-8",
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(self.harness, "ROOT", self.temp_dir):
            with mock.patch.object(sys, "argv", [str(SCRIPT), "status"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = self.harness.main()

        self.assertNotEqual(code, 0)
        error = json.loads(stderr.getvalue())
        self.assertIs(error["ok"], False)
        self.assertIsInstance(error["error"], str)

    def test_empty_test_discovery_cannot_pass_check(self):
        empty_tests = self.temp_dir / "empty-tests"
        empty_tests.mkdir()
        with self.assertRaises(ValueError):
            self.harness.run_tests(empty_tests)

    def test_forced_staged_mri_data_is_rejected_even_when_ignored(self):
        repo = self.temp_dir / "repo"
        repo.mkdir()
        data = repo / "data"
        data.mkdir()
        (repo / ".gitignore").write_text("data/*\n", encoding="utf-8")
        (data / "study.dcm").write_bytes(b"synthetic-placeholder")
        (data / "README.md").write_text("Data directory policy.\n", encoding="utf-8")
        subprocess.run(
            ["git", "init", "--quiet", str(repo)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(repo), "add", "-f", "data/study.dcm", "data/README.md"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with mock.patch.object(self.harness, "ROOT", repo):
            self.assertEqual(self.harness.unsafe_candidates(), ["data/study.dcm"])


if __name__ == "__main__":
    unittest.main()
