"""The one-command tool coordinates checked stages without MRI dependencies."""

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from mri_fusion.runner import run_fusion as checked_fusion
from mri_root_impact.contracts import load_document, manifest_digest
from tests.fusion_fixtures import build_case, canonical_bytes, case_by_id


ROOT = Path(__file__).resolve().parents[1]


class MRIToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mri-tool-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.inputs = self.directory / "inputs"
        self.config_dir = self.directory / "config"
        self.assets = self.directory / "model-assets"
        for path in (self.inputs, self.config_dir, self.assets):
            path.mkdir()
        self.study, self.results = copy.deepcopy(build_case(case_by_id("F001-13"))[:2])
        self.study_path = self.write(self.inputs / "study.json", self.study)
        self.spine_source = self.assets / "spine-source"
        self.rsna_source = self.assets / "rsna-source"
        self.weights = self.assets / "weights"
        self.runtime = self.assets / "runtime"
        for path in (self.spine_source, self.rsna_source, self.weights, self.runtime / "bin"):
            path.mkdir(parents=True)
        self.python = self.runtime / "bin" / "python"
        self.python.symlink_to(sys.executable)
        self.config = {
            "schema_version": "1.0", "mode": "inference",
            "spinenet": {"source": "../model-assets/spine-source", "python": "../model-assets/runtime/bin/python"},
            "rsna": {"source": "../model-assets/rsna-source", "weights": "../model-assets/weights",
                     "python": "../model-assets/runtime/bin/python"},
        }
        self.config_path = self.write(self.config_dir / "models.json", self.config)
        self.output = self.directory / "output"
        self.events = []
        from mri_tool import runner
        self.tool = runner
        self.actual_command = runner._run_command
        self.command = self.patch("_run_command", side_effect=self.fake_command)
        self.spine = self.patch("run_spinenet", side_effect=self.fake_spine)
        self.rsna = self.patch("run_rsna", side_effect=self.fake_rsna)
        self.fusion = self.patch("run_fusion", side_effect=self.real_fusion_with_checked_fixture_sources)

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(self.tool, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    @staticmethod
    def write(path, value):
        path.write_bytes(canonical_bytes(value))
        return path

    @staticmethod
    def snapshot(path):
        return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()}

    def run_tool(self, **kwargs):
        return self.tool.run_mri(self.study_path, self.config_path, kwargs.pop("output", self.output), **kwargs)

    def fake_run(self, output, stage):
        output = Path(output)
        output.mkdir(parents=True)
        self.write(output / "run.json", {"status": "completed", "fictitious_test_stage": stage})
        if stage == "localization":
            self.write(output / "localization.json", self.results[0])
        elif stage == "stenosis":
            self.write(output / "subarticular.json", self.results[1])
        else:
            self.write(output / "crosscheck.json", {"fictitious_test_comparison": True})

    def fake_command(self, python, arguments, log_path, timeout):
        arguments = [str(value) for value in arguments]
        output = Path(arguments[arguments.index("--output") + 1])
        Path(log_path).write_text("Fictitious stage; no image or model was run.\n")
        if "preflight" in arguments:
            self.events.append("preflight")
            count = sum(len(series["images"]) for series in self.study["series"])
            self.write(output, {"kind": "input_preflight", "study_digest": manifest_digest(self.study),
                                "files_verified": count, "files_decoded": count})
        elif 'mri_disc.worker' in arguments:
            self.events.append('disc_analysis')
            self.fake_run(output, 'disc_analysis')
        else:
            self.events.append("crosscheck")
            self.fake_run(output, "crosscheck")

    def fake_spine(self, study_path, output, **kwargs):
        self.events.append("localization")
        self.fake_run(output, "localization")
        return {"ok": True, "inference_performed": True}

    def fake_rsna(self, study_path, output, **kwargs):
        self.events.append("stenosis")
        self.fake_run(output, "stenosis")
        return {"ok": True, "inference_performed": True}

    def checked_fixture_sources(self, study, spine, rsna, crosscheck, files):
        """Replace expensive upstream geometry checks, retaining real C001 fusion."""
        for directory in (spine, rsna, crosscheck):
            directory = Path(directory)
            if load_document(directory / "run.json").get("status") != "completed":
                raise ValueError("Fictitious checked upstream is incomplete")
            for path in directory.iterdir():
                files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        documents = []
        for index, path in enumerate((Path(spine) / "localization.json", Path(rsna) / "subarticular.json")):
            documents.append(("fixture-" + str(index), load_document(path), files[str(path.resolve())]))
        return documents

    def real_fusion_with_checked_fixture_sources(self, *args, **kwargs):
        self.events.append("fusion-verify" if kwargs.get("verify_only") else "fusion")
        with mock.patch("mri_fusion.runner._frozen_runs", side_effect=self.checked_fixture_sources):
            return checked_fusion(*args, **kwargs)

    def frozen(self):
        directories = {}
        for key, stage in (("spinenet_run", "localization"), ("rsna_run", "stenosis"), ("crosscheck_run", "crosscheck")):
            path = self.directory / "frozen" / stage
            self.fake_run(path, stage)
            directories[key] = "../frozen/" + stage
        self.config = {"schema_version": "1.0", "mode": "frozen", **directories}
        self.write(self.config_path, self.config)

    def test_inference_order_relative_paths_and_venv_python_symlink_are_preserved(self):
        progress = []
        result = self.run_tool(progress=progress.append)
        expected = ["preflight", "localization", "stenosis", "crosscheck", "fusion"]
        self.assertEqual(self.events, expected)
        self.assertEqual(progress, expected)
        self.assertEqual(self.spine.call_args.kwargs["source"], self.spine_source)
        self.assertEqual(self.rsna.call_args.kwargs["weights"], self.weights)
        self.assertEqual(self.spine.call_args.kwargs["python"], self.python)
        self.assertEqual(self.command.call_args_list[0].args[0], self.python)
        self.assertEqual(self.command.call_args_list[1].args[0], self.python)
        self.assertTrue(self.python.is_symlink())
        self.assertEqual(Path(result["report"]), self.output / "fusion" / "report.html")
        self.assertTrue(Path(result["report"]).is_absolute())
        journal = load_document(self.output / "tool-run.json")
        self.assertEqual(journal["status"], "completed")
        self.assertEqual(journal["completed_stages"], expected)
        self.assertEqual(journal["herniation_analysis"], "not_configured")

    def test_research_flags_are_default_false_and_only_explicitly_enabled(self):
        self.run_tool()
        self.assertIs(self.spine.call_args.kwargs["allow_missing_modality"], False)
        self.assertIs(self.rsna.call_args.kwargs["allow_unverified_registration"], False)
        self.assertTrue(all("--allow-missing-modality" not in call.args[1] for call in self.command.call_args_list))
        self.config["options"] = {"allow_missing_modality": True, "allow_unverified_registration": True,
                                  "threads": 3, "timeout": 123}
        self.write(self.config_path, self.config)
        self.run_tool(output=self.directory / "opt-in-output")
        self.assertIs(self.rsna.call_args.kwargs["allow_unverified_registration"], True)
        self.assertIs(self.spine.call_args.kwargs["allow_missing_modality"], True)
        self.assertEqual(self.rsna.call_args.kwargs["threads"], 3)
        self.assertEqual(self.rsna.call_args.kwargs["timeout"], 123)
        for call in self.command.call_args_list[-2:]:
            self.assertIn("--allow-missing-modality", call.args[1])
            self.assertEqual(call.args[3], 123)

    def test_generated_report_uses_real_fusion_and_retains_source_indeterminacy(self):
        self.run_tool()
        report = load_document(self.output / "fusion" / "fusion.json")
        self.assertEqual(len(report["branches"]), 20)
        self.assertEqual(report["coverage"]["indeterminate"], 20)
        self.assertEqual(report["root_candidates"], [])
        self.assertIs(report["inference_performed"], False)
        self.assertTrue((self.output / "fusion" / "report.html").is_file())

    def enable_disc_fixture(self):
        self.config['disc_analysis']='spinenet_binary'
        self.write(self.config_path,self.config)
        actual=self.fusion.side_effect
        def fixture_fusion(*args,**kwargs):
            kwargs.pop('disc_run',None)
            return actual(*args,**kwargs)
        self.fusion.side_effect=fixture_fusion

    def test_disc_worker_uses_spinenet_runtime_after_localization_and_is_not_repeated_in_verify(self):
        self.enable_disc_fixture()
        with mock.patch('mri_disc.contract.validate_run') as validate:
            self.run_tool()
            self.assertEqual(self.events,['preflight','localization','disc_analysis','stenosis','crosscheck','fusion'])
            call=next(c for c in self.command.call_args_list if 'mri_disc.worker' in c.args[1])
            self.assertEqual(call.args[0],self.python)
            self.assertIn(str(self.spine_source),[str(v) for v in call.args[1]])
            validate.assert_called_once()
            before=self.snapshot(self.output);count=self.command.call_count
            self.run_tool(verify_only=True)
            self.assertEqual(self.command.call_count,count)
            self.assertEqual(self.snapshot(self.output),before)
        self.assertEqual(load_document(self.output/'tool-run.json')['herniation_analysis'],'spinenet_binary_unreviewed')

    def test_invalid_disc_artifact_stops_pipeline_before_stenosis(self):
        self.enable_disc_fixture()
        with mock.patch('mri_disc.contract.validate_run',side_effect=ValueError('Fictitious corrupt disc artifact')):
            with self.assertRaises(ValueError):self.run_tool()
        journal=load_document(self.output/'tool-run.json')
        self.assertEqual(journal['failed_stage'],'disc_analysis')
        self.assertEqual(journal['completed_stages'],['preflight','localization'])
        self.rsna.assert_not_called();self.fusion.assert_not_called()

    def test_frozen_disc_mode_reuses_all_stages_without_model_calls(self):
        self.enable_disc_fixture();self.frozen()
        self.config['disc_run']='../frozen/disc'
        self.write(self.config_path,self.config)
        self.run_tool()
        self.command.assert_not_called();self.spine.assert_not_called();self.rsna.assert_not_called()
        self.assertEqual(self.fusion.call_args.kwargs['disc_run'],self.directory/'frozen/disc')
        self.assertEqual(load_document(self.output/'tool-run.json')['completed_stages'],['fusion'])

    def test_unknown_disc_backend_is_rejected_before_any_stage(self):
        self.config['disc_analysis']='ldh_mock'
        self.write(self.config_path,self.config)
        with self.assertRaises(ValueError):self.run_tool()
        self.command.assert_not_called()

    def test_frozen_mode_calls_only_fusion_and_never_models_or_commands(self):
        self.frozen()
        result = self.run_tool()
        self.assertEqual(self.events, ["fusion"])
        self.command.assert_not_called()
        self.spine.assert_not_called()
        self.rsna.assert_not_called()
        self.assertIs(result["inference_performed"], False)
        journal = load_document(self.output / "tool-run.json")
        self.assertEqual(journal["completed_stages"], ["fusion"])
        self.assertIs(journal["inference_performed"], False)

    def test_verify_only_has_no_writes_or_new_inference_after_inference_run(self):
        self.run_tool()
        before = self.snapshot(self.directory)
        counts = (self.spine.call_count, self.rsna.call_count, self.command.call_count)
        result = self.run_tool(verify_only=True)
        self.assertIs(result["inference_performed"], False)
        self.assertEqual((self.spine.call_count, self.rsna.call_count, self.command.call_count), counts)
        self.assertEqual(self.events[-1], "fusion-verify")
        self.assertEqual(self.snapshot(self.directory), before)

    def test_changed_config_or_study_bytes_invalidate_verify_even_if_json_equal(self):
        for name in ("config", "study"):
            with self.subTest(changed=name):
                output = self.directory / ("verify-" + name)
                self.run_tool(output=output)
                path = self.config_path if name == "config" else self.study_path
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                before = self.snapshot(output)
                with self.assertRaises(ValueError):
                    self.run_tool(output=output, verify_only=True)
                self.assertEqual(self.snapshot(output), before)
                path.write_bytes(original)

    def test_changed_tool_code_binding_invalidates_verify(self):
        self.frozen()
        self.run_tool()
        code = self.tool._code_hashes()
        code[next(iter(code))] = "f" * 64
        before = self.snapshot(self.directory)
        with mock.patch.object(self.tool, "_code_hashes", return_value=code):
            with self.assertRaises(ValueError):
                self.run_tool(verify_only=True)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_changed_fusion_run_or_html_is_rejected_without_repair(self):
        self.frozen()
        self.run_tool()
        for name in ("run.json", "report.html"):
            with self.subTest(changed=name):
                path = self.output / "fusion" / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                before = self.snapshot(self.output)
                with self.assertRaises(ValueError):
                    self.run_tool(verify_only=True)
                self.assertEqual(self.snapshot(self.output), before)
                path.write_bytes(original)

    def test_forged_tool_metadata_is_rejected_without_writes(self):
        self.frozen()
        self.run_tool()
        original = load_document(self.output / "tool-run.json")
        for changed in ([], None, {**original, "status": "error"},
                        {**original, "stage_directories": {}}, {**original, "report": "/wrong/report.html"}):
            with self.subTest(metadata=type(changed).__name__):
                self.write(self.output / "tool-run.json", changed)
                before = self.snapshot(self.output)
                with self.assertRaises(ValueError):
                    self.run_tool(verify_only=True)
                self.assertEqual(self.snapshot(self.output), before)

    def test_existing_output_is_not_used_as_an_implicit_cache_or_overwritten(self):
        self.output.mkdir()
        (self.output / "marker.txt").write_text("keep")
        before = self.snapshot(self.output)
        with self.assertRaises(ValueError):
            self.run_tool()
        self.assertEqual(self.snapshot(self.output), before)
        self.command.assert_not_called()
        self.fusion.assert_not_called()

    def test_output_cannot_be_inside_git_input_source_weights_or_runtime(self):
        candidates = [ROOT / "mri-tool-must-not-create", self.inputs / "report", self.config_dir / "report",
                      self.spine_source / "report", self.weights / "report", self.runtime / "report"]
        for path in candidates:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.run_tool(output=path)
                self.assertFalse(path.exists())
        self.command.assert_not_called()

    def test_invalid_configuration_types_keys_and_policy_values_are_rejected_before_execution(self):
        changed = [None, [], {**self.config, "extra": True}, {**self.config, "mode": "automatic"},
                   {**self.config, "schema_version": "2.0"}]
        for options in ({"allow_missing_modality": 1}, {"allow_unverified_registration": "yes"},
                        {"threads": True}, {"threads": 0}, {"timeout": 7201}, {"timeout": 0}, {"other": 1}):
            changed.append({**self.config, "options": options})
        for index, config in enumerate(changed):
            with self.subTest(config=config):
                self.write(self.config_path, config)
                output = self.directory / ("invalid-" + str(index))
                with self.assertRaises(ValueError):
                    self.run_tool(output=output)
                self.assertFalse(output.exists())
        self.command.assert_not_called()
        self.fusion.assert_not_called()

    def test_frozen_configuration_must_supply_all_three_runs_and_no_inference_options(self):
        self.frozen()
        for field in ("spinenet_run", "rsna_run", "crosscheck_run"):
            changed = dict(self.config)
            changed.pop(field)
            self.write(self.config_path, changed)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.run_tool()
        self.write(self.config_path, {**self.config, "options": {}})
        with self.assertRaises(ValueError):
            self.run_tool()
        self.fusion.assert_not_called()

    def test_invalid_or_missing_preflight_blocks_models_and_records_failed_stage(self):
        for name, effect in (("no-artifact", lambda *args: None),
                             ("wrong-count", lambda python, args, log, timeout: self.write(self.output / "preflight.json", {
                                 "kind": "input_preflight", "study_digest": manifest_digest(self.study),
                                 "files_verified": 0, "files_decoded": 0}))):
            with self.subTest(case=name):
                self.output = self.directory / name
                self.command.side_effect = effect
                with self.assertRaises((OSError, ValueError)):
                    self.run_tool()
                journal = load_document(self.output / "tool-run.json")
                self.assertEqual(journal["status"], "error")
                self.assertEqual(journal["failed_stage"], "preflight")
                self.assertEqual(journal["completed_stages"], [])
        self.spine.assert_not_called()
        self.rsna.assert_not_called()
        self.fusion.assert_not_called()

    def test_failed_localization_stops_later_stages_and_keeps_error_journal(self):
        self.spine.side_effect = ValueError("Fictitious model failure")
        with self.assertRaises(ValueError):
            self.run_tool()
        journal = load_document(self.output / "tool-run.json")
        self.assertEqual(journal["status"], "error")
        self.assertEqual(journal["failed_stage"], "localization")
        self.assertEqual(journal["completed_stages"], ["preflight"])
        self.rsna.assert_not_called()
        self.fusion.assert_not_called()
        self.assertEqual(self.command.call_count, 1)

    def test_missing_crosscheck_artifacts_cannot_be_accepted_from_exit_zero(self):
        def only_preflight(*args):
            if "preflight" in args[1]:
                self.fake_command(*args)
        self.command.side_effect = only_preflight
        with self.assertRaises((OSError, ValueError)):
            self.run_tool()
        journal = load_document(self.output / "tool-run.json")
        self.assertEqual(journal["status"], "error")
        self.assertEqual(journal["failed_stage"], "fusion")
        self.assertFalse((self.output / "fusion" / "report.html").exists())

    def test_input_change_during_stages_never_produces_completed_tool_run(self):
        def change_config(*args):
            self.fake_command(*args)
            self.config_path.write_bytes(self.config_path.read_bytes() + b"\n")
        self.command.side_effect = change_config
        with self.assertRaises(ValueError):
            self.run_tool()
        journal = load_document(self.output / "tool-run.json")
        self.assertEqual(journal["status"], "error")
        self.assertTrue(self.config_path.read_bytes().endswith(b"\n"))

    def test_verify_only_requires_boolean_and_existing_run(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_tool(verify_only=value)
        with self.assertRaises(ValueError):
            self.run_tool(verify_only=True)
        self.assertFalse(self.output.exists())

    def test_command_timeout_terminates_process_group_without_waiting_for_real_model(self):
        process = mock.Mock(pid=4321)
        process.wait.side_effect = [subprocess.TimeoutExpired("fictitious-stage", 1), 0]
        with mock.patch.object(self.tool.subprocess, "Popen", return_value=process) as launch:
            with mock.patch.object(self.tool.os, "killpg") as kill:
                with self.assertRaisesRegex(ValueError, "timed out"):
                    self.actual_command(self.python, ["-c", "pass"], self.directory / "timeout.log", 1)
                kill.assert_called_once_with(4321, signal.SIGTERM)
                self.assertIs(launch.call_args.kwargs["start_new_session"], True)

    def test_cli_reports_invalid_configuration_as_structured_error_two(self):
        self.write(self.config_path, {"schema_version": "1.0", "mode": "unknown"})
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/run_mri.py"),
                                    "--study", str(self.study_path), "--config", str(self.config_path),
                                    "--output", str(self.output)], cwd=ROOT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertNotIn("Traceback", completed.stderr)
        error = json.loads(completed.stderr)
        self.assertIs(error["ok"], False)
        self.assertTrue(error["error"].strip())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
