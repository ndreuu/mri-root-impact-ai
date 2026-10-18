#!/usr/bin/env python3
"""Offline development harness. No MRI models or clinical inference are run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("localization", "disc_analysis", "stenosis", "fusion")
STATUSES = {"todo", "ready", "in_progress", "done"}
FORBIDDEN_SUFFIXES = (
    ".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".nrrd", ".mha", ".mhd",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npy", ".npz",
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_tasks(path):
    registry = read_json(path)
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("Task registry requires schema_version=1")
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task registry must contain a nonempty tasks list")
    by_id = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be an object")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Z]\d{3}", task_id):
            raise ValueError("Invalid task id: {!r}".format(task_id))
        if task_id in by_id:
            raise ValueError("Duplicate task id: " + task_id)
        if not isinstance(task.get("title"), str) or not task["title"].strip():
            raise ValueError("Missing title: " + task_id)
        if not isinstance(task.get("status"), str) or task["status"] not in STATUSES:
            raise ValueError("Invalid status: " + task_id)
        dependencies = task.get("depends_on")
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise ValueError("Invalid dependencies: " + task_id)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("Duplicate dependency: " + task_id)
        evidence = task.get("evidence")
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) or not item.strip() for item in evidence
        ):
            raise ValueError("Invalid evidence: " + task_id)
        if task["status"] == "done" and not evidence:
            raise ValueError("Completed task requires evidence: " + task_id)
        by_id[task_id] = task

    visited = set()
    active = set()

    def visit(task_id):
        if task_id in active:
            raise ValueError("Dependency cycle at " + task_id)
        if task_id in visited:
            return
        active.add(task_id)
        for dependency in by_id[task_id]["depends_on"]:
            if dependency not in by_id:
                raise ValueError("Unknown dependency: " + dependency)
            visit(dependency)
        active.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task["id"])
        if task["status"] in {"ready", "in_progress", "done"}:
            for dependency in task["depends_on"]:
                if by_id[dependency]["status"] != "done":
                    raise ValueError("Unresolved dependency for " + task["id"])
    return tasks


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=False,
    )


def doctor():
    git_available = shutil.which("git") is not None
    version = git("--version") if git_available else None
    toplevel = git("rev-parse", "--show-toplevel") if git_available else None
    correct_repo = bool(
        toplevel and toplevel.returncode == 0
        and Path(toplevel.stdout.strip()).resolve() == ROOT
    )
    return {
        "ok": sys.version_info >= (3, 9) and git_available and correct_repo,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "minimum_python": "3.9",
        "git": version.stdout.strip() if version else "unavailable",
        "repository": str(ROOT),
        "repository_initialized": correct_repo,
        "dependencies": "standard_library_only",
        "mri_models": "optional_spinenet_requires_explicit_external_paths",
        "inference_ready": False,
    }


def unsafe_candidates():
    result = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if result.returncode:
        raise ValueError("Cannot inspect Git candidates: " + result.stderr.strip())
    unsafe = []
    for name in set(result.stdout.split("\0")) - {""}:
        path = Path(name)
        lower_name = name.lower()
        protected_directory = path.parts[0] in {"data", "models", "runs"}
        allowed_readme = name in {"data/README.md", "models/README.md"}
        env_file = path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        )
        if (
            lower_name.endswith(FORBIDDEN_SUFFIXES) or env_file
            or (protected_directory and not allowed_readme)
        ):
            unsafe.append(name)
    return sorted(unsafe)


def read_fixture(path):
    fixture = read_json(path)
    fields = {"case_id", "synthetic", "input_kind", "series"}
    if not isinstance(fixture, dict) or set(fixture) != fields:
        raise ValueError("Smoke fixture must contain only case_id, synthetic, input_kind, series")
    if fixture["synthetic"] is not True:
        raise ValueError("Smoke accepts synthetic fixtures only")
    if fixture["input_kind"] != "manifest_only" or fixture["series"] != []:
        raise ValueError("Smoke accepts an empty manifest only; it does not run MRI inference")
    if not isinstance(fixture["case_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", fixture["case_id"]
    ):
        raise ValueError("Fixture requires a valid case_id")
    return fixture


def validate_smoke_report(report):
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Invalid smoke report schema")
    if report.get("kind") != "harness_smoke" or report.get("synthetic") is not True:
        raise ValueError("Smoke report must be explicitly synthetic")
    if report.get("inference_performed") is not False:
        raise ValueError("Harness cannot claim model inference")
    if "root_impacts" not in report or report["root_impacts"] is not None:
        raise ValueError("Unassessed root impacts must remain null")
    expected_stages = {name: "not_configured" for name in STAGES}
    if report.get("stages") != expected_stages:
        raise ValueError("MRI stages must remain not_configured in harness smoke")
    if not isinstance(report.get("case_id"), str) or not report["case_id"]:
        raise ValueError("Smoke report requires case_id")


def smoke(fixture_path, output_dir):
    fixture_path = Path(fixture_path).resolve()
    fixture = read_fixture(fixture_path)
    revision = git("rev-parse", "--verify", "HEAD")
    report = {
        "schema_version": 1,
        "kind": "harness_smoke",
        "case_id": fixture["case_id"],
        "synthetic": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(git("status", "--porcelain").stdout.strip()),
        "python": sys.version.split()[0],
        "inference_performed": False,
        "root_impacts": None,
        "stages": {name: "not_configured" for name in STAGES},
    }
    validate_smoke_report(report)
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / "report.json"
    if report_path == fixture_path:
        raise ValueError("Report must not overwrite its input fixture")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(output_dir),
            prefix=".report-", suffix=".json", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(report_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    persisted = read_json(report_path)
    validate_smoke_report(persisted)
    if persisted != report:
        raise ValueError("Smoke report round trip failed")
    return {
        "ok": True,
        "kind": "harness_smoke",
        "report": str(report_path),
        "inference_performed": False,
        "message": "Harness passed; model inference not performed.",
    }


def run_tests(test_dir):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    suite = unittest.TestLoader().discover(str(test_dir))
    count = suite.countTestCases()
    if count == 0:
        raise ValueError("No tests discovered; check cannot pass without tests")
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    return result.wasSuccessful(), count


def check():
    environment = doctor()
    if not environment["ok"]:
        raise ValueError("Environment check failed; run doctor")
    tasks = load_tasks(ROOT / "harness/tasks.json")
    unsafe = unsafe_candidates()
    if unsafe:
        raise ValueError("Forbidden Git candidates: " + ", ".join(unsafe))
    for args in (("diff", "--check"), ("diff", "--cached", "--check")):
        result = git(*args)
        if result.returncode:
            raise ValueError(result.stdout.strip() or result.stderr.strip())
    sys.dont_write_bytecode = True
    passed, test_count = run_tests(ROOT / "tests")
    if not passed:
        return {"ok": False, "failed": "unit_tests"}
    examples = ROOT / "harness/fixtures/contracts"
    command = [sys.executable, "-m", "mri_root_impact", "validate", "--study", str(examples / "study.json")]
    for name in ("localization", "stenosis-assessed", "stenosis-indeterminate", "stenosis-not-run"):
        command.extend(["--result", str(examples / (name + ".json"))])
    validation = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if validation.returncode:
        raise ValueError("Contract examples failed: " + validation.stderr.strip())
    with tempfile.TemporaryDirectory(prefix="mri-harness-check-") as directory:
        smoke(ROOT / "harness/fixtures/synthetic-study.json", directory)
    return {
        "ok": True,
        "checks": ["environment", "task_registry", "git_candidates", "diff_whitespace", "unit_tests", "contract_examples", "synthetic_smoke"],
        "tasks": len(tasks),
        "tests_run": test_count,
        "inference_performed": False,
        "message": "Development harness verified; MRI model quality has not been evaluated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local tools and repository")
    commands.add_parser("status", help="Validate and list development tasks")
    commands.add_parser("check", help="Run offline repository checks and tests")
    smoke_parser = commands.add_parser("smoke", help="Exercise synthetic harness artifact round trip")
    smoke_parser.add_argument("--fixture", type=Path, default=ROOT / "harness/fixtures/synthetic-study.json")
    smoke_parser.add_argument("--output", type=Path, default=ROOT / "runs/smoke")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "status":
            result = {"ok": True, "tasks": load_tasks(ROOT / "harness/tasks.json")}
        elif args.command == "smoke":
            result = smoke(args.fixture, args.output)
        else:
            result = check()
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
