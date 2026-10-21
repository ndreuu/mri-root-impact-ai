"""Orchestrate existing checked MRI modules without changing their decisions."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from mri_fusion.runner import file_sha256, run_fusion
from mri_root_impact.contracts import load_document, manifest_digest, validate_bundle
from mri_root_impact.rsna_runner import run_rsna
from mri_root_impact.spinenet_runner import run_spinenet
from . import __version__


ROOT = Path(__file__).resolve().parents[1]


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) <= set(required) | set(optional):
        raise ValueError("Invalid configuration keys; required: " + ", ".join(required))


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configuration paths must be nonempty strings")
    # Preserve a venv's final Python symlink: resolving it selects the base runtime.
    return Path(os.path.abspath(base / value))


def _config(path):
    config = load_document(path)
    if not isinstance(config, dict) or config.get("schema_version") != "1.0":
        raise ValueError("Tool configuration requires schema_version 1.0")
    base = path.parent
    mode = config.get("mode")
    if mode == "frozen":
        names = ("spinenet_run", "rsna_run", "crosscheck_run")
        _object(config, ("schema_version", "mode") + names)
        return {"mode": mode, **{name: _path(config[name], base) for name in names}}
    if mode != "inference":
        raise ValueError("Tool mode must be inference or frozen")
    _object(config, ("schema_version", "mode", "spinenet", "rsna"), ("options",))
    resolved = {"mode": mode}
    for model, fields in (("spinenet", ("source", "python")), ("rsna", ("source", "weights", "python"))):
        _object(config[model], fields)
        resolved[model] = {key: _path(config[model][key], base) for key in fields}
    options = config.get("options", {})
    _object(options, (), ("allow_missing_modality", "allow_unverified_registration", "threads", "timeout"))
    options = {"allow_missing_modality": False, "allow_unverified_registration": False,
               "threads": 2, "timeout": 600, **options}
    if any(type(options[key]) is not bool for key in ("allow_missing_modality", "allow_unverified_registration")):
        raise ValueError("Research policies must be explicit booleans")
    if type(options["threads"]) is not int or not 1 <= options["threads"] <= 8:
        raise ValueError("threads must be an integer from 1 to 8")
    if type(options["timeout"]) is not int or not 1 <= options["timeout"] <= 7200:
        raise ValueError("timeout must be an integer from 1 to 7200 seconds")
    resolved["options"] = options
    return resolved


def _code_hashes():
    paths = sorted((ROOT / "mri_tool").glob("*.py")) + [ROOT / "scripts/run_mri.py"]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in paths}


def _write(path, document):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _run_command(python, arguments, log_path, timeout):
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    environment.pop("PYTHONHOME", None)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen([str(python)] + [str(v) for v in arguments], cwd=ROOT,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise ValueError("Stage timed out; inspect " + str(log_path))
    if code != 0:
        raise ValueError("Stage failed; inspect " + str(log_path))


def _protect(output, study_path, study, config_path, config, verify_only):
    raw = Path(output)
    output = raw.resolve()
    if raw.is_symlink() or any((p / ".git").exists() for p in (output,) + tuple(output.parents)):
        raise ValueError("Tool output must be outside Git and not a symlink")
    if (verify_only and not output.is_dir()) or (not verify_only and output.exists()):
        raise ValueError("Verification needs an existing output; a run needs a new output directory")
    files = [study_path, config_path] + [Path(image["path"]).resolve()
                                      for series in study["series"] for image in series["images"]]
    for path in files:
        if output == path or output in path.parents or path.parent in output.parents:
            raise ValueError("Tool output overlaps an input file directory")
    directories = ([config[key].resolve() for key in ("spinenet_run", "rsna_run", "crosscheck_run")]
                   if config["mode"] == "frozen" else
                   [config["spinenet"]["source"].resolve(), config["rsna"]["source"].resolve(), config["rsna"]["weights"].resolve()])
    if config["mode"] == "inference":
        for model in ("spinenet", "rsna"):
            parent = config[model]["python"].parent
            directories.append((parent.parent if parent.name in ("bin", "Scripts") else parent).resolve())
    for directory in directories:
        if output == directory or directory in output.parents or output in directory.parents:
            raise ValueError("Tool output overlaps a source, weights or upstream run")
    return output


def _unchanged(bindings):
    if any(file_sha256(path) != digest for path, digest in bindings.items()):
        raise ValueError("Study or configuration changed during execution")


def run_mri(study_path, config_path, output, verify_only=False, progress=None):
    """A prepared DICOM study and local model config produce a checked report."""
    if type(verify_only) is not bool:
        raise ValueError("verify_only must be a boolean")
    study_path, config_path = Path(study_path).resolve(), Path(config_path).resolve()
    bindings = {str(p): file_sha256(p) for p in (study_path, config_path)}
    study = load_document(study_path)
    validate_bundle(study, [])
    config = _config(config_path)
    output = _protect(output, study_path, study, config_path, config, verify_only)
    runs = ({key: config[key] for key in ("spinenet_run", "rsna_run", "crosscheck_run")}
            if config["mode"] == "frozen" else
            {"spinenet_run": output / "localization", "rsna_run": output / "stenosis", "crosscheck_run": output / "crosscheck"})
    identity = {"kind": "mri_report_tool_run", "schema_version": "1.0", "tool_version": __version__, "mode": config["mode"],
                "study_id": study["study_id"], "study_digest": manifest_digest(study),
                "input_files_sha256": bindings, "code_files_sha256": _code_hashes(),
                "stage_directories": {k: str(p) for k, p in runs.items()},
                "report": str(output / "fusion/report.html"),
                "markdown_report": str(output / "fusion/report.md"),
                "herniation_analysis": "not_configured"}
    if verify_only:
        saved = load_document(output / "tool-run.json")
        if (not isinstance(saved, dict) or saved.get("status") != "completed"
                or any(saved.get(key) != value for key, value in identity.items())
                or saved.get("inference_performed") is not (config["mode"] == "inference")
                or saved.get("completed_stages") != (["preflight", "localization", "stenosis", "crosscheck", "fusion"]
                                                     if config["mode"] == "inference" else ["fusion"])
                or saved.get("fusion_run_sha256") != file_sha256(output / "fusion/run.json")):
            raise ValueError("Tool run metadata, inputs, code or fusion binding differ")
        result = run_fusion(study_path, output / "fusion", **runs, verify_only=True)
        _unchanged(bindings)
        return {**result, "operation": "verify-mri-report", "output": str(output),
                "tool_run": str(output / "tool-run.json")}
    output.mkdir(parents=True)
    started = time.monotonic()
    stage = "preflight" if config["mode"] == "inference" else "fusion"
    completed = []
    def advance(name):
        nonlocal stage
        stage = name
        if progress is not None:
            progress(name)
    try:
        if config["mode"] == "inference":
            options = config["options"]
            policy = ["--allow-missing-modality"] if options["allow_missing_modality"] else []
            python = config["rsna"]["python"]
            for model in ("spinenet", "rsna"):
                executable = config[model]["python"]
                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise ValueError("Model Python executable is unavailable: " + str(executable))
            advance("preflight")
            _run_command(python, ["-m", "mri_root_impact", "preflight", "--study", study_path,
                                 "--output", output / "preflight.json"] + policy,
                         output / "preflight.log", options["timeout"])
            preflight = load_document(output / "preflight.json")
            image_count = sum(len(series["images"]) for series in study["series"])
            if (not isinstance(preflight, dict) or preflight.get("kind") != "input_preflight"
                    or preflight.get("study_digest") != manifest_digest(study)
                    or preflight.get("files_verified") != image_count
                    or preflight.get("files_decoded") != image_count):
                raise ValueError("Preflight did not confirm this study")
            completed.append(stage)
            advance("localization")
            common = {key: options[key] for key in ("allow_missing_modality", "threads", "timeout")}
            run_spinenet(study_path, output=runs["spinenet_run"], **config["spinenet"], **common)
            completed.append(stage)
            advance("stenosis")
            run_rsna(study_path, output=runs["rsna_run"], **config["rsna"], **common,
                     allow_unverified_registration=options["allow_unverified_registration"])
            completed.append(stage)
            advance("crosscheck")
            _run_command(python, [ROOT / "scripts/compare_localizers.py", "--study", study_path,
                                 "--spinenet-run", runs["spinenet_run"], "--rsna-run", runs["rsna_run"],
                                 "--output", runs["crosscheck_run"]] + policy,
                         output / "crosscheck.log", options["timeout"])
            completed.append(stage)
        advance("fusion")
        result = run_fusion(study_path, output / "fusion", **runs)
        completed.append(stage)
        _unchanged(bindings)
        journal = {**identity, "status": "completed", "completed_stages": completed,
                   "inference_performed": config["mode"] == "inference",
                   "duration_seconds": time.monotonic() - started,
                   "fusion_run_sha256": file_sha256(output / "fusion/run.json")}
        _write(output / "tool-run.json", journal)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        _write(output / "tool-run.json", {**identity, "status": "error", "failed_stage": stage,
                                         "completed_stages": completed, "error": str(exc),
                                         "duration_seconds": time.monotonic() - started})
        raise
    return {**result, "operation": "mri-report", "output": str(output),
            "inference_performed": config["mode"] == "inference",
            "tool_run": str(output / "tool-run.json")}
