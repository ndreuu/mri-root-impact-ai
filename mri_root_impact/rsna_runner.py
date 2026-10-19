"""Checked separate-process RSNA research inference; core remains stdlib-only."""

import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .contracts import load_document, manifest_digest, validate_bundle
from .control_study import write_json_external
from .rsna_assets import ROOT, file_sha256, verify_rsna_assets
from .rsna_results import build_results
from .rsna_spatial import validate_rsna_spatial
from .spinenet_runner import _png_dimensions, external_run_directory


ARTIFACTS = {"localization.json", "foraminal.json", "subarticular.json", "native.json", "spatial.json", "overlay.png", "roi_montage.png"}


def validate_worker_outputs(study, output, expected_assets=None, allow_unverified_registration=False):
    execution = load_document(output / "execution.json")
    if (not isinstance(execution, dict) or execution.get("kind") != "rsna_mri_execution"
            or execution.get("status") != "completed" or execution.get("study_id") != study["study_id"]
            or execution.get("study_digest") != manifest_digest(study)
            or type(execution.get("inference_performed")) is not bool
            or execution.get("source_bytes_unchanged") is not True
            or execution.get("root_impacts", "missing") is not None
            or execution.get("anatomical_review") != "indeterminate"
            or execution.get("registration_status") != "unreviewed"):
        raise ValueError("Worker has not completed a source-bound RSNA run with preserved uncertainty")
    hashes = execution.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != ARTIFACTS:
        raise ValueError("Worker must record all seven required RSNA artifacts")
    for name in ARTIFACTS:
        path = output / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != hashes[name]:
            raise ValueError("RSNA artifact checksum mismatch")
    native, spatial = load_document(output / "native.json"), load_document(output / "spatial.json")
    results = [load_document(output / filename) for filename in ("localization.json", "foraminal.json", "subarticular.json")]
    if build_results(study, native) != results:
        raise ValueError("C001 results differ from the checked native model outputs")
    validate_rsna_spatial(study, native, spatial)
    if execution.get("assets") != native["assets"] or (expected_assets is not None and execution["assets"] != expected_assets):
        raise ValueError("Execution provenance differs from verified model assets")
    predicted = sum(p["status"] == "predicted" for p in native["predictions"])
    calls = native.get("classification_calls")
    if (type(execution.get("classification_calls")) is not int or type(calls) is not int
            or not predicted <= calls <= 10 or execution["classification_calls"] != calls
            or execution.get("predicted_pairs") != predicted
            or (predicted and execution["inference_performed"] is not True)):
        raise ValueError("Execution model-call count disagrees with native predictions")
    keypoint_calls = native.get("keypoint_calls")
    if (type(keypoint_calls) is not int or not 0 <= keypoint_calls <= 7
            or execution.get("keypoint_calls") != keypoint_calls
            or execution["inference_performed"] != (keypoint_calls + calls > 0)):
        raise ValueError("Inference flag must reflect actual keypoint/classifier calls")
    if execution.get("parameters", {}).get("allow_unverified_registration") is not allow_unverified_registration:
        raise ValueError("Registration policy differs from the explicitly selected run policy")
    policy = "explicit_research_identity_LPS_unverified" if allow_unverified_registration else "require_same_declared_FrameOfReferenceUID"
    if native.get("registration_policy") != policy or spatial.get("registration_policy") != policy:
        raise ValueError("Native/spatial registration policy is inconsistent")
    if native.get("different_frame_of_reference_uids") and not allow_unverified_registration and predicted:
        raise ValueError("Unregistered cross-series prediction requires explicit research opt-in")
    adapter = {p.name: file_sha256(p) for p in sorted((ROOT / "mri_root_impact").glob("*.py"))}
    if execution.get("adapter_sources_sha256") != adapter:
        raise ValueError("Worker adapter code differs from the validating runner")
    overlays = execution.get("overlays")
    frames = spatial["frames"]
    if not isinstance(overlays, list) or len(overlays) != len(frames):
        raise ValueError("Every decoded source frame requires a recorded overlay")
    for index, (entry, frame) in enumerate(zip(overlays, frames)):
        name = "overlays/frame-{:03d}.png".format(index)
        path = output / name
        expected_size = frame["columns"], frame["rows"] + 32
        if (entry.get("file") != name or entry.get("image_ref") != frame["image_ref"]
                or entry.get("header_pixels") != 32 or path.is_symlink() or not path.is_file()
                or file_sha256(path) != entry.get("sha256") or _png_dimensions(path) != expected_size):
            raise ValueError("Source-frame overlay bytes/dimensions/mapping disagree")
    if _png_dimensions(output / "overlay.png") != (1280, max(1, math.ceil(len(frames) / 5)) * 288):
        raise ValueError("Incorrect source overlay montage")
    if _png_dimensions(output / "roi_montage.png") != (960, 940):
        raise ValueError("Incorrect classifier ROI montage")
    for series in study["series"]:
        for image in series["images"]:
            if image["sha256"] is None or file_sha256(image["path"]) != image["sha256"]:
                raise ValueError("Input source bytes changed")
    return execution


def run_rsna(study_path, source, weights, python, output, allow_missing_modality=False,
             allow_unverified_registration=False, threads=2, timeout=600):
    study_path = Path(study_path).resolve()
    study = load_document(study_path)
    validate_bundle(study, [])
    if type(threads) is not int or not 1 <= threads <= 8 or type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("Threads must be 1..8 and timeout 1..7200 seconds")
    if type(allow_missing_modality) is not bool or type(allow_unverified_registration) is not bool:
        raise ValueError("Input/registration policies must be explicit booleans")
    output = external_run_directory(study_path, study, output, source)
    weights = Path(weights).resolve()
    if weights == output or weights in output.parents:
        raise ValueError("RSNA run must not modify the model weights")
    python = Path(python).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Isolated RSNA Python is unavailable")
    assets = verify_rsna_assets(source, weights)
    output.mkdir(parents=True)
    command = [str(python), "-m", "mri_root_impact.rsna_worker", "--study", str(study_path),
               "--source", str(Path(source).resolve()), "--weights", str(weights), "--output", str(output), "--threads", str(threads)]
    if allow_missing_modality:
        command.append("--allow-missing-modality")
    if allow_unverified_registration:
        command.append("--allow-unverified-registration")
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                        "PYTHONPYCACHEPREFIX": str(output / ".source-pycache"),
                        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    environment.pop("PYTHONHOME", None)
    started = time.monotonic()
    timed_out, process, launch_error = False, None, None
    try:
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=output, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    except OSError as exc:
        launch_error = type(exc).__name__
    report = {"kind": "rsna_inference_run", "study_id": study["study_id"], "status": "error",
              "command": command, "worker_exit_code": process.returncode if process else None,
              "timed_out": timed_out, "launch_error_type": launch_error,
              "duration_seconds": time.monotonic() - started, "inference_performed": None,
              "worker_log_sha256": file_sha256(output / "worker.log"), "root_impacts": None}
    try:
        if not timed_out and process is not None and process.returncode == 0:
            execution = validate_worker_outputs(study, output, assets, allow_unverified_registration)
            report.update(status="completed", inference_performed=execution["inference_performed"],
                          stenosis_predictions=execution["predicted_pairs"] * 2,
                          anatomical_review="indeterminate", registration_status="unreviewed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["validation_error"] = str(exc)
    write_json_external(report, output / "run.json")
    if report["status"] != "completed":
        raise ValueError("RSNA run failed; inspect external run.json and worker.log at " + str(output))
    return {"ok": True, "operation": "stenosis-rsna", "output": str(output),
            "inference_performed": report["inference_performed"], "stenosis_predictions": report["stenosis_predictions"],
            "anatomical_review": "indeterminate", "registration_status": "unreviewed", "root_impacts": None}
