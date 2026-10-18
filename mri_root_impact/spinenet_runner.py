"""Launch optional SpineNet inference in a separate Python and fresh external run directory."""

import os
from pathlib import Path
import signal
import struct
import subprocess
import time

from .contracts import load_document, manifest_digest, validate_bundle
from .spinenet_assets import ROOT, file_sha256


def _png_dimensions(path):
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR" or len(header) != 24:
        raise ValueError("Localization overlay is not a PNG")
    return struct.unpack(">II", header[16:24])


def validate_worker_outputs(study, output):
    from .spatial import build_localization, validate_spatial
    expected_files = {"localization.json", "spatial.json", "native.json", "overlay.png"}
    execution = load_document(output / "execution.json")
    if (not isinstance(execution, dict) or execution.get("kind") != "localization_execution"
            or execution.get("study_id") != study["study_id"] or execution.get("study_digest") != manifest_digest(study)
            or execution.get("status") != "completed" or execution.get("inference_performed") is not True):
        raise ValueError("Worker did not record completed inference for this manifest")
    hashes = execution.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise ValueError("Worker must record all four required localization artifacts")
    for name, digest in hashes.items():
        path = output / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError("Localization artifact checksum mismatch")
    result = load_document(output / "localization.json")
    spatial = load_document(output / "spatial.json")
    validate_spatial(study, result, spatial)
    native = load_document(output / "native.json")
    if (not isinstance(native, dict) or native.get("kind") != "native_spinenet_detections"
            or native.get("study_id") != study["study_id"] or native.get("study_digest") != manifest_digest(study)):
        raise ValueError("Native detections are not bound to this manifest")
    rebuilt, _ = build_localization(study, spatial["geometry"], native.get("detections"), spatial["model_provenance"])
    if rebuilt != result or type(execution.get("native_detection_count")) is not int or execution["native_detection_count"] != len(spatial["detections"]):
        raise ValueError("Native detections disagree with the validated localization")
    geometry = spatial["geometry"]
    assets = execution.get("assets")
    if (not isinstance(assets, dict) or execution.get("series_id") != geometry["series_id"]
            or any(assets.get(key) != spatial["model_provenance"][key] for key in ("source_commit", "weights_sha256"))
            or execution.get("source_bytes_unchanged") is not True):
        raise ValueError("Execution provenance disagrees with the source-bound localization")
    height, width, count = geometry["volume_shape"]
    overlays = execution.get("overlays")
    if not isinstance(overlays, list) or len(overlays) != count:
        raise ValueError("Every source slice must have a recorded overlay")
    for index, entry in enumerate(overlays):
        name = "overlays/slice-{:03d}.png".format(index)
        if (not isinstance(entry, dict) or entry.get("slice_index") != index or entry.get("file") != name
                or entry.get("image_ref") != geometry["slices"][index]["image_ref"]
                or entry.get("header_pixels") != 28 or entry.get("image_scale") != 1):
            raise ValueError("Overlay source-frame mapping is inconsistent")
        path = output / name
        if (path.is_symlink() or output.resolve() not in path.resolve().parents
                or file_sha256(path) != entry.get("sha256") or _png_dimensions(path) != (width, height + 28)):
            raise ValueError("Overlay bytes or dimensions are inconsistent")
    columns = min(5, count)
    if _png_dimensions(output / "overlay.png") != (columns * width, ((count + columns - 1) // columns) * (height + 28)):
        raise ValueError("Overlay montage dimensions are inconsistent")
    return execution


def external_run_directory(study_path, study, output, source):
    output = Path(output).resolve()
    source = Path(source).resolve()
    if output == source or source in output.parents:
        raise ValueError("Localization output must not modify the upstream source")
    if output.exists():
        raise ValueError("Localization requires a new output directory for every run")
    for parent in output.parents:
        if (parent / ".git").exists():
            raise ValueError("Real localization artifacts must be written outside Git")
    inputs = [Path(study_path).resolve()] + [Path(image["path"]).resolve()
                                          for series in study["series"] for image in series["images"]]
    if any(output == path or output in path.parents for path in inputs):
        raise ValueError("Localization output overlaps an input")
    return output


def run_spinenet(study_path, source, python, output, series_id=None,
                 allow_missing_modality=False, threads=2, timeout=600):
    study_path = Path(study_path).resolve()
    study = load_document(study_path)
    validate_bundle(study, [])
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("SpineNet threads must be an integer from 1 to 8")
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("SpineNet timeout must be an integer from 1 to 7200 seconds")
    if type(allow_missing_modality) is not bool:
        raise ValueError("Modality policy must be an explicit boolean")
    output = external_run_directory(study_path, study, output, source)
    python = Path(python).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("SpineNet Python executable is unavailable")
    output.mkdir(parents=True)
    command = [str(python), "-m", "mri_root_impact.spinenet_worker", "--study", str(study_path),
               "--source", str(Path(source).resolve()), "--output", str(output), "--threads", str(threads)]
    if series_id is not None:
        command.extend(["--series-id", series_id])
    if allow_missing_modality:
        command.append("--allow-missing-modality")
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(output / ".mpl-cache"),
                        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    environment.pop("PYTHONHOME", None)
    started = time.monotonic()
    timed_out, launch_error, process = False, None, None
    try:
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=output, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    except OSError as error:
        launch_error = type(error).__name__
    from .control_study import write_json_external
    report = {"kind": "localization_run", "study_id": study["study_id"],
              "worker_exit_code": process.returncode if process else None, "timed_out": timed_out,
              "duration_seconds": time.monotonic() - started,
              "command": command, "worker_log_sha256": file_sha256(output / "worker.log"),
              "inference_performed": None, "root_impacts": None, "status": "error",
              "launch_error_type": launch_error}
    try:
        if (output / "execution.json").is_file():
            execution = load_document(output / "execution.json")
            if isinstance(execution, dict):
                report["inference_performed"] = execution.get("inference_performed")
        if not timed_out and process is not None and process.returncode == 0:
            execution = validate_worker_outputs(study, output)
            report.update({"status": "completed", "inference_performed": True,
                           "native_detection_count": execution["native_detection_count"],
                           "anatomical_review": execution["anatomical_review"]})
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["validation_error_type"] = type(error).__name__
    write_json_external(report, output / "run.json")
    if report["status"] != "completed":
        raise ValueError("SpineNet run failed; inspect external run.json and worker.log at " + str(output))
    return {"ok": True, "operation": "localize-spinenet", "output": str(output),
            "inference_performed": True, "native_detection_count": report["native_detection_count"],
            "anatomical_review": report["anatomical_review"], "root_impacts": None}
