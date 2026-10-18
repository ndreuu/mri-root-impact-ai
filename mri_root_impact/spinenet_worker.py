"""Isolated, pinned CPU localization worker. Invoke through localize-spinenet."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .contracts import load_document, manifest_digest
from .control_study import write_json_external
from .spinenet_assets import ROOT, file_sha256, verify_assets


def progress(phase):
    print(json.dumps({"phase": phase}), flush=True)


def runtime_versions():
    expected = {}
    for line in (ROOT / "requirements-spinenet.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, pinned = line.split("==")
            actual = version(name)
            if actual != pinned:
                raise ValueError("SpineNet runtime version differs from requirements-spinenet.txt: " + name)
            expected[name] = actual
    if sys.version_info[:2] != (3, 12):
        raise ValueError("SpineNet worker requires the pinned Python 3.12 runtime")
    return {"python": sys.version.split()[0], "packages": expected, "device": "cpu"}


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if type(value) is float and not math.isfinite(value):
        raise ValueError("SpineNet produced nonfinite output")
    return value


def render_overlays(volume, geometry, native, output):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    height, width, count = volume.shape
    header = 28
    columns = min(5, count)
    montage = Image.new("RGB", (columns * width, math.ceil(count / columns) * (height + header)), "#101820")
    folder = output / "overlays"
    folder.mkdir()
    font = ImageFont.load_default(size=13)
    colors = [(255, 204, 64), (66, 225, 215), (246, 144, 199), (146, 225, 101), (127, 177, 255), (255, 156, 86)]
    files = []
    for index in range(count):
        image = volume[:, :, index].astype(float)
        lo, hi = np.percentile(image, [.5, 99.5])
        display = np.rint(np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)
        canvas = Image.new("RGB", (width, height + header), "#101820")
        canvas.paste(Image.fromarray(display).convert("RGB"), (0, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 4), "Slice {} | model labels, unreviewed".format(index), fill="white", font=font)
        for detection_index, detection in enumerate(native):
            if index not in detection["slice_nos"]:
                continue
            polygon = detection["polys"][detection["slice_nos"].index(index)]
            points = [(float(point[0]), float(point[1]) + header) for point in polygon]
            color = colors[detection_index % len(colors)]
            draw.line(points + [points[0]], fill=color, width=2)
            draw.text((max(0, min(point[0] for point in points)), max(header, min(point[1] for point in points))),
                      detection["predicted_label"], fill=color, font=font)
        name = "slice-{:03d}.png".format(index)
        canvas.save(folder / name)
        files.append({"slice_index": index, "image_ref": geometry["slices"][index]["image_ref"],
                      "file": "overlays/" + name, "sha256": file_sha256(folder / name),
                      "header_pixels": header, "image_scale": 1})
        montage.paste(canvas, ((index % columns) * width, (index // columns) * (height + header)))
    montage.save(output / "overlay.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--series-id")
    parser.add_argument("--allow-missing-modality", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    attempted, inference_completed = False, False
    study = load_document(args.study)
    try:
        progress("verify_runtime_and_assets")
        runtime = runtime_versions()
        assets = verify_assets(args.source)
        adapter_sources = {str(path.relative_to(ROOT)): file_sha256(path)
                           for path in sorted((ROOT / "mri_root_impact").glob("*.py"))}
        adapter_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=15).strip()
        from .sagittal import load_sagittal_series
        from .spatial import build_localization, validate_spatial
        progress("prepare_sagittal_geometry")
        volume, geometry = load_sagittal_series(study, series_id=args.series_id,
                                              allow_missing_modality=args.allow_missing_modality)
        import numpy as np
        import torch
        if int(np.__version__.split(".")[0]) >= 2:
            raise ValueError("Pinned SpineNet requires NumPy below 2")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        # Existing timestamp-valid upstream .pyc files must not bypass source verification.
        sys.pycache_prefix = str(args.output / ".source-pycache")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.source.resolve()))
        import spinenet
        if Path(spinenet.__file__).resolve() != args.source.resolve() / "spinenet/__init__.py":
            raise ValueError("Imported SpineNet does not match the verified source")
        progress("load_checkpoints")
        network = spinenet.SpineNet(device="cpu", scan_type="lumbar", verbose=True)
        progress("detect_and_label_vertebrae")
        attempted = True
        inference_started = time.monotonic()
        with torch.inference_mode():
            native = plain(network.detect_vb(volume, geometry["pixel_spacing_mm"]))
        inference_completed = True
        inference_seconds = time.monotonic() - inference_started
        progress("validate_native_geometry")
        result, spatial = build_localization(study, geometry, native,
                                             {key: assets[key] for key in ("source_commit", "weights_sha256")})
        validate_spatial(study, result, spatial)
        # Re-read source hashes after inference; nothing writes patient inputs.
        selected = next(series for series in study["series"] if series["series_id"] == geometry["series_id"])
        if any(file_sha256(image["path"]) != image["sha256"] for image in selected["images"]):
            raise ValueError("Source bytes changed during localization")
        progress("save_source_overlays")
        overlays = render_overlays(volume, geometry, native, args.output)
        write_json_external(result, args.output / "localization.json")
        write_json_external(spatial, args.output / "spatial.json")
        write_json_external({"kind": "native_spinenet_detections", "study_id": study["study_id"],
                             "study_digest": manifest_digest(study), "detections": native}, args.output / "native.json")
        execution = {
            "kind": "localization_execution", "study_id": study["study_id"],
            "study_digest": manifest_digest(study), "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime, "assets": assets, "adapter_sources_sha256": adapter_sources,
            "adapter_git_revision": adapter_revision,
            "series_id": geometry["series_id"],
            "parameters": {"scan_type": "lumbar", "device": "cpu", "threads": args.threads,
                           "penalise_skips": True, "remove_single_slice_detections": True,
                           "allow_missing_modality": args.allow_missing_modality, "seed": 0},
            "inference_attempted": True, "inference_performed": True,
            "inference_seconds": inference_seconds, "duration_seconds": time.monotonic() - started,
            "source_bytes_unchanged": True, "anatomical_review": "not_performed",
            "pathology_inference_performed": False, "root_impacts": None,
            "native_detection_count": len(native), "overlays": overlays,
            "artifacts_sha256": {name: file_sha256(args.output / name)
                                 for name in ("localization.json", "spatial.json", "native.json", "overlay.png")},
        }
        write_json_external(execution, args.output / "execution.json")
        progress("completed")
        return 0
    except Exception as error:
        traceback.print_exc()
        write_json_external({"kind": "localization_execution", "study_id": study["study_id"],
                             "study_digest": manifest_digest(study), "status": "error",
                             "error_type": type(error).__name__, "inference_attempted": attempted,
                             "inference_performed": inference_completed, "duration_seconds": time.monotonic() - started,
                             "root_impacts": None}, args.output / "execution.json")
        return 2


if __name__ == "__main__":
    sys.exit(main())
