"""Isolated CPU MRI inference. Invoked by the checked stenosis-rsna runner."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from .contracts import load_document, manifest_digest, validate_bundle
from .control_study import write_json_external
from .rsna_assets import ROOT, file_sha256, verify_rsna_assets
from .rsna_probe import runtime_versions


def progress(phase):
    print(json.dumps({"phase": phase}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "weights", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-missing-modality", action="store_true")
    parser.add_argument("--allow-unverified-registration", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    attempted, performed = False, False
    study = load_document(args.study)
    try:
        validate_bundle(study, [])
        progress("verify_runtime_and_assets")
        runtime = runtime_versions()
        assets = verify_rsna_assets(args.source, args.weights)
        import torch
        import numpy as np
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        from .rsna_dicom import load_frames
        from .rsna_pipeline import infer, load_networks
        from .rsna_results import build_results
        from .rsna_spatial import validate_rsna_spatial
        from .rsna_qc import render_qc
        progress("read_and_verify_source_frames")
        frames, by_sequence = load_frames(study, allow_missing_modality=args.allow_missing_modality)
        progress("load_three_medical_checkpoints")
        networks, model_loading = load_networks(args.source, args.weights, assets)
        progress("automatic_keypoints_ROIs_and_stenosis")
        inference_started = time.monotonic()
        attempted = True
        native, spatial, examples = infer(study, frames, by_sequence, networks, assets,
                                          allow_unverified_registration=args.allow_unverified_registration)
        performed = native["keypoint_calls"] + native["classification_calls"] > 0
        inference_seconds = time.monotonic() - inference_started
        progress("validate_native_results_and_source_geometry")
        results = build_results(study, native)
        validate_rsna_spatial(study, native, spatial)
        for series in study["series"]:
            for image in series["images"]:
                if image["sha256"] is None or file_sha256(image["path"]) != image["sha256"]:
                    raise ValueError("Source bytes changed or are not pinned")
        progress("save_external_overlays_and_results")
        overlays = render_qc(frames, spatial, examples, args.output)
        artifacts = {"localization.json": results[0], "foraminal.json": results[1],
                     "subarticular.json": results[2], "native.json": native, "spatial.json": spatial}
        for name, value in artifacts.items():
            write_json_external(value, args.output / name)
        names = list(artifacts) + ["overlay.png", "roi_montage.png"]
        execution = {"kind": "rsna_mri_execution", "schema_version": "1.0", "status": "completed",
                     "created_at": datetime.now(timezone.utc).isoformat(), "study_id": study["study_id"],
                     "study_digest": manifest_digest(study), "runtime": runtime, "assets": assets,
                     "model_loading": model_loading, "threads": args.threads,
                     "parameters": {"allow_missing_modality": args.allow_missing_modality,
                                    "allow_unverified_registration": args.allow_unverified_registration,
                                    "seed": 0, "device": "cpu"},
                     "inference_attempted": attempted, "inference_performed": performed,
                     "classification_calls": native["classification_calls"], "keypoint_calls": native["keypoint_calls"],
                     "predicted_pairs": sum(p["status"] == "predicted" for p in native["predictions"]),
                     "inference_seconds": inference_seconds,
                     "elapsed_seconds": time.monotonic() - started, "source_bytes_unchanged": True,
                     "anatomical_review": "indeterminate", "registration_status": "unreviewed",
                     "root_impacts": None, "overlays": overlays,
                     "adapter_sources_sha256": {p.name: file_sha256(p) for p in sorted((ROOT / "mri_root_impact").glob("*.py"))},
                     "artifacts_sha256": {name: file_sha256(args.output / name) for name in names}}
        write_json_external(execution, args.output / "execution.json")
        progress("completed")
        return 0
    except Exception as exc:
        traceback.print_exc()
        error = {"kind": "rsna_mri_execution", "status": "error", "study_id": study["study_id"],
                 "study_digest": manifest_digest(study), "inference_attempted": attempted,
                 "inference_performed": True if performed else (None if attempted else False),
                 "error_type": type(exc).__name__, "error": str(exc), "root_impacts": None}
        write_json_external(error, args.output / "execution.json")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
