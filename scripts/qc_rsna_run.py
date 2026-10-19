#!/usr/bin/env python3
"""Re-read source headers and independently verify saved RSNA pixel/LPS geometry."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_root_impact.contracts import load_document
from mri_root_impact.control_study import write_json_external
from mri_root_impact.rsna_runner import validate_worker_outputs


def check(study, run):
    import numpy as np
    import pydicom
    execution = load_document(run / "execution.json")
    policy = execution["parameters"]["allow_unverified_registration"]
    validate_worker_outputs(study, run, allow_unverified_registration=policy)
    spatial = load_document(run / "spatial.json")
    manifest = {(s["series_id"], i["image_id"], 0): i for s in study["series"] for i in s["images"]}
    headers = {}
    for frame in spatial["frames"]:
        ref = frame["image_ref"]
        key = (ref["series_id"], ref["image_id"], ref["frame_index"])
        image = manifest[key]
        payload = Path(image["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != image["sha256"]:
            raise ValueError("Original source bytes changed")
        d = pydicom.dcmread(BytesIO(payload), stop_before_pixels=True)
        for actual, saved in [(d.ImagePositionPatient, frame["position"]),
                              (d.ImageOrientationPatient, frame["orientation"]),
                              (d.PixelSpacing, frame["spacing"])]:
            if not np.allclose(np.asarray(actual, dtype=float), saved, atol=1e-9, rtol=0):
                raise ValueError("Saved source geometry differs from original DICOM header")
        if (int(d.Rows), int(d.Columns)) != (frame["rows"], frame["columns"]):
            raise ValueError("Saved source matrix differs from DICOM header")
        uid = str(getattr(d, "FrameOfReferenceUID", "") or "").strip() or None
        if uid != frame["frame_of_reference_uid"]:
            raise ValueError("Saved FrameOfReferenceUID differs from original header")
        headers[key] = d

    def header(reference):
        return headers[(reference["series_id"], reference["image_id"], reference["frame_index"])]

    def transform(d, x, y):
        # Independent DICOM formula: column x uses PixelSpacing[1], row y [0].
        return (np.asarray(d.ImagePositionPatient, dtype=float)
                + np.asarray(d.ImageOrientationPatient[:3], dtype=float) * float(d.PixelSpacing[1]) * x
                + np.asarray(d.ImageOrientationPatient[3:], dtype=float) * float(d.PixelSpacing[0]) * y)

    max_corner_error, max_pixel_error = 0., 0.
    for roi in spatial["rois"]:
        d = header(roi["image_ref"])
        left, top, right, bottom = roi["source_crop_box"]
        if not (0 <= left < right <= d.Columns and 0 <= top < bottom <= d.Rows):
            raise ValueError("ROI exceeds original source FOV")
        corners = [transform(d, x - .5, y - .5) for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]
        max_corner_error = max(max_corner_error, float(np.max(np.abs(np.asarray(corners) - roi["source_corners_lps_mm"]))))
    for point in spatial["keypoints"]:
        d = header(point["image_ref"])
        basis = np.column_stack([np.asarray(d.ImageOrientationPatient[:3]) * float(d.PixelSpacing[1]),
                                 np.asarray(d.ImageOrientationPatient[3:]) * float(d.PixelSpacing[0])])
        xy, _, _, _ = np.linalg.lstsq(basis, np.asarray(point["lps_mm"]) - np.asarray(d.ImagePositionPatient), rcond=None)
        max_pixel_error = max(max_pixel_error, float(np.max(np.abs(xy - [point["column"], point["row"]]))))
    if max_corner_error > 1e-5 or max_pixel_error > 1e-5:
        raise ValueError("Independent source geometry roundtrip failed")
    return {"kind": "rsna_technical_qc", "study_id": study["study_id"], "status": "passed",
            "source_files_verified": len(headers), "roi_footprints_verified": len(spatial["rois"]),
            "native_keypoints_verified": len(spatial["keypoints"]),
            "max_corner_error_mm": max_corner_error, "max_pixel_roundtrip_error": max_pixel_error,
            "source_bytes_unchanged": True, "numbering_confirmed": False, "registration_confirmed": False,
            "explicit_unverified_registration": policy, "model_inference_performed_by_qc": False,
            "root_impacts": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run, output = args.run.resolve(), args.output.resolve()
        if output.parent != run or output.exists() or args.output.is_symlink():
            raise ValueError("QC output must be a new file inside the external run directory")
        report = check(load_document(args.study), run)
        write_json_external(report, output)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
