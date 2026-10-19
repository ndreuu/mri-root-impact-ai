"""Checked single-frame inputs for the pinned RSNA research configuration."""

import hashlib
from io import BytesIO
from pathlib import Path

from .contracts import validate_study
from .preflight import _uid
from .rsna_geometry import _axes, dot


SEQUENCES = ("sagittal_t1", "sagittal_t2", "axial_t2")


def normalize_pixels(pixels):
    import numpy as np
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 2 or not pixels.size or not np.isfinite(pixels).all():
        raise ValueError("RSNA requires finite nonempty grayscale frames")
    low, high = np.percentile(pixels, (0.1, 99.9))
    clipped = np.clip(pixels, low, high)
    clipped -= clipped.min()
    maximum = float(clipped.max())
    if maximum <= 0:
        raise ValueError("Constant/empty-intensity frame cannot be evaluated")
    return (clipped / maximum * 255).astype(np.uint8)


def load_frames(study, allow_missing_modality=False):
    import numpy as np
    import pydicom
    validate_study(study)
    if type(allow_missing_modality) is not bool:
        raise ValueError("Modality policy must be explicit")
    frames, by_sequence, study_uids = [], {}, set()
    for sequence in SEQUENCES:
        candidates = [s for s in study["series"] if s["sequence"] == sequence]
        if len(candidates) > 1:
            raise ValueError("Select a manifest containing exactly one series per required sequence")
        if not candidates:
            by_sequence[sequence] = []
            continue
        series, selected, series_uids = candidates[0], [], set()
        for image in series["images"]:
            if image["frame_count"] != 1 or image["sha256"] is None:
                raise ValueError("RSNA requires single-frame DICOM with pinned image SHA-256")
            data = Path(image["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != image["sha256"]:
                raise ValueError("Input SHA-256 mismatch: " + image["image_id"])
            d = pydicom.dcmread(BytesIO(data))
            modality = str(getattr(d, "Modality", "") or "").strip()
            if modality != "MR" and not (allow_missing_modality and not modality):
                raise ValueError("RSNA input must be MR; missing modality requires an explicit flag")
            if (int(getattr(d, "NumberOfFrames", 1)) != 1
                    or getattr(d, "SamplesPerPixel", None) != 1
                    or getattr(d, "PhotometricInterpretation", None) != "MONOCHROME2"
                    or hasattr(d, "PerFrameFunctionalGroupsSequence") or hasattr(d, "SharedFunctionalGroupsSequence")):
                raise ValueError("Only conventional single-frame MONOCHROME2 is supported")
            study_uids.add(_uid(d, "StudyInstanceUID"))
            series_uids.add(_uid(d, "SeriesInstanceUID"))
            frame = {"image_ref": {"series_id": series["series_id"], "image_id": image["image_id"], "frame_index": 0},
                     "sequence": sequence, "sha256": image["sha256"], "path": image["path"],
                     "position": [float(v) for v in d.ImagePositionPatient],
                     "orientation": [float(v) for v in d.ImageOrientationPatient],
                     "spacing": [float(v) for v in d.PixelSpacing],
                     "rows": int(d.Rows), "columns": int(d.Columns),
                     "frame_of_reference_uid": _uid(d, "FrameOfReferenceUID", required=False),
                     "instance_number": int(d.InstanceNumber),
                     "modality_status": "dicom_declared" if modality else "dataset_declared"}
            _axes(frame)
            c, r = frame["orientation"][:3], frame["orientation"][3:]
            if sequence.startswith("sagittal"):
                if c[1] <= .99 or r[2] >= -.99:
                    raise ValueError("Unsupported sagittal orientation; no silent side/polarity transform")
            elif c[0] <= .98 or r[1] <= .8:
                raise ValueError("Unsupported axial orientation; patient-side mapping is unverified")
            pixels = d.pixel_array
            if tuple(pixels.shape) != (frame["rows"], frame["columns"]):
                raise ValueError("DICOM pixels disagree with declared matrix")
            frame["pixels"] = normalize_pixels(pixels)
            selected.append(frame)
            frames.append(frame)
        if len(series_uids) != 1:
            raise ValueError("One manifest series contains different DICOM SeriesInstanceUIDs")
        selected.sort(key=lambda f: (f["position"][0] if sequence.startswith("sagittal") else -f["position"][2], f["image_ref"]["image_id"]))
        if sequence.startswith("sagittal") and selected:
            first = selected[0]
            _, _, _, normal = _axes(first)
            projections = []
            for f in selected:
                if (f["rows"] != first["rows"] or f["columns"] != first["columns"]
                        or not np.allclose(f["orientation"], first["orientation"], atol=1e-6, rtol=0)
                        or not np.allclose(f["spacing"], first["spacing"], atol=1e-6, rtol=0)):
                    raise ValueError("Sagittal source geometry is inconsistent")
                delta = [a - b for a, b in zip(f["position"], first["position"])]
                projected = dot(delta, normal)
                if sum((v - projected * n) ** 2 for v, n in zip(delta, normal)) ** .5 > .01:
                    raise ValueError("Sagittal in-plane drift requires explicit resampling")
                projections.append(projected)
            if len({round(v, 4) for v in projections}) != len(projections):
                raise ValueError("Duplicate sagittal planes")
        by_sequence[sequence] = selected
    if len(study_uids) > 1:
        raise ValueError("Selected frames belong to different DICOM studies")
    return frames, by_sequence


def frame_metadata(frame):
    return {key: value for key, value in frame.items() if key not in {"pixels", "path"}}
