"""Read and verify local DICOM inputs without producing MRI findings."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import re
import sys

from .contracts import manifest_digest, validate_study


REQUIRED_SEQUENCES = ("sagittal_t2", "sagittal_t1", "axial_t2")


def _dependencies():
    try:
        import numpy
        import pydicom
    except ImportError:
        raise ValueError("DICOM preflight requires optional dependencies from requirements-data.txt") from None
    return numpy, pydicom


def _uid(dataset, name, required=True):
    value = str(getattr(dataset, name, "")).strip()
    valid = bool(value) and len(value) <= 64 and bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value))
    if not valid:
        if required:
            raise ValueError("DICOM is missing a valid " + name)
        return None
    return value


def _vector(value, length, positive=False):
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != length or not all(math.isfinite(item) for item in values):
        return None
    if positive and not all(item > 0 for item in values):
        return None
    return values


def _geometry_value(dataset, per_frame, shared, sequence, attribute):
    for group in (per_frame, shared):
        nested = getattr(group, sequence, None)
        if nested:
            value = getattr(nested[0], attribute, None)
            if value is not None:
                return value
    return getattr(dataset, attribute, None)


def _inspect_image(path, declared_frames, allow_missing_modality=False):
    """Decode pixels; return internal metadata that must not be serialized directly."""
    numpy, pydicom = _dependencies()
    try:
        dataset = pydicom.dcmread(path)
    except Exception:
        raise ValueError("Cannot read DICOM file") from None
    modality = str(getattr(dataset, "Modality", "") or "").strip()
    if modality != "MR" and not (allow_missing_modality and not modality):
        raise ValueError("DICOM Modality must be MR")
    if (getattr(dataset, "SamplesPerPixel", None) != 1
            or getattr(dataset, "PhotometricInterpretation", None) not in {"MONOCHROME1", "MONOCHROME2"}):
        raise ValueError("DICOM must contain grayscale MR pixels")
    try:
        frames = int(getattr(dataset, "NumberOfFrames", 1))
        rows, columns = int(dataset.Rows), int(dataset.Columns)
    except (TypeError, ValueError, AttributeError, OverflowError):
        raise ValueError("DICOM has invalid dimensions or NumberOfFrames") from None
    if frames < 1 or rows < 1 or columns < 1 or frames != declared_frames:
        raise ValueError("DICOM dimensions or NumberOfFrames disagree with manifest frame_count")
    try:
        pixels = dataset.pixel_array
    except Exception:
        raise ValueError("Cannot decode DICOM pixels; check transfer syntax and available decoders") from None
    expected_shape = (rows, columns) if frames == 1 else (frames, rows, columns)
    if tuple(pixels.shape) != expected_shape or pixels.size == 0:
        raise ValueError("Decoded grayscale pixel shape disagrees with DICOM dimensions")
    try:
        finite = bool(numpy.isfinite(pixels).all())
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("Decoded DICOM pixels must be finite numbers")
    study_uid = _uid(dataset, "StudyInstanceUID")
    series_uid = _uid(dataset, "SeriesInstanceUID")
    frame_uid = _uid(dataset, "FrameOfReferenceUID", required=False)
    shared_groups = getattr(dataset, "SharedFunctionalGroupsSequence", None)
    shared = shared_groups[0] if shared_groups else None
    per_frame_groups = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if per_frame_groups is not None and len(per_frame_groups) != frames:
        raise ValueError("DICOM per-frame geometry count disagrees with NumberOfFrames")
    orientations, positions, spacings = [], [], []
    for index in range(frames):
        per_frame = per_frame_groups[index] if per_frame_groups else None
        orientation = _vector(_geometry_value(
            dataset, per_frame, shared, "PlaneOrientationSequence", "ImageOrientationPatient"
        ), 6)
        if orientation is not None:
            row, column = orientation[:3], orientation[3:]
            if (abs(sum(value * value for value in row) - 1) > 1e-3
                    or abs(sum(value * value for value in column) - 1) > 1e-3
                    or abs(sum(a * b for a, b in zip(row, column))) > 1e-3):
                orientation = None
        orientations.append(orientation)
        positions.append(_vector(_geometry_value(
            dataset, per_frame, shared, "PlanePositionSequence", "ImagePositionPatient"
        ), 3))
        spacings.append(_vector(_geometry_value(
            dataset, per_frame, shared, "PixelMeasuresSequence", "PixelSpacing"
        ), 2, positive=True))
    return {
        "modality_verified": modality == "MR",
        "study_uid": study_uid,
        "series_uid": series_uid,
        "frame_uid": frame_uid,
        "frames": frames,
        "matrix": (rows, columns),
        "orientations": orientations,
        "positions": positions,
        "spacings": spacings,
    }


def _rounded(values):
    return tuple(round(value, 6) for value in values)


def _runtime():
    result = {"python": sys.version.split()[0]}
    for name in ("pydicom", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def verify_study(study, allow_missing_modality=False):
    """Validate manifest, hash files and decode every image; do not write or infer."""
    validate_study(study)
    if type(allow_missing_modality) is not bool:
        raise ValueError("Modality policy must be an explicit boolean")
    if not study["series"]:
        raise ValueError("Preflight requires at least one real image series")
    summaries, study_uids, declared_series_uids, frame_uids = [], set(), set(), set()
    files_verified, files_decoded, missing_modality = 0, 0, 0
    geometry_review, missing_frame_uid = False, False
    for series in study["series"]:
        series_uids, orientations, positions, spacings, matrices = set(), set(), set(), set(), set()
        frames, complete_geometry = 0, True
        for image in series["images"]:
            if image["sha256"] is None:
                raise ValueError("Preflight requires SHA-256 for every image")
            path = Path(image["path"])
            try:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError:
                raise ValueError("Image file is missing or unreadable: " + image["image_id"]) from None
            if digest.hexdigest() != image["sha256"]:
                raise ValueError("Image SHA-256 mismatch: " + image["image_id"])
            files_verified += 1
            inspected = _inspect_image(path, image["frame_count"], allow_missing_modality=allow_missing_modality)
            files_decoded += 1
            if not inspected["modality_verified"]:
                missing_modality += 1
            study_uids.add(inspected["study_uid"])
            series_uids.add(inspected["series_uid"])
            if len(study_uids) != 1:
                raise ValueError("Manifest contains images from different DICOM studies")
            if len(series_uids) != 1:
                raise ValueError("Manifest series contains different DICOM SeriesInstanceUIDs")
            if inspected["frame_uid"] is None:
                missing_frame_uid = True
            else:
                frame_uids.add(inspected["frame_uid"])
            frames += inspected["frames"]
            matrices.add(inspected["matrix"])
            for orientation, position, spacing in zip(
                    inspected["orientations"], inspected["positions"], inspected["spacings"]):
                complete_geometry = complete_geometry and all(value is not None for value in (orientation, position, spacing))
                if orientation is not None:
                    orientations.add(_rounded(orientation))
                if position is not None:
                    positions.add(_rounded(position))
                if spacing is not None:
                    spacings.add(_rounded(spacing))
        series_uid = next(iter(series_uids))
        if series_uid in declared_series_uids:
            raise ValueError("The same DICOM series is declared more than once")
        declared_series_uids.add(series_uid)
        consistent = (complete_geometry and len(orientations) == 1 and len(spacings) == 1
                      and len(matrices) == 1 and len(positions) == frames)
        geometry_review = geometry_review or not consistent
        summaries.append({
            "series_id": series["series_id"], "sequence": series["sequence"],
            "images": len(series["images"]), "frames": frames,
            "unique_orientations": len(orientations),
            "geometry_status": "consistent" if consistent else "needs_review",
        })
    sequences = {series["sequence"] for series in study["series"]}
    missing = [sequence for sequence in REQUIRED_SEQUENCES if sequence not in sequences]
    registration_review = missing_frame_uid or len(frame_uids) != 1 or geometry_review
    limitations = [
        "Sequence labels come from the manifest; acquisition contrast is not independently validated.",
        "Anatomical level coverage and suitability for any model have not been assessed.",
        "Matching FrameOfReferenceUIDs do not establish image registration; no transform is computed.",
        "No model inference, stenosis assessment or nerve-root impact assessment is performed.",
    ]
    if missing:
        limitations.append("Required Stage I sequences are missing from the manifest.")
    if missing_modality:
        limitations.append("DICOM Modality is missing; MR is declared by the source dataset, not verified from those headers.")
    if registration_review:
        limitations.append("Spatial geometry or frame-of-reference consistency requires review before fusion.")
    return {
        "kind": "input_preflight", "synthetic": study["synthetic"],
        "runtime": _runtime(),
        "study_id": study["study_id"], "study_digest": manifest_digest(study),
        "files_verified": files_verified, "files_decoded": files_decoded,
        "missing_modality_files": missing_modality,
        "modality_status": "dataset_declared" if missing_modality else "verified_mr",
        "allow_missing_modality": allow_missing_modality,
        "series": summaries,
        "registration_status": "needs_review" if registration_review else "consistent_frame_of_reference",
        "missing_sequences": missing,
        "localization_candidate": "sagittal_t2" in sequences,
        "full_stage_i_input": False, "inference_performed": False,
        "limitations": list(study["limitations"]) + limitations,
    }
