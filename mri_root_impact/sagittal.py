"""Build one checked native sagittal volume without resampling or inference."""

import hashlib
from io import BytesIO
import math
from pathlib import Path

from .contracts import validate_study
from .preflight import _dependencies, _uid, _vector


_MM_TOLERANCE = 1e-3


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _same(left, right, tolerance=1e-6):
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _orientation(value):
    orientation = _vector(value, 6)
    if orientation is None:
        raise ValueError("Sagittal input requires finite ImageOrientationPatient")
    column_direction, row_direction = orientation[:3], orientation[3:]
    if (abs(_dot(column_direction, column_direction) - 1) > 1e-3
            or abs(_dot(row_direction, row_direction) - 1) > 1e-3
            or abs(_dot(column_direction, row_direction)) > 1e-3):
        raise ValueError("ImageOrientationPatient must contain orthonormal directions")
    if column_direction[1] <= .99 or -row_direction[2] <= .99:
        raise ValueError("Only near-standard sagittal orientation is supported: columns posterior, rows inferior")
    normal = (
        column_direction[1] * row_direction[2] - column_direction[2] * row_direction[1],
        column_direction[2] * row_direction[0] - column_direction[0] * row_direction[2],
        column_direction[0] * row_direction[1] - column_direction[1] * row_direction[0],
    )
    norm = math.sqrt(_dot(normal, normal))
    return orientation, tuple(item / norm for item in normal)


def _read_image(image, allow_missing_modality, numpy, pydicom):
    if image["frame_count"] != 1:
        raise ValueError("Sagittal loader supports only single-frame DICOM images")
    if image["sha256"] is None:
        raise ValueError("Sagittal loader requires SHA-256 for every image")
    try:
        payload = Path(image["path"]).read_bytes()
    except OSError:
        raise ValueError("Sagittal image is missing or unreadable: " + image["image_id"]) from None
    if hashlib.sha256(payload).hexdigest() != image["sha256"]:
        raise ValueError("Image SHA-256 mismatch: " + image["image_id"])
    try:
        dataset = pydicom.dcmread(BytesIO(payload))
    except Exception:
        raise ValueError("Cannot read sagittal DICOM image") from None
    modality = str(getattr(dataset, "Modality", "") or "").strip()
    if modality != "MR" and not (allow_missing_modality and not modality):
        raise ValueError("DICOM Modality must be MR")
    if (getattr(dataset, "SamplesPerPixel", None) != 1
            or getattr(dataset, "PhotometricInterpretation", None) != "MONOCHROME2"):
        raise ValueError("Sagittal input requires MONOCHROME2 grayscale pixels; polarity transforms are unsupported")
    try:
        frame_count = int(getattr(dataset, "NumberOfFrames", 1))
        rows, columns = int(dataset.Rows), int(dataset.Columns)
    except (ValueError, TypeError, OverflowError, AttributeError):
        raise ValueError("Invalid sagittal DICOM dimensions or NumberOfFrames") from None
    if frame_count != 1 or rows < 1 or columns < 1:
        raise ValueError("Sagittal loader supports only positive single-frame dimensions")
    # Enhanced multi-frame objects are not interpreted by this first loader.
    if (getattr(dataset, "SharedFunctionalGroupsSequence", None) is not None
            or getattr(dataset, "PerFrameFunctionalGroupsSequence", None) is not None):
        raise ValueError("Enhanced DICOM geometry is not supported by the sagittal loader")
    orientation, normal = _orientation(getattr(dataset, "ImageOrientationPatient", None))
    position = _vector(getattr(dataset, "ImagePositionPatient", None), 3)
    spacing = _vector(getattr(dataset, "PixelSpacing", None), 2, positive=True)
    thickness = _vector([getattr(dataset, "SliceThickness", None)], 1, positive=True)
    if position is None or spacing is None or thickness is None:
        raise ValueError("Sagittal input requires finite position and positive pixel spacing and slice thickness")
    if not _same(spacing[:1], spacing[1:]):
        raise ValueError("Anisotropic in-plane spacing requires explicit resampling; averaging is not supported")
    study_uid = _uid(dataset, "StudyInstanceUID")
    series_uid = _uid(dataset, "SeriesInstanceUID")
    try:
        pixels = dataset.pixel_array
    except Exception:
        raise ValueError("Cannot decode sagittal DICOM pixels") from None
    if tuple(pixels.shape) != (rows, columns) or pixels.size == 0:
        raise ValueError("Decoded sagittal shape disagrees with DICOM dimensions")
    try:
        finite = bool(numpy.isfinite(pixels).all())
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("Decoded sagittal pixels must be finite numbers")
    return {
        "image": image, "pixels": pixels, "matrix": (rows, columns),
        "study_uid": study_uid, "series_uid": series_uid,
        "orientation": orientation, "normal": normal,
        "position": position, "spacing": spacing, "thickness": thickness[0],
    }


def load_sagittal_series(study, series_id=None, allow_missing_modality=False):
    """Return native H×W×S pixels and per-slice DICOM LPS geometry.

    Select exactly one declared sagittal_t2 series, optionally by identifier.
    Pixel values, axes and raw files are unchanged. Ascending slice order follows
    the normal computed from ImageOrientationPatient, rather than InstanceNumber.
    No intensity transform, registration, anatomical assessment or inference runs.
    """
    validate_study(study)
    if type(allow_missing_modality) is not bool:
        raise ValueError("Modality policy must be an explicit boolean")
    if series_id is not None and (not isinstance(series_id, str) or not series_id):
        raise ValueError("series_id must be a nonempty string or None")
    candidates = [series for series in study["series"] if series["sequence"] == "sagittal_t2"
                  and (series_id is None or series["series_id"] == series_id)]
    if len(candidates) != 1:
        raise ValueError("Select exactly one sagittal_t2 series")
    selected = candidates[0]
    if len(selected["images"]) < 2:
        raise ValueError("At least two sagittal slices are required to measure center spacing")
    numpy, pydicom = _dependencies()
    slices = [_read_image(image, allow_missing_modality, numpy, pydicom)
              for image in selected["images"]]
    first = slices[0]
    for current in slices[1:]:
        if current["study_uid"] != first["study_uid"]:
            raise ValueError("Sagittal images belong to different DICOM studies")
        if current["series_uid"] != first["series_uid"]:
            raise ValueError("Sagittal images belong to different DICOM series")
        if (current["matrix"] != first["matrix"]
                or not _same(current["spacing"], first["spacing"])
                or not _same(current["orientation"], first["orientation"])):
            raise ValueError("Sagittal matrix, pixel spacing and orientation must be consistent")
    normal = first["normal"]
    slices.sort(key=lambda item: _dot(item["position"], normal))
    positions = [_dot(item["position"], normal) for item in slices]
    steps = [right - left for left, right in zip(positions, positions[1:])]
    if any(step <= _MM_TOLERANCE for step in steps):
        raise ValueError("Sagittal slice positions are duplicated or too close")
    center_spacing = sum(steps) / len(steps)
    if any(abs(step - center_spacing) > _MM_TOLERANCE for step in steps):
        raise ValueError("Irregular sagittal slice center spacing is unsupported")
    # A scalar slice spacing cannot describe a sheared stack without resampling.
    origin, origin_projection = slices[0]["position"], positions[0]
    for current, projection in zip(slices[1:], positions[1:]):
        residual = [value - base - (projection - origin_projection) * direction
                    for value, base, direction in zip(current["position"], origin, normal)]
        if math.sqrt(_dot(residual, residual)) > _MM_TOLERANCE:
            raise ValueError("In-plane slice-origin drift requires explicit resampling")
    volume = numpy.stack([item["pixels"] for item in slices], axis=2)
    geometry = {
        "series_id": selected["series_id"],
        "volume_shape": list(volume.shape),
        "pixel_spacing_mm": first["spacing"][0],
        "slice_center_spacing_mm": center_spacing,
        "coordinate_system": "DICOM_LPS_mm",
        "slices": [{
            "slice_index": index,
            "image_ref": {"series_id": selected["series_id"], "image_id": item["image"]["image_id"], "frame_index": 0},
            "image_position_patient_mm": list(item["position"]),
            "image_orientation_patient": list(item["orientation"]),
            "pixel_spacing_mm": list(item["spacing"]),
            "slice_thickness_mm": item["thickness"],
        } for index, item in enumerate(slices)],
    }
    return volume, geometry
