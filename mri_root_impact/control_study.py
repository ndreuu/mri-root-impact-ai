"""Prepare an external RSNA control study from a pinned download inventory."""

import hashlib
import json
from pathlib import Path
import tempfile

from .contracts import load_document, validate_study


DESCRIPTIONS = {
    "Sagittal T2/STIR": "sagittal_t2",
    "Sagittal T1": "sagittal_t1",
    "Axial T2": "axial_t2",
}


def _file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Inventory requires a relative file path")
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Inventory file escapes the dataset directory")
    return path


def _frame_count(path):
    try:
        import pydicom
    except ImportError:
        raise ValueError("Study preparation requires optional dependencies from requirements-data.txt") from None
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["NumberOfFrames"])
        count = int(getattr(dataset, "NumberOfFrames", 1))
    except Exception:
        raise ValueError("Cannot read DICOM frame count") from None
    if count < 1:
        raise ValueError("Invalid DICOM frame count")
    return count


def prepare_rsna_study(data_root):
    """Verify the download inventory and build the existing C001 manifest."""
    root = Path(data_root).resolve()
    inventory = load_document(root / "provenance.json")
    if not isinstance(inventory, dict):
        raise ValueError("RSNA inventory must be an object")
    study_id = inventory.get("study_id")
    if not isinstance(study_id, str) or not study_id.isdigit():
        raise ValueError("RSNA inventory requires a numeric study_id")
    if not inventory.get("repo_id") or not inventory.get("revision"):
        raise ValueError("RSNA inventory requires a repository and pinned revision")
    series = inventory.get("series")
    files = inventory.get("files")
    if not isinstance(series, list) or not series or not isinstance(files, list) or not files:
        raise ValueError("RSNA inventory requires series and files")
    outputs, seen_paths, descriptions = {}, set(), set()
    for entry in series:
        if not isinstance(entry, dict):
            raise ValueError("Invalid inventory series record")
        native_id = entry.get("series_id")
        description = entry.get("series_description")
        if not isinstance(native_id, str) or not native_id.isdigit() or native_id in outputs:
            raise ValueError("Invalid or duplicate RSNA series identifier")
        if not isinstance(description, str) or description not in DESCRIPTIONS:
            raise ValueError("Unexpected RSNA control series description")
        descriptions.add(description)
        outputs[native_id] = {"series_id": native_id, "sequence": DESCRIPTIONS[description], "images": []}
    if descriptions != set(DESCRIPTIONS):
        raise ValueError("RSNA control requires all three declared series types")
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid inventory file record")
        path = _file(root, entry.get("relative_path"))
        if path in seen_paths:
            raise ValueError("Duplicate inventory file path")
        seen_paths.add(path)
        relative = path.relative_to(root)
        if (len(relative.parts) != 4 or relative.parts[:2] != ("train_images", study_id)
                or relative.parts[2] not in outputs or path.suffix.lower() != ".dcm"):
            raise ValueError("Inventory image does not belong to selected study/series")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(digest, str) or type(size) is not int or size < 1:
            raise ValueError("Inventory requires image size and SHA-256")
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Image differs from pinned download inventory")
        native_id = relative.parts[2]
        outputs[native_id]["images"].append({
            "image_id": native_id + "-" + path.stem,
            "path": str(path), "frame_count": _frame_count(path), "sha256": digest,
        })
    study = {
        "schema_version": "1.0", "kind": "study_manifest",
        "study_id": "rsna-" + study_id, "synthetic": False,
        "series": sorted(outputs.values(), key=lambda item: item["series_id"]),
        "limitations": [
            "Sagittal T2/STIR is represented as a T2-like input; conventional T2 contrast is not established by the CSV label.",
            "Anatomical level coverage is unconfirmed; coordinate labels are not a geometric coverage validation.",
            "Spatial correspondence between series requires review before fusion.",
            "RSNA stenosis labels are not herniation morphology or clinical nerve-root ground truth.",
            "Source is a pinned third-party Hugging Face mirror; upstream dataset conditions remain recorded in provenance.json.",
        ],
    }
    validate_study(study)
    return study


def write_json_external(value, filename):
    """Keep real input manifests and reports outside all Git working trees."""
    filename = Path(filename).resolve()
    for parent in (filename.parent, *filename.parents):
        if (parent / ".git").exists():
            raise ValueError("Real study metadata must be written outside Git repositories")
    if filename.exists():
        try:
            previous = load_document(filename)
        except (OSError, ValueError):
            raise ValueError("Refusing to overwrite an existing input or unrecognized file") from None
        if (not isinstance(previous, dict) or previous.get("kind") != value.get("kind")
                or previous.get("study_id") != value.get("study_id")):
            raise ValueError("Refusing to overwrite metadata of another kind or study")
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(filename.parent), delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(filename)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return filename
