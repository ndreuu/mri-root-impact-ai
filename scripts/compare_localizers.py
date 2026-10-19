#!/usr/bin/env python3
"""Compare frozen RSNA disc points with SpineNet on exact source frames."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_root_impact.contracts import load_document, manifest_digest, validate_bundle
from mri_root_impact.control_study import write_json_external
from mri_root_impact.rsna_assets import file_sha256
from mri_root_impact.rsna_geometry import LEVELS, pixel_to_lps
from mri_root_impact.rsna_runner import validate_worker_outputs as validate_rsna
from mri_root_impact.spinenet_runner import external_run_directory, validate_worker_outputs as validate_spinenet

EPSILON_MM = 1e-6
UNCERTAINTY = "Native model agreement does not independently confirm anatomical numbering or cross-series registration."


def ref_key(ref):
    return ref["series_id"], ref["image_id"], ref["frame_index"]


def _vector(value, count):
    try:
        valid = (isinstance(value, list) and len(value) == count
                 and all(type(v) in (int, float) and math.isfinite(v) for v in value))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Invalid finite source geometry")
    return value


def _same_geometry(a, b):
    if (a["rows"], a["columns"]) != (b["rows"], b["columns"]):
        raise ValueError("Source matrix differs between localizers")
    for name, count in (("position", 3), ("orientation", 6), ("spacing", 2)):
        x, y = _vector(a[name], count), _vector(b[name], count)
        if any(abs(i - j) > EPSILON_MM for i, j in zip(x, y)):
            raise ValueError("Source geometry differs between localizers: " + name)
    pixel_to_lps(a, 0, 0)
    pixel_to_lps(b, 0, 0)


def polygon_centroid(polygon):
    """Shoelace area centroid; reject collapsed, concave or crossing quads."""
    if not isinstance(polygon, list) or len(polygon) != 4:
        raise ValueError("Vertebral polygon requires four points")
    points = [_vector(p, 2) for p in polygon]
    crosses = []
    for a, b, c in zip(points, points[1:] + points[:1], points[2:] + points[:2]):
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(v > 1e-9 for v in crosses) or all(v < -1e-9 for v in crosses)):
        raise ValueError("Degenerate or non-convex vertebral polygon")
    terms = [a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])]
    area2 = sum(terms)
    if abs(area2) <= 1e-9:
        raise ValueError("Degenerate vertebral polygon")
    return [sum((a[i] + b[i]) * t for a, b, t in zip(points, points[1:] + points[:1], terms)) / (3 * area2)
            for i in range(2)]


def _axial_links(level, spatial):
    rois = {r["roi_id"]: r for r in spatial.get("rois", [])}
    frames = {ref_key(f["image_ref"]): f for f in spatial.get("frames", [])}
    links = []
    for entry in spatial.get("inputs", []):
        if entry["level"] != level:
            continue
        ids = entry["roi_ids"]["axial_t2"]
        refs = {ref_key(rois[rid]["image_ref"]): rois[rid]["image_ref"] for rid in ids}
        links.append({"side": entry["side"], "roi_ids": list(ids),
                      "image_refs": [refs[k].copy() for k in sorted(refs)],
                      "orientations": [list(v) for v in sorted({tuple(frames[k]["orientation"]) for k in refs})],
                      "registration_status": "unreviewed"})
    return sorted(links, key=lambda x: x["side"])


def compare_levels(study, spine_spatial, rsna_spatial):
    """Arithmetic comparison only. File/native validation belongs to run()."""
    digest = manifest_digest(study)
    for spatial in (spine_spatial, rsna_spatial):
        if spatial.get("study_id") != study["study_id"] or spatial.get("study_digest") != digest:
            raise ValueError("Localizer artifact belongs to a different manifest")
    geometry = spine_spatial["geometry"]
    detections = spine_spatial["detections"]
    result = {"kind": "localizer_crosscheck", "schema_version": "1.0", "study_id": study["study_id"],
              "study_digest": digest, "levels": [], "method": "same-frame area-centroid bracketing along patient SI (-LPS Z)",
              "numerical_epsilon_mm": EPSILON_MM, "numbering_confirmed": False,
              "registration_confirmed": False, "inference_performed": False, "root_impacts": None,
              "limitations": [UNCERTAINTY, "No vertebral midpoint or polygon-containment threshold is used to validate posterior disc points.",
                              "No slice-index match, nearest-slice substitution, inter-series registration, renumbering or new disc/herniation inference."]}
    for level in LEVELS:
        labels = level.split("-")
        expected = {label: [d["detection_id"] for d in detections if d["native_label"] == label] for label in labels}
        entry = {"level": level, "comparison": "indeterminate", "status": "indeterminate",
                 "reason": UNCERTAINTY, "sagittal_reference": None, "expected_adjacent_labels": labels,
                 "expected_detection_ids": expected, "bracketing_pair": None, "source_polygons": [],
                 "axial_roi_links": _axial_links(level, rsna_spatial)}
        result["levels"].append(entry)
        try:
            points = [p for p in rsna_spatial["keypoints"] if p["sequence"] == "sagittal_t2" and p["label"] == level]
            if len(points) != 1:
                raise ValueError("No unique RSNA SagT2 point for this native level")
            point = points[0]
            reference = point["image_ref"]
            entry["sagittal_reference"] = reference.copy()
            slices = [s for s in geometry["slices"] if ref_key(s["image_ref"]) == ref_key(reference)]
            frames = [f for f in rsna_spatial["frames"] if ref_key(f["image_ref"]) == ref_key(reference)]
            if len(slices) != 1 or len(frames) != 1 or frames[0]["sequence"] != "sagittal_t2":
                raise ValueError("No unique exact source frame shared by both localizers")
            saved, frame = slices[0], frames[0]
            source = {"position": saved["image_position_patient_mm"], "orientation": saved["image_orientation_patient"],
                      "spacing": saved["pixel_spacing_mm"], "rows": geometry["volume_shape"][0], "columns": geometry["volume_shape"][1]}
            _same_geometry(source, frame)
            xyz = _vector(point["lps_mm"], 3)
            x, y = point["column"], point["row"]
            _vector([x, y], 2)
            if not (0 <= x < frame["columns"] and 0 <= y < frame["rows"]):
                raise ValueError("RSNA point lies outside the source frame")
            if any(abs(a - b) > 1e-5 for a, b in zip(xyz, pixel_to_lps(frame, x, y))):
                raise ValueError("RSNA pixel/LPS coordinates disagree")
            if any(len(ids) != 1 for ids in expected.values()):
                raise ValueError("Expected adjacent vertebra label is missing or duplicated")
            bodies = []
            for detection in detections:
                polygons = [p for p in detection["polygons"] if ref_key(p["image_ref"]) == ref_key(reference)]
                if not polygons:
                    continue
                if len(polygons) != 1:
                    raise ValueError("Duplicate exact-frame vertebral polygon")
                polygon = polygons[0]
                center = polygon_centroid(polygon["polygon_xy"])
                if not isinstance(polygon["polygon_lps_mm"], list) or len(polygon["polygon_lps_mm"]) != 4:
                    raise ValueError("Vertebral polygon requires four physical vertices")
                for xy, lps in zip(polygon["polygon_xy"], polygon["polygon_lps_mm"]):
                    if any(abs(a - b) > 1e-5 for a, b in zip(_vector(lps, 3), pixel_to_lps(frame, *xy))):
                        raise ValueError("Vertebral polygon pixel/LPS coordinates disagree")
                if (polygon["in_bounds"] is not True or any(not (0 <= p[0] <= frame["columns"] - 1
                        and 0 <= p[1] <= frame["rows"] - 1) for p in polygon["polygon_xy"])):
                    raise ValueError("Vertebral polygon is outside the source frame")
                center_lps = pixel_to_lps(frame, *center)
                body = {"detection_id": detection["detection_id"], "native_label": detection["native_label"],
                        "image_ref": reference.copy(), "polygon_xy": [xy[:] for xy in polygon["polygon_xy"]],
                        "center_xy": center, "center_lps_mm": center_lps}
                bodies.append(body)
            entry["source_polygons"] = bodies
            if any(sum(b["native_label"] == label for b in bodies) != 1 for label in labels):
                raise ValueError("Expected adjacent vertebra has no unique exact-frame polygon")
            # Sort using physical SI coordinates, independently of native labels.
            bodies.sort(key=lambda b: -b["center_lps_mm"][2])
            scalars = [-b["center_lps_mm"][2] for b in bodies]
            value = -xyz[2]
            if any(b - a <= EPSILON_MM for a, b in zip(scalars, scalars[1:])):
                raise ValueError("Vertebral centroid order is ambiguous")
            if any(abs(value - z) <= EPSILON_MM for z in scalars):
                raise ValueError("RSNA point lies on a vertebral centroid boundary")
            brackets = [(i, a, b) for i, (a, b) in enumerate(zip(scalars, scalars[1:])) if a < value < b]
            if len(brackets) != 1:
                raise ValueError("RSNA point is outside the visible centroid brackets")
            i, a, b = brackets[0]
            upper, lower = bodies[i:i + 2]
            pair = upper["native_label"] + "-" + lower["native_label"]
            if pair not in LEVELS:
                raise ValueError("Observed neighboring labels do not form a supported canonical adjacent pair")
            if any(sum(d["native_label"] == name for d in detections) != 1 for name in (upper["native_label"], lower["native_label"])):
                raise ValueError("Observed neighbor label is duplicated")
            axis = [v - u for u, v in zip(upper["center_lps_mm"], lower["center_lps_mm"])]
            offset = [v - u for u, v in zip(upper["center_lps_mm"], xyz)]
            fraction = sum(u * v for u, v in zip(axis, offset)) / sum(v * v for v in axis)
            residual = [q - fraction * v for q, v in zip(offset, axis)]
            entry["bracketing_pair"] = {"upper_detection_id": upper["detection_id"], "lower_detection_id": lower["detection_id"],
                "upper_native_label": upper["native_label"], "lower_native_label": lower["native_label"],
                "centroid_fraction": (value - a) / (b - a), "upper_margin_mm": value - a, "lower_margin_mm": b - value,
                "centroid_axis_fraction_diagnostic": fraction, "perpendicular_distance_mm_diagnostic": math.sqrt(sum(v * v for v in residual))}
            entry["comparison"] = "native_agreement" if pair == level else "native_disagreement"
            entry["reason"] = ("RSNA native level matches the independently ordered SpineNet neighbor pair. " if pair == level
                               else "RSNA native level differs from the independently ordered SpineNet neighbor pair. ") + UNCERTAINTY
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            entry["reason"] = str(exc) + ". " + UNCERTAINTY
    return result


def _read_runs(study, spine_run, rsna_run):
    spine_execution = validate_spinenet(study, spine_run)
    raw_execution = load_document(rsna_run / "execution.json")
    policy = raw_execution["parameters"]["allow_unverified_registration"]
    if type(policy) is not bool:
        raise ValueError("RSNA source registration policy is not an explicit boolean")
    rsna_execution = validate_rsna(study, rsna_run, allow_unverified_registration=policy)
    sources = {}
    for name, directory, execution in (("spinenet", spine_run, spine_execution), ("rsna", rsna_run, rsna_execution)):
        parent = load_document(directory / "run.json")
        if (parent.get("status") != "completed" or parent.get("study_id") != study["study_id"]
                or type(parent.get("worker_exit_code")) is not int or parent["worker_exit_code"] != 0
                or parent.get("timed_out") is not False or parent.get("inference_performed") is not True):
            raise ValueError("Input requires a completed checked model run")
        names = set(execution["artifacts_sha256"]) | {"execution.json", "run.json"}
        sources[name] = {"run_directory": str(directory), "files_sha256": {n: file_sha256(directory / n) for n in sorted(names)}}
    results = [load_document(spine_run / "localization.json")] + [load_document(rsna_run / n) for n in ("localization.json", "foraminal.json", "subarticular.json")]
    validate_bundle(study, results)
    return load_document(spine_run / "spatial.json"), load_document(rsna_run / "spatial.json"), sources


def render_overlays(report, spine_spatial, rsna_spatial, frames, output):
    """Render source images and existing model geometry; never derive new ROIs."""
    from PIL import Image, ImageDraw, ImageOps
    lookup = {ref_key(f["image_ref"]): f for f in frames}
    panels, entries = [], []
    references = {ref_key(e["sagittal_reference"]): e["sagittal_reference"] for e in report["levels"] if e["sagittal_reference"] is not None}
    panels_to_draw = [(ref, None) for key, ref in sorted(references.items())]
    panels_to_draw += [(p["image_ref"], p["native_level"]) for p in rsna_spatial["keypoints"] if p["sequence"] == "axial_t2"]
    for index, (ref, axial_level) in enumerate(panels_to_draw):
        frame = lookup[ref_key(ref)]
        panel = Image.new("RGB", (frame["columns"], frame["rows"] + 48), "#101820")
        panel.paste(Image.fromarray(frame["pixels"]).convert("RGB"), (0, 48))
        draw = ImageDraw.Draw(panel)
        label = axial_level or "SagT2 exact common source frame"
        draw.text((3, 2), label, fill="white")
        draw.text((3, 15), "NATIVE LABELS / UNREVIEWED", fill="#ffb34d")
        draw.text((3, 28), ref["image_id"], fill="white")
        if axial_level is None:
            used = set()
            for entry in report["levels"]:
                if entry["sagittal_reference"] != ref:
                    continue
                for body in entry["source_polygons"]:
                    if body["detection_id"] in used:
                        continue
                    used.add(body["detection_id"])
                    xy = [(x, y + 48) for x, y in body["polygon_xy"]]
                    draw.line(xy + xy[:1], fill="#5ee48c", width=1)
                    x, y = body["center_xy"]
                    draw.text((x - 15, y + 45), body["native_label"], fill="#5ee48c")
                points = [p for p in rsna_spatial["keypoints"] if p["sequence"] == "sagittal_t2" and p["label"] == entry["level"] and p["image_ref"] == ref]
                if len(points) == 1:
                    x, y = points[0]["column"], points[0]["row"] + 48
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#ff6b6b")
                    draw.text((x + 4, y - 5), entry["level"], fill="#ff6b6b")
        else:
            entry = next(e for e in report["levels"] if e["level"] == axial_level)
            ids = {rid for link in entry["axial_roi_links"] for rid in link["roi_ids"]}
            for roi in rsna_spatial["rois"]:
                if roi["roi_id"] in ids and roi["image_ref"] == ref:
                    left, top, right, bottom = roi["source_crop_box"]
                    color = "#ffcc66" if roi["side"] == "left" else "#80bfff"
                    draw.rectangle((left, top + 48, right - 1, bottom + 47), outline=color, width=1)
                    draw.text((left, bottom + 48), roi["side"], fill=color)
        name = "overlays/frame-{:03d}.png".format(index)
        (output / "overlays").mkdir(exist_ok=True)
        panel.save(output / name)
        entries.append({"file": name, "image_ref": ref, "native_level": axial_level, "header_pixels": 48,
                        "image_scale": 1, "sha256": file_sha256(output / name)})
        panels.append(panel)
    rows = max(1, math.ceil(len(panels) / 3))
    montage = Image.new("RGB", (1152, rows * 456), "#101820")
    for i, panel in enumerate(panels):
        tile = ImageOps.contain(panel, (384, 456), Image.Resampling.BILINEAR)
        montage.paste(tile, ((i % 3) * 384 + (384 - tile.width) // 2,
                             (i // 3) * 456 + (456 - tile.height) // 2))
    montage.save(output / "overlay.png")
    return entries


def run(study_path, spine_run, rsna_run, output, allow_missing_modality=False):
    from mri_root_impact.rsna_dicom import load_frames, frame_metadata
    started = time.monotonic()
    if type(allow_missing_modality) is not bool:
        raise ValueError("Missing Modality policy must be an explicit boolean")
    study_path, spine_run, rsna_run = (Path(p).resolve() for p in (study_path, spine_run, rsna_run))
    study = load_document(study_path)
    validate_bundle(study, [])
    output = external_run_directory(study_path, study, output, spine_run)
    if rsna_run == output or rsna_run in output.parents:
        raise ValueError("Comparison output must not modify the RSNA input run")
    spine, rsna, sources = _read_runs(study, spine_run, rsna_run)
    frames, _ = load_frames(study, allow_missing_modality=allow_missing_modality)
    saved = {ref_key(f["image_ref"]): f for f in rsna["frames"]}
    if {ref_key(f["image_ref"]): frame_metadata(f) for f in frames} != saved:
        raise ValueError("Recorded source headers differ from original DICOM files")
    for source in spine["geometry"]["slices"]:
        _same_geometry({"position": source["image_position_patient_mm"], "orientation": source["image_orientation_patient"],
                        "spacing": source["pixel_spacing_mm"], "rows": spine["geometry"]["volume_shape"][0],
                        "columns": spine["geometry"]["volume_shape"][1]}, saved[ref_key(source["image_ref"])])
    report = compare_levels(study, spine, rsna)
    report["sources"] = sources
    report["source_files_verified"] = len(frames)
    report["source_bytes_unchanged"] = True
    report["script_sha256"] = file_sha256(__file__)
    output.mkdir(parents=True)
    try:
        overlays = render_overlays(report, spine, rsna, frames, output)
        for series in study["series"]:
            for image in series["images"]:
                if file_sha256(image["path"]) != image["sha256"]:
                    raise ValueError("Original DICOM bytes changed")
        write_json_external(report, output / "crosscheck.json")
        evidence = {"kind": "localizer_crosscheck_run", "status": "completed", "study_id": study["study_id"],
                    "inference_performed": False, "root_impacts": None, "duration_seconds": time.monotonic() - started,
                    "parameters": {"allow_missing_modality": allow_missing_modality}, "overlays": overlays,
                    "artifacts_sha256": {name: file_sha256(output / name) for name in ("crosscheck.json", "overlay.png")},
                    "source_bytes_unchanged": True, "numbering_confirmed": False, "registration_confirmed": False}
        write_json_external(evidence, output / "run.json")
    except Exception as exc:
        write_json_external({"kind": "localizer_crosscheck_run", "status": "error", "error_type": type(exc).__name__,
                             "error": str(exc), "inference_performed": False, "root_impacts": None}, output / "run.json")
        raise
    return {"ok": True, "operation": "compare-localizers", "output": str(output), "inference_performed": False,
            "comparisons": {name: sum(e["comparison"] == name for e in report["levels"]) for name in ("native_agreement", "native_disagreement", "indeterminate")},
            "numbering_confirmed": False, "registration_confirmed": False, "root_impacts": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "spinenet-run", "rsna-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-missing-modality", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.study, args.spinenet_run, args.rsna_run, args.output, args.allow_missing_modality), indent=2))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
