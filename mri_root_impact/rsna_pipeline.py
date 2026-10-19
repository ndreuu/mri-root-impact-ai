"""Automatic single-fold RSNA ROI/inference with explicit source-frame geometry."""

import math

from .contracts import manifest_digest
from .rsna_dicom import SEQUENCES, frame_metadata
from .rsna_geometry import (LEVELS, box_corners_lps, crop_box, dot, level_segments,
                            lps_to_pixel, pixel_to_lps, segment_plane_intersection)


CLASS_ORDER = ["normal_mild", "moderate", "severe"]
TEMPERATURE = 0.9964720472122109


def unique_refs(refs):
    values = {(r["series_id"], r["image_id"], r["frame_index"]): r for r in refs}
    return [values[key] for key in sorted(values)]


def load_networks(source, weights, assets):
    import torch
    import timm
    from omegaconf import OmegaConf
    from .contracts import load_document
    from .rsna_assets import ROOT
    from .rsna_probe import IGNORABLE_LOSS_KEYS, extract_model_definitions
    tree = extract_model_definitions((source / "src/predict.py").read_text())
    scope = {"__name__": "_verified_rsna_models", "torch": torch,
             "nn": torch.nn, "F": torch.nn.functional, "timm": timm}
    exec(compile(tree, str(source / "src/predict.py") + "::<classes_only>", "exec"), scope)
    inventory = load_document(ROOT / "harness/upstreams/rsna-assets.json")
    networks, loading = {}, {}
    for role, expected in inventory["weights"].items():
        cfg = OmegaConf.to_container(OmegaConf.load(source / expected["config"]), resolve=False)["model"]
        cfg.pop("name", None)
        cfg["timm_model"]["pretrained"] = False
        with torch.no_grad():
            model = scope[expected["model_class"]](**cfg)
        state = torch.load(weights / expected["filename"], weights_only=True, map_location="cpu")
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected medical tensor state_dict")
        removed = sorted(set(state) & IGNORABLE_LOSS_KEYS)
        state = {k: v for k, v in state.items() if k not in IGNORABLE_LOSS_KEYS}
        model.load_state_dict(state, strict=True)
        model.eval()
        networks[role] = model
        loading[role] = {"strict_load": True, "removed_training_only_keys": removed,
                         "weights_sha256": assets["weights"][role]["sha256"]}
    return networks, loading


def locate(frames, model, labels):
    import albumentations as A
    import cv2
    import numpy as np
    import torch
    if not frames:
        return None
    reference = frames[len(frames) // 2]
    image = cv2.resize(reference["pixels"], (512, 512), interpolation=cv2.INTER_LINEAR)
    transform = A.Compose([A.Resize(380, 380), A.Normalize(mean=.5, std=.5)])
    image = transform(image=image[:, :, None])["image"]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)[None])).float()
    with torch.inference_mode():
        heatmaps = model(tensor)["logits"].sigmoid().cpu().numpy()[0]
    if heatmaps.shape[0] != len(labels) or not np.isfinite(heatmaps).all():
        raise ValueError("Invalid keypoint heatmaps")
    points = {}
    height, width = heatmaps.shape[-2:]
    for label, heatmap in zip(labels, heatmaps):
        if float(np.max(heatmap) - np.min(heatmap)) <= 1e-7:
            points[label] = None
            continue
        ys, xs = np.where(heatmap >= np.percentile(heatmap, 99.8))
        nx = float((xs.min() + xs.max() + 1) / 2 / width)
        ny = float((ys.min() + ys.max() + 1) / 2 / height)
        x, y = nx * reference["columns"], ny * reference["rows"]
        points[label] = {"label": label, "image_ref": reference["image_ref"],
                         "column": x, "row": y, "norm_xy": [nx, ny],
                         "lps_mm": pixel_to_lps(reference, x, y),
                         "heatmap_peak": float(heatmap.max()),
                         "heatmap_shape": [height, width], "percentile_mask": 99.8}
    return {"reference": reference, "points": points}


def _group_axial(frames, segments, points):
    groups, ambiguous, outside, outside_fov = {level: {} for level in LEVELS}, [], [], []
    for frame in frames:
        matches = [level for level, (upper, lower) in segments.items()
                   if segment_plane_intersection(frame, upper, lower)]
        if len(matches) != 1:
            (ambiguous if matches else outside).append(frame["image_ref"])
            continue
        level = matches[0]
        a, b = segments[level]
        da, db = lps_to_pixel(frame, a)[2], lps_to_pixel(frame, b)[2]
        t = da / (da - db) if abs(da - db) > 1e-6 else .5
        intersection = [a[i] + t * (b[i] - a[i]) for i in range(3)]
        x, y, _ = lps_to_pixel(frame, intersection)
        if not (0 <= x < frame["columns"] and 0 <= y < frame["rows"]):
            outside_fov.append({"image_ref": frame["image_ref"], "level": level})
            continue
        key = tuple(round(v, 6) for v in frame["orientation"])
        groups[level].setdefault(key, []).append(frame)
    chosen, details = {}, {}
    for level in LEVELS:
        def rank(item):
            key, group = item
            distances = [abs(lps_to_pixel(f, points[level])[2]) for f in group]
            return (-len(group), sum(distances) / len(distances), key)
        ordered = sorted(groups[level].items(), key=rank)
        chosen[level] = ordered[0][1] if ordered else []
        details[level] = {"selection": "largest constant-IOP group, nearest mean plane distance, lexicographic IOP tie-break",
                          "groups": [{"orientation": list(key), "frames": [f["image_ref"] for f in group],
                                      "selected": i == 0} for i, (key, group) in enumerate(ordered)]}
    return chosen, {"per_level": details, "ambiguous_boundary_frames": ambiguous,
                    "outside_level_segments": outside, "outside_field_of_view_frames": outside_fov}


def _crop_records(frames, level, sequence, norm_xy=None, world_center=None):
    records, rejected = [], []
    for frame in frames:
        try:
            center = norm_xy
            if world_center is not None:
                x, y, _ = lps_to_pixel(frame, world_center)
                center = [x / frame["columns"], y / frame["rows"]]
            sagittal = sequence.startswith("sagittal")
            box = crop_box(frame, center, [35, 50] if sagittal else [45, 75], [-5, 0] if sagittal else [5, 0])
            left, top, right, bottom = box
            image = frame["pixels"][top:bottom, left:right]
            if not image.size or image.max() == image.min():
                raise ValueError("Empty/constant crop")
            records.append({"frame": frame, "box": box, "image": image, "level": level, "sequence": sequence})
        except ValueError as exc:
            rejected.append({"image_ref": frame["image_ref"], "level": level, "sequence": sequence, "reason": str(exc)})
    return records, rejected


def _input(records, sequence, side, level, spatial_rois):
    import albumentations as A
    import cv2
    import numpy as np
    if not records:
        raise ValueError("No valid " + sequence + " crops for " + level)
    count = 10 if sequence == "axial_t2" else 15
    records = sorted(records, key=lambda r: (r["frame"]["position"][0] if sequence != "axial_t2"
                                             else r["frame"]["position"][2]), reverse=side == "left" if sequence != "axial_t2" else True)
    if sequence != "axial_t2":
        records = records[:math.ceil(len(records) / 2)]
    indices = np.round(np.linspace(0, len(records) - 1, count)).astype(int).tolist()
    arrays, refs, roi_ids = [], [], []
    normalize = A.Normalize(mean=.5, std=.5)
    for index in indices:
        record = records[index]
        image, box = record["image"], list(record["box"])
        if sequence == "axial_t2":
            width = image.shape[1]
            if side == "right":
                side_width = int(width * .6)
                if side_width < 1:
                    raise ValueError("Right side crop is empty")
                image = cv2.flip(image[:, :side_width], 1)
                box[2] = box[0] + side_width
            else:
                image = image[:, int(width * .4):]
                box[0] += int(width * .4)
        if not image.size or image.max() == image.min():
            raise ValueError("Side ROI is empty or constant; no zero fallback")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = normalize(image=image[:, :, None])["image"][:, :, 0]
        arrays.append(image)
        frame = record["frame"]
        ref = frame["image_ref"]
        rid = "{}:{}:{}:{}".format(level, side, sequence, ref["image_id"])
        entry = {"roi_id": rid, "level": level, "side": side, "sequence": sequence,
                 "image_ref": ref, "base_crop_box": record["box"], "source_crop_box": box,
                 "source_corners_lps_mm": box_corners_lps(frame, box),
                 "horizontal_flip": sequence == "axial_t2" and side == "right",
                 "resize_shape": [224, 224]}
        spatial_rois[rid] = entry
        refs.append(ref)
        roi_ids.append(rid)
    return np.stack(arrays), refs, roi_ids


def infer(study, frames, by_sequence, networks, assets, allow_unverified_registration=False):
    import numpy as np
    import torch
    native = {"kind": "rsna_native_predictions", "schema_version": "1.0",
              "study_id": study["study_id"], "study_digest": manifest_digest(study),
              "assets": assets, "class_order": CLASS_ORDER, "localizations": [], "predictions": [],
              "limitations": ["Native RSNA disc numbering is not independently reviewed anatomical root numbering.",
                              "Cross-series registration is not independently reviewed.",
                              "Single classifier fold1 and two keypoint fold0 models; training-set overlap unverified.",
                              "All three series contribute to both classifier outputs; no herniation or clinical diagnosis.",
                              "Normal/Mild combines normal and mild disease; not absence of all pathology.",
                              "Axial frames selected within one orientation group per level; per-frame LPS projection corrects source spacing assumptions."]}
    spatial = {"kind": "rsna_source_geometry", "schema_version": "1.0", "study_id": study["study_id"],
               "study_digest": manifest_digest(study), "coordinate_system": "DICOM_LPS_mm",
               "frames": [frame_metadata(f) for f in frames], "keypoints": [], "rois": [], "inputs": [],
               "registration_status": "unreviewed", "anatomical_numbering_status": "unreviewed",
               "pixel_coordinate_convention": "DICOM IPP is pixel-center (0,0); boxes [left,top,right,bottom) are half-open indices; physical footprint edges are index minus 0.5; OpenCV resize uses half-pixel centers; horizontal_flip is applied after side crop",
               "rejected_crops": [], "geometry_changes_from_upstream": [
                   "DICOM row/column spacing corrected for x=column,y=row.",
                   "Axial level selection uses each source plane; a deterministic constant-IOP group is selected.",
                   "Axial ROI center is projected in LPS onto each selected source frame."]}
    declared_frames = {f["frame_of_reference_uid"] for f in frames if f["frame_of_reference_uid"]}
    registration_conflict = len(declared_frames) > 1
    registration_missing = any(not f["frame_of_reference_uid"] for f in frames)
    native["registration_policy"] = "explicit_research_identity_LPS_unverified" if allow_unverified_registration else "require_same_declared_FrameOfReferenceUID"
    native["different_frame_of_reference_uids"] = registration_conflict
    if registration_conflict:
        native["limitations"].append("Declared FrameOfReferenceUIDs differ; identity-LPS alignment is an explicit unverified research assumption, not established registration.")
    if registration_missing:
        native["limitations"].append("FrameOfReferenceUID is missing on source frames; a common declared coordinate frame cannot be established.")
    spatial["registration_policy"] = native["registration_policy"]
    sagittal, keypoint_refs, keypoint_calls = {}, [], 0
    for sequence in SEQUENCES[:2]:
        located = locate(by_sequence[sequence], networks["sagittal_keypoints"], LEVELS)
        keypoint_calls += int(bool(by_sequence[sequence]))
        sagittal[sequence] = located
        if located:
            keypoint_refs.append(located["reference"]["image_ref"])
            for point in located["points"].values():
                if point:
                    spatial["keypoints"].append({"sequence": sequence, **point})
    reference = sagittal.get("sagittal_t2")
    selection_reason = None
    try:
        if reference is None or any(reference["points"][level] is None for level in LEVELS):
            raise ValueError("Five SagT2 keypoints are unavailable; axial level assignment not performed")
        points = {level: reference["points"][level]["lps_mm"] for level in LEVELS}
        if (registration_conflict or registration_missing) and not allow_unverified_registration:
            raise ValueError("Different or missing FrameOfReferenceUIDs require registration; cross-series level projection not run")
        segments = level_segments(points)
        axial_groups, spatial["axial_selection"] = _group_axial(by_sequence["axial_t2"], segments, points)
        spatial["level_segments_lps_mm"] = segments
    except ValueError as exc:
        axial_groups = {level: [] for level in LEVELS}
        selection_reason = str(exc)
        spatial["axial_selection"] = {"reason": selection_reason}
    roi_records, examples, classification_calls = {}, {}, 0
    for level in LEVELS:
        refs = list(keypoint_refs)
        crop_sets, issues = {}, []
        for sequence in SEQUENCES[:2]:
            located = sagittal[sequence]
            point = located["points"][level] if located else None
            if point is None:
                crop_sets[sequence] = []
                issues.append("Missing " + sequence + " native keypoint")
            else:
                crop_sets[sequence], rejected = _crop_records(by_sequence[sequence], level, sequence, norm_xy=point["norm_xy"])
                spatial["rejected_crops"].extend(rejected)
        axial = locate(axial_groups[level], networks["axial_keypoints"], ["center"])
        keypoint_calls += int(bool(axial_groups[level]))
        center = axial["points"]["center"] if axial else None
        if center:
            refs.append(center["image_ref"])
            spatial["keypoints"].append({"sequence": "axial_t2", "native_level": level, **center})
            crop_sets["axial_t2"], rejected = _crop_records(axial_groups[level], level, "axial_t2", world_center=center["lps_mm"])
            spatial["rejected_crops"].extend(rejected)
        else:
            crop_sets["axial_t2"] = []
            issues.append(selection_reason or "No axial group/keypoint at this native level")
        has_level = reference is not None and reference["points"].get(level) is not None
        native["localizations"].append({"level": level, "status": "located" if has_level else "not_located",
                                        "reason": None if has_level else "Native sagittal disc point unavailable",
                                        "image_refs": unique_refs(refs)})
        for side in ("left", "right"):
            record = {"level": level, "side": side, "status": "not_run", "reason": None,
                      "probabilities": None, "image_refs": unique_refs(refs)}
            try:
                if issues:
                    raise ValueError("; ".join(issues))
                arrays, consumed, sample_ids, pair_examples = [], [], {}, {}
                for sequence in SEQUENCES:
                    image, source_refs, ids = _input(crop_sets[sequence], sequence, side, level, roi_records)
                    arrays.append(image)
                    consumed.extend(source_refs)
                    sample_ids[sequence] = ids
                    pair_examples[(level, side, sequence)] = image[len(image) // 2]
                tensors = [torch.from_numpy(np.ascontiguousarray(image[None])).float() for image in arrays]
                record["image_refs"] = unique_refs(refs + consumed)
                classification_calls += 1
                with torch.inference_mode():
                    probabilities = (networks["classifier"](*tensors)["logits"] / TEMPERATURE).softmax(-1).cpu().numpy()[0, :, 0]
                if probabilities.shape != (2, 3) or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid classifier output")
                record.update(status="predicted", reason=None,
                              probabilities={"foraminal_stenosis": probabilities[0].tolist(),
                                             "subarticular_stenosis": probabilities[1].tolist()},
                              image_refs=unique_refs(refs + consumed))
                spatial["inputs"].append({"level": level, "side": side, "roi_ids": sample_ids})
                examples.update(pair_examples)
            except ValueError as exc:
                record["reason"] = str(exc)
            native["predictions"].append(record)
    spatial["rois"] = list(roi_records.values())
    native["classification_calls"] = classification_calls
    native["keypoint_calls"] = keypoint_calls
    native["temperature"] = TEMPERATURE
    native["tta"] = False
    return native, spatial, examples
