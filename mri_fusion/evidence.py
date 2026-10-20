"""Traceable model facts, conditional anatomy and existing MRI illustrations."""

import copy

from .contracts import FusionContractError, load_policy


def ref_key(ref):
    return tuple(ref[k] for k in ("series_id", "image_id", "frame_index"))


def anatomical_facts(report):
    """Explain source findings without promoting any formal fusion decision.

    A conditional relation assumes that native level/side labels are correct.
    It is not a verified root assignment or a positive impact classification.
    """
    policy, _ = load_policy()
    anatomy = {row["level"]: row for row in policy["anatomy"]}
    facts = []
    for branch in report["branches"]:
        observations = [e["observation"] for e in branch["evidence"] if e["observation"] is not None]
        accepted = [o for o in observations if o["scale_id"] == policy["accepted_input"]["scale_id"]
                    and o["label"] in policy["accepted_input"]["labels"]]
        grades = sorted({o["label"] for o in accepted})
        grade = grades[0] if len(grades) == 1 and len(accepted) == len(observations) else None
        target = branch["target"]
        relation = None
        if grade is not None:
            relation = dict(anatomy[target["level"]][target["zone"]], side=target["side"])
        effect = {"severe": "severe_research_threshold", "moderate": "moderate_relation_only",
                  "normal_mild": "no_severe_criterion"}.get(grade, "unresolved_source")
        facts.append({"fact_id": "fact-" + branch["branch_id"], "branch_id": branch["branch_id"],
                      "native_target": copy.deepcopy(target), "native_grade": grade,
                      "native_observations": copy.deepcopy(observations),
                      "source_refs": [copy.deepcopy(e["source_ref"]) for e in branch["evidence"]],
                      "conditional_anatomy": relation, "relation_requires_correct_native_labels": True,
                      "rule_effect": effect, "formal_decision": branch["decision"],
                      "verified_candidate": copy.deepcopy(branch["candidate_root"]),
                      "herniation_assessed_by_this_rule": False})
    return facts


def build_evidence(report, spatial=None, overlays=(), read_image=None):
    """Select exact model input frames, never an arbitrary nearby MRI slice.

    read_image receives a checked overlay record and returns JSON metadata with
    a SHA-bound PNG data URI. The runner owns actual file-byte verification.
    """
    bundle = {"schema_version": "1.0", "kind": "mri_report_evidence", "study_id": report["study_id"],
              "study_digest": report["study_digest"], "fusion_id": report["fusion_id"],
              "facts": anatomical_facts(report), "stage_views": [], "disc_views": [], "images": {},
              "image_status": "not_supplied", "new_mri_inference_performed": False,
              "disc_analysis_status": "not_evaluated"}
    if spatial is None:
        return bundle
    if (spatial["study_id"] != report["study_id"] or spatial["study_digest"] != report["study_digest"]
            or read_image is None):
        raise FusionContractError("Illustrations must match this study and a checked image reader")
    by_roi = {}
    for roi in spatial["rois"]:
        if roi["roi_id"] in by_roi:
            raise FusionContractError("Duplicate ROI identity in report evidence")
        by_roi[roi["roi_id"]] = roi
    by_overlay = {}
    for overlay in overlays:
        key = ref_key(overlay["image_ref"])
        if key in by_overlay:
            raise FusionContractError("Duplicate source frame overlay")
        by_overlay[key] = overlay
    inputs = {}
    for item in spatial["inputs"]:
        key = item["level"], item["side"]
        if key in inputs:
            raise FusionContractError("Duplicate native model input group")
        inputs[key] = item

    def image_for(ref):
        overlay = by_overlay.get(ref_key(ref))
        if overlay is None:
            raise FusionContractError("Model source frame has no checked overlay")
        image_id = "png-" + overlay["sha256"]
        if image_id not in bundle["images"]:
            image = read_image(overlay)
            if image["sha256"] != overlay["sha256"]:
                raise FusionContractError("Illustration image differs from its checked overlay")
            bundle["images"][image_id] = image
        return image_id, overlay

    for branch in report["branches"]:
        if not branch["evidence"]:
            continue
        target = branch["target"]
        sequence = "sagittal_t1" if target["zone"] == "foraminal" else "axial_t2"
        item = inputs.get((target["level"], target["side"]))
        roi_ids = [] if item is None else item["roi_ids"].get(sequence, [])
        if not roi_ids:
            continue
        # Median in the actual ordered model input, preserving repeated views.
        # This is a reproducible illustration, not a search for the worst slice.
        index = len(roi_ids) // 2
        roi = by_roi.get(roi_ids[index])
        if (roi is None or roi["level"] != target["level"] or roi["side"] != target["side"]
                or roi["sequence"] != sequence):
            raise FusionContractError("Illustration ROI does not match the model input target")
        refs = {ref_key(ref) for evidence in branch["evidence"] for ref in evidence["image_refs"]}
        if ref_key(roi["image_ref"]) not in refs:
            if all(e["observation"] is None for e in branch["evidence"]):
                continue
            raise FusionContractError("Illustration is not a source of the displayed branch assessment")
        image_id, overlay = image_for(roi["image_ref"])
        box = roi["source_crop_box"]
        image = bundle["images"][image_id]
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= image["width"]
                or not 0 <= box[1] < box[3] <= image["height"] - overlay["header_pixels"]):
            raise FusionContractError("Illustration crop lies outside its source image")
        bundle["stage_views"].append({"branch_id": branch["branch_id"], "native_target": copy.deepcopy(target),
                                      "sequence": sequence, "image_id": image_id,
                                      "image_ref": copy.deepcopy(roi["image_ref"]), "roi_id": roi["roi_id"],
                                      "source_crop_box": copy.deepcopy(roi["source_crop_box"]),
                                      "header_pixels": overlay["header_pixels"],
                                      "model_horizontal_flip": roi["horizontal_flip"],
                                      "model_sample_index": index, "model_sample_count": len(roi_ids),
                                      "selection": "middle_of_ordered_model_input_not_maximum_stenosis",
                                      "overlay_is_pathology_segmentation": False})
    for point in spatial["keypoints"]:
        if point["sequence"] != "axial_t2":
            continue
        image_id, _ = image_for(point["image_ref"])
        bundle["disc_views"].append({"native_level": point["native_level"], "sequence": "axial_t2",
                                     "image_id": image_id, "image_ref": copy.deepcopy(point["image_ref"]),
                                     "purpose": "disc_visual_review_only", "herniation_detection_performed": False})
    bundle["image_status"] = "checked_model_source_overlays"
    return bundle
