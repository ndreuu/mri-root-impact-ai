"""Synthetic physical geometry checks; no image reads or model execution."""

import copy
import math
import unittest

from mri_root_impact.contracts import manifest_digest
from mri_root_impact.rsna_geometry import box_corners_lps, crop_box, pixel_to_lps
from mri_root_impact.rsna_results import LEVELS, SIDES
from mri_root_impact.rsna_spatial import SEQUENCES, validate_rsna_spatial
from tests.test_rsna_results import example


def spatial_example(predicted=True):
    study, native = example()
    study["series"] = []
    frames = []
    specifications = [("sagittal_t1", None), ("sagittal_t2", None)] + [("axial_t2", level) for level in LEVELS]
    for index, (sequence, frame_level) in enumerate(specifications):
        ref = {"series_id": sequence, "image_id": "image-" + str(index), "frame_index": 0}
        sha = format(index + 1, "064x")
        series = next((item for item in study["series"] if item["sequence"] == sequence), None)
        if series is None:
            series = {"series_id": sequence, "sequence": sequence, "images": []}
            study["series"].append(series)
        series["images"].append({"image_id": ref["image_id"], "path": "/synthetic/" + ref["image_id"] + ".dcm", "frame_count": 1, "sha256": sha})
        frames.append({"image_ref": ref, "sequence": sequence, "sha256": sha,
                       "position": [0, 0, 0 if frame_level is None else -40 - 35 * LEVELS.index(frame_level)],
                       "orientation": [0, 1, 0, 0, 0, -1] if index < 2 else [1, 0, 0, 0, 1, 0],
                       "spacing": [1.0, 1.0], "rows": 256, "columns": 256,
                       "frame_of_reference_uid": "1.2.3.4"})
    native["study_digest"] = manifest_digest(study)
    native["registration_policy"] = "require_same_declared_FrameOfReferenceUID"
    native["different_frame_of_reference_uids"] = False
    spatial = {"kind": "rsna_source_geometry", "schema_version": "1.0", "study_id": study["study_id"],
               "study_digest": manifest_digest(study), "coordinate_system": "DICOM_LPS_mm", "frames": frames,
               "keypoints": [], "rois": [], "inputs": [], "registration_status": "unreviewed", "anatomical_numbering_status": "unreviewed",
               "registration_policy": native["registration_policy"]}
    for location in native["localizations"]:
        level_frames = frames[:2] + [frames[2 + LEVELS.index(location["level"])]]
        all_refs = [copy.deepcopy(f["image_ref"]) for f in level_frames]
        location.update(image_refs=copy.deepcopy(all_refs) if predicted else [], status="located" if predicted else "not_located",
                        reason=None if predicted else "No automatic point")
    for prediction in native["predictions"]:
        all_refs = [copy.deepcopy(f["image_ref"]) for f in frames[:2] + [frames[2 + LEVELS.index(prediction["level"])]]]
        prediction["image_refs"] = copy.deepcopy(all_refs) if predicted else []
        if not predicted:
            prediction.update(status="not_run", reason="No automatic point", probabilities=None)
    if not predicted:
        return study, native, spatial
    for level in LEVELS:
        level_frames = frames[:2] + [frames[2 + LEVELS.index(level)]]
        for frame in level_frames:
            sequence = frame["sequence"]
            row = 128. if sequence == "axial_t2" else 40. + 35. * LEVELS.index(level)
            point = {"sequence": sequence, "label": level if sequence != "axial_t2" else "center",
                     "image_ref": copy.deepcopy(frame["image_ref"]), "column": 128., "row": row, "norm_xy": [.5, row / 256],
                     "lps_mm": pixel_to_lps(frame, 128, row), "heatmap_peak": .9, "heatmap_shape": [190, 190], "percentile_mask": 99.8}
            if sequence == "axial_t2":
                point["native_level"] = level
            spatial["keypoints"].append(point)
        for side in SIDES:
            sample = {"level": level, "side": side, "roi_ids": {}}
            for frame in level_frames:
                sequence = frame["sequence"]
                sagittal = sequence != "axial_t2"
                center = [.5, (40. + 35. * LEVELS.index(level)) / 256] if sagittal else [.5, .5]
                base = crop_box(frame, center, [35, 50] if sagittal else [45, 75], [-5, 0] if sagittal else [5, 0])
                box = list(base)
                if not sagittal:
                    width = base[2] - base[0]
                    if side == "right":
                        box[2] = base[0] + int(width * .6)
                    else:
                        box[0] += int(width * .4)
                rid = level + ":" + side + ":" + sequence
                spatial["rois"].append({"roi_id": rid, "level": level, "side": side, "sequence": sequence,
                                         "image_ref": copy.deepcopy(frame["image_ref"]), "base_crop_box": base, "source_crop_box": box,
                                         "source_corners_lps_mm": box_corners_lps(frame, box),
                                         "horizontal_flip": not sagittal and side == "right", "resize_shape": [224, 224]})
                sample["roi_ids"][sequence] = [rid] * (15 if sagittal else 10)
            spatial["inputs"].append(sample)
    return study, native, spatial


class RSNASpatialTests(unittest.TestCase):
    def test_valid_source_geometry_and_repeated_sampling_do_not_read_images(self):
        study, native, spatial = spatial_example()
        before = copy.deepcopy((study, native, spatial))
        self.assertIsNone(validate_rsna_spatial(study, native, spatial))
        self.assertEqual((study, native, spatial), before)

    def test_no_usable_roi_and_not_run_is_a_valid_explicit_outcome(self):
        study, native, spatial = spatial_example(predicted=False)
        self.assertIsNone(validate_rsna_spatial(study, native, spatial))
        spatial["inputs"] = [{"level": LEVELS[0], "side": "left", "roi_ids": {}}]
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)

    def test_manifest_binding_and_finite_physical_frame_geometry_are_required(self):
        mutations = [lambda s: s.update(study_digest="e" * 64),
                     lambda s: s["frames"][0].update(sha256="e" * 64),
                     lambda s: s["frames"][0].update(sequence="axial_t2"),
                     lambda s: s["frames"][0].update(position=[float("nan"), 0, 0]),
                     lambda s: s["frames"][0].update(orientation=[0, 2, 0, 0, 0, -1]),
                     lambda s: s["frames"][0].update(spacing=[0, 1]),
                     lambda s: s["frames"].append(copy.deepcopy(s["frames"][0])),
                     lambda s: s["frames"].pop()]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, native, spatial = spatial_example()
                mutate(spatial)
                with self.assertRaises(ValueError):
                    validate_rsna_spatial(study, native, spatial)

    def test_keypoint_pixels_normalized_values_and_world_points_must_agree(self):
        mutations = [lambda p: p.update(column=129), lambda p: p.update(norm_xy=[.6, .5]),
                     lambda p: p.update(lps_mm=[0, 129, -128]), lambda p: p.update(column=True),
                     lambda p: p.update(row=256), lambda p: p.update(heatmap_peak=float("inf")),
                     lambda p: p["image_ref"].update(image_id="missing")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, native, spatial = spatial_example()
                mutate(spatial["keypoints"][0])
                with self.assertRaises(ValueError):
                    validate_rsna_spatial(study, native, spatial)

    def test_half_pixel_corners_base_crop_and_patient_side_are_verified(self):
        mutations = [lambda r: r["base_crop_box"].__setitem__(0, r["base_crop_box"][0] + 1),
                     lambda r: r["source_crop_box"].__setitem__(2, 300),
                     lambda r: r["source_corners_lps_mm"][0].__setitem__(0, r["source_corners_lps_mm"][0][0] + .5),
                     lambda r: r.update(horizontal_flip=True), lambda r: r.update(resize_shape=[256, 256]),
                     lambda r: r.update(side="right")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, native, spatial = spatial_example()
                # First left axial ROI: a physical footprint and asymmetric side crop.
                mutate(spatial["rois"][2])
                with self.assertRaises(ValueError):
                    validate_rsna_spatial(study, native, spatial)

    def test_exact_sequence_counts_roi_ids_targets_and_native_refs_are_checked(self):
        mutations = [lambda s: s["inputs"][0]["roi_ids"]["sagittal_t1"].pop(),
                     lambda s: s["inputs"][0]["roi_ids"]["axial_t2"].__setitem__(0, "missing"),
                     lambda s: s["inputs"][0]["roi_ids"]["axial_t2"].__setitem__(0, LEVELS[0] + ":right:axial_t2"),
                     lambda s: s["inputs"].pop(),
                     lambda s: s["inputs"].append(copy.deepcopy(s["inputs"][0])),
                     lambda s: s["rois"].append(copy.deepcopy(s["rois"][0]))]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, native, spatial = spatial_example()
                mutate(spatial)
                with self.assertRaises(ValueError):
                    validate_rsna_spatial(study, native, spatial)
        study, native, spatial = spatial_example()
        native["predictions"][0]["image_refs"].pop()
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)

    def test_unsupported_side_orientation_and_fake_review_are_rejected(self):
        study, native, spatial = spatial_example()
        spatial["frames"][2]["orientation"] = [-1, 0, 0, 0, -1, 0]
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)
        for field in ("registration_status", "anatomical_numbering_status"):
            study, native, spatial = spatial_example()
            spatial[field] = "confirmed"
            with self.assertRaises(ValueError):
                validate_rsna_spatial(study, native, spatial)
        study, native, spatial = spatial_example()
        for index, frame in enumerate(spatial["frames"]):
            frame["frame_of_reference_uid"] = "1.2.3." + str(index)
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)
        native["different_frame_of_reference_uids"] = True
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)
        native["registration_policy"] = "explicit_research_identity_LPS_unverified"
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)
        spatial["registration_policy"] = native["registration_policy"]
        self.assertIsNone(validate_rsna_spatial(study, native, spatial))
        study, native, spatial = spatial_example(predicted=False)
        for index, frame in enumerate(spatial["frames"]):
            frame["frame_of_reference_uid"] = "1.2.3." + str(index)
        native["different_frame_of_reference_uids"] = True
        self.assertIsNone(validate_rsna_spatial(study, native, spatial))
        study, native, spatial = spatial_example()
        spatial["frames"][0].pop("frame_of_reference_uid")
        with self.assertRaisesRegex(ValueError, "missing or conflicting"):
            validate_rsna_spatial(study, native, spatial)
        native["registration_policy"] = spatial["registration_policy"] = "explicit_research_identity_LPS_unverified"
        self.assertIsNone(validate_rsna_spatial(study, native, spatial))
        study, native, spatial = spatial_example(predicted=False)
        for frame in spatial["frames"]:
            frame["frame_of_reference_uid"] = None
        self.assertIsNone(validate_rsna_spatial(study, native, spatial))

    def test_native_probability_mismatch_is_not_bypassed_by_spatial_validation(self):
        study, native, spatial = spatial_example()
        native["predictions"][0]["probabilities"]["foraminal_stenosis"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            validate_rsna_spatial(study, native, spatial)

    def test_consistent_axial_crop_cannot_be_assigned_to_a_different_level_plane(self):
        for change_keypoint in (False, True):
            with self.subTest(keypoint=change_keypoint):
                study, native, spatial = spatial_example()
                wrong_frame = spatial["frames"][-1]
                if change_keypoint:
                    point = spatial["keypoints"][2]
                    point["image_ref"] = copy.deepcopy(wrong_frame["image_ref"])
                    point["lps_mm"] = pixel_to_lps(wrong_frame, point["column"], point["row"])
                else:
                    roi = spatial["rois"][2]
                    roi["image_ref"] = copy.deepcopy(wrong_frame["image_ref"])
                    roi["source_corners_lps_mm"] = box_corners_lps(wrong_frame, roi["source_crop_box"])
                native["localizations"][0]["image_refs"].append(copy.deepcopy(wrong_frame["image_ref"]))
                for prediction in native["predictions"][:2]:
                    prediction["image_refs"].append(copy.deepcopy(wrong_frame["image_ref"]))
                with self.assertRaisesRegex(ValueError, "native SagT2 level segment"):
                    validate_rsna_spatial(study, native, spatial)

    def test_axial_keypoint_must_share_input_orientation_and_level_intersection_fov(self):
        study, native, spatial = spatial_example()
        frame = copy.deepcopy(spatial["frames"][2])
        frame["image_ref"]["image_id"] = "rotated-frame"
        angle = .1
        frame["orientation"] = [math.cos(angle), math.sin(angle), 0, -math.sin(angle), math.cos(angle), 0]
        image = {"image_id": "rotated-frame", "path": "/synthetic/rotated-frame.dcm", "frame_count": 1, "sha256": frame["sha256"]}
        study["series"][2]["images"].append(image)
        native["study_digest"] = spatial["study_digest"] = manifest_digest(study)
        spatial["frames"].append(frame)
        roi = spatial["rois"][2]
        point = spatial["keypoints"][2]
        from mri_root_impact.rsna_geometry import lps_to_pixel
        x, y, _ = lps_to_pixel(frame, point["lps_mm"])
        roi["base_crop_box"] = crop_box(frame, [x / 256, y / 256], [45, 75], [5, 0])
        roi["source_crop_box"] = list(roi["base_crop_box"])
        width = roi["base_crop_box"][2] - roi["base_crop_box"][0]
        roi["source_crop_box"][0] += int(width * .4)
        roi["image_ref"] = copy.deepcopy(frame["image_ref"])
        roi["source_corners_lps_mm"] = box_corners_lps(frame, roi["source_crop_box"])
        native["predictions"][0]["image_refs"].append(copy.deepcopy(frame["image_ref"]))
        with self.assertRaisesRegex(ValueError, "orientation group"):
            validate_rsna_spatial(study, native, spatial)

        study, native, spatial = spatial_example()
        # Sagittal x is a constant LPS plane; translating both source sagittal
        # planes preserves all points/crops but moves the segment beyond Ax FOV.
        for frame in spatial["frames"][:2]:
            frame["position"][0] = 300.
        for point in spatial["keypoints"]:
            if point["sequence"].startswith("sagittal"):
                frame = next(f for f in spatial["frames"] if f["image_ref"] == point["image_ref"])
                point["lps_mm"] = pixel_to_lps(frame, point["column"], point["row"])
        with self.assertRaisesRegex(ValueError, "field of view"):
            validate_rsna_spatial(study, native, spatial)


if __name__ == "__main__":
    unittest.main()
