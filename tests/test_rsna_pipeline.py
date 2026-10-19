"""Synthetic RSNA preprocessing checks; no models or patient data are loaded."""

import importlib.util
import math
import unittest
from unittest.mock import Mock, patch

from mri_root_impact.rsna_dicom import normalize_pixels
from mri_root_impact.rsna_geometry import LEVELS, level_segments
from mri_root_impact.rsna_pipeline import _group_axial, _input, infer

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
    import albumentations
except ImportError:
    cv2 = None
    albumentations = None


def frame(image_id, position, orientation=None):
    return {
        "image_ref": {"series_id": "synthetic-series", "image_id": image_id,
                      "frame_index": 0},
        "position": position,
        "orientation": orientation or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "spacing": [1.0, 1.0], "rows": 20, "columns": 20,
    }


def points():
    return {level: [0.0, 0.0, 100.0 - 20.0 * i]
            for i, level in enumerate(LEVELS)}


@unittest.skipIf(np is None, "numpy is an optional RSNA dependency")
class PixelNormalizationTests(unittest.TestCase):
    def test_percentiles_clip_outliers_and_stretch_grayscale_to_uint8(self):
        # 1,001 samples place the .1 and 99.9 percentiles at exactly 0 and 100.
        pixels = np.concatenate(([-100.0], np.linspace(0, 100, 999), [1000.0])).reshape(7, 143)
        before = pixels.copy()
        normalized = normalize_pixels(pixels)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, pixels.shape)
        self.assertEqual(normalized.flat[0], 0)
        self.assertEqual(normalized.flat[1], 0)
        self.assertEqual(normalized.flat[500], 127)
        self.assertEqual(normalized.flat[999], 255)
        self.assertEqual(normalized.flat[1000], 255)
        np.testing.assert_array_equal(pixels, before)

    def test_each_frame_is_normalized_independently_of_intensity_scale(self):
        first = np.arange(100, dtype=np.float32).reshape(10, 10)
        second = first * 8 + 32
        np.testing.assert_array_equal(normalize_pixels(first), normalize_pixels(second))

    def test_constant_nonfinite_empty_and_nongrayscale_inputs_are_rejected(self):
        invalid = [np.zeros((4, 4)), np.full((4, 4), 15.0),
                   np.array([[0.0, float("nan")], [1.0, 2.0]]),
                   np.array([[0.0, float("inf")], [1.0, 2.0]]),
                   np.array([[0.0, -float("inf")], [1.0, 2.0]]),
                   np.empty((0, 4)), np.arange(12), np.zeros((2, 2, 3))]
        for pixels in invalid:
            with self.subTest(shape=pixels.shape), self.assertRaises(ValueError):
                normalize_pixels(pixels)


class AxialGroupSelectionTests(unittest.TestCase):
    def test_largest_constant_orientation_group_wins_without_merging_planes(self):
        centers = points()
        segments = level_segments(centers)
        angle = math.radians(8)
        oblique = [1.0, 0.0, 0.0, 0.0, math.cos(angle), math.sin(angle)]
        frames = [frame("flat-1", [0.0, 0.0, 96.0]),
                  frame("tilted-1", [0.0, 0.0, 99.0], oblique),
                  frame("flat-2", [0.0, 0.0, 100.0]),
                  frame("tilted-2", [0.0, 0.0, 101.0], oblique),
                  frame("flat-3", [0.0, 0.0, 104.0])]
        chosen, details = _group_axial(frames, segments, centers)
        self.assertEqual({f["image_ref"]["image_id"] for f in chosen["L1-L2"]},
                         {"flat-1", "flat-2", "flat-3"})
        self.assertEqual(len({tuple(f["orientation"]) for f in chosen["L1-L2"]}), 1)
        self.assertTrue(all(not chosen[level] for level in LEVELS[1:]))
        groups = details["per_level"]["L1-L2"]["groups"]
        self.assertEqual([g["selected"] for g in groups], [True, False])
        self.assertEqual([len(g["frames"]) for g in groups], [3, 2])
        reordered, _ = _group_axial(list(reversed(frames)), segments, centers)
        self.assertEqual({f["image_ref"]["image_id"] for f in reordered["L1-L2"]},
                         {"flat-1", "flat-2", "flat-3"})

    def test_equal_size_groups_use_distance_then_deterministic_orientation_tie(self):
        centers = points()
        segments = level_segments(centers)
        angle = math.radians(8)
        oblique = [1.0, 0.0, 0.0, 0.0, math.cos(angle), math.sin(angle)]
        farther = frame("farther", [0.0, 0.0, 105.0])
        nearer = frame("nearer", [0.0, 0.0, 100.0], oblique)
        selected, _ = _group_axial([farther, nearer], segments, centers)
        self.assertEqual(selected["L1-L2"][0]["image_ref"]["image_id"], "nearer")
        flat = frame("flat", [0.0, 0.0, 100.0])
        for inputs in ([flat, nearer], [nearer, flat]):
            selected, _ = _group_axial(inputs, segments, centers)
            self.assertEqual(selected["L1-L2"][0]["image_ref"]["image_id"], "nearer")

    def test_shared_level_boundary_and_uncovered_frames_are_excluded(self):
        centers = points()
        boundary = frame("boundary", [0.0, 0.0, 90.0])
        outside = frame("outside", [0.0, 0.0, 200.0])
        selected, details = _group_axial([boundary, outside], level_segments(centers), centers)
        self.assertTrue(all(not group for group in selected.values()))
        self.assertEqual(details["ambiguous_boundary_frames"], [boundary["image_ref"]])
        self.assertEqual(details["outside_level_segments"], [outside["image_ref"]])


@unittest.skipIf(np is None or cv2 is None or albumentations is None,
                 "numpy/OpenCV/albumentations are optional RSNA dependencies")
class ClassifierInputTests(unittest.TestCase):
    def records(self, count, sequence, image=None):
        values = []
        for i in range(count):
            pixels = (np.tile(np.arange(10, dtype=np.uint8), (10, 1)) + i * 20
                      if image is None else image.copy())
            values.append({"frame": frame("slice-" + str(i), [float(i), 0.0, float(i * 10)]),
                           "box": [2, 3, 12, 13], "image": pixels,
                           "level": "L1-L2", "sequence": sequence})
        return values

    def test_sagittal_patient_side_uses_outer_half_and_preserves_repeated_samples(self):
        records = self.records(5, "sagittal_t1")
        for side, expected in (("left", ["slice-4"] * 4 + ["slice-3"] * 7 + ["slice-2"] * 4),
                               ("right", ["slice-0"] * 4 + ["slice-1"] * 7 + ["slice-2"] * 4)):
            with self.subTest(side=side):
                rois = {}
                image, refs, ids = _input(records, "sagittal_t1", side, "L1-L2", rois)
                self.assertEqual(image.shape, (15, 224, 224))
                self.assertEqual(image.dtype, np.float32)
                self.assertEqual([r["image_id"] for r in refs], expected)
                self.assertEqual(len(ids), 15)
                self.assertEqual(len(set(ids)), 3)
                self.assertEqual(set(ids), set(rois))
                for index, image_id in enumerate(expected):
                    gray = int(image_id.rsplit("-", 1)[1]) * 20
                    self.assertAlmostEqual(float(image[index, 0, 0]), gray / 127.5 - 1, places=6)
                self.assertTrue(all(not roi["horizontal_flip"] for roi in rois.values()))

    def test_one_sagittal_slice_is_repeated_without_inventing_source_references(self):
        records = self.records(1, "sagittal_t2")
        rois = {}
        image, refs, ids = _input(records, "sagittal_t2", "left", "L1-L2", rois)
        self.assertEqual(image.shape, (15, 224, 224))
        self.assertEqual(refs, [records[0]["frame"]["image_ref"]] * 15)
        self.assertEqual(len(rois), 1)
        self.assertEqual(ids, [next(iter(rois))] * 15)

    def test_axial_right_sixty_percent_flip_and_left_overlap_retain_source_boxes(self):
        gradient = np.tile(np.arange(10, dtype=np.uint8) * 20, (10, 1))
        records = self.records(3, "axial_t2", gradient)
        expected_refs = ["slice-2"] * 3 + ["slice-1"] * 4 + ["slice-0"] * 3
        for side, endpoints, box in (("right", [100, 0], [2, 3, 8, 13]),
                                     ("left", [80, 180], [6, 3, 12, 13])):
            with self.subTest(side=side):
                rois = {}
                image, refs, ids = _input(records, "axial_t2", side, "L1-L2", rois)
                self.assertEqual(image.shape, (10, 224, 224))
                self.assertEqual([r["image_id"] for r in refs], expected_refs)
                self.assertAlmostEqual(float(image[0, 0, 0]), endpoints[0] / 127.5 - 1, places=6)
                self.assertAlmostEqual(float(image[0, 0, -1]), endpoints[1] / 127.5 - 1, places=6)
                self.assertEqual(len(ids), 10)
                for roi in rois.values():
                    self.assertEqual(roi["base_crop_box"], [2, 3, 12, 13])
                    self.assertEqual(roi["source_crop_box"], box)
                    self.assertEqual(roi["horizontal_flip"], side == "right")
                for record in records:
                    self.assertEqual(record["box"], [2, 3, 12, 13])
                    np.testing.assert_array_equal(record["image"], gradient)

    def test_missing_sequence_records_raise_instead_of_returning_zero_input(self):
        for sequence in ("sagittal_t1", "sagittal_t2", "axial_t2"):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                _input([], sequence, "left", "L1-L2", {})

    def test_narrow_right_side_crop_is_rejected_before_opencv_resize(self):
        vertical_gradient = np.arange(10, dtype=np.uint8)[:, None]
        records = self.records(1, "axial_t2", vertical_gradient)
        records[0]["box"] = [2, 3, 3, 13]
        with self.assertRaisesRegex(ValueError, "empty"):
            _input(records, "axial_t2", "right", "L1-L2", {})


@unittest.skipUnless(np is not None and cv2 is not None and albumentations is not None
                     and importlib.util.find_spec("torch") is not None,
                     "the optional RSNA tensor runtime is not installed")
class MissingSagittalInferenceTests(unittest.TestCase):
    def test_failed_classifier_keeps_consumed_refs_without_grades_or_preview(self):
        sequences = ("sagittal_t1", "sagittal_t2", "axial_t2")
        grouped = {sequence: [frame(sequence + suffix, [0., 0., 100.])
                              for suffix in ("-keypoint", "-consumed")] for sequence in sequences}
        frames = [source for group in grouped.values() for source in group]
        for source in frames:
            source["frame_of_reference_uid"] = "1.2.3.4"

        def located(group, model, labels):
            source = group[0]
            return {"reference": source, "points": {
                label: {"label": label, "norm_xy": [.5, .5],
                        "image_ref": source["image_ref"], "lps_mm": [0., 0., 100. - 20. * i]}
                for i, label in enumerate(labels)}}

        def crops(group, *args, **kwargs):
            return [{"frame": source} for source in group], []

        def prepared(records, sequence, side, level, rois):
            return np.ones((2, 2, 2), dtype=np.float32), [records[-1]["frame"]["image_ref"]], []

        networks = {"sagittal_keypoints": Mock(), "axial_keypoints": Mock(),
                    "classifier": Mock(side_effect=ValueError("Synthetic invalid logits"))}
        with patch("mri_root_impact.rsna_pipeline.locate", side_effect=located), \
                patch("mri_root_impact.rsna_pipeline._group_axial",
                      return_value=({level: grouped["axial_t2"] for level in LEVELS}, {})), \
                patch("mri_root_impact.rsna_pipeline._crop_records", side_effect=crops), \
                patch("mri_root_impact.rsna_pipeline._input", side_effect=prepared):
            native, spatial, examples = infer({"study_id": "synthetic-failed-classifier"}, frames, grouped, networks, {})
        self.assertEqual(networks["classifier"].call_count, 10)
        self.assertEqual(native["classification_calls"], 10)
        self.assertTrue(all(p["status"] == "not_run" and p["probabilities"] is None for p in native["predictions"]))
        for prediction in native["predictions"]:
            self.assertIn(grouped["sagittal_t2"][-1]["image_ref"], prediction["image_refs"])
            self.assertIn("Synthetic invalid logits", prediction["reason"])
        self.assertEqual(spatial["inputs"], [])
        self.assertEqual(examples, {})

    def test_missing_frame_uid_blocks_cross_series_assignment_by_default(self):
        sequences = ("sagittal_t1", "sagittal_t2", "axial_t2")
        frames = [frame(sequence, [0., 0., 100.]) for sequence in sequences]
        for source in frames:
            source["frame_of_reference_uid"] = "1.2.3.4"
        frames[0]["frame_of_reference_uid"] = None
        grouped = dict(zip(sequences, ([source] for source in frames)))
        networks = {role: Mock() for role in ("sagittal_keypoints", "axial_keypoints", "classifier")}

        def located(group, model, labels):
            if not group:
                return None
            source = group[0]
            return {"reference": source, "points": {
                level: {"label": level, "norm_xy": [.5, .5],
                        "image_ref": source["image_ref"], "lps_mm": [0., 0., 100. - 20. * i]}
                for i, level in enumerate(LEVELS)}}

        with patch("mri_root_impact.rsna_pipeline.locate", side_effect=located), \
                patch("mri_root_impact.rsna_pipeline._crop_records", return_value=([], [])), \
                patch("mri_root_impact.rsna_pipeline._group_axial") as transfer:
            native, spatial, _ = infer({"study_id": "synthetic-missing-uid"}, frames, grouped, networks, {})
        transfer.assert_not_called()
        networks["classifier"].assert_not_called()
        self.assertEqual(native["classification_calls"], 0)
        self.assertFalse(native["different_frame_of_reference_uids"])
        self.assertIn("missing FrameOfReferenceUIDs", spatial["axial_selection"]["reason"])
        self.assertTrue(all(p["status"] == "not_run" and p["probabilities"] is None for p in native["predictions"]))

    def test_axial_only_input_does_not_claim_or_call_any_model_inference(self):
        axial = frame("axial-only", [0.0, 0.0, 100.0])
        axial["frame_of_reference_uid"] = "1.2.3.4"
        axial["pixels"] = np.arange(400, dtype=np.uint16).reshape(20, 20)
        study = {"schema_version": "1.0", "kind": "study_manifest",
                 "study_id": "synthetic-axial-only", "synthetic": True,
                 "limitations": ["Synthetic input; both sagittal sequences are missing."],
                 "series": [{"series_id": "synthetic-series", "sequence": "axial_t2",
                             "images": [{"image_id": "axial-only", "frame_count": 1,
                                         "path": "/synthetic/axial-only.dcm", "sha256": "a" * 64}]}]}
        networks = {role: Mock(side_effect=AssertionError("No model may run without sagittal inputs"))
                    for role in ("sagittal_keypoints", "axial_keypoints", "classifier")}
        native, spatial, examples = infer(
            study, [axial], {"sagittal_t1": [], "sagittal_t2": [], "axial_t2": [axial]},
            networks, {})
        self.assertEqual(native["keypoint_calls"], 0)
        self.assertEqual(native["classification_calls"], 0)
        self.assertEqual(len(native["predictions"]), 10)
        self.assertTrue(all(p["status"] == "not_run" and p["probabilities"] is None
                            for p in native["predictions"]))
        self.assertEqual(spatial["keypoints"], [])
        self.assertEqual(spatial["rois"], [])
        self.assertEqual(spatial["inputs"], [])
        self.assertEqual(examples, {})
        for model in networks.values():
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
