"""Counterfactual source-frame comparisons, using synthetic metadata only."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from mri_root_impact.contracts import manifest_digest
from mri_root_impact.spatial import build_localization


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_localizers.py"
_SPEC = importlib.util.spec_from_file_location("compare_localizers_test_module", _SCRIPT)
_COMPARE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COMPARE
_SPEC.loader.exec_module(_COMPARE)
compare_levels = _COMPARE.compare_levels

LABELS = ("L1", "L2", "L3", "L4", "L5", "S1")
LEVELS = tuple(a + "-" + b for a, b in zip(LABELS, LABELS[1:]))


def example():
    """A straight, fully synthetic lumbar stack with exact source references."""
    images = [{"image_id": "sag-" + str(i), "path": "/synthetic/sag-" + str(i) + ".dcm",
               "frame_count": 1, "sha256": format(i + 1, "064x")} for i in range(3)]
    study = {"schema_version": "1.0", "kind": "study_manifest", "study_id": "synthetic-crosscheck",
             "synthetic": True, "series": [{"series_id": "sag-t2", "sequence": "sagittal_t2", "images": images}],
             "limitations": ["Synthetic geometry; no anatomical or registration review."]}
    geometry = {"series_id": "sag-t2", "volume_shape": [256, 256, 3], "pixel_spacing_mm": 1.,
                "slice_center_spacing_mm": 5., "coordinate_system": "DICOM_LPS_mm", "slices": []}
    frames = []
    for i, image in enumerate(images):
        ref = {"series_id": "sag-t2", "image_id": image["image_id"], "frame_index": 0}
        ipp = [5. - 5. * i, 0., 0.]
        iop = [0., 1., 0., 0., 0., -1.]
        geometry["slices"].append({"slice_index": i, "image_ref": copy.deepcopy(ref),
                                    "image_position_patient_mm": ipp, "image_orientation_patient": iop,
                                    "pixel_spacing_mm": [1., 1.], "slice_thickness_mm": 4.})
        frames.append({"image_ref": copy.deepcopy(ref), "sequence": "sagittal_t2", "sha256": image["sha256"],
                       "position": list(ipp), "orientation": list(iop), "spacing": [1., 1.],
                       "rows": 256, "columns": 256, "frame_of_reference_uid": "1.2.3.4"})
    native = []
    for i, label in enumerate(LABELS):
        row = 25. + 35. * i
        polygon = [[80., row - 10.], [120., row - 10.], [120., row + 10.], [80., row + 10.]]
        native.append({"predicted_label": label, "slice_nos": [0, 1, 2],
                       "polys": [copy.deepcopy(polygon) for _ in range(3)], "average_polygon": copy.deepcopy(polygon)})
    _, spine = build_localization(study, geometry, native, {"source_commit": "a" * 40, "weights_sha256": "b" * 64})
    rsna = {"kind": "rsna_source_geometry", "schema_version": "1.0", "study_id": study["study_id"],
            "study_digest": manifest_digest(study), "coordinate_system": "DICOM_LPS_mm", "frames": frames,
            "keypoints": [], "rois": [], "inputs": [], "registration_status": "unreviewed",
            "anatomical_numbering_status": "unreviewed", "registration_policy": "require_same_declared_FrameOfReferenceUID"}
    for i, level in enumerate(LEVELS):
        row = 42.5 + 35. * i
        rsna["keypoints"].append({"sequence": "sagittal_t2", "label": level,
                                  "image_ref": copy.deepcopy(frames[1]["image_ref"]), "column": 100., "row": row,
                                  "norm_xy": [100. / 256., row / 256.], "lps_mm": [0., 100., -row],
                                  "heatmap_peak": .95, "heatmap_shape": [190, 190], "percentile_mask": 99.8})
    return study, spine, rsna


def move_point(point, row):
    point["row"] = row
    point["norm_xy"][1] = row / 256.
    point["lps_mm"][2] = -row


def set_polygon_rows(polygon, rows):
    for pixel, lps, row in zip(polygon["polygon_xy"], polygon["polygon_lps_mm"], rows):
        pixel[1] = row
        lps[2] = -row


class LevelCrosscheckTests(unittest.TestCase):
    def compare(self, study=None, spine=None, rsna=None):
        if study is None:
            study, spine, rsna = example()
        report = compare_levels(study, spine, rsna)
        self.assertEqual([entry["level"] for entry in report["levels"]], list(LEVELS))
        self.assertEqual(len(report["levels"]), 5)
        self.assertTrue(all(entry["status"] == "indeterminate" and isinstance(entry["reason"], str)
                            and entry["reason"] for entry in report["levels"]))
        for flag in ("inference_performed", "numbering_confirmed", "registration_confirmed"):
            self.assertIs(report[flag], False)
        self.assertIsNone(report["root_impacts"])
        return report

    def test_all_five_native_agreements_retain_the_exact_frame_and_labels(self):
        study, spine, rsna = example()
        report = self.compare(study, spine, rsna)
        for i, item in enumerate(report["levels"]):
            with self.subTest(level=item["level"]):
                self.assertEqual(item["comparison"], "native_agreement")
                self.assertEqual(item["sagittal_reference"], rsna["keypoints"][i]["image_ref"])
                self.assertEqual(item["expected_adjacent_labels"], list(LABELS[i:i + 2]))
                self.assertEqual(item["expected_detection_ids"], {
                    LABELS[i]: [spine["detections"][i]["detection_id"]],
                    LABELS[i + 1]: [spine["detections"][i + 1]["detection_id"]]})
                pair = item["bracketing_pair"]
                self.assertEqual((pair["upper_native_label"], pair["lower_native_label"]), LABELS[i:i + 2])
                self.assertEqual(pair["upper_detection_id"], spine["detections"][i]["detection_id"])
                self.assertEqual(pair["lower_detection_id"], spine["detections"][i + 1]["detection_id"])
                self.assertAlmostEqual(pair["centroid_fraction"], .5)
                self.assertTrue(item["source_polygons"])
                self.assertEqual(item["axial_roi_links"], [])

    def test_polygon_centroid_is_area_weighted_instead_of_vertex_mean(self):
        study, spine, rsna = example()
        for polygon in spine["detections"][0]["polygons"]:
            polygon["polygon_xy"] = [[80., 15.], [120., 15.], [110., 35.], [90., 35.]]
            x = polygon["polygon_lps_mm"][0][0]
            polygon["polygon_lps_mm"] = [[x, column, -row] for column, row in polygon["polygon_xy"]]
        first = self.compare(study, spine, rsna)["levels"][0]
        self.assertEqual(first["comparison"], "native_agreement")
        # The trapezoid centroid is y=15 + 20*(40+2*20)/(3*(40+20)),
        # so the point at y=42.5 divides the centroid interval in ratio 67/130.
        self.assertAlmostEqual(first["bracketing_pair"]["centroid_fraction"], 67. / 130.)
        self.assertNotAlmostEqual(first["bracketing_pair"]["centroid_fraction"], .5)

    def test_one_disc_shift_is_visible_as_disagreement_without_relabeling(self):
        study, spine, rsna = example()
        move_point(rsna["keypoints"][0], 77.5)
        item = self.compare(study, spine, rsna)["levels"][0]
        self.assertEqual(item["comparison"], "native_disagreement")
        self.assertEqual(item["expected_adjacent_labels"], ["L1", "L2"])
        self.assertEqual(item["bracketing_pair"]["upper_native_label"], "L2")
        self.assertEqual(item["bracketing_pair"]["lower_native_label"], "L3")
        self.assertEqual(rsna["keypoints"][0]["label"], "L1-L2")

    def test_array_and_slice_order_do_not_replace_exact_image_references(self):
        study, spine, rsna = example()
        expected = self.compare(study, spine, rsna)
        spine["geometry"]["slices"].reverse()
        positions = {}
        for index, frame in enumerate(spine["geometry"]["slices"]):
            frame["slice_index"] = index
            positions[frame["image_ref"]["image_id"]] = index
        for detection in spine["detections"]:
            detection["polygons"].reverse()
            for polygon in detection["polygons"]:
                polygon["slice_index"] = positions[polygon["image_ref"]["image_id"]]
        spine["detections"].reverse()
        rsna["frames"].reverse()
        rsna["keypoints"].reverse()
        actual = self.compare(study, spine, rsna)
        for before, after in zip(expected["levels"], actual["levels"]):
            self.assertEqual(before["comparison"], after["comparison"])
            self.assertEqual(before["sagittal_reference"], after["sagittal_reference"])
            self.assertEqual(before["bracketing_pair"], after["bracketing_pair"])

    def test_same_sized_unrelated_frame_does_not_count_as_a_common_source(self):
        study, spine, rsna = example()
        unrelated = copy.deepcopy(rsna["frames"][1])
        unrelated["image_ref"]["image_id"] = "different-source"
        rsna["frames"].append(unrelated)
        image = copy.deepcopy(study["series"][0]["images"][1])
        image["image_id"] = "different-source"
        study["series"][0]["images"].append(image)
        for point in rsna["keypoints"]:
            point["image_ref"] = copy.deepcopy(unrelated["image_ref"])
        rsna["study_digest"] = spine["study_digest"] = manifest_digest(study)
        self.assertTrue(all(item["comparison"] == "indeterminate" for item in self.compare(study, spine, rsna)["levels"]))

    def test_nearby_slice_polygons_cannot_fill_in_an_absent_exact_frame_polygon(self):
        study, spine, rsna = example()
        for detection in spine["detections"]:
            detection["polygons"] = [p for p in detection["polygons"] if p["image_ref"]["image_id"] != "sag-1"]
        self.assertTrue(all(item["comparison"] == "indeterminate" for item in self.compare(study, spine, rsna)["levels"]))

    def test_duplicate_or_missing_expected_vertebra_is_explicitly_unknown(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                study, spine, rsna = example()
                if duplicate:
                    copied = copy.deepcopy(spine["detections"][1])
                    copied["detection_id"] = "duplicate-L2"
                    spine["detections"].append(copied)
                else:
                    spine["detections"].pop(1)
                first = self.compare(study, spine, rsna)["levels"][0]
                self.assertEqual(first["comparison"], "indeterminate")
                self.assertEqual(len(first["expected_detection_ids"]["L2"]), 2 if duplicate else 0)

    def test_missing_and_duplicate_rsna_points_do_not_force_a_level(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                study, spine, rsna = example()
                if duplicate:
                    rsna["keypoints"].append(copy.deepcopy(rsna["keypoints"][0]))
                else:
                    rsna["keypoints"].pop(0)
                first = self.compare(study, spine, rsna)["levels"][0]
                self.assertEqual(first["comparison"], "indeterminate")
                self.assertIsNone(first["bracketing_pair"])

    def test_geometry_mismatch_even_on_the_same_image_id_is_not_comparable(self):
        mutations = [lambda f: f["position"].__setitem__(2, 1e-4),
                     lambda f: f["spacing"].__setitem__(1, 2.),
                     lambda f: f["orientation"].__setitem__(1, -1.),
                     lambda f: f.update(rows=255), lambda f: f.update(columns=255)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, spine, rsna = example()
                mutate(rsna["frames"][1])
                self.assertTrue(all(item["comparison"] == "indeterminate" for item in self.compare(study, spine, rsna)["levels"]))

    def test_invalid_point_coordinates_are_not_used_to_claim_native_agreement(self):
        mutations = [lambda p: p.update(row=float("nan")),
                     lambda p: p.update(column=300., norm_xy=[300. / 256., p["row"] / 256.],
                                        lps_mm=[0., 300., -p["row"]]),
                     lambda p: p["lps_mm"].__setitem__(2, -100.),
                     lambda p: p["image_ref"].update(frame_index=1)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, spine, rsna = example()
                mutate(rsna["keypoints"][0])
                self.assertEqual(self.compare(study, spine, rsna)["levels"][0]["comparison"], "indeterminate")

    def test_malformed_out_of_bounds_degenerate_and_inconsistent_polygons_are_unknown(self):
        mutations = [lambda p: p.update(in_bounds=False),
                     lambda p: (p["polygon_xy"][0].__setitem__(0, -1.),
                                p["polygon_lps_mm"][0].__setitem__(1, -1.)),
                     lambda p: p["polygon_xy"][0].__setitem__(1, float("nan")),
                     lambda p: p["polygon_lps_mm"][0].__setitem__(2, 999.),
                     lambda p: p["polygon_xy"].pop(),
                     lambda p: set_polygon_rows(p, [25., 25., 25., 25.])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, spine, rsna = example()
                mutate(spine["detections"][0]["polygons"][1])
                self.assertEqual(self.compare(study, spine, rsna)["levels"][0]["comparison"], "indeterminate")

    def test_body_centroid_and_equal_centroid_ties_do_not_get_arbitrary_brackets(self):
        study, spine, rsna = example()
        move_point(rsna["keypoints"][0], 25.)
        self.assertEqual(self.compare(study, spine, rsna)["levels"][0]["comparison"], "indeterminate")
        study, spine, rsna = example()
        for polygon in spine["detections"][1]["polygons"]:
            set_polygon_rows(polygon, [15., 15., 35., 35.])
        self.assertEqual(self.compare(study, spine, rsna)["levels"][0]["comparison"], "indeterminate")

    def test_centroid_boundary_uses_only_the_declared_numeric_epsilon(self):
        for delta, expected in ((5e-7, "indeterminate"), (2e-6, "native_agreement")):
            with self.subTest(delta=delta):
                study, spine, rsna = example()
                move_point(rsna["keypoints"][0], 25. + delta)
                self.assertEqual(self.compare(study, spine, rsna)["levels"][0]["comparison"], expected)

    def test_canonical_labels_in_nonadjacent_physical_order_do_not_make_a_valid_pair(self):
        study, spine, rsna = example()
        spine["detections"][1]["polygons"], spine["detections"][2]["polygons"] = (
            spine["detections"][2]["polygons"], spine["detections"][1]["polygons"])
        first = self.compare(study, spine, rsna)["levels"][0]
        self.assertTrue(all(len(ids) == 1 for ids in first["expected_detection_ids"].values()))
        self.assertEqual(first["comparison"], "indeterminate")
        self.assertIsNone(first["bracketing_pair"])

    def test_noncanonical_observed_pair_stays_unknown_instead_of_renumbering(self):
        study, spine, rsna = example()
        spine["detections"][1]["native_label"] = "T12"
        self.assertEqual(self.compare(study, spine, rsna)["levels"][0]["comparison"], "indeterminate")

    def test_all_five_levels_remain_listed_when_no_points_or_detections_exist(self):
        study, spine, rsna = example()
        rsna["keypoints"] = []
        spine["detections"] = []
        report = self.compare(study, spine, rsna)
        self.assertTrue(all(item["comparison"] == "indeterminate" and item["bracketing_pair"] is None
                            for item in report["levels"]))

    def test_comparison_is_deterministic_and_never_mutates_inputs(self):
        inputs = example()
        before = copy.deepcopy(inputs)
        first = self.compare(*inputs)
        self.assertEqual(first, self.compare(*inputs))
        self.assertEqual(inputs, before)
        first["levels"][0]["expected_detection_ids"]["L1"].append("foreign-id")
        self.assertEqual(inputs, before)


class LevelCrosscheckRunnerTests(unittest.TestCase):
    def read_run_loader(self, spine, rsna, parents, registration_policy=False):
        def load(path):
            path = Path(path)
            if path.name == "execution.json":
                return {"parameters": {"allow_unverified_registration": registration_policy}}
            if path.name == "run.json":
                return copy.deepcopy(parents[path.parent.name])
            if path.name == "spatial.json":
                return copy.deepcopy(spine if path.parent.name == "spinenet" else rsna)
            return {"kind": "mock-validated-result"}
        return load

    def test_worker_success_cannot_override_a_failed_or_incomplete_parent_run(self):
        study, spine, rsna = example()
        good = {"study_id": study["study_id"], "status": "completed", "worker_exit_code": 0, "timed_out": False, "inference_performed": True}
        mutations = ({"status": "error"}, {"status": "running"}, {"worker_exit_code": 1},
                     {"timed_out": True}, {"inference_performed": False}, {"study_id": "unrelated-study"},
                     {"worker_exit_code": False}, {"worker_exit_code": 0.0})
        worker = {"artifacts_sha256": {"spatial.json": "c" * 64}}
        for source in ("spinenet", "rsna"):
            for changes in mutations:
                with self.subTest(source=source, changes=changes):
                    parents = {name: dict(good) for name in ("spinenet", "rsna")}
                    parents[source].update(changes)
                    with mock.patch.object(_COMPARE, "validate_spinenet", return_value=worker), \
                         mock.patch.object(_COMPARE, "validate_rsna", return_value=worker), \
                         mock.patch.object(_COMPARE, "load_document", side_effect=self.read_run_loader(spine, rsna, parents)), \
                         mock.patch.object(_COMPARE, "file_sha256", return_value="d" * 64), \
                         mock.patch.object(_COMPARE, "validate_bundle"):
                        with self.assertRaisesRegex(ValueError, "completed checked model run"):
                            _COMPARE._read_runs(study, Path("/synthetic/spinenet"), Path("/synthetic/rsna"))

    def test_completed_input_runs_bind_both_executions_and_parent_records_in_provenance(self):
        study, spine, rsna = example()
        parents = {name: {"study_id": study["study_id"], "status": "completed", "worker_exit_code": 0, "timed_out": False, "inference_performed": True}
                   for name in ("spinenet", "rsna")}
        worker = {"artifacts_sha256": {"spatial.json": "c" * 64, "overlay.png": "d" * 64}}
        with mock.patch.object(_COMPARE, "validate_spinenet", return_value=worker), \
             mock.patch.object(_COMPARE, "validate_rsna", return_value=worker) as checked_rsna, \
             mock.patch.object(_COMPARE, "load_document", side_effect=self.read_run_loader(spine, rsna, parents)), \
             mock.patch.object(_COMPARE, "file_sha256", return_value="e" * 64), \
             mock.patch.object(_COMPARE, "validate_bundle"):
            actual_spine, actual_rsna, sources = _COMPARE._read_runs(
                study, Path("/synthetic/spinenet"), Path("/synthetic/rsna"))
            checked_rsna.assert_called_once_with(study, Path("/synthetic/rsna"), allow_unverified_registration=False)
        self.assertEqual((actual_spine, actual_rsna), (spine, rsna))
        self.assertEqual(set(sources), {"spinenet", "rsna"})
        for name, source in sources.items():
            self.assertEqual(source["run_directory"], "/synthetic/" + name)
            self.assertEqual(source["files_sha256"], {
                filename: "e" * 64 for filename in ("spatial.json", "overlay.png", "execution.json", "run.json")})

    def test_registration_policy_from_input_run_must_be_an_explicit_boolean(self):
        study, spine, rsna = example()
        worker = {"artifacts_sha256": {}}
        for policy in (None, 0, 1, "false", []):
            with self.subTest(policy=policy):
                with mock.patch.object(_COMPARE, "validate_spinenet", return_value=worker), \
                     mock.patch.object(_COMPARE, "validate_rsna") as validate, \
                     mock.patch.object(_COMPARE, "load_document", side_effect=self.read_run_loader(spine, rsna, {}, policy)):
                    with self.assertRaisesRegex(ValueError, "explicit boolean"):
                        _COMPARE._read_runs(study, Path("/synthetic/spinenet"), Path("/synthetic/rsna"))
                    validate.assert_not_called()

    def test_output_inside_either_upstream_run_is_rejected_before_read_or_write(self):
        study, _, _ = example()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spine_run, rsna_run = root / "spinenet", root / "rsna"
            for directory in (spine_run, rsna_run):
                directory.mkdir()
                (directory / "sentinel").write_bytes(b"unchanged synthetic input")
            for output in (spine_run, spine_run / "nested", rsna_run, rsna_run / "nested"):
                with self.subTest(output=output):
                    with mock.patch.object(_COMPARE, "load_document", return_value=study), \
                         mock.patch.object(_COMPARE, "_read_runs") as read:
                        with self.assertRaises(ValueError):
                            _COMPARE.run(root / "study.json", spine_run, rsna_run, output)
                        read.assert_not_called()
                    for directory in (spine_run, rsna_run):
                        self.assertEqual(list(directory.iterdir()), [directory / "sentinel"])
                        self.assertEqual((directory / "sentinel").read_bytes(), b"unchanged synthetic input")

    def test_direct_runner_modality_policy_is_strictly_boolean(self):
        study, _, _ = example()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for policy in (None, 0, 1, "false", []):
                with self.subTest(policy=policy):
                    with mock.patch.object(_COMPARE, "load_document", return_value=study), \
                         mock.patch.object(_COMPARE, "_read_runs") as read, \
                         mock.patch("mri_root_impact.rsna_dicom.load_frames") as frames:
                        with self.assertRaisesRegex(ValueError, "boolean"):
                            _COMPARE.run(root / "study.json", root / "spinenet", root / "rsna", root / "output", policy)
                        read.assert_not_called()
                        frames.assert_not_called()
                    self.assertFalse((root / "output").exists())

    def test_source_header_mismatch_is_rejected_before_output_directory_creation(self):
        study, spine, rsna = example()
        frames = copy.deepcopy(rsna["frames"])
        frames[0]["position"][0] += .5
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp), Path(tmp) / "output"
            with mock.patch.object(_COMPARE, "load_document", return_value=study), \
                 mock.patch.object(_COMPARE, "_read_runs", return_value=(spine, rsna, {})), \
                 mock.patch("mri_root_impact.rsna_dicom.load_frames", return_value=(frames, {})), \
                 mock.patch.object(_COMPARE, "render_overlays") as render:
                with self.assertRaisesRegex(ValueError, "source headers differ"):
                    _COMPARE.run(root / "study.json", root / "spinenet", root / "rsna", output)
                render.assert_not_called()
            self.assertFalse(output.exists())

    def test_valid_bounded_run_records_checked_sources_without_model_inference(self):
        study, spine, rsna = example()
        frames = copy.deepcopy(rsna["frames"])
        sources = {"spinenet": {"files_sha256": {"run.json": "e" * 64}},
                   "rsna": {"files_sha256": {"run.json": "f" * 64}}}
        input_hashes = {image["path"]: image["sha256"] for series in study["series"] for image in series["images"]}
        original_hash = _COMPARE.file_sha256
        def checked_hash(path):
            return input_hashes[str(path)] if str(path) in input_hashes else original_hash(path)
        def render(report, spine_data, rsna_data, decoded, output):
            (output / "overlay.png").write_bytes(b"synthetic renderer boundary")
            return []
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp), Path(tmp) / "output"
            before = copy.deepcopy((study, spine, rsna, frames))
            with mock.patch.object(_COMPARE, "load_document", return_value=study), \
                 mock.patch.object(_COMPARE, "_read_runs", return_value=(spine, rsna, sources)), \
                 mock.patch("mri_root_impact.rsna_dicom.load_frames", return_value=(frames, {})) as loader, \
                 mock.patch.object(_COMPARE, "render_overlays", side_effect=render), \
                 mock.patch.object(_COMPARE, "file_sha256", side_effect=checked_hash):
                result = _COMPARE.run(root / "study.json", root / "spinenet", root / "rsna", output, True)
                loader.assert_called_once_with(study, allow_missing_modality=True)
            report = json.loads((output / "crosscheck.json").read_text())
            execution = json.loads((output / "run.json").read_text())
            self.assertEqual(report["sources"], sources)
            self.assertEqual(report["source_files_verified"], 3)
            self.assertIs(report["source_bytes_unchanged"], True)
            self.assertEqual(execution["status"], "completed")
            self.assertEqual(result["comparisons"], {"native_agreement": 5, "native_disagreement": 0, "indeterminate": 0})
            for value in (result, report, execution):
                self.assertIs(value["inference_performed"], False)
                self.assertIs(value["numbering_confirmed"], False)
                self.assertIs(value["registration_confirmed"], False)
                self.assertIsNone(value["root_impacts"])
            self.assertEqual((study, spine, rsna, frames), before)


if __name__ == "__main__":
    unittest.main()
