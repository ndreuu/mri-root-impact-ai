"""Synthetic coordinate and uncertainty tests; no MRI or ML packages needed."""

import copy
import unittest

from mri_root_impact.contracts import ContractError, validate_bundle
from mri_root_impact.spatial import build_localization, validate_spatial


def inputs():
    images = [{"image_id": "image-{}".format(i), "path": "/synthetic/image-{}.dcm".format(i),
               "frame_count": 1, "sha256": "a" * 64} for i in range(3)]
    study = {"schema_version": "1.0", "kind": "study_manifest", "study_id": "synthetic-spatial",
             "synthetic": True, "series": [{"series_id": "sag-t2", "sequence": "sagittal_t2", "images": images}],
             "limitations": ["Synthetic geometry only."]}
    geometry = {"series_id": "sag-t2", "volume_shape": [100, 120, 3], "pixel_spacing_mm": 2,
                "slice_center_spacing_mm": 5, "coordinate_system": "DICOM_LPS_mm", "slices": []}
    for i in range(3):
        geometry["slices"].append({
            "slice_index": i, "image_ref": {"series_id": "sag-t2", "image_id": "image-{}".format(i), "frame_index": 0},
            "image_position_patient_mm": [10 - 5 * i, 20, 30],
            "image_orientation_patient": [0, 1, 0, 0, 0, -1],
            "pixel_spacing_mm": [2, 2], "slice_thickness_mm": 4})
    native = [{"predicted_label": "L4", "slice_nos": [0, 2],
               "polys": [[[3, 4], [7, 4], [7, 8], [3, 8]], [[4, 5], [8, 5], [8, 9], [4, 9]]],
               "average_polygon": [[3.5, 4.5], [7.5, 4.5], [7.5, 8.5], [3.5, 8.5]]}]
    provenance = {"source_commit": "b" * 40, "weights_sha256": "c" * 64}
    return study, geometry, native, provenance


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.study, self.geometry, self.native, self.provenance = inputs()

    def build(self):
        return build_localization(self.study, self.geometry, self.native, self.provenance)

    def test_coordinates_use_column_then_row_with_source_frame_mapping(self):
        result, artifact = self.build()
        validate_bundle(self.study, [result])
        self.assertIsNone(validate_spatial(self.study, result, artifact))
        first, last = artifact["detections"][0]["polygons"]
        self.assertEqual(first["polygon_lps_mm"][0], [10, 26, 22])
        self.assertEqual(last["polygon_lps_mm"][0], [0, 28, 20])
        self.assertEqual([r["image_id"] for r in result["assessments"][0]["image_refs"]], ["image-0", "image-2"])
        self.assertEqual(artifact["geometry"]["slice_center_spacing_mm"], 5)
        self.assertEqual(artifact["geometry"]["slices"][0]["slice_thickness_mm"], 4)

    def test_build_is_deterministic_and_does_not_mutate_or_alias_inputs(self):
        before = copy.deepcopy((self.study, self.geometry, self.native, self.provenance))
        result, artifact = self.build()
        self.assertEqual((result, artifact), self.build())
        self.assertEqual(before, (self.study, self.geometry, self.native, self.provenance))
        artifact["geometry"]["slices"][0]["image_position_patient_mm"][0] = 999
        artifact["detections"][0]["polygons"][0]["polygon_xy"][0][0] = 999
        self.assertEqual(before, (self.study, self.geometry, self.native, self.provenance))

    def test_all_expected_labels_stay_indeterminate_until_review_and_discs_not_run(self):
        base = self.native[0]
        self.native = [dict(copy.deepcopy(base), predicted_label=label) for label in ("L1", "L2", "L3", "L4", "L5", "S1")]
        result, artifact = self.build()
        self.assertEqual(len(result["assessments"]), 11)
        self.assertTrue(all(a["status"] == "indeterminate" and a["observation"] for a in result["assessments"][:6]))
        self.assertTrue(all(a["status"] == "not_run" and a["observation"] is None and a["image_refs"] == []
                            for a in result["assessments"][6:]))
        self.assertIs(artifact["coverage"]["full_expected_numbering"], False)
        self.assertEqual(artifact["coverage"]["anatomical_review_status"], "not_reviewed")
        self.assertEqual(artifact["coverage"]["disc_levels"][0]["adjacent_detection_ids"],
                         {"L1": ["detection-0001"], "L2": ["detection-0002"]})
        validate_spatial(self.study, result, artifact)

    def test_missing_labels_are_explicit_unknown_assessments_not_negative(self):
        result, artifact = self.build()
        missing = next(a for a in result["assessments"] if a["assessment_id"] == "missing-S1")
        self.assertEqual(missing["status"], "indeterminate")
        self.assertIsNone(missing["observation"])
        self.assertEqual(missing["image_refs"], [])
        self.assertIn("not a negative", missing["reason"])
        self.assertEqual(next(v for v in artifact["coverage"]["vertebrae"] if v["level"] == "S1")["detection_ids"], [])

    def test_duplicate_numbering_preserves_both_observations_without_promoting_either(self):
        self.native.append(copy.deepcopy(self.native[0]))
        result, artifact = self.build()
        self.assertEqual([a["observation"]["label"] for a in result["assessments"][:2]], ["L4", "L4"])
        self.assertTrue(all(a["status"] == "indeterminate" and "Duplicate" in a["reason"] for a in result["assessments"][:2]))
        self.assertEqual(next(v for v in artifact["coverage"]["vertebrae"] if v["level"] == "L4")["detection_ids"],
                         ["detection-0001", "detection-0002"])
        validate_spatial(self.study, result, artifact)

    def test_unsupported_labels_t12_s2_are_retained_without_renumbering(self):
        self.native = [dict(copy.deepcopy(self.native[0]), predicted_label=label) for label in ("T12", "S2", " L5")]
        result, artifact = self.build()
        self.assertEqual([d["native_label"] for d in artifact["detections"]], ["T12", "S2", " L5"])
        self.assertEqual([a["target"]["level"] for a in result["assessments"][:3]], ["unknown"] * 3)
        self.assertEqual([a["observation"]["label"] for a in result["assessments"][:3]], ["T12", "S2", " L5"])
        self.assertEqual(len(artifact["coverage"]["unsupported_detection_ids"]), 3)
        self.assertIn("missing-L5", [a["assessment_id"] for a in result["assessments"]])
        validate_spatial(self.study, result, artifact)

    def test_empty_detection_output_retains_all_missing_levels_and_unrun_discs(self):
        self.native = []
        result, artifact = self.build()
        self.assertEqual(len(result["assessments"]), 11)
        self.assertEqual(artifact["detections"], [])
        self.assertTrue(all(a["observation"] is None for a in result["assessments"]))
        validate_spatial(self.study, result, artifact)

    def test_native_average_is_retained_as_summary_without_forcing_it_onto_a_slice(self):
        self.native[0]["average_polygon"][0] = [17, 19]
        result, artifact = self.build()
        self.assertEqual(artifact["detections"][0]["native_average_polygon_xy"][0], [17, 19])
        self.assertEqual(artifact["detections"][0]["polygons"][0]["polygon_xy"][0], [3, 4])
        validate_spatial(self.study, result, artifact)

    def test_polygon_outside_image_is_preserved_flagged_and_not_clipped(self):
        self.native[0]["polys"][0][0] = [-4, 120]
        result, artifact = self.build()
        polygon = artifact["detections"][0]["polygons"][0]
        self.assertEqual(polygon["polygon_xy"][0], [-4, 120])
        self.assertIs(polygon["in_bounds"], False)
        self.assertEqual(result["assessments"][0]["status"], "indeterminate")

    def test_bad_native_coordinates_shapes_and_slices_are_rejected(self):
        mutations = [
            lambda n: n[0]["polys"][0][0].__setitem__(0, float("nan")),
            lambda n: n[0]["average_polygon"][0].__setitem__(1, float("inf")),
            lambda n: n[0]["polys"][0].pop(),
            lambda n: n[0]["polys"][0][0].append(0),
            lambda n: n[0]["polys"].pop(),
            lambda n: n[0]["slice_nos"].__setitem__(1, 0),
            lambda n: n[0]["slice_nos"].__setitem__(1, 3),
            lambda n: n[0]["slice_nos"].__setitem__(0, True),
            lambda n: n[0].update(predicted_label=""),
            lambda n: n[0].update(extra_field="unknown"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                native = copy.deepcopy(self.native)
                mutation(native)
                with self.assertRaises(ContractError):
                    build_localization(self.study, self.geometry, native, self.provenance)

    def test_geometry_requires_valid_source_frames_and_consistent_physical_stack(self):
        mutations = [
            lambda g: g.update(series_id="unknown"),
            lambda g: g["slices"][0]["image_ref"].update(image_id="unknown"),
            lambda g: g["slices"][1]["image_ref"].update(image_id="image-0"),
            lambda g: g["slices"][0]["image_ref"].update(frame_index=True),
            lambda g: g["slices"][0].update(slice_index=True),
            lambda g: g["volume_shape"].__setitem__(2, 2),
            lambda g: g["slices"][0]["pixel_spacing_mm"].__setitem__(1, 3),
            lambda g: g["slices"][0]["image_orientation_patient"].__setitem__(1, 2),
            lambda g: g["slices"][1]["image_position_patient_mm"].__setitem__(0, 4),
            lambda g: g["slices"][1]["image_position_patient_mm"].__setitem__(1, 21),
            lambda g: g["slices"][2]["image_position_patient_mm"].__setitem__(0, 10),
            lambda g: g.update(coordinate_system="RAS"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                geometry = copy.deepcopy(self.geometry)
                mutation(geometry)
                with self.assertRaises(ContractError):
                    build_localization(self.study, geometry, self.native, self.provenance)

    def test_spatial_validator_rejects_lps_xy_swap_and_source_reference_tampering(self):
        result, artifact = self.build()
        mutations = [
            lambda a: a["detections"][0]["polygons"][0]["polygon_lps_mm"][0].__setitem__(1, 28),
            lambda a: a["detections"][0]["polygons"][0]["image_ref"].update(image_id="image-1"),
            lambda a: a["detections"][0].update(assessment_id="missing-L1"),
            lambda a: a["detections"][0]["polygons"][0].update(in_bounds=1),
            lambda a: a["coverage"].update(full_expected_numbering=True),
            lambda a: a["coverage"]["disc_levels"][0].update(status="assessed"),
            lambda a: a.update(schema_version="2.0"),
            lambda a: a.update(extra_field=True),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(artifact)
                mutation(changed)
                with self.assertRaises(ContractError):
                    validate_spatial(self.study, result, changed)

    def test_c001_cannot_be_promoted_or_retargeted_independently_of_spatial_artifact(self):
        result, artifact = self.build()
        for changes in ({"status": "assessed", "reason": None}, {"observation": None}):
            changed = copy.deepcopy(result)
            changed["assessments"][0].update(changes)
            with self.assertRaises(ContractError):
                validate_spatial(self.study, changed, artifact)
        changed = copy.deepcopy(result)
        changed["assessments"][0]["target"]["level"] = "L3"
        with self.assertRaises(ContractError):
            validate_spatial(self.study, changed, artifact)

    def test_changed_manifest_cannot_be_reused_with_old_results(self):
        result, artifact = self.build()
        self.study["limitations"].append("Changed source identity.")
        with self.assertRaises(ContractError):
            validate_spatial(self.study, result, artifact)

    def test_source_and_weight_provenance_are_strict_and_bound_to_result(self):
        for changes in ({"source_commit": "a" * 39}, {"weights_sha256": "A" * 64}, {"extra": True}):
            provenance = dict(self.provenance, **changes)
            with self.assertRaises(ContractError):
                build_localization(self.study, self.geometry, self.native, provenance)
        result, artifact = self.build()
        artifact["model_provenance"]["source_commit"] = "d" * 40
        with self.assertRaises(ContractError):
            validate_spatial(self.study, result, artifact)

    def test_nonfinite_derived_coordinates_are_rejected(self):
        self.geometry["pixel_spacing_mm"] = 1e308
        for frame in self.geometry["slices"]:
            frame["pixel_spacing_mm"] = [1e308, 1e308]
        with self.assertRaises(ContractError):
            self.build()


if __name__ == "__main__":
    unittest.main()
