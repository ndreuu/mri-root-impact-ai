"""Synthetic geometry checks for RSNA crops; no DICOM or model is loaded."""

import copy
import math
import unittest

from mri_root_impact.rsna_geometry import (
    box_corners_lps,
    crop_box,
    level_segments,
    lps_to_pixel,
    pixel_to_lps,
    segment_plane_intersection,
)


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")


def frame(position=None, orientation=None, spacing=None, rows=200, columns=160):
    return {
        "position": [10.0, -20.0, 30.0] if position is None else position,
        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] if orientation is None else orientation,
        "spacing": [2.0, 0.5] if spacing is None else spacing,
        "rows": rows,
        "columns": columns,
    }


def level_points():
    return {level: [5.0 * (index + 1), 2.0 * (index + 1), 100.0 - 20.0 * index]
            for index, level in enumerate(LEVELS)}


class PixelGeometryTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=10)

    def test_anisotropic_spacing_uses_columns_for_x_and_rows_for_y(self):
        geometry = frame()
        point = pixel_to_lps(geometry, 4.0, 3.0)
        self.assertVectorAlmostEqual(point, [12.0, -14.0, 30.0])
        self.assertVectorAlmostEqual(lps_to_pixel(geometry, point), [4.0, 3.0, 0.0])

    def test_oblique_roundtrip_preserves_fractional_pixel_coordinates(self):
        sine, cosine = math.sin(math.pi / 6), math.cos(math.pi / 6)
        diagonal = 1.0 / math.sqrt(2.0)
        geometry = frame(
            orientation=[diagonal, diagonal, 0.0,
                         -cosine * diagonal, cosine * diagonal, sine],
            spacing=[1.6, 0.7],
        )
        x, y = 12.25, 8.75
        point = pixel_to_lps(geometry, x, y)
        expected = [
            10.0 + x * 0.7 * diagonal - y * 1.6 * cosine * diagonal,
            -20.0 + x * 0.7 * diagonal + y * 1.6 * cosine * diagonal,
            30.0 + y * 1.6 * sine,
        ]
        self.assertVectorAlmostEqual(point, expected)
        self.assertVectorAlmostEqual(lps_to_pixel(geometry, point), [x, y, 0.0])

        normal = [sine * diagonal, -sine * diagonal, cosine]
        off_plane = [value + 7.25 * direction for value, direction in zip(point, normal)]
        self.assertVectorAlmostEqual(lps_to_pixel(geometry, off_plane), [x, y, 7.25])

    def test_plane_distance_sign_tracks_frame_normal(self):
        geometry = frame(position=[0.0, 0.0, 10.0])
        self.assertVectorAlmostEqual(lps_to_pixel(geometry, [2.0, 4.0, 13.0]), [4.0, 2.0, 3.0])
        self.assertVectorAlmostEqual(lps_to_pixel(geometry, [2.0, 4.0, 7.0]), [4.0, 2.0, -3.0])


class SegmentSelectionTests(unittest.TestCase):
    def test_axial_group_rejects_intersection_outside_finite_source_image(self):
        from mri_root_impact.rsna_pipeline import _group_axial
        points = level_points()
        within = frame(position=[0.0, 0.0, 60.0], spacing=[1.0, 1.0])
        outside = frame(position=[500.0, 0.0, 60.0], spacing=[1.0, 1.0])
        within["image_ref"] = {"series_id": "ax", "image_id": "within", "frame_index": 0}
        outside["image_ref"] = {"series_id": "ax", "image_id": "outside", "frame_index": 0}

        selected, details = _group_axial([outside, within], level_segments(points), points)

        self.assertEqual(selected["L3-L4"], [within])
        self.assertEqual(details["outside_field_of_view_frames"], [
            {"image_ref": outside["image_ref"], "level": "L3-L4"},
        ])

    def test_each_frame_orientation_changes_segment_selection(self):
        # Same ImagePositionPatient, different planes: a stack-wide affine is wrong.
        axial = frame(position=[5.0, 0.0, 0.0])
        sagittal = frame(position=[5.0, 0.0, 0.0],
                         orientation=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        upper, lower = [0.0, 0.0, 10.0], [0.0, 0.0, -10.0]
        self.assertTrue(segment_plane_intersection(axial, upper, lower))
        self.assertFalse(segment_plane_intersection(sagittal, upper, lower))
        self.assertFalse(segment_plane_intersection(frame(position=[0.0, 0.0, 20.0]), upper, lower))

    def test_parallel_coplanar_and_reversed_segments(self):
        geometry = frame(position=[0.0, 0.0, 0.0])
        self.assertFalse(segment_plane_intersection(geometry, [0.0, 0.0, 3.0], [10.0, 0.0, 3.0]))
        self.assertTrue(segment_plane_intersection(geometry, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]))
        self.assertTrue(segment_plane_intersection(geometry, [0.0, 0.0, -2.0], [0.0, 0.0, 2.0]))
        self.assertTrue(segment_plane_intersection(geometry, [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]))

    def test_shared_endpoint_matches_both_segments_so_caller_can_flag_ambiguity(self):
        geometry = frame(position=[0.0, 0.0, 0.0])
        endpoint = [0.0, 0.0, 0.0]
        self.assertTrue(segment_plane_intersection(geometry, [0.0, 0.0, 10.0], endpoint))
        self.assertTrue(segment_plane_intersection(geometry, endpoint, [0.0, 0.0, -10.0]))

    def test_plane_tolerance_is_in_millimeters(self):
        geometry = frame(position=[0.0, 0.0, 0.0], spacing=[3.0, 0.3])
        self.assertTrue(segment_plane_intersection(
            geometry, [0.0, 0.0, 0.5e-6], [5.0, 0.0, 0.5e-6], tolerance=1e-6))
        self.assertFalse(segment_plane_intersection(
            geometry, [0.0, 0.0, 2e-6], [5.0, 0.0, 2e-6], tolerance=1e-6))

    def test_endpoint_tolerance_does_not_grow_with_segment_length(self):
        geometry = frame(position=[0.0, 0.0, 0.0])
        for direction in (1.0, -1.0):
            with self.subTest(direction=direction):
                self.assertFalse(segment_plane_intersection(
                    geometry, [0.0, 0.0, direction * 2e-6],
                    [0.0, 0.0, direction * 20.0], tolerance=1e-6))
                self.assertTrue(segment_plane_intersection(
                    geometry, [0.0, 0.0, direction * 1e-6],
                    [0.0, 0.0, direction * 20.0], tolerance=1e-6))


class LevelSegmentTests(unittest.TestCase):
    def test_midpoint_boundaries_and_terminal_extrapolation_preserve_lps(self):
        points = level_points()
        before = copy.deepcopy(points)
        segments = level_segments(points)
        self.assertEqual(set(segments), set(LEVELS))
        self.assertEqual(segments["L1-L2"], [[2.5, 1.0, 110.0], [7.5, 3.0, 90.0]])
        self.assertEqual(segments["L3-L4"], [[12.5, 5.0, 70.0], [17.5, 7.0, 50.0]])
        self.assertEqual(segments["L5-S1"], [[22.5, 9.0, 30.0], [27.5, 11.0, 10.0]])
        for current, following in zip(LEVELS, LEVELS[1:]):
            self.assertEqual(segments[current][1], segments[following][0])
        self.assertEqual(points, before)

    def test_missing_or_extra_levels_cannot_create_invented_segments(self):
        missing = level_points()
        del missing["L3-L4"]
        extra = dict(level_points(), L6_S1=[0.0, 0.0, 0.0])
        for points in (missing, extra, {}, []):
            with self.subTest(points=points), self.assertRaises(ValueError):
                level_segments(points)

    def test_nonfinite_malformed_or_nondescending_points_are_rejected(self):
        for bad_point in ([1.0, 2.0, float("nan")], [1.0, float("inf"), 60.0],
                          [1.0, 2.0], [15.0, 6.0, 80.0], [15.0, 6.0, 90.0]):
            with self.subTest(point=bad_point):
                points = level_points()
                points["L3-L4"] = bad_point
                with self.assertRaises(ValueError):
                    level_segments(points)

    def test_minimum_five_millimeter_level_separation_is_inclusive(self):
        points = {level: [0.0, 0.0, 30.0 - 5.0 * index]
                  for index, level in enumerate(LEVELS)}
        self.assertEqual(set(level_segments(points)), set(LEVELS))
        points["L3-L4"][2] = points["L2-L3"][2] - 4.999
        with self.assertRaises(ValueError):
            level_segments(points)


class CropGeometryTests(unittest.TestCase):
    def test_half_open_box_world_corners_are_pixel_footprint_edges(self):
        # DICOM IPP locates a pixel center. A slice [top:bottom,left:right]
        # starts half a pixel before its first center and ends after its last.
        self.assertEqual(box_corners_lps(frame(), [3, 4, 13, 14]), [
            [11.25, -13.0, 30.0], [16.25, -13.0, 30.0],
            [16.25, 7.0, 30.0], [11.25, 7.0, 30.0],
        ])

    def test_mm_crop_and_offset_respect_anisotropic_axis_order(self):
        geometry = frame(columns=200)
        # [height,width] and [vertical,horizontal] are upstream millimeter order.
        self.assertEqual(crop_box(geometry, [0.5, 0.5], [40.0, 50.0], [-4.0, 2.0]),
                         [54, 88, 154, 108])

    def test_normalized_center_uses_columns_and_rows(self):
        self.assertEqual(crop_box(frame(), [0.25, 0.75], [20.0, 10.0], [0.0, 0.0]),
                         [30, 145, 50, 155])

    def test_outside_or_terminal_boundary_crop_is_rejected_without_padding(self):
        geometry = frame(columns=200)
        cases = (
            ([0.0, 0.5], [20.0, 10.0], [0.0, 0.0]),
            ([0.5, 0.0], [20.0, 10.0], [0.0, 0.0]),
            ([0.95, 0.5], [20.0, 10.0], [0.0, 0.0]),
            ([0.5, 0.975], [20.0, 10.0], [0.0, 0.0]),
            ([0.5, 0.5], [20.0, 10.0], [0.0, 100.0]),
        )
        for center, size, offset in cases:
            with self.subTest(center=center, size=size, offset=offset):
                with self.assertRaises(ValueError):
                    crop_box(geometry, center, size, offset)

    def test_zero_negative_or_rounded_empty_crop_is_not_a_valid_input(self):
        for size in ([0.0, 10.0], [-2.0, 10.0], [10.0, 0.0], [0.01, 0.01]):
            with self.subTest(size=size), self.assertRaises(ValueError):
                crop_box(frame(), [0.5, 0.5], size, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
