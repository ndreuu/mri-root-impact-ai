"""Native DICOM loading must preserve geometry and reject unsupported inputs."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mri_root_impact.sagittal import load_sagittal_series


HAS_DICOM = (importlib.util.find_spec("pydicom") is not None
             and importlib.util.find_spec("numpy") is not None)


class SagittalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.study = {
            "schema_version": "1.0", "kind": "study_manifest", "study_id": "synthetic-sagittal",
            "synthetic": True, "limitations": ["Synthetic test only."],
            "series": [{"series_id": "sag-t2", "sequence": "sagittal_t2", "images": [
                {"image_id": "image-0", "path": "/external/0.dcm", "frame_count": 1, "sha256": "a" * 64},
                {"image_id": "image-1", "path": "/external/1.dcm", "frame_count": 1, "sha256": "b" * 64},
            ]}],
        }

    def test_invalid_contract_fails_before_optional_import_or_file_reads(self):
        self.study["series"][0]["images"][0]["frame_count"] = 0
        with patch("mri_root_impact.sagittal._dependencies") as dependencies:
            with self.assertRaises(ValueError):
                load_sagittal_series(self.study)
            dependencies.assert_not_called()

    def test_modality_policy_requires_an_actual_boolean(self):
        for value in (None, 0, 1, "true", []):
            with self.subTest(value=value), patch("mri_root_impact.sagittal._dependencies") as dependencies:
                with self.assertRaisesRegex(ValueError, "explicit boolean"):
                    load_sagittal_series(self.study, allow_missing_modality=value)
                dependencies.assert_not_called()

    def test_ambiguous_sagittal_series_rejected_before_loading(self):
        duplicate = copy.deepcopy(self.study["series"][0])
        duplicate["series_id"] = "second-sag-t2"
        for image in duplicate["images"]:
            image["image_id"] += "-second"
        self.study["series"].append(duplicate)
        with patch("mri_root_impact.sagittal._dependencies") as dependencies:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_sagittal_series(self.study)
            dependencies.assert_not_called()

    def test_non_t2_or_unknown_identifier_cannot_select_other_sequences(self):
        self.study["series"][0]["sequence"] = "sagittal_t1"
        with patch("mri_root_impact.sagittal._dependencies") as dependencies:
            for identifier in (None, "sag-t2", "missing"):
                with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "exactly one"):
                    load_sagittal_series(self.study, series_id=identifier)
            dependencies.assert_not_called()

    def test_single_slice_does_not_invent_center_spacing_from_thickness(self):
        self.study["series"][0]["images"] = self.study["series"][0]["images"][:1]
        with patch("mri_root_impact.sagittal._dependencies") as dependencies:
            with self.assertRaisesRegex(ValueError, "At least two"):
                load_sagittal_series(self.study)
            dependencies.assert_not_called()


@unittest.skipUnless(HAS_DICOM, "optional DICOM dependencies are not installed")
class SagittalDicomTests(unittest.TestCase):
    def setUp(self):
        import numpy
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

        self.numpy = numpy
        self.temp = tempfile.TemporaryDirectory(prefix="mri-native-sagittal-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = {
            "schema_version": "1.0", "kind": "study_manifest", "study_id": "generated-sagittal",
            "synthetic": True, "limitations": ["Generated DICOM, no patient data."],
            "series": [{"series_id": "sag-t2", "sequence": "sagittal_t2", "images": []}],
        }
        self.datasets = []
        study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
        # Manifest order and InstanceNumber both disagree with physical order.
        for index, x in enumerate((0, 10, 5)):
            path = self.root / (str(index) + ".dcm")
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            dataset.SOPClassUID, dataset.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
            dataset.StudyInstanceUID, dataset.SeriesInstanceUID = study_uid, series_uid
            dataset.FrameOfReferenceUID = frame_uid
            dataset.Modality, dataset.PatientID = "MR", "synthetic-not-for-geometry"
            dataset.Rows, dataset.Columns, dataset.SamplesPerPixel = 2, 3, 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit, dataset.PixelRepresentation = 16, 16, 15, 0
            dataset.InstanceNumber = index + 1
            dataset.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
            dataset.ImagePositionPatient = [x, -90, 123]
            dataset.PixelSpacing, dataset.SliceThickness = [0.7, 0.7], 4
            dataset.PixelData = numpy.full((2, 3), index, dtype=numpy.uint16).tobytes()
            self.datasets.append(dataset)
            self.study["series"][0]["images"].append({
                "image_id": "image-" + str(index), "path": str(path), "frame_count": 1, "sha256": None,
            })
        self.save()

    def save(self):
        for dataset, image in zip(self.datasets, self.study["series"][0]["images"]):
            path = Path(image["path"])
            dataset.save_as(path, enforce_file_format=True)
            image["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_sorting_preserves_pixels_refs_and_thickness_distinct_from_center_spacing(self):
        before = copy.deepcopy(self.study)
        bytes_before = {image["path"]: Path(image["path"]).read_bytes()
                        for image in self.study["series"][0]["images"]}
        volume, geometry = load_sagittal_series(self.study)
        self.assertEqual(volume.shape, (2, 3, 3))
        self.assertEqual(volume.dtype, self.numpy.uint16)
        self.assertEqual(volume[0, 0, :].tolist(), [1, 2, 0])
        self.assertEqual(set(geometry), {"series_id", "volume_shape", "pixel_spacing_mm",
                                       "slice_center_spacing_mm", "coordinate_system", "slices"})
        self.assertEqual(geometry["volume_shape"], [2, 3, 3])
        self.assertEqual(geometry["pixel_spacing_mm"], .7)
        self.assertEqual(geometry["slice_center_spacing_mm"], 5)
        self.assertEqual(geometry["coordinate_system"], "DICOM_LPS_mm")
        self.assertEqual([item["image_ref"]["image_id"] for item in geometry["slices"]],
                         ["image-1", "image-2", "image-0"])
        for index, item in enumerate(geometry["slices"]):
            self.assertEqual(set(item), {"slice_index", "image_ref", "image_position_patient_mm",
                                         "image_orientation_patient", "pixel_spacing_mm", "slice_thickness_mm"})
            self.assertEqual(item["slice_index"], index)
            self.assertEqual(item["image_ref"]["frame_index"], 0)
            self.assertEqual(item["slice_thickness_mm"], 4)
            self.assertEqual(item["image_orientation_patient"], [0, 1, 0, 0, 0, -1])
        self.assertEqual(self.study, before)
        self.assertEqual(bytes_before, {path: Path(path).read_bytes() for path in bytes_before})

    def test_explicit_series_selection_ignores_other_declared_sagittal_t2(self):
        duplicate = copy.deepcopy(self.study["series"][0])
        duplicate["series_id"] = "second-sag-t2"
        for image in duplicate["images"]:
            image["image_id"] += "-second"
            image["path"] = str(self.root / "intentionally-missing.dcm")
        self.study["series"].append(duplicate)
        volume, geometry = load_sagittal_series(self.study, series_id="sag-t2")
        self.assertEqual(volume.shape[2], 3)
        self.assertEqual(geometry["series_id"], "sag-t2")

    def test_checksum_mismatch_rejected_before_decoder(self):
        self.study["series"][0]["images"][0]["sha256"] = "0" * 64
        with patch("pydicom.dcmread") as decoder:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_sagittal_series(self.study)
            decoder.assert_not_called()

    def test_missing_checksum_cannot_be_silently_accepted(self):
        self.study["series"][0]["images"][0]["sha256"] = None
        with self.assertRaisesRegex(ValueError, "requires SHA-256"):
            load_sagittal_series(self.study)

    def test_missing_modality_requires_explicit_policy_and_ct_is_always_rejected(self):
        for dataset in self.datasets:
            del dataset.Modality
        self.save()
        with self.assertRaisesRegex(ValueError, "must be MR"):
            load_sagittal_series(self.study)
        volume, _ = load_sagittal_series(self.study, allow_missing_modality=True)
        self.assertEqual(volume.shape[2], 3)
        self.datasets[0].Modality = "CT"
        self.save()
        with self.assertRaisesRegex(ValueError, "must be MR"):
            load_sagittal_series(self.study, allow_missing_modality=True)

    def test_actual_multiframe_and_manifest_multiframe_are_rejected(self):
        self.datasets[0].NumberOfFrames = 2
        self.datasets[0].PixelData = self.numpy.zeros((2, 2, 3), dtype=self.numpy.uint16).tobytes()
        self.save()
        with self.assertRaisesRegex(ValueError, "single-frame"):
            load_sagittal_series(self.study)
        self.study["series"][0]["images"][0]["frame_count"] = 2
        with self.assertRaisesRegex(ValueError, "single-frame"):
            load_sagittal_series(self.study)

    def test_anisotropic_spacing_is_not_averaged(self):
        self.datasets[0].PixelSpacing = [.7, .9]
        self.save()
        with self.assertRaisesRegex(ValueError, "Anisotropic"):
            load_sagittal_series(self.study)

    def test_inverse_grayscale_requires_an_explicit_polarity_transform(self):
        self.datasets[0].PhotometricInterpretation = "MONOCHROME1"
        self.save()
        with self.assertRaisesRegex(ValueError, "MONOCHROME2"):
            load_sagittal_series(self.study)

    def test_reverse_orientation_requires_resampling_instead_of_silent_flip(self):
        for dataset in self.datasets:
            dataset.ImageOrientationPatient = [0, -1, 0, 0, 0, -1]
        self.save()
        with self.assertRaisesRegex(ValueError, "near-standard sagittal"):
            load_sagittal_series(self.study)

    def test_non_sagittal_orientation_is_rejected(self):
        self.datasets[0].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        self.save()
        with self.assertRaisesRegex(ValueError, "near-standard sagittal"):
            load_sagittal_series(self.study)

    def test_uid_mismatch_rejected_without_exposing_uids(self):
        for name in ("StudyInstanceUID", "SeriesInstanceUID"):
            original = getattr(self.datasets[1], name)
            setattr(self.datasets[1], name, "1.2.999")
            self.save()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "different DICOM") as caught:
                load_sagittal_series(self.study)
            self.assertNotIn("1.2.999", str(caught.exception))
            setattr(self.datasets[1], name, original)

    def test_inconsistent_in_plane_spacing_is_rejected(self):
        self.datasets[0].PixelSpacing = [.8, .8]
        self.save()
        with self.assertRaisesRegex(ValueError, "must be consistent"):
            load_sagittal_series(self.study)

    def test_duplicate_positions_are_rejected(self):
        self.datasets[1].ImagePositionPatient = self.datasets[0].ImagePositionPatient
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            load_sagittal_series(self.study)

    def test_irregular_center_spacing_is_rejected(self):
        self.datasets[2].ImagePositionPatient = [4.99, -90, 123]
        self.save()
        with self.assertRaisesRegex(ValueError, "Irregular"):
            load_sagittal_series(self.study)

    def test_in_plane_origin_drift_is_not_hidden_by_projection(self):
        self.datasets[2].ImagePositionPatient = [5, -89.9, 123]
        self.save()
        with self.assertRaisesRegex(ValueError, "drift"):
            load_sagittal_series(self.study)

    def test_missing_and_nonfinite_geometry_is_rejected(self):
        originals = copy.deepcopy(self.datasets)
        for attribute, replacement in (("ImagePositionPatient", [float("nan"), -90, 123]),
                                       ("ImageOrientationPatient", [0, 1, 0, 0, 0, float("inf")]),
                                       ("SliceThickness", 0), ("PixelSpacing", [0, .7]),
                                       ("ImagePositionPatient", None)):
            self.datasets = copy.deepcopy(originals)
            if replacement is None:
                delattr(self.datasets[0], attribute)
            else:
                setattr(self.datasets[0], attribute, replacement)
            self.save()
            with self.subTest(attribute=attribute, replacement=replacement), self.assertRaises(ValueError):
                load_sagittal_series(self.study)

    def test_nonfinite_pixels_are_rejected(self):
        dataset = self.datasets[0]
        del dataset.PixelData
        dataset.BitsAllocated = 32
        dataset.FloatPixelData = self.numpy.array([0, 1, 2, 3, 4, float("nan")], dtype=self.numpy.float32).tobytes()
        self.save()
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            load_sagittal_series(self.study)


if __name__ == "__main__":
    unittest.main()
