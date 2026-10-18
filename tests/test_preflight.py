"""Input verification is distinct from model inference and clinical assessment."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mri_root_impact.preflight import _dependencies, verify_study


HAS_DICOM = (importlib.util.find_spec("pydicom") is not None
             and importlib.util.find_spec("numpy") is not None)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mri-input-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = {
            "schema_version": "1.0", "kind": "study_manifest", "study_id": "synthetic-input",
            "synthetic": True, "series": [], "limitations": ["Synthetic test input."],
        }
        self.metadata = {}
        for index, sequence in enumerate(("sagittal_t2", "axial_t2")):
            path = self.root / (str(index) + ".dcm")
            path.write_bytes(("synthetic file " + str(index)).encode("ascii"))
            image = {
                "image_id": "image-" + str(index), "path": str(path), "frame_count": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            self.study["series"].append({"series_id": "series-" + str(index), "sequence": sequence, "images": [image]})
            self.metadata[path] = {
                "modality_verified": True,
                "study_uid": "1.2.3", "series_uid": "1.2.3." + str(index), "frame_uid": "1.2.4",
                "frames": 1, "matrix": (2, 3), "orientations": [(1, 0, 0, 0, 1, 0)],
                "positions": [(0, 0, index)], "spacings": [(1, 1)],
            }

    def verify(self):
        def inspect(path, frames, allow_missing_modality=False):
            metadata = self.metadata[path]
            self.assertEqual(frames, metadata["frames"])
            return metadata

        with patch("mri_root_impact.preflight._inspect_image", side_effect=inspect):
            return verify_study(self.study)

    def test_valid_input_is_hashed_and_reported_without_findings_or_sensitive_metadata(self):
        before = copy.deepcopy(self.study)
        files_before = {path: path.read_bytes() for path in self.metadata}
        report = self.verify()
        self.assertEqual(report["kind"], "input_preflight")
        self.assertEqual(report["files_verified"], 2)
        self.assertEqual(report["files_decoded"], 2)
        self.assertEqual(report["missing_sequences"], ["sagittal_t1"])
        self.assertTrue(report["localization_candidate"])
        self.assertFalse(report["full_stage_i_input"])
        self.assertFalse(report["inference_performed"])
        self.assertEqual(report["registration_status"], "consistent_frame_of_reference")
        self.assertEqual(report["series"][0]["geometry_status"], "consistent")
        self.assertEqual(self.study, before)
        self.assertEqual(files_before, {path: path.read_bytes() for path in self.metadata})
        serialized = json.dumps(report)
        for forbidden in ("1.2.3", "1.2.4", str(self.root), "PatientID", "root_impacts", "assessments"):
            self.assertNotIn(forbidden, serialized)

    def test_checksum_required_and_mismatch_rejected_before_decoding(self):
        for checksum in (None, "0" * 64):
            with self.subTest(checksum=checksum):
                self.study["series"][0]["images"][0]["sha256"] = checksum
                with patch("mri_root_impact.preflight._inspect_image") as reader:
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        verify_study(self.study)
                    reader.assert_not_called()

    def test_missing_file_rejected_without_leaking_path(self):
        self.study["series"][0]["images"][0]["path"] = str(self.root / "missing.dcm")
        with self.assertRaisesRegex(ValueError, "missing or unreadable") as caught:
            self.verify()
        self.assertNotIn(str(self.root), str(caught.exception))

    def test_contract_checked_before_any_file_reads(self):
        self.study["series"][0]["images"][0]["frame_count"] = 0
        with patch("mri_root_impact.preflight._inspect_image") as reader:
            with self.assertRaises(ValueError):
                verify_study(self.study)
            reader.assert_not_called()

    def test_empty_manifest_cannot_pass_preflight_even_with_declared_limitations(self):
        self.study["series"] = []
        with patch("mri_root_impact.preflight._inspect_image") as reader:
            with self.assertRaisesRegex(ValueError, "at least one real image series"):
                verify_study(self.study)
            reader.assert_not_called()

    def test_source_limitations_are_preserved_alongside_preflight_limitations(self):
        self.study["limitations"] = ["Control source has no herniation morphology reference."]
        before = list(self.study["limitations"])
        report = self.verify()
        self.assertEqual(report["limitations"][:len(before)], before)
        self.assertGreater(len(report["limitations"]), len(before))
        self.assertEqual(self.study["limitations"], before)

    def test_different_actual_studies_rejected(self):
        list(self.metadata.values())[1]["study_uid"] = "1.2.999"
        with self.assertRaisesRegex(ValueError, "different DICOM studies") as caught:
            self.verify()
        self.assertNotIn("1.2.999", str(caught.exception))

    def test_different_actual_series_within_manifest_series_rejected(self):
        second = self.study["series"].pop()
        self.study["series"][0]["images"].extend(second["images"])
        with self.assertRaisesRegex(ValueError, "different DICOM SeriesInstanceUIDs"):
            self.verify()

    def test_same_actual_series_cannot_be_declared_twice(self):
        list(self.metadata.values())[1]["series_uid"] = "1.2.3.0"
        with self.assertRaisesRegex(ValueError, "declared more than once"):
            self.verify()

    def test_missing_or_different_frame_of_reference_requires_review(self):
        for value in (None, "1.2.888"):
            with self.subTest(frame_reference=value):
                list(self.metadata.values())[1]["frame_uid"] = value
                self.assertEqual(self.verify()["registration_status"], "needs_review")

    def test_missing_geometry_requires_review(self):
        list(self.metadata.values())[0]["positions"] = [None]
        report = self.verify()
        self.assertEqual(report["series"][0]["geometry_status"], "needs_review")
        self.assertEqual(report["registration_status"], "needs_review")

    def test_variable_orientation_is_reported_without_raw_geometry(self):
        image = self.study["series"][0]["images"][0]
        image["frame_count"] = 2
        metadata = self.metadata[Path(image["path"])]
        metadata.update(frames=2, orientations=[(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 1)],
                        positions=[(0, 0, 0), (0, 0, 1)], spacings=[(1, 1), (1, 1)])
        report = self.verify()
        self.assertEqual(report["series"][0]["unique_orientations"], 2)
        self.assertEqual(report["series"][0]["geometry_status"], "needs_review")
        self.assertNotIn("positions", json.dumps(report))

    def test_unknown_sequence_is_not_a_negative_assessment_or_localization_input(self):
        for series in self.study["series"]:
            series["sequence"] = "unknown"
        report = self.verify()
        self.assertFalse(report["localization_candidate"])
        self.assertEqual(set(report["missing_sequences"]), {"sagittal_t1", "sagittal_t2", "axial_t2"})
        self.assertFalse(report["full_stage_i_input"])

    def test_optional_dependency_error_explains_how_to_install(self):
        real_import = __import__

        def without_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("simulated missing optional dependency")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_numpy):
            with self.assertRaisesRegex(ValueError, "requirements-data.txt"):
                _dependencies()


@unittest.skipUnless(HAS_DICOM, "optional DICOM dependencies are not installed")
class RealDicomPreflightTests(unittest.TestCase):
    def setUp(self):
        import numpy
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

        self.numpy = numpy
        self.temp = tempfile.TemporaryDirectory(prefix="mri-dicom-decode-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "generated.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        self.dataset = FileDataset(str(self.path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds = self.dataset
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = generate_uid(), generate_uid(), generate_uid()
        ds.Modality, ds.PatientID = "MR", "synthetic-patient-not-for-report"
        ds.Rows, ds.Columns, ds.SamplesPerPixel = 2, 3, 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
        ds.ImagePositionPatient, ds.PixelSpacing = [12.5, -90.3, 123.4], [0.7, 0.7]
        ds.PixelData = numpy.arange(6, dtype=numpy.uint16).tobytes()
        self.study = {
            "schema_version": "1.0", "kind": "study_manifest", "study_id": "generated-dicom",
            "synthetic": True, "limitations": ["Generated DICOM, no patient data."],
            "series": [{"series_id": "sag-t2", "sequence": "sagittal_t2", "images": [{
                "image_id": "image-1", "path": str(self.path), "frame_count": 1, "sha256": None,
            }]}],
        }

    def save(self):
        self.dataset.save_as(self.path, enforce_file_format=True)
        self.study["series"][0]["images"][0]["sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_actual_decoder_reads_generated_pixels_and_redacts_dicom_identity(self):
        self.save()
        report = verify_study(self.study)
        self.assertEqual(report["files_decoded"], 1)
        self.assertEqual(report["series"][0]["frames"], 1)
        self.assertEqual(report["series"][0]["geometry_status"], "consistent")
        serialized = json.dumps(report)
        for hidden in (str(self.dataset.StudyInstanceUID), str(self.dataset.SeriesInstanceUID),
                       self.dataset.PatientID, "123.4", str(self.path)):
            self.assertNotIn(hidden, serialized)

    def test_missing_number_of_frames_cannot_satisfy_multiframe_manifest(self):
        self.save()
        self.study["series"][0]["images"][0]["frame_count"] = 2
        with self.assertRaisesRegex(ValueError, "NumberOfFrames"):
            verify_study(self.study)

    def test_bad_pixel_length_is_rejected_by_actual_decoder(self):
        self.dataset.PixelData = b"\0\0"
        self.save()
        with self.assertRaisesRegex(ValueError, "decode DICOM pixels"):
            verify_study(self.study)

    def test_nonfinite_float_pixels_are_rejected(self):
        del self.dataset.PixelData
        self.dataset.BitsAllocated = 32
        self.dataset.FloatPixelData = self.numpy.array([0, 1, 2, 3, 4, float("nan")], dtype=self.numpy.float32).tobytes()
        self.save()
        with self.assertRaisesRegex(ValueError, "finite"):
            verify_study(self.study)

    def test_required_uid_is_not_silently_replaced_with_manifest_id(self):
        del self.dataset.StudyInstanceUID
        self.save()
        with self.assertRaisesRegex(ValueError, "StudyInstanceUID"):
            verify_study(self.study)

    def test_missing_frame_of_reference_is_reviewable_not_a_decode_failure(self):
        del self.dataset.FrameOfReferenceUID
        self.save()
        report = verify_study(self.study)
        self.assertEqual(report["files_decoded"], 1)
        self.assertEqual(report["registration_status"], "needs_review")

    def test_non_mr_input_rejected(self):
        self.dataset.Modality = "CT"
        self.save()
        with self.assertRaisesRegex(ValueError, "must be MR"):
            verify_study(self.study)
        with self.assertRaisesRegex(ValueError, "must be MR"):
            verify_study(self.study, allow_missing_modality=True)

    def test_missing_modality_requires_explicit_policy_and_remains_declared(self):
        del self.dataset.Modality
        self.save()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "must be MR"):
            verify_study(self.study)
        report = verify_study(self.study, allow_missing_modality=True)
        self.assertEqual(report["files_decoded"], 1)
        self.assertEqual(report["missing_modality_files"], 1)
        self.assertEqual(report["modality_status"], "dataset_declared")
        self.assertTrue(report["allow_missing_modality"])
        self.assertFalse(report["full_stage_i_input"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_enhanced_multiframe_geometry_and_pixels_are_read(self):
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence

        self.dataset.NumberOfFrames = 2
        self.dataset.PixelData = self.numpy.arange(12, dtype=self.numpy.uint16).tobytes()
        shared = Dataset()
        orientation, measures = Dataset(), Dataset()
        orientation.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
        measures.PixelSpacing = [0.7, 0.7]
        shared.PlaneOrientationSequence = Sequence([orientation])
        shared.PixelMeasuresSequence = Sequence([measures])
        self.dataset.SharedFunctionalGroupsSequence = Sequence([shared])
        groups = []
        for index in range(2):
            group, position = Dataset(), Dataset()
            position.ImagePositionPatient = [12.5 + index, -90.3, 123.4]
            group.PlanePositionSequence = Sequence([position])
            groups.append(group)
        self.dataset.PerFrameFunctionalGroupsSequence = Sequence(groups)
        self.study["series"][0]["images"][0]["frame_count"] = 2
        self.save()
        report = verify_study(self.study)
        self.assertEqual(report["series"][0]["frames"], 2)
        self.assertEqual(report["series"][0]["unique_orientations"], 1)
        self.assertEqual(report["series"][0]["geometry_status"], "consistent")


if __name__ == "__main__":
    unittest.main()
