"""Generated DICOM inputs for RSNA; no model or real study is opened."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

from mri_root_impact.rsna_dicom import frame_metadata, load_frames


HAS_DICOM = (importlib.util.find_spec("numpy") is not None
             and importlib.util.find_spec("pydicom") is not None)


@unittest.skipUnless(HAS_DICOM, "numpy/pydicom are optional RSNA dependencies")
class RsnaDicomTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

        self.np = np
        temporary = tempfile.TemporaryDirectory(prefix="mri-rsna-generated-dicom-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = {"schema_version": "1.0", "kind": "study_manifest",
                      "study_id": "synthetic-rsna-input", "synthetic": True,
                      "limitations": ["Generated test DICOM; no patient data."], "series": []}
        self.pairs = []
        study_uid = generate_uid()
        for sequence, positions in (("sagittal_t1", [0.0, 10.0, 5.0]),
                                    ("axial_t2", [25.0])):
            series = {"series_id": "series-" + sequence, "sequence": sequence, "images": []}
            series_uid, frame_uid = generate_uid(), generate_uid()
            self.study["series"].append(series)
            for index, position in enumerate(positions):
                image_id = sequence + "-" + str(index)
                path = self.root / (image_id + ".dcm")
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = generate_uid()
                dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                dataset.SOPClassUID = meta.MediaStorageSOPClassUID
                dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                dataset.StudyInstanceUID, dataset.SeriesInstanceUID = study_uid, series_uid
                dataset.FrameOfReferenceUID = frame_uid
                dataset.Modality = "MR"
                dataset.Rows, dataset.Columns, dataset.SamplesPerPixel = 4, 5, 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated, dataset.BitsStored = 16, 16
                dataset.HighBit, dataset.PixelRepresentation = 15, 0
                dataset.InstanceNumber = index + 1
                dataset.PixelSpacing = [0.7, 1.2]
                if sequence.startswith("sagittal"):
                    dataset.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
                    dataset.ImagePositionPatient = [position, -30, 80]
                else:
                    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                    dataset.ImagePositionPatient = [-40, -30, position]
                dataset.PixelData = np.arange(20, dtype=np.uint16).reshape(4, 5).tobytes()
                image = {"image_id": image_id, "path": str(path), "frame_count": 1, "sha256": None}
                series["images"].append(image)
                self.pairs.append((dataset, image))
        self.save()

    def save(self):
        for dataset, image in self.pairs:
            path = Path(image["path"])
            dataset.save_as(path, enforce_file_format=True)
            image["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_physical_sagittal_order_and_per_frame_geometry_are_preserved(self):
        before = copy.deepcopy(self.study)
        bytes_before = {image["path"]: Path(image["path"]).read_bytes()
                        for _, image in self.pairs}
        frames, by_sequence = load_frames(self.study)
        sagittal = by_sequence["sagittal_t1"]
        self.assertEqual([f["image_ref"]["image_id"] for f in sagittal],
                         ["sagittal_t1-0", "sagittal_t1-2", "sagittal_t1-1"])
        self.assertEqual([f["instance_number"] for f in sagittal], [1, 3, 2])
        self.assertEqual([f["position"] for f in sagittal],
                         [[0.0, -30.0, 80.0], [5.0, -30.0, 80.0], [10.0, -30.0, 80.0]])
        self.assertEqual(by_sequence["sagittal_t2"], [])
        self.assertEqual(len(frames), 4)
        expected_frames = {image["image_id"]: str(dataset.FrameOfReferenceUID)
                           for dataset, image in self.pairs}
        self.assertEqual(len(set(expected_frames.values())), 2)
        for f in frames:
            self.assertEqual(f["spacing"], [0.7, 1.2])
            self.assertEqual(f["frame_of_reference_uid"], expected_frames[f["image_ref"]["image_id"]])
            self.assertEqual(f["image_ref"]["frame_index"], 0)
            self.assertEqual(f["pixels"].shape, (4, 5))
            self.assertEqual(f["pixels"].dtype, self.np.uint8)
            self.assertEqual(f["modality_status"], "dicom_declared")
            metadata = frame_metadata(f)
            self.assertEqual(metadata["sha256"], f["sha256"])
            self.assertNotIn("pixels", metadata)
            self.assertNotIn("path", metadata)
        self.assertEqual(self.study, before)
        self.assertEqual(bytes_before, {path: Path(path).read_bytes() for path in bytes_before})

    def test_missing_or_wrong_sha_is_rejected_before_dicom_decoder(self):
        image = self.pairs[0][1]
        for checksum in ("0" * 64, None):
            with self.subTest(checksum=checksum), patch("pydicom.dcmread") as decoder:
                image["sha256"] = checksum
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_frames(self.study)
                decoder.assert_not_called()

    def test_missing_modality_needs_flag_and_ct_is_rejected_even_with_flag(self):
        for dataset, _ in self.pairs:
            del dataset.Modality
        self.save()
        with self.assertRaisesRegex(ValueError, "must be MR"):
            load_frames(self.study)
        frames, _ = load_frames(self.study, allow_missing_modality=True)
        self.assertTrue(all(f["modality_status"] == "dataset_declared" for f in frames))
        self.pairs[0][0].Modality = "CT"
        self.save()
        with self.assertRaisesRegex(ValueError, "must be MR"):
            load_frames(self.study, allow_missing_modality=True)

    def test_inverse_grayscale_requires_explicit_polarity_conversion(self):
        self.pairs[0][0].PhotometricInterpretation = "MONOCHROME1"
        self.save()
        with self.assertRaisesRegex(ValueError, "MONOCHROME2"):
            load_frames(self.study)

    def test_cross_study_uid_is_rejected_without_exposing_the_uid(self):
        self.pairs[-1][0].StudyInstanceUID = "1.2.999.123"
        self.save()
        with self.assertRaisesRegex(ValueError, "different DICOM studies") as caught:
            load_frames(self.study)
        self.assertNotIn("1.2.999.123", str(caught.exception))

    def test_single_frame_requirement_checks_manifest_and_actual_dicom(self):
        dataset, image = self.pairs[0]
        image["frame_count"] = 2
        with patch("pydicom.dcmread") as decoder:
            with self.assertRaisesRegex(ValueError, "single-frame"):
                load_frames(self.study)
            decoder.assert_not_called()
        image["frame_count"] = 1
        dataset.NumberOfFrames = 2
        dataset.PixelData = self.np.arange(40, dtype=self.np.uint16).reshape(2, 4, 5).tobytes()
        self.save()
        with self.assertRaisesRegex(ValueError, "single-frame"):
            load_frames(self.study)

    def test_decoded_pixels_must_match_declared_rows_and_columns(self):
        from pydicom.dataset import Dataset
        with patch.object(Dataset, "pixel_array", new_callable=PropertyMock,
                          return_value=self.np.ones((3, 5), dtype=self.np.uint16)):
            with self.assertRaisesRegex(ValueError, "declared matrix"):
                load_frames(self.study)

    def test_one_manifest_series_cannot_combine_different_series_uids(self):
        self.pairs[1][0].SeriesInstanceUID = "1.2.999.456"
        self.save()
        with self.assertRaisesRegex(ValueError, "different DICOM SeriesInstanceUIDs") as caught:
            load_frames(self.study)
        self.assertNotIn("1.2.999.456", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
