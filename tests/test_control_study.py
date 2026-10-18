"""Prepare and replay external control metadata without changing source files."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mri_root_impact.__main__ import main
from mri_root_impact.control_study import prepare_rsna_study, write_json_external


class ControlStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mri-control-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory_path = self.root / "provenance.json"
        self.inventory = {
            "study_id": "1234", "repo_id": "synthetic/control", "revision": "a" * 40,
            "series": [], "files": [],
        }
        for index, description in enumerate(("Sagittal T2/STIR", "Sagittal T1", "Axial T2")):
            native_id = str(100 + index)
            self.inventory["series"].append({"series_id": native_id, "series_description": description})
            path = self.root / "train_images" / "1234" / native_id / "1.dcm"
            path.parent.mkdir(parents=True)
            path.write_bytes(("synthetic DICOM placeholder " + native_id).encode("ascii"))
            self.inventory["files"].append({
                "relative_path": str(path.relative_to(self.root)), "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        self.save_inventory(self.inventory)

    def save_inventory(self, value):
        self.inventory_path.write_text(json.dumps(value), encoding="utf-8")

    def prepare_without_decoding(self):
        with patch("mri_root_impact.control_study._frame_count", return_value=1):
            return prepare_rsna_study(self.root)

    def test_inventory_builds_reproducible_manifest_and_detects_changed_source(self):
        originals = {self.root / item["relative_path"]: (self.root / item["relative_path"]).read_bytes()
                     for item in self.inventory["files"]}
        study = self.prepare_without_decoding()
        self.assertEqual(study, self.prepare_without_decoding())
        self.assertEqual({item["sequence"] for item in study["series"]}, {"sagittal_t1", "sagittal_t2", "axial_t2"})
        self.assertEqual(len(study["series"]), 3)
        self.assertFalse(study["synthetic"])
        self.assertTrue(study["limitations"])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        changed = next(iter(originals))
        changed.write_bytes(b"changed")
        with patch("mri_root_impact.control_study._frame_count") as reader:
            with self.assertRaisesRegex(ValueError, "differs from pinned download inventory"):
                prepare_rsna_study(self.root)
            reader.assert_not_called()

    def test_json_writer_preserves_source_bytes_and_unrelated_metadata(self):
        report = {"kind": "input_preflight", "study_id": "rsna-1234", "files_decoded": 3}
        source = self.root / self.inventory["files"][0]["relative_path"]
        other = self.root / "other-study.json"
        other.write_text(json.dumps({"kind": "input_preflight", "study_id": "rsna-other"}))
        for target in (source, self.inventory_path, other):
            with self.subTest(target=target.name):
                before = target.read_bytes()
                with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                    write_json_external(report, target)
                self.assertEqual(target.read_bytes(), before)

    def test_report_replay_replaces_only_matching_kind_and_study(self):
        target = self.root / "outputs" / "preflight.json"
        first = {"kind": "input_preflight", "study_id": "rsna-1234", "files_decoded": 1}
        second = dict(first, files_decoded=3)
        self.assertEqual(write_json_external(first, target), target.resolve())
        write_json_external(second, target)
        self.assertEqual(json.loads(target.read_text()), second)
        self.assertEqual(list(target.parent.iterdir()), [target])

    def test_cli_rejects_input_image_and_manifest_output_before_decoder(self):
        study = self.prepare_without_decoding()
        manifest = self.root / "study.json"
        manifest.write_text(json.dumps(study), encoding="utf-8")
        image = Path(study["series"][0]["images"][0]["path"])
        alias = self.root / "image-alias.json"
        alias.symlink_to(image)
        for output in (image, alias, manifest):
            with self.subTest(output=output.name):
                original = image.read_bytes(), manifest.read_bytes()
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", ["mri_root_impact", "preflight", "--study", str(manifest), "--output", str(output)]):
                    with patch("mri_root_impact.preflight.verify_study") as decoder:
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            status = main()
                        decoder.assert_not_called()
                self.assertEqual(status, 2)
                self.assertEqual(stdout.getvalue(), "")
                error = json.loads(stderr.getvalue())
                self.assertFalse(error["ok"])
                self.assertIn("must not overwrite", error["error"])
                self.assertEqual((image.read_bytes(), manifest.read_bytes()), original)

    def test_malformed_inventory_objects_raise_value_error_without_attribute_errors(self):
        malformed = ([], None, dict(self.inventory, series=[None]))
        for inventory in malformed:
            with self.subTest(inventory=inventory):
                self.save_inventory(inventory)
                with self.assertRaises(ValueError):
                    prepare_rsna_study(self.root)


if __name__ == "__main__":
    unittest.main()
