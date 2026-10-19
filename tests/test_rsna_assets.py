import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mri_root_impact.rsna_assets import RSNAAssetsError, verify_rsna_assets
from mri_root_impact.rsna_probe import MODEL_CLASSES, extract_model_definitions, probe


class RSNAAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.weights = self.root / "weights"
        self.weights.mkdir()
        (self.source / "model.py").write_text("raise RuntimeError('must not be imported')\n")
        (self.source / "LICENSE").write_text("Synthetic test license\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "synthetic")
        checkpoint = self.weights / "nested/model.pth"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic checkpoint bytes, not a pickle")
        self.inventory = {"schema_version": 1, "kind": "rsna_asset_inventory",
                          "repository": "synthetic", "commit": self.git("rev-parse", "HEAD").strip(),
                          "source_files": {"model.py": self.info(self.source / "model.py"),
                                           "LICENSE": self.info(self.source / "LICENSE")},
                          "weights": {"classifier": {"filename": "nested/model.pth", **self.info(checkpoint)}}}
        self.inventory_path = self.root / "inventory.json"
        self.save()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    @staticmethod
    def info(path):
        content = path.read_bytes()
        return {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def save(self):
        self.inventory_path.write_text(json.dumps(self.inventory))

    def verify(self):
        return verify_rsna_assets(self.source, self.weights, self.inventory_path)

    def test_verifies_bytes_without_import_or_unpickle(self):
        report = self.verify()
        self.assertEqual(report["source_commit"], self.inventory["commit"])
        self.assertEqual(report["weights"]["classifier"]["sha256"], self.inventory["weights"]["classifier"]["sha256"])

    def test_index_assume_unchanged_cannot_hide_tampering(self):
        self.git("update-index", "--assume-unchanged", "model.py")
        (self.source / "model.py").write_text("# changed\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_unlisted_tracked_python_is_also_verified(self):
        extra = self.source / "other.py"
        extra.write_text("# pinned\n")
        self.git("add", "other.py")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "extra")
        self.inventory["commit"] = self.git("rev-parse", "HEAD").strip()
        self.save()
        self.verify()
        extra.write_text("# altered\n")
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_untracked_python_is_rejected(self):
        (self.source / "extra.py").write_text("# unexpected\n")
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_cached_bytecode_is_never_used(self):
        (self.source / "__pycache__").mkdir()
        (self.source / "__pycache__/model.cpython-312.pyc").write_bytes(b"invalid bytecode")
        self.verify()

    def test_wrong_commit_is_rejected(self):
        self.inventory["commit"] = "0" * 40
        self.save()
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_bad_source_inventory_hash_is_rejected(self):
        self.inventory["source_files"]["LICENSE"]["sha256"] = "0" * 64
        self.save()
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_checkpoint_hash_and_size_are_both_required(self):
        path = self.weights / "nested/model.pth"
        original = path.read_bytes()
        for replacement in (b"x" * len(original), original[:-1]):
            with self.subTest(length=len(replacement)):
                path.write_bytes(replacement)
                with self.assertRaises(RSNAAssetsError):
                    self.verify()

    def test_missing_checkpoint_and_null_hash_are_rejected(self):
        self.inventory["weights"]["classifier"]["sha256"] = None
        self.save()
        with self.assertRaises(RSNAAssetsError):
            self.verify()
        (self.weights / "nested/model.pth").unlink()
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_traversal_paths_are_rejected(self):
        for name in ("../model.pth", "/tmp/model.pth", "nested/../model.pth", "nested//model.pth", "nested\\model.pth"):
            with self.subTest(name=name):
                self.inventory["weights"]["classifier"]["filename"] = name
                self.save()
                with self.assertRaises(RSNAAssetsError):
                    self.verify()

    def test_checkpoint_symlink_is_rejected(self):
        path = self.weights / "nested/model.pth"
        target = self.root / "same-bytes.pth"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_directory_symlink_is_rejected(self):
        folder = self.weights / "nested"
        target = self.root / "elsewhere"
        folder.rename(target)
        folder.symlink_to(target, target_is_directory=True)
        with self.assertRaises(RSNAAssetsError):
            self.verify()

    def test_probe_verifies_before_runtime_or_model_import(self):
        with patch("mri_root_impact.rsna_probe.verify_rsna_assets", side_effect=RSNAAssetsError("invalid")), \
                patch("mri_root_impact.rsna_probe.runtime_versions") as runtime:
            with self.assertRaises(RSNAAssetsError):
                probe(self.source, self.weights)
            runtime.assert_not_called()


class ModelExtractionTests(unittest.TestCase):
    @staticmethod
    def definitions():
        return "\n".join("class {}:\n    pass\n".format(name) for name in MODEL_CLASSES)

    def test_only_explicit_classes_are_extracted(self):
        text = "raise RuntimeError('top-level must never execute')\n" + self.definitions()
        text += "\nclass Other:\n    raise RuntimeError('not selected')\n"
        module = extract_model_definitions(text)
        self.assertTrue(all(isinstance(node, ast.ClassDef) for node in module.body))
        namespace = {}
        exec(compile(module, "synthetic", "exec"), namespace)
        self.assertNotIn("Other", namespace)
        self.assertTrue(all(name in namespace for name in MODEL_CLASSES))

    def test_missing_or_duplicate_model_definition_is_rejected(self):
        for text in (self.definitions().replace("RSNA2024KeypointNetTest", "Absent"),
                     self.definitions() + "\nclass GeM:\n    pass\n"):
            with self.assertRaises(ValueError):
                extract_model_definitions(text)


if __name__ == "__main__":
    unittest.main()
