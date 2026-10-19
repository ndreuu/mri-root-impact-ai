import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("rsna_asset_fetch", Path(__file__).resolve().parents[1] / "scripts/fetch_rsna_assets.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


class Response:
    status = 200

    def __init__(self, data, expected_length):
        self.data = data
        self.offset = 0
        self.headers = {"Content-Length": str(expected_length)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, length):
        chunk = self.data[self.offset:self.offset + length]
        self.offset += len(chunk)
        return chunk


class RSNAFetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"synthetic checkpoint content"
        self.expected = {
            "filename": "nested/model.pth", "size_bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "source_url": "https://www.kaggle.com/api/v1/datasets/download/akinosora/rsna2024-lsdc-3rd-models-pub/synthetic?datasetVersionNumber=2",
        }

    def test_download_is_atomic_and_offline_recheck_does_not_open_network(self):
        with patch.object(fetcher.urllib.request, "urlopen", return_value=Response(self.data, len(self.data))) as network:
            fetcher.fetch_one(self.root, self.expected)
            network.assert_called_once()
        self.assertEqual((self.root / "nested/model.pth").read_bytes(), self.data)
        with patch.object(fetcher.urllib.request, "urlopen") as network:
            fetcher.fetch_one(self.root, self.expected, verify_only=True)
            network.assert_not_called()

    def test_corrupt_existing_file_is_not_overwritten(self):
        target = self.root / "nested/model.pth"
        target.parent.mkdir()
        target.write_bytes(b"other bytes")
        with patch.object(fetcher.urllib.request, "urlopen") as network:
            with self.assertRaises(ValueError):
                fetcher.fetch_one(self.root, self.expected)
            network.assert_not_called()
        self.assertEqual(target.read_bytes(), b"other bytes")

    def test_missing_file_verify_only_does_not_open_network(self):
        with patch.object(fetcher.urllib.request, "urlopen") as network:
            with self.assertRaises(ValueError):
                fetcher.fetch_one(self.root, self.expected, verify_only=True)
            network.assert_not_called()

    def test_truncated_oversized_and_wrong_hash_responses_leave_no_checkpoint(self):
        for data in (self.data[:-1], self.data + b"excess", b"x" * len(self.data)):
            with self.subTest(data=data), \
                    patch.object(fetcher.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(data, len(self.data))), \
                    patch.object(fetcher.time, "sleep"):
                with self.assertRaises(ValueError):
                    fetcher.fetch_one(self.root, self.expected)
                self.assertFalse((self.root / "nested/model.pth").exists())
                self.assertFalse((self.root / "nested/model.pth.part").exists())

    def test_symlink_parent_and_traversal_are_rejected_before_network(self):
        (self.root / "nested").symlink_to(self.root, target_is_directory=True)
        with patch.object(fetcher.urllib.request, "urlopen") as network:
            with self.assertRaises(ValueError):
                fetcher.fetch_one(self.root, self.expected)
            self.expected["filename"] = "../outside.pth"
            with self.assertRaises(ValueError):
                fetcher.fetch_one(self.root, self.expected)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
