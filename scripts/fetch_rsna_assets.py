#!/usr/bin/env python3
"""Fetch only pinned RSNA medical checkpoints; never load or execute them."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_root_impact.contracts import load_document
from mri_root_impact.rsna_assets import _relative_path, _verify_file


def fetch_one(root, expected, verify_only=False):
    name = expected["filename"]
    # Validate the relative name before creating directories or opening a URL.
    if (not isinstance(name, str) or name.startswith("/") or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe checkpoint path")
    target = root / name
    if target.exists() or target.is_symlink():
        _verify_file(_relative_path(root, name), expected)
        return {"filename": name, "sha256": expected["sha256"], "status": "verified_existing"}
    if verify_only:
        raise ValueError("Missing checkpoint: " + name)
    url = expected["source_url"]
    if not url.startswith("https://www.kaggle.com/api/v1/datasets/download/akinosora/rsna2024-lsdc-3rd-models-pub/") or not url.endswith("?datasetVersionNumber=2"):
        raise ValueError("Only the pinned public versioned model source is supported")
    # Existing parent symlinks must not redirect writes outside the chosen root.
    parent = root
    for part in name.split("/")[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError("Checkpoint parent must not be a symlink")
        parent.mkdir(exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists() or partial.is_symlink():
        raise ValueError("A partial checkpoint already exists; use a fresh output directory")
    size = expected["size_bytes"]
    for attempt in range(3):
        try:
            offset = 0
            with partial.open("xb") as handle:
                while offset < size:
                    headers = {"Accept-Encoding": "identity", "Cache-Control": "no-cache"}
                    if offset:
                        headers["Range"] = "bytes={}-{}".format(offset, size - 1)
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                        length = size
                        if response.status == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if not match:
                                raise ValueError("Missing Content-Range")
                            start, end, total = map(int, match.groups())
                            if start != offset or total != size or not start <= end < size:
                                raise ValueError("Unexpected checkpoint range")
                            length = end - start + 1
                        elif response.status != 200 or offset:
                            raise ValueError("Unexpected HTTP status")
                        if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) != length:
                            raise ValueError("Unexpected checkpoint Content-Length")
                        remaining = length
                        while remaining:
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("Truncated checkpoint")
                            handle.write(block)
                            remaining -= len(block)
                        if response.read(1):
                            raise ValueError("Checkpoint exceeds the expected boundary")
                        offset += length
                handle.flush()
                os.fsync(handle.fileno())
            _verify_file(partial, expected)
            if target.exists():
                raise ValueError("Checkpoint destination appeared during download")
            partial.replace(target)
            return {"filename": name, "sha256": expected["sha256"], "status": "downloaded_verified"}
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.output.is_symlink():
            raise ValueError("Output directory must not be a symlink")
        output = args.output.resolve()
        if output.is_relative_to(ROOT):
            raise ValueError("Weights must remain outside the repository")
        inventory = load_document(ROOT / "harness/upstreams/rsna-assets.json")
        output.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            files = list(pool.map(lambda item: fetch_one(output, item, args.verify_only), inventory["weights"].values()))
        print(json.dumps({"ok": True, "files": files, "inference_performed": False}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "inference_performed": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
