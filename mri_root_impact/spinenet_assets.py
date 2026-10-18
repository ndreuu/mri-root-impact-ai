"""Verify pinned external SpineNet code and every checkpoint before importing it."""

import hashlib
from pathlib import Path
import subprocess

from .contracts import load_document, manifest_digest


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(source):
    source = Path(source).resolve()
    inventory = load_document(ROOT / "harness/upstreams/spinenet.json")
    def git(*args):
        process = subprocess.run(["git", "-C", str(source), *args], capture_output=True,
                                 text=True, timeout=15, check=False)
        if process.returncode:
            raise ValueError("Pinned SpineNet source verification failed")
        return process.stdout.strip()
    if git("rev-parse", "HEAD") != inventory["commit"]:
        raise ValueError("SpineNet is not the pinned revision")
    git("diff", "--quiet", "HEAD", "--")
    # Read the pinned tree, not index flags such as assume-unchanged/skip-worktree.
    tracked = [name for name in git("ls-tree", "-r", "--name-only", inventory["commit"]).splitlines()
               if name.endswith(".py") or name == "LICENCE.md"]
    expected_python = {name for name in tracked if name.endswith(".py")}
    actual_python = {str(path.relative_to(source)) for path in source.rglob("*.py")}
    if expected_python != actual_python:
        raise ValueError("SpineNet contains additional or missing Python source files")
    code = {}
    for name in tracked:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Pinned source files must be regular files")
        pinned_bytes = subprocess.check_output(["git", "-C", str(source), "show", inventory["commit"] + ":" + name], timeout=15)
        if path.read_bytes() != pinned_bytes:
            raise ValueError("SpineNet source bytes differ from the pinned Git object")
        code[name] = file_sha256(path)
    weights = {}
    for relative, expected in inventory["files"].items():
        path = source / "spinenet/weights" / relative
        if path.is_symlink() or not path.is_file() or sorted(path.parent.glob("*.pt")) != [path]:
            raise ValueError("SpineNet checkpoint selection must be unambiguous")
        actual = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
            raise ValueError("SpineNet checkpoint differs from the pinned inventory")
        weights[relative] = actual
    return {
        "source_commit": inventory["commit"], "source_files_sha256": code,
        "source_tree_sha256": manifest_digest(code), "weights": weights,
        "weights_sha256": manifest_digest(weights),
        "weights_digest_definition": "SHA-256 of canonical JSON mapping relative filenames to sha256 and size_bytes",
        "license_url": inventory["license_url"],
    }
