"""Verify pinned RSNA source and medical checkpoints without importing a model."""

import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess

from .contracts import load_document, manifest_digest


ROOT = Path(__file__).resolve().parents[1]


class RSNAAssetsError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root, name):
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise RSNAAssetsError("Asset paths must be safe relative POSIX paths")
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise RSNAAssetsError("Asset paths must not contain symlinks")
    if not path.is_file():
        raise RSNAAssetsError("Missing regular asset file: " + name)
    return path


def _verify_file(path, expected):
    if (not isinstance(expected, dict)
            or type(expected.get("size_bytes")) is not int or expected["size_bytes"] <= 0
            or not isinstance(expected.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
        raise RSNAAssetsError("Inventory requires positive sizes and pinned SHA-256")
    if path.stat().st_size != expected["size_bytes"]:
        raise RSNAAssetsError("Asset size differs from the pinned inventory")
    actual = file_sha256(path)
    if actual != expected["sha256"]:
        raise RSNAAssetsError("Asset hash differs from the pinned inventory")
    return actual


def verify_rsna_assets(source_root, weights_root, inventory_path=None):
    """Check Git blobs and bytes before AST compilation or weights-only loading.

    Check all tracked Python, not merely files selected for execution. Index flags
    cannot hide changes. No upstream Python import or cached bytecode is used.
    """
    try:
        source, weights = Path(source_root), Path(weights_root)
        if source.is_symlink() or weights.is_symlink():
            raise RSNAAssetsError("Asset roots must not be symlinks")
        source, weights = source.resolve(), weights.resolve()
        if not source.is_dir() or not weights.is_dir():
            raise RSNAAssetsError("External source and weights directories are required")
        inventory = load_document(inventory_path or ROOT / "harness/upstreams/rsna-assets.json")
        if (not isinstance(inventory, dict) or inventory.get("schema_version") != 1
                or inventory.get("kind") != "rsna_asset_inventory"
                or not isinstance(inventory.get("commit"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", inventory["commit"])):
            raise RSNAAssetsError("Invalid pinned RSNA inventory")
        for field in ("source_files", "weights"):
            if not isinstance(inventory.get(field), dict) or not inventory[field]:
                raise RSNAAssetsError("Inventory requires nonempty " + field)

        def git(*args):
            result = subprocess.run(["git", "-C", str(source), *args],
                                    capture_output=True, timeout=20, check=False)
            if result.returncode:
                raise RSNAAssetsError("Pinned RSNA Git source verification failed")
            return result.stdout

        commit = inventory["commit"]
        if git("rev-parse", "HEAD").decode().strip() != commit:
            raise RSNAAssetsError("RSNA source is not the pinned revision")
        names = set(git("ls-tree", "-r", "--name-only", commit).decode().splitlines())
        python = {name for name in names if name.endswith(".py")}
        actual_python = {p.relative_to(source).as_posix() for p in source.rglob("*.py")}
        if python != actual_python:
            raise RSNAAssetsError("Additional or missing Python source files")
        if set(inventory["source_files"]) - names:
            raise RSNAAssetsError("Inventory source is not in the pinned tree")
        code = {}
        for name in sorted(python | set(inventory["source_files"])):
            path = _relative_path(source, name)
            if path.read_bytes() != git("show", commit + ":" + name):
                raise RSNAAssetsError("Source bytes differ from the pinned Git object")
            if name in inventory["source_files"]:
                code[name] = _verify_file(path, inventory["source_files"][name])
            else:
                code[name] = file_sha256(path)
        verified_weights, filenames = {}, set()
        for role, expected in inventory["weights"].items():
            if not isinstance(expected, dict):
                raise RSNAAssetsError("Invalid checkpoint inventory")
            name = expected.get("filename")
            path = _relative_path(weights, name)
            if name in filenames:
                raise RSNAAssetsError("Checkpoint roles must select distinct files")
            filenames.add(name)
            sha = _verify_file(path, expected)
            verified_weights[role] = {"filename": name, "size_bytes": path.stat().st_size,
                                      "sha256": sha}
        return {"source_commit": commit, "source_files_sha256": code,
                "source_tree_sha256": manifest_digest(code), "weights": verified_weights,
                "weights_sha256": manifest_digest(verified_weights)}
    except RSNAAssetsError:
        raise
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise RSNAAssetsError("Cannot verify RSNA assets: " + str(exc)) from exc
