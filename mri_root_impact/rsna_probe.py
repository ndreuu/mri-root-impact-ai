"""Explicit CPU checkpoint compatibility probe using synthetic tensors only."""

import argparse
import ast
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from .contracts import load_document
from .rsna_assets import ROOT, file_sha256, verify_rsna_assets


MODEL_CLASSES = ("GeM", "RSNA2024TransformerBlock", "RSNA2024AttentionNet",
                 "RSNA2024KeypointNeck", "RSNA2024KeypointNetTest")
IGNORABLE_LOSS_KEYS = frozenset({"target_loss.weight", "target_loss.ce_loss.weight",
                               "target_loss.ce_loss.class_weights", "target_loss.ce_loss.distance_matrix",
                               "level_fc.weight", "level_fc.bias", "target_loss.condition_weight",
                               "target_loss.slice_ce_loss.weight"})


def extract_model_definitions(source_text):
    """Return only reviewed class definitions, excluding imports/top-level calls."""
    parsed = ast.parse(source_text)
    selected = [node for node in parsed.body if isinstance(node, ast.ClassDef)
                and node.name in MODEL_CLASSES]
    if tuple(node.name for node in selected) != MODEL_CLASSES:
        raise ValueError("Pinned model class definitions are missing, duplicated or reordered")
    return ast.Module(body=selected, type_ignores=[])


def runtime_versions():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("RSNA probe requires the pinned Python 3.12 runtime")
    packages = {}
    for line in (ROOT / "requirements-rsna.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, pinned = line.split("==")
            actual = version(name)
            if actual != pinned:
                raise ValueError("RSNA runtime differs from requirements-rsna.txt: " + name)
            packages[name] = actual
    return {"python": sys.version.split()[0], "packages": packages, "device": "cpu"}


def probe(source, weights, threads=2):
    started = time.monotonic()
    assets = verify_rsna_assets(source, weights)
    runtime = runtime_versions()
    inventory = load_document(ROOT / "harness/upstreams/rsna-assets.json")
    import torch
    import timm
    from omegaconf import OmegaConf
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    tree = extract_model_definitions((source / "src/predict.py").read_text())
    definitions = {"__name__": "_pinned_rsna_models", "torch": torch, "nn": torch.nn,
                   "F": torch.nn.functional, "timm": timm}
    exec(compile(tree, str(source / "src/predict.py") + "::<model_classes_only>", "exec"), definitions)
    reports = {}
    for role, expected in inventory["weights"].items():
        cfg = OmegaConf.to_container(OmegaConf.load(source / expected["config"]), resolve=False)
        args = cfg["model"]
        args.pop("name", None)
        args["timm_model"]["pretrained"] = False
        before_load = time.monotonic()
        with torch.no_grad():
            model = definitions[expected["model_class"]](**args)
        checkpoint = weights / expected["filename"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) for k in state):
            raise ValueError("Expected a tensor state_dict, not an arbitrary checkpoint object")
        removed = sorted(set(state) & IGNORABLE_LOSS_KEYS)
        state = {key: value for key, value in state.items() if key not in IGNORABLE_LOSS_KEYS}
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("Checkpoint contains unexpected nontensor values")
        model.load_state_dict(state, strict=True)
        model.eval()
        shapes = expected["synthetic_input_shapes"]
        inputs = [torch.randn(*shape) for shape in shapes]
        forward_started = time.monotonic()
        with torch.inference_mode():
            output = model(*inputs)["logits"]
        if list(output.shape) != expected["expected_output_shape"] or not torch.isfinite(output).all().item():
            raise ValueError("Checkpoint synthetic output shape/finite check failed")
        reports[role] = {"checkpoint_sha256": file_sha256(checkpoint),
                         "state_dict_tensor_count": len(state), "removed_training_only_keys": removed,
                         "strict_load": True, "synthetic_input_shapes": shapes,
                         "output_shape": list(output.shape), "finite": True,
                         "synthetic_forward_seconds": time.monotonic() - forward_started,
                         "load_and_forward_seconds": time.monotonic() - before_load}
        del model, state, output, inputs
    return {"kind": "rsna_checkpoint_compatibility_probe", "status": "passed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_kind": "synthetic_random_tensors", "patient_images_read": False,
            "synthetic_forward_performed": True, "mri_inference_performed": False,
            "root_impacts": None, "runtime": runtime, "threads": threads,
            "assets": assets, "models": reports, "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        if not 1 <= args.threads <= 16:
            raise ValueError("threads must be between 1 and 16")
        output = args.output.resolve()
        if output.exists() or output.is_relative_to(ROOT):
            raise ValueError("Probe output must be a new file outside the repository")
        if any(output.is_relative_to(root.resolve()) for root in (args.source, args.weights)):
            raise ValueError("Probe output must not be inside source/weights")
        report = probe(args.source.resolve(), args.weights.resolve(), args.threads)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"ok": True, "report": str(output), "mri_inference_performed": False}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "mri_inference_performed": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
