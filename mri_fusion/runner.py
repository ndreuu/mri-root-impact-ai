"""Checked JSON I/O for the separate, dependency-free research fusion engine."""

import hashlib
import json
import math
from pathlib import Path
import re
import time

from mri_root_impact.contracts import manifest_digest, validate_bundle
from mri_root_impact.control_study import write_json_external
from .contracts import FusionContractError, artifact_binding, load_policy, validate_output
from .engine import fuse


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path, files):
    path = Path(path).resolve()
    raw = path.read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FusionContractError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise FusionContractError("Nonfinite JSON constant: " + value)
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise FusionContractError("Invalid input JSON: " + str(exc)) from exc
    digest = hashlib.sha256(raw).hexdigest()
    if str(path) in files and files[str(path)] != digest:
        raise FusionContractError("Input changed while loading: " + str(path))
    files[str(path)] = digest
    return document, digest


def _assert_unchanged(files):
    for path, expected in files.items():
        if file_sha256(path) != expected:
            raise FusionContractError("Input source bytes changed: " + path)


def _pin_file(path, digest, files):
    path = Path(path).resolve()
    if file_sha256(path) != digest or (str(path) in files and files[str(path)] != digest):
        raise FusionContractError("Input changed across frozen validation: " + str(path))
    files[str(path)] = digest


def _equivalent(a, b):
    """Only crosscheck arithmetic permits tiny 3.9/3.12 summation differences."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equivalent(x, y) for x, y in zip(a, b))
    if type(a) is float and type(b) is float:
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-9
    return type(a) is type(b) and a == b


def _frozen_runs(study, spine_path, rsna_path, crosscheck_path, files):
    from scripts.compare_localizers import _read_runs, compare_levels
    from mri_root_impact.spinenet_runner import _png_dimensions
    spine_path, rsna_path, crosscheck_path = (Path(p).resolve() for p in (spine_path, rsna_path, crosscheck_path))
    # Anchor metadata before upstream validators open it, then require their
    # checked inventory and every subsequent read to refer to the same bytes.
    common = {"execution.json", "run.json", "localization.json", "native.json", "spatial.json"}
    for directory, names in ((spine_path, common), (rsna_path, common | {"foraminal.json", "subarticular.json"})):
        for name in sorted(names):
            _load(directory / name, files)
    spine, rsna, sources = _read_runs(study, spine_path, rsna_path)
    for key, directory in (("spinenet", spine_path), ("rsna", rsna_path)):
        for name, digest in sources[key]["files_sha256"].items():
            _pin_file(directory / name, digest, files)
    crosscheck, _ = _load(crosscheck_path / "crosscheck.json", files)
    run, _ = _load(crosscheck_path / "run.json", files)
    if (not isinstance(run, dict) or run.get("kind") != "localizer_crosscheck_run" or run.get("status") != "completed"
            or run.get("study_id") != study["study_id"] or run.get("inference_performed") is not False
            or run.get("numbering_confirmed") is not False or run.get("registration_confirmed") is not False):
        raise FusionContractError("L003 is not a completed unreviewed comparison for this study")
    hashes = run.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"crosscheck.json", "overlay.png"}:
        raise FusionContractError("L003 must bind both required artifacts")
    for name, digest in hashes.items():
        path = crosscheck_path / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise FusionContractError("L003 artifact checksum mismatch")
        _pin_file(path, digest, files)
    rebuilt = compare_levels(study, spine, rsna)
    rebuilt.update(sources=sources, source_files_verified=len(rsna["frames"]), source_bytes_unchanged=True,
                   script_sha256=file_sha256(ROOT / "scripts/compare_localizers.py"))
    if not _equivalent(rebuilt, crosscheck):
        raise FusionContractError("L003 differs from recomputed same-source localizer evidence")
    # Resolve expected overlay references from source keypoints, not arbitrary
    # paths supplied by the saved JSON.
    refs = {tuple(e["sagittal_reference"][k] for k in ("series_id", "image_id", "frame_index")): e["sagittal_reference"]
            for e in crosscheck["levels"] if e["sagittal_reference"] is not None}
    expected = [(r, None) for _, r in sorted(refs.items())]
    expected += [(p["image_ref"], p["native_level"]) for p in rsna["keypoints"] if p["sequence"] == "axial_t2"]
    overlays = run.get("overlays")
    if not isinstance(overlays, list) or len(overlays) != len(expected):
        raise FusionContractError("L003 source overlay coverage mismatch")
    frames = {tuple(f["image_ref"][k] for k in ("series_id", "image_id", "frame_index")): f for f in rsna["frames"]}
    for i, (overlay, (ref, level)) in enumerate(zip(overlays, expected)):
        name = "overlays/frame-{:03d}.png".format(i)
        path = crosscheck_path / name
        frame = frames[tuple(ref[k] for k in ("series_id", "image_id", "frame_index"))]
        if (not isinstance(overlay, dict) or overlay.get("file") != name or overlay.get("image_ref") != ref or overlay.get("native_level") != level
                or overlay.get("header_pixels") != 48 or overlay.get("image_scale") != 1 or path.is_symlink()
                or file_sha256(path) != overlay.get("sha256")
                or _png_dimensions(path) != (frame["columns"], frame["rows"] + 48)):
            raise FusionContractError("L003 source overlay bytes/mapping disagree")
        _pin_file(path, overlay["sha256"], files)
    if _png_dimensions(crosscheck_path / "overlay.png") != (1152, max(1, math.ceil(len(expected) / 3)) * 456):
        raise FusionContractError("L003 montage dimensions mismatch")
    directories = [("spine", spine_path), ("rsna", rsna_path), ("crosscheck", crosscheck_path)]
    documents = []
    for prefix, directory in directories:
        names = ({"crosscheck.json", "run.json"} if prefix == "crosscheck" else
                 {n for n in sources["spinenet" if prefix == "spine" else "rsna"]["files_sha256"] if n.endswith(".json")})
        for name in sorted(names):
            document, digest = _load(directory / name, files)
            documents.append((prefix + "-" + Path(name).stem, document, digest))
        if prefix != "crosscheck":
            execution, _ = _load(directory / "execution.json", files)
            for name, digest in execution["artifacts_sha256"].items():
                path = directory / name
                if file_sha256(path) != digest:
                    raise FusionContractError("Frozen artifact changed after validation")
                _pin_file(path, digest, files)
            for overlay in execution["overlays"]:
                path = directory / overlay["file"]
                if file_sha256(path) != overlay["sha256"]:
                    raise FusionContractError("Frozen overlay changed after validation")
                _pin_file(path, overlay["sha256"], files)
    return documents


def _protect_output(output, input_paths, input_dirs, verify_only):
    if type(verify_only) is not bool:
        raise FusionContractError("verify_only must be an explicit boolean")
    raw = Path(output)
    output = raw.resolve()
    if raw.is_symlink() or any((p / ".git").exists() for p in [output] + list(output.parents)):
        raise FusionContractError("Fusion output must be outside Git and not a symlink")
    if output.exists() and not verify_only:
        raise FusionContractError("Fusion requires a new output directory")
    if verify_only and not output.is_dir():
        raise FusionContractError("Verification requires an existing output directory")
    for path in input_paths:
        path = Path(path).resolve()
        if output == path or output in path.parents or path.parent in output.parents:
            raise FusionContractError("Fusion output overlaps an input file")
    for directory in input_dirs:
        directory = Path(directory).resolve()
        if output == directory or directory in output.parents:
            raise FusionContractError("Fusion output must not modify an upstream run")
    return output


def _identifier(prefix, value):
    name = prefix + "-" + value
    return name if len(name) <= 128 else prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:32]


def _proof(study, document):
    fields = {"schema_version", "kind", "evidence_id", "study_id", "study_digest", "synthetic", "description"}
    if (not isinstance(document, dict) or set(document) != fields or document["schema_version"] != "1.0"
            or document["kind"] != "fusion_review_evidence" or document["study_id"] != study["study_id"]
            or document["study_digest"] != manifest_digest(study) or type(document["synthetic"]) is not bool
            or document["synthetic"] != study["synthetic"]
            or not isinstance(document["evidence_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", document["evidence_id"])
            or not isinstance(document["description"], str) or not document["description"].strip()):
        raise FusionContractError("Review evidence must match the study and its synthetic flag")


def render_report(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Stage I — исследовательский MRI fusion", "",
             "Применена policy `{}`. Нового MRI inference нет.".format(report["policy_id"]), "",
             "Кандидаты по двум стенозным правилам: **{}**. Неопределенные ветви: **{} из 20**.".format(
                 len(report["root_candidates"]), report["coverage"]["indeterminate"]), "",
             "Отдельно сохранены находки без установленного target: **{}**; исходные оценки дисков: **{}**.".format(
                 len(report["unresolved_native_evidence"]), len(report["raw_disc_evidence"])), "",
             "Отсутствие кандидатов при неопределенных входах не является отрицательной находкой. "
             "Кандидат не подтверждает компрессию или радикулопатию. Severe-only — исследовательский порог.", "",
             "| Native уровень | Сторона | Зона | Native grade | Решение | Кандидат | Причины |",
             "|---|---|---|---|---|---|---|"]
    for row in report["branches"]:
        grades = sorted({e["observation"]["label"] for e in row["evidence"] if e["observation"] is not None})
        root = row["candidate_root"]
        candidate = "—" if root is None else "{} {} ({})".format(root["side"], root["root"], root["role"])
        values = [row["target"]["level"], row["target"]["side"], row["target"]["zone"],
                  ", ".join(grades) or "нет оценки", row["decision"], candidate, ", ".join(row["reason_codes"])]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += ["", "Контекст грыжи: `not_evaluated`; причина стеноза грыже не приписывается. "
              "Согласие локализаторов и неподтвержденное identity-LPS не заменяют review.", "",
              "Policy SHA-256: `{}`.".format(report["policy_sha256"]),
              "Review SHA-256: `{}`.".format(report["review_context_sha256"] or "не предоставлен"),
              "Полная provenance, native probabilities и source refs сохранены в `fusion.json`.", ""]
    return "\n".join(lines)


def _code_hashes():
    paths = (sorted((ROOT / "mri_fusion").glob("*.py"))
             + sorted((ROOT / "mri_root_impact").glob("*.py"))
             + [ROOT / "scripts/fuse_mri.py", ROOT / "scripts/compare_localizers.py"])
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in paths}


def validate_saved_run(study, output, expected, files):
    metadata = {}
    run, _ = _load(output / "run.json", metadata)
    if (not isinstance(run, dict) or run.get("kind") != "mri_fusion_run" or run.get("status") != "completed"
            or run.get("study_id") != study["study_id"] or run.get("study_digest") != manifest_digest(study)
            or run.get("inference_performed") is not False or run.get("fusion_performed") is not True
            or run.get("source_bytes_unchanged") is not True or run.get("code_files_sha256") != _code_hashes()
            or run.get("input_files_sha256") != files):
        raise FusionContractError("Fusion run metadata is not bound to these unchanged inputs/code")
    hashes = run.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"fusion.json", "report.md"}:
        raise FusionContractError("Fusion requires both output artifacts")
    for name, digest in hashes.items():
        if (output / name).is_symlink() or file_sha256(output / name) != digest:
            raise FusionContractError("Fusion output checksum mismatch")
    actual, _ = _load(output / "fusion.json", metadata)
    validate_output(study, actual)
    if manifest_digest(actual) != manifest_digest(expected):
        raise FusionContractError("Saved fusion differs from recomputed source evidence")
    if (output / "report.md").read_text(encoding="utf-8") != render_report(expected):
        raise FusionContractError("Human report differs from checked fusion JSON")
    _assert_unchanged(files)
    return actual


def run_fusion(study_path, output, result_paths=(), review_path=None, evidence_paths=(),
               spinenet_run=None, rsna_run=None, crosscheck_run=None, policy_path=None, verify_only=False):
    try:
        return _run(study_path, output, result_paths, review_path, evidence_paths,
                    spinenet_run, rsna_run, crosscheck_run, policy_path, verify_only)
    except FusionContractError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise FusionContractError("Fusion input/run error: " + str(exc)) from exc


def _run(study_path, output, result_paths, review_path, evidence_paths, spinenet_run, rsna_run, crosscheck_run, policy_path, verify_only):
    started = time.monotonic()
    frozen = [p for p in (spinenet_run, rsna_run, crosscheck_run) if p is not None]
    if frozen and len(frozen) != 3:
        raise FusionContractError("Frozen mode requires all three upstream run directories")
    from .contracts import POLICY_PATH
    policy_path = Path(policy_path or POLICY_PATH).resolve()
    paths = [study_path, policy_path] + list(result_paths) + list(evidence_paths) + ([] if review_path is None else [review_path])
    output = _protect_output(output, paths, frozen, verify_only)
    files = {}
    study, _ = _load(study_path, files)
    validate_bundle(study, [])
    _, policy_sha = load_policy(policy_path)
    files[str(policy_path)] = policy_sha
    documents = _frozen_runs(study, *frozen, files) if frozen else []
    for path in result_paths:
        document, digest = _load(path, files)
        if not isinstance(document, dict) or document.get("kind") != "module_result":
            raise FusionContractError("--result requires a C001 module_result")
        documents.append(("result", document, digest))
    for path in evidence_paths:
        document, digest = _load(path, files)
        _proof(study, document)
        documents.append((_identifier("evidence", document["evidence_id"]), document, digest))
    results, bindings, by_id = [], [], {}
    for name, document, digest in documents:
        if document["kind"] == "module_result":
            results.append(document)
            name = _identifier("result", document["result_id"])
        binding = artifact_binding(name, document, digest)
        if name in by_id:
            if by_id[name] != binding:
                raise FusionContractError("Artifact identity has conflicting file/document content")
            continue
        by_id[name] = binding
        bindings.append(binding)
    review, review_sha = (None, None) if review_path is None else _load(review_path, files)
    # Real images remain outside Git and must have pinned, unchanged source bytes.
    if not study["synthetic"]:
        for series in study["series"]:
            for image in series["images"]:
                if image["sha256"] is None or file_sha256(image["path"]) != image["sha256"]:
                    raise FusionContractError("Real image source hash is missing or changed")
                files[str(Path(image["path"]).resolve())] = image["sha256"]
    # Images and frozen artifacts are discovered after the initial path check.
    # Keep new artifacts out of every source directory, including DICOM folders.
    _protect_output(output, list(files), frozen, verify_only)
    expected = fuse(study, results, bindings, review, review_sha, policy_path)
    validate_output(study, expected)
    if verify_only:
        validate_saved_run(study, output, expected, files)
    else:
        _assert_unchanged(files)
        output.mkdir(parents=True)
        try:
            write_json_external(expected, output / "fusion.json")
            (output / "report.md").write_text(render_report(expected), encoding="utf-8")
            _assert_unchanged(files)
            report = {"schema_version": "1.0", "kind": "mri_fusion_run", "status": "completed",
                      "study_id": study["study_id"], "study_digest": manifest_digest(study),
                      "inference_performed": False, "fusion_performed": True,
                      "duration_seconds": time.monotonic() - started, "source_bytes_unchanged": True,
                      "input_files_sha256": files, "code_files_sha256": _code_hashes(),
                      "parameters": {"input_mode": "frozen_mri" if frozen else "c001_files", "review_provided": review_path is not None},
                      "artifacts_sha256": {name: file_sha256(output / name) for name in ("fusion.json", "report.md")}}
            write_json_external(report, output / "run.json")
            validate_saved_run(study, output, expected, files)
        except Exception as exc:
            write_json_external({"kind": "mri_fusion_run", "status": "error", "inference_performed": False,
                                 "fusion_performed": False, "error": str(exc)}, output / "run.json")
            raise
    return {"ok": True, "operation": "verify-fusion" if verify_only else "fuse-mri", "output": str(output),
            "inference_performed": False, "fusion_performed": True, "coverage": expected["coverage"],
            "named_root_candidates": len(expected["root_candidates"]), "global_root_clearance": False}
