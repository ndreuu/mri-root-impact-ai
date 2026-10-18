"""Validate contracts and verify local DICOM inputs. No models are run."""

import argparse
import json
from pathlib import Path
import sys

from .contracts import load_document, manifest_digest, validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate study and optional module results")
    validate.add_argument("--study", type=Path, required=True)
    validate.add_argument("--result", type=Path, action="append", default=[])
    prepare = commands.add_parser("prepare-rsna", help="Build and verify an external RSNA control manifest")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True, help="External directory for study.json and preflight.json")
    prepare.add_argument("--allow-missing-modality", action="store_true", help="Record dataset-declared MR when the DICOM Modality tag is absent; never accept another modality")
    preflight = commands.add_parser("preflight", help="Hash and decode all DICOM inputs in a study manifest")
    preflight.add_argument("--study", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True, help="External JSON report path")
    preflight.add_argument("--allow-missing-modality", action="store_true", help="Explicit policy for stripped DICOM metadata; recorded in the report")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            study = load_document(args.study)
            results = [load_document(filename) for filename in args.result]
            validate_bundle(study, results)
            report = {
                "ok": True, "operation": "validate", "study_id": study["study_id"],
                "study_digest": manifest_digest(study), "results_validated": len(results),
                "synthetic": study["synthetic"], "inference_performed": False,
            }
        else:
            from .control_study import prepare_rsna_study, write_json_external
            from .preflight import verify_study
            if args.command == "prepare-rsna":
                study = prepare_rsna_study(args.data_root)
                verification = verify_study(study, allow_missing_modality=args.allow_missing_modality)
                manifest_path = write_json_external(study, args.output / "study.json")
                report_path = write_json_external(verification, args.output / "preflight.json")
                report = {"ok": True, "manifest": str(manifest_path), "report": str(report_path), **verification}
            else:
                if args.study.resolve() == args.output.resolve():
                    raise ValueError("Preflight report must not overwrite the input manifest")
                study = load_document(args.study)
                from .contracts import validate_study
                validate_study(study)
                if any(Path(image["path"]).resolve() == args.output.resolve()
                       for series in study["series"] for image in series["images"]):
                    raise ValueError("Preflight report must not overwrite an input image")
                verification = verify_study(study, allow_missing_modality=args.allow_missing_modality)
                report_path = write_json_external(verification, args.output)
                report = {"ok": True, "report": str(report_path), **verification}
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
