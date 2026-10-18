"""Validate project data boundaries; no images are opened and no models are run."""

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
    args = parser.parse_args()
    try:
        study = load_document(args.study)
        results = [load_document(filename) for filename in args.result]
        validate_bundle(study, results)
        report = {
            "ok": True,
            "operation": "validate",
            "study_id": study["study_id"],
            "study_digest": manifest_digest(study),
            "results_validated": len(results),
            "synthetic": study["synthetic"],
            "inference_performed": False,
        }
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
