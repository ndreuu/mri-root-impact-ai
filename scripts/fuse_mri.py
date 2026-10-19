#!/usr/bin/env python3
"""Run or verify research MRI fusion against frozen or explicit C001 inputs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_fusion.contracts import FusionContractError
from mri_fusion.runner import run_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--review", type=Path)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--policy", type=Path)
    for name in ("spinenet-run", "rsna-run", "crosscheck-run"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run_fusion(args.study, args.output, args.result, args.review, args.evidence,
                            args.spinenet_run, args.rsna_run, args.crosscheck_run, args.policy, args.verify_only)
        print(json.dumps(result, indent=2))
    except FusionContractError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
