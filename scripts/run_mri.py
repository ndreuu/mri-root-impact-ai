#!/usr/bin/env python3
"""A prepared MRI study → checked four-section physician report."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_tool.runner import run_mri
from mri_tool import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="mri-report " + __version__)
    for name in ("study", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check a saved run without inference or writes")
    args = parser.parse_args()
    try:
        result = run_mri(args.study, args.config, args.output, args.verify_only,
                         progress=lambda stage: print("[mri-report] " + stage, file=sys.stderr, flush=True))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
